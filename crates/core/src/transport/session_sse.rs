use axum::response::sse::Event as SseEvent;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::event_store::Event;
use crate::llm::TokenDelta;

/// Terminates when `event_rx` closes (Turn scopes auto-close on
/// `turn.completed`). Delta-side closure is ignored — the transport outlives
/// any single turn.
pub fn merge_session_stream(
    mut event_rx: mpsc::Receiver<Event>,
    mut delta_rx: mpsc::Receiver<TokenDelta>,
    scope_turn_id: Option<String>,
    shutdown: CancellationToken,
) -> mpsc::Receiver<SseEvent> {
    let (out_tx, out_rx) = mpsc::channel(64);
    tokio::spawn(async move {
        loop {
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => return,
                ev = event_rx.recv() => match ev {
                    Some(event) => {
                        let event_type = event.payload_type().to_owned();
                        let data = serde_json::to_string(&event).unwrap_or_default();
                        let sse = SseEvent::default()
                            .id(event.sequence.to_string())
                            .event(event_type)
                            .data(data);
                        if out_tx.send(sse).await.is_err() {
                            return;
                        }
                    }
                    None => return,
                },
                delta = delta_rx.recv() => match delta {
                    Some(delta) => {
                        if let Some(ref scope) = scope_turn_id {
                            if delta.turn_id.as_deref() != Some(scope.as_str()) {
                                continue;
                            }
                        }
                        if out_tx.send(token_delta_to_sse(delta)).await.is_err() {
                            return;
                        }
                    }
                    None => continue,
                },
            }
        }
    });
    out_rx
}

fn token_delta_to_sse(delta: TokenDelta) -> SseEvent {
    let payload = serde_json::json!({
        "type": "llm.token.delta",
        "session_id": delta.session_id,
        "agent_id": delta.agent_id,
        "turn_id": delta.turn_id,
        "call_id": delta.call_id,
        "attempt": delta.attempt,
        "seq": delta.seq,
        "text": delta.text,
        "reasoning": delta.reasoning,
        "tool_calls": delta.tool_calls,
        "finish_reason": delta.finish_reason,
    });
    SseEvent::default()
        .event("llm.token.delta")
        .data(payload.to_string())
}
