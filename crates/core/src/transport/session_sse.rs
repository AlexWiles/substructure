use axum::http::HeaderMap;
use axum::response::sse::Event as SseEvent;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::api::v1::RUN_DONE_EVENT;
use crate::event_store::Seq;
use crate::protocol::TokenDelta;
use crate::session::SessionEvent;

/// The resume cursor for a stream request. A reconnecting EventSource
/// re-requests the original URL — whose `after_seq` is stale by then — and
/// sends the id of the last frame it saw as a `Last-Event-ID` header, so the
/// header, when present and parseable, wins over the query param.
pub fn resume_cursor(headers: &HeaderMap, after_seq: Option<u64>) -> Option<Seq> {
    let last_event_id = headers
        .get("last-event-id")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok());
    last_event_id.or(after_seq).map(Seq)
}

pub fn run_event_stream(
    mut event_rx: mpsc::Receiver<SessionEvent>,
    shutdown: CancellationToken,
) -> mpsc::Receiver<SseEvent> {
    let (tx, rx) = mpsc::channel(64);
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = shutdown.cancelled() => return,
                _ = tx.closed() => return,
                event = event_rx.recv() => {
                    let Some(event) = event else { return };
                    let ends = event.ends_run();
                    let sse = SseEvent::default()
                        .id(event.seq.to_string())
                        .event(event.payload_type())
                        .data(serde_json::to_string(&event).unwrap_or_default());
                    if tx.send(sse).await.is_err() {
                        return;
                    }
                    if ends {
                        let _ = tx.send(SseEvent::default().event(RUN_DONE_EVENT).data("")).await;
                        return;
                    }
                }
            }
        }
    });
    rx
}

/// Terminates when `event_rx` closes (Turn scopes auto-close on
/// `turn.completed`). Delta-side closure is ignored — the transport outlives
/// any single turn.
pub fn merge_session_stream(
    mut event_rx: mpsc::Receiver<SessionEvent>,
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
                _ = out_tx.closed() => return,
                ev = event_rx.recv() => match ev {
                    Some(event) => {
                        let event_type = event.payload_type();
                        let data = serde_json::to_string(&event).unwrap_or_default();
                        let sse = SseEvent::default()
                            .id(event.seq.to_string())
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
