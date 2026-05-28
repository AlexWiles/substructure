//! Shared SSE plumbing for session event streaming. Fans in persisted
//! events from the event store with transient `llm.token.delta` events
//! from the [`TokenDeltaTransport`], emitting them as SSE messages with
//! a uniform Event-shaped envelope.

use axum::response::sse::Event as SseEvent;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::event_store::Event;
use crate::llm::TokenDelta;

/// Spawn a fan-in task that forwards persisted events and transient token
/// deltas onto a single SSE channel. Terminates when the event side closes
/// (mirrors the existing Turn-scoped completion semantics — on
/// `turn.completed` the session subscription's sender drops, the merged
/// stream ends, and the SSE response closes so client `for-await` loops
/// can exit).
///
/// When `scope_turn_id` is `Some`, deltas whose `turn_id` doesn't match
/// are filtered out so a Turn-scoped subscriber doesn't see deltas from
/// other concurrent calls in the same root session.
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

/// Wrap a transient [`TokenDelta`] in an envelope mirroring the persisted
/// `Event` shape so SSE consumers can branch on `payload.type` uniformly.
/// `sequence` is omitted because transient deltas are not persisted.
fn token_delta_to_sse(delta: TokenDelta) -> SseEvent {
    let envelope = serde_json::json!({
        "aggregate_type": "session",
        "aggregate_id": delta.session_id,
        "event_type": "llm.token.delta",
        "occurred_at": chrono::Utc::now(),
        "payload": {
            "type": "llm.token.delta",
            "call_id": delta.call_id,
            "attempt": delta.attempt,
            "seq": delta.seq,
            "agent_id": delta.agent_id,
            "turn_id": delta.turn_id,
            "text": delta.text,
            "finish_reason": delta.finish_reason,
        },
    });
    SseEvent::default()
        .event("llm.token.delta")
        .data(envelope.to_string())
}
