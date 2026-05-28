use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc};

/// Transient LLM token delta payload. Travels through the
/// [`TokenDeltaTransport`] and is emitted on the SSE stream inside an
/// envelope that mirrors the persisted [`Event`](crate::event_store::Event)
/// shape — same `event_type`, `aggregate_id`, `occurred_at`, `payload`
/// discrimination. Sequence is absent because transient deltas are not
/// persisted and cannot be resumed via `sequence_after`; reconnecting
/// clients see the final assembled content via the persisted
/// `llm.call.completed` event.
///
/// The `(call_id, attempt)` pair joins back to the `llm.call.requested`
/// event the client already received; that's how the frontend folds
/// deltas into the in-progress assistant message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenDelta {
    /// Root session — used as the transport routing key.
    pub root_session_id: String,
    /// Session that owns the LLM call (may be a sub-agent of root).
    pub session_id: String,
    pub agent_id: String,
    /// Present when the call was issued inside a turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    pub call_id: String,
    pub attempt: u32,
    /// Monotonic per-call sequence (0-based).
    pub seq: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Pluggable transport for live token deltas. The in-memory impl is fine for
/// single-node; a distributed impl (Redis pub/sub, NATS, etc.) routes deltas
/// across nodes so that an LLM call running on node A can stream tokens to a
/// frontend connected to node B.
///
/// Deltas are lossy by design — slow subscribers may drop intermediate deltas
/// and reconnecting clients do not see deltas already emitted. The
/// authoritative final content always arrives via the persisted
/// `llm.call.completed` event.
#[async_trait]
pub trait TokenDeltaTransport: Send + Sync + 'static {
    /// Fire-and-forget publish. Routing key is `delta.root_session_id`.
    async fn publish(&self, delta: TokenDelta);

    /// Subscribe to deltas for a single root session.
    async fn subscribe(&self, root_session_id: &str) -> mpsc::Receiver<TokenDelta>;
}

const IN_MEMORY_BROADCAST_CAPACITY: usize = 1024;
const PER_SUBSCRIBER_CAPACITY: usize = 64;

/// Single-node implementation backed by a tokio broadcast channel. Every
/// publish fans out to every subscriber; the subscribe task filters by
/// `root_session_id` before forwarding to the per-subscriber mpsc.
pub struct InMemoryTokenDeltaTransport {
    tx: broadcast::Sender<TokenDelta>,
}

impl InMemoryTokenDeltaTransport {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(IN_MEMORY_BROADCAST_CAPACITY);
        Self { tx }
    }
}

impl Default for InMemoryTokenDeltaTransport {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TokenDeltaTransport for InMemoryTokenDeltaTransport {
    async fn publish(&self, delta: TokenDelta) {
        let _ = self.tx.send(delta);
    }

    async fn subscribe(&self, root_session_id: &str) -> mpsc::Receiver<TokenDelta> {
        let mut rx = self.tx.subscribe();
        let (out_tx, out_rx) = mpsc::channel(PER_SUBSCRIBER_CAPACITY);
        let target = root_session_id.to_owned();
        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(delta) => {
                        if delta.root_session_id != target {
                            continue;
                        }
                        if out_tx.send(delta).await.is_err() {
                            return;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => return,
                }
            }
        });
        out_rx
    }
}
