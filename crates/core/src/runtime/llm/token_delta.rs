use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenDelta {
    /// Tenant isolation key — subscribers must match.
    pub tenant_id: String,
    /// Transport routing key.
    pub root_session_id: String,
    /// May be a sub-agent of root.
    pub session_id: String,
    pub agent_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    pub call_id: String,
    pub attempt: u32,
    /// Per-call counter, distinct from event-store sequence.
    pub seq: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[async_trait]
pub trait TokenDeltaTransport: Send + Sync + 'static {
    async fn publish(&self, delta: TokenDelta);
    async fn subscribe(&self, tenant_id: &str, root_session_id: &str)
        -> mpsc::Receiver<TokenDelta>;
}

const IN_MEMORY_BROADCAST_CAPACITY: usize = 1024;
const PER_SUBSCRIBER_CAPACITY: usize = 64;

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

    async fn subscribe(
        &self,
        tenant_id: &str,
        root_session_id: &str,
    ) -> mpsc::Receiver<TokenDelta> {
        let mut rx = self.tx.subscribe();
        let (out_tx, out_rx) = mpsc::channel(PER_SUBSCRIBER_CAPACITY);
        let tenant = tenant_id.to_owned();
        let target = root_session_id.to_owned();
        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(delta) => {
                        if delta.tenant_id != tenant || delta.root_session_id != target {
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
