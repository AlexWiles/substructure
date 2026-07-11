use async_trait::async_trait;
use tokio::sync::{broadcast, mpsc};

use crate::protocol::TokenDelta;

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
