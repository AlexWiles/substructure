//! Best-effort fan-out of freshly appended events.
//!
//! Lossy by contract: a tap may miss batches under lag, and a distributed bus
//! may drop or delay notifications arbitrarily. Correctness always comes from
//! cursor replay against the [`EventStore`](super::EventStore); batch payloads
//! exist to cut latency, never to guarantee delivery.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::broadcast;

use crate::runtime::session::SessionEvent;

pub trait EventBus: Send + Sync {
    /// Hand a committed batch to every live tap. Must not block the appender.
    fn publish(&self, batch: Arc<Vec<SessionEvent>>);

    fn subscribe(&self) -> EventTap;
}

/// The receiving half of an [`EventBus`] subscription.
pub struct EventTap(Box<dyn TapSource>);

impl EventTap {
    pub fn new(source: impl TapSource + 'static) -> Self {
        Self(Box::new(source))
    }

    /// The next batch, `None` once the bus is gone. Lag is absorbed here: a
    /// slow tap skips ahead rather than surfacing an error.
    pub async fn recv(&mut self) -> Option<Arc<Vec<SessionEvent>>> {
        self.0.recv().await
    }
}

#[async_trait]
pub trait TapSource: Send {
    async fn recv(&mut self) -> Option<Arc<Vec<SessionEvent>>>;
}

/// In-process bus over `tokio::sync::broadcast` — the single-node default.
pub struct BroadcastBus {
    tx: broadcast::Sender<Arc<Vec<SessionEvent>>>,
}

impl BroadcastBus {
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }
}

impl Default for BroadcastBus {
    fn default() -> Self {
        Self::new(1024)
    }
}

impl EventBus for BroadcastBus {
    fn publish(&self, batch: Arc<Vec<SessionEvent>>) {
        let _ = self.tx.send(batch);
    }

    fn subscribe(&self) -> EventTap {
        EventTap::new(BroadcastTap(self.tx.subscribe()))
    }
}

struct BroadcastTap(broadcast::Receiver<Arc<Vec<SessionEvent>>>);

#[async_trait]
impl TapSource for BroadcastTap {
    async fn recv(&mut self) -> Option<Arc<Vec<SessionEvent>>> {
        loop {
            match self.0.recv().await {
                Ok(batch) => return Some(batch),
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => return None,
            }
        }
    }
}
