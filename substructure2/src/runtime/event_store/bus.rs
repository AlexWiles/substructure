use std::sync::Arc;

use tokio::sync::broadcast;

use super::store::Event;

pub struct EventBus {
    tx: broadcast::Sender<Arc<Vec<Event>>>,
}

impl EventBus {
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }

    pub fn publish(&self, events: Vec<Event>) {
        let _ = self.tx.send(Arc::new(events));
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Arc<Vec<Event>>> {
        self.tx.subscribe()
    }
}
