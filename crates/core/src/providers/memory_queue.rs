use tokio::sync::{mpsc, Mutex};

pub struct InMemoryQueue<T> {
    tx: mpsc::UnboundedSender<T>,
    rx: Mutex<mpsc::UnboundedReceiver<T>>,
}

impl<T: Send + 'static> InMemoryQueue<T> {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        Self {
            tx,
            rx: Mutex::new(rx),
        }
    }

    pub async fn enqueue(&self, item: T) -> Result<(), String> {
        self.tx.send(item).map_err(|_| "queue closed".to_string())
    }

    pub async fn dequeue(&self) -> Option<T> {
        self.rx.lock().await.recv().await
    }
}
