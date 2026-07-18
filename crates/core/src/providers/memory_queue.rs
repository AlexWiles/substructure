use async_trait::async_trait;
use tokio::sync::{mpsc, Mutex};

use crate::shard::shard_of;

// ---------------------------------------------------------------------------
// Original unsharded queue (used by WorkerQueue, etc.)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Generic sharded task queue
// ---------------------------------------------------------------------------

#[async_trait]
pub trait TaskQueue<T: Send>: Send + Sync {
    async fn enqueue(&self, shard_key: &str, task: T) -> Result<(), String>;
    fn subscribe(&self) -> mpsc::UnboundedReceiver<T>;
}

pub struct ShardedInMemoryQueue<T> {
    shard_count: u32,
    senders: Vec<mpsc::UnboundedSender<T>>,
    receivers: Mutex<Vec<mpsc::UnboundedReceiver<T>>>,
}

impl<T: Send + 'static> ShardedInMemoryQueue<T> {
    pub fn new(shard_count: u32) -> Self {
        assert!(shard_count > 0, "shard_count must be > 0");
        let mut senders = Vec::with_capacity(shard_count as usize);
        let mut receivers = Vec::with_capacity(shard_count as usize);
        for _ in 0..shard_count {
            let (tx, rx) = mpsc::unbounded_channel();
            senders.push(tx);
            receivers.push(rx);
        }
        Self {
            shard_count,
            senders,
            receivers: Mutex::new(receivers),
        }
    }
}

#[async_trait]
impl<T: Send + 'static> TaskQueue<T> for ShardedInMemoryQueue<T> {
    async fn enqueue(&self, shard_key: &str, task: T) -> Result<(), String> {
        let shard = shard_of(shard_key, self.shard_count);
        self.senders[shard]
            .send(task)
            .map_err(|_| "queue closed".to_string())
    }

    fn subscribe(&self) -> mpsc::UnboundedReceiver<T> {
        self.receivers
            .try_lock()
            .expect("subscribe must not be called concurrently")
            .pop()
            .expect("no more shards available — subscribe called more times than shard_count")
    }
}
