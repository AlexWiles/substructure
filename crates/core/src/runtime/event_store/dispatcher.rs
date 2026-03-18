use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::runtime::aggregate::{AggregateState, DomainEvent, EventHandler};

use super::store::EventStore;

pub fn spawn_handler_pool<R: AggregateState>(
    store: Arc<dyn EventStore>,
    handler: Arc<dyn EventHandler<R>>,
    pool_size: usize,
) -> JoinHandle<()> {
    let mut worker_txs = Vec::with_capacity(pool_size);

    for _ in 0..pool_size {
        let (tx, rx) = mpsc::channel(256);
        let h = handler.clone();
        tokio::spawn(worker_loop(rx, h));
        worker_txs.push(tx);
    }

    let mut rx = store.subscribe();
    tokio::spawn(async move {
        while let Ok(batch) = rx.recv().await {
            for raw in batch.iter() {
                if raw.aggregate_type != R::AGGREGATE_TYPE {
                    continue;
                }
                let event = match DomainEvent::<R>::from_raw(raw) {
                    Ok(e) => e,
                    Err(_) => continue,
                };
                let idx = route(&event.aggregate_id, worker_txs.len());

                let _ = worker_txs[idx].send(event).await;
            }
        }
    })
}

fn route(aggregate_id: &str, pool_size: usize) -> usize {
    let mut hasher = DefaultHasher::new();
    aggregate_id.hash(&mut hasher);
    hasher.finish() as usize % pool_size
}

async fn worker_loop<R: AggregateState>(
    mut rx: mpsc::Receiver<DomainEvent<R>>,
    handler: Arc<dyn EventHandler<R>>,
) {
    while let Some(event) = rx.recv().await {
        handler.on_event(&event).await;
    }
}
