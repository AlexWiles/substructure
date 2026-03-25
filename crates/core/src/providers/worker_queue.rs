use std::collections::VecDeque;
use async_trait::async_trait;
use tokio::sync::{oneshot, Mutex};

use crate::worker::{DequeueFilter, WorkerDecisionRequest, WorkerQueue};

struct Waiter {
    tenant_id: String,
    tx: oneshot::Sender<WorkerDecisionRequest>,
}

impl Waiter {
    fn matches(&self, decision: &WorkerDecisionRequest) -> bool {
        self.tenant_id == decision.tenant_id
    }
}

struct Inner {
    items: VecDeque<WorkerDecisionRequest>,
    waiters: VecDeque<Waiter>,
}

pub struct InMemoryWorkerQueue {
    inner: Mutex<Inner>,
}

impl InMemoryWorkerQueue {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                items: VecDeque::new(),
                waiters: VecDeque::new(),
            }),
        }
    }
}

#[async_trait]
impl WorkerQueue for InMemoryWorkerQueue {
    async fn enqueue(&self, mut decision: WorkerDecisionRequest) {
        let mut inner = self.inner.lock().await;
        // Remove stale waiters and try to hand off to a live one
        inner.waiters.retain(|w| !w.tx.is_closed());
        while let Some(idx) = inner.waiters.iter().position(|w| w.matches(&decision)) {
            let waiter = inner.waiters.remove(idx).unwrap();
            match waiter.tx.send(decision) {
                Ok(()) => return,
                Err(d) => decision = d,
            }
        }
        inner.items.push_back(decision);
    }

    async fn dequeue(&self, filter: &DequeueFilter) -> Option<WorkerDecisionRequest> {
        let rx = {
            let mut inner = self.inner.lock().await;
            // Check buffered items first
            if let Some(idx) = inner.items.iter().position(|d| d.tenant_id == filter.tenant_id) {
                return inner.items.remove(idx);
            }
            // No match — register as a waiter
            let (tx, rx) = oneshot::channel();
            inner.waiters.push_back(Waiter {
                tenant_id: filter.tenant_id.clone(),
                tx,
            });
            rx
        };
        rx.await.ok()
    }
}
