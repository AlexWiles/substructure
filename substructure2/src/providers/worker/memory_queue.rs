use std::collections::VecDeque;

use async_trait::async_trait;
use tokio::sync::{oneshot, Mutex};

use crate::runtime::worker::{DequeueFilter, PendingDecision, WorkerQueue};

struct Waiter {
    tenant_id: String,
    agent_ids: Vec<String>,
    tx: oneshot::Sender<PendingDecision>,
}

impl Waiter {
    fn matches(&self, decision: &PendingDecision) -> bool {
        self.tenant_id == decision.tenant_id
            && self.agent_ids.iter().any(|id| id == &decision.agent_id)
    }
}

struct Inner {
    items: VecDeque<PendingDecision>,
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

fn matches_filter(decision: &PendingDecision, filter: &DequeueFilter) -> bool {
    decision.tenant_id == filter.tenant_id
        && filter.agent_ids.iter().any(|id| id == &decision.agent_id)
}

#[async_trait]
impl WorkerQueue for InMemoryWorkerQueue {
    async fn enqueue(&self, mut decision: PendingDecision) {
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

    async fn dequeue(&self, filter: &DequeueFilter) -> Option<PendingDecision> {
        let rx = {
            let mut inner = self.inner.lock().await;
            // Check buffered items first
            if let Some(idx) = inner.items.iter().position(|d| matches_filter(d, filter)) {
                return inner.items.remove(idx);
            }
            // No match — register as a waiter
            let (tx, rx) = oneshot::channel();
            inner.waiters.push_back(Waiter {
                tenant_id: filter.tenant_id.clone(),
                agent_ids: filter.agent_ids.clone(),
                tx,
            });
            rx
        };
        rx.await.ok()
    }
}
