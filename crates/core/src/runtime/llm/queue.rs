use std::collections::VecDeque;

use async_trait::async_trait;
use tokio::sync::{oneshot, Mutex};

use crate::runtime::identity::ClientIdentity;
use crate::runtime::llm::LlmRequest;
use crate::runtime::span::SpanContext;

#[derive(Debug, Clone)]
pub struct LlmTask {
    pub session_id: String,
    pub tenant_id: String,
    pub call_id: String,
    pub llm_client: String,
    pub request: LlmRequest,
    pub auth: ClientIdentity,
    pub span: SpanContext,
}

impl LlmTask {
    pub fn dedupe_key(&self) -> String {
        format!("llm:{}:{}", self.session_id, self.call_id)
    }
}

#[async_trait]
pub trait LlmTaskQueue: Send + Sync {
    async fn enqueue(&self, task: LlmTask) -> Result<(), String>;
    async fn dequeue(&self) -> Option<LlmTask>;
}

struct Waiter {
    tx: oneshot::Sender<LlmTask>,
}

struct Inner {
    items: VecDeque<LlmTask>,
    waiters: VecDeque<Waiter>,
}

pub struct InMemoryLlmTaskQueue {
    inner: Mutex<Inner>,
}

impl InMemoryLlmTaskQueue {
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
impl LlmTaskQueue for InMemoryLlmTaskQueue {
    async fn enqueue(&self, mut task: LlmTask) -> Result<(), String> {
        let mut inner = self.inner.lock().await;
        inner.waiters.retain(|w| !w.tx.is_closed());
        while let Some(waiter) = inner.waiters.pop_front() {
            match waiter.tx.send(task) {
                Ok(()) => return Ok(()),
                Err(t) => task = t,
            }
        }
        inner.items.push_back(task);
        Ok(())
    }

    async fn dequeue(&self) -> Option<LlmTask> {
        let rx = {
            let mut inner = self.inner.lock().await;
            if let Some(task) = inner.items.pop_front() {
                return Some(task);
            }
            let (tx, rx) = oneshot::channel();
            inner.waiters.push_back(Waiter { tx });
            rx
        };
        rx.await.ok()
    }
}
