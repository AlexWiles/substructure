use std::collections::VecDeque;

use async_trait::async_trait;
use rust_decimal::Decimal;
use tokio::sync::{oneshot, Mutex};
use uuid::Uuid;

use crate::runtime::identity::ClientIdentity;
use crate::runtime::retry::RetryPolicy;
use crate::runtime::session::events::Artifact;
use crate::runtime::session::message::Message;
use crate::runtime::span::SpanContext;

#[derive(Debug, Clone)]
pub enum SubAgentTask {
    SpawnSubAgent {
        source_event_id: Uuid,
        parent_session_id: String,
        tenant_id: String,
        child_session_id: String,
        agent_id: String,
        auth: ClientIdentity,
        ancestry: Vec<String>,
        retry: RetryPolicy,
        span: SpanContext,
    },
    SendSessionMessage {
        source_event_id: Uuid,
        tenant_id: String,
        target_session_id: String,
        message: Message,
        span: SpanContext,
    },
    ResolveToolCall {
        source_event_id: Uuid,
        tenant_id: String,
        target_session_id: String,
        tool_call_id: String,
        result: String,
        span: SpanContext,
    },
    CompleteSubAgentTurn {
        source_event_id: Uuid,
        parent_session_id: String,
        tenant_id: String,
        child_session_id: String,
        agent_id: String,
        turn_id: String,
        artifacts: Vec<Artifact>,
        cost: Decimal,
        token_usage: std::collections::BTreeMap<String, u64>,
        span: SpanContext,
    },
}

impl SubAgentTask {
    pub fn dedupe_key(&self) -> String {
        match self {
            SubAgentTask::SpawnSubAgent {
                source_event_id,
                parent_session_id,
                child_session_id,
                ..
            } => format!(
                "subagent:spawn:{parent_session_id}:{child_session_id}:{source_event_id}"
            ),
            SubAgentTask::SendSessionMessage {
                source_event_id,
                target_session_id,
                ..
            } => format!("subagent:send_message:{target_session_id}:{source_event_id}"),
            SubAgentTask::ResolveToolCall {
                source_event_id,
                target_session_id,
                tool_call_id,
                ..
            } => format!(
                "subagent:resolve_tool:{target_session_id}:{tool_call_id}:{source_event_id}"
            ),
            SubAgentTask::CompleteSubAgentTurn {
                source_event_id,
                parent_session_id,
                child_session_id,
                turn_id,
                ..
            } => format!(
                "subagent:complete_turn:{parent_session_id}:{child_session_id}:{turn_id}:{source_event_id}"
            ),
        }
    }
}

#[async_trait]
pub trait SubAgentTaskQueue: Send + Sync {
    async fn enqueue(&self, task: SubAgentTask) -> Result<(), String>;
    async fn dequeue(&self) -> Option<SubAgentTask>;
}

struct Waiter {
    tx: oneshot::Sender<SubAgentTask>,
}

struct Inner {
    items: VecDeque<SubAgentTask>,
    waiters: VecDeque<Waiter>,
}

pub struct InMemorySubAgentTaskQueue {
    inner: Mutex<Inner>,
}

impl InMemorySubAgentTaskQueue {
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
impl SubAgentTaskQueue for InMemorySubAgentTaskQueue {
    async fn enqueue(&self, mut task: SubAgentTask) -> Result<(), String> {
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

    async fn dequeue(&self) -> Option<SubAgentTask> {
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
