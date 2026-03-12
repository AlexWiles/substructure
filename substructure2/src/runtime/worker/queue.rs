use async_trait::async_trait;
use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::runtime::identity::ClientIdentity;
use crate::runtime::session::command::WorkerAction;
use crate::runtime::session::decision::DecisionTrigger;
use crate::runtime::span::SpanContext;

#[derive(Clone)]
pub struct PendingDecision {
    pub session_id: Uuid,
    pub tenant_id: String,
    pub decision_id: String,
    pub agent_id: String,
    pub auth: ClientIdentity,
    pub trigger: DecisionTrigger,
    pub worker_state: Vec<u8>,
    pub span: SpanContext,
    pub attempts: u32,
    pub deadline: Option<DateTime<Utc>>,
}

pub struct DequeueFilter {
    pub tenant_id: String,
    pub agent_ids: Vec<String>,
}

pub struct SubmitDecision {
    pub session_id: Uuid,
    pub tenant_id: String,
    pub decision_id: String,
    pub actions: Vec<WorkerAction>,
    pub state: Vec<u8>,
    pub span: SpanContext,
}

#[async_trait]
pub trait WorkerQueue: Send + Sync {
    async fn enqueue(&self, decision: PendingDecision);
    async fn dequeue(&self, filter: &DequeueFilter) -> Option<PendingDecision>;
}
