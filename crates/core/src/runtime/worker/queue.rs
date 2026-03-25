use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::runtime::identity::ClientIdentity;
use crate::runtime::serde_helpers::base64_bytes;
use crate::runtime::session::decision::DecisionTrigger;
use crate::runtime::session::decision::WorkerAction;
use crate::runtime::span::SpanContext;

/// Wire format sent to workers (via poll or push) when a decision is needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerDecisionRequest {
    pub session_id: String,
    pub tenant_id: String,
    pub decision_id: String,
    pub agent_id: String,
    pub auth: ClientIdentity,
    pub trigger: DecisionTrigger,
    #[serde(with = "base64_bytes")]
    pub worker_state: Vec<u8>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ancestry: Vec<String>,
    pub span: SpanContext,
    pub attempts: u32,
    pub deadline: Option<DateTime<Utc>>,
}

pub struct DequeueFilter {
    pub tenant_id: String,
}

#[derive(Debug, Deserialize)]
pub struct SubmitDecision {
    pub session_id: String,
    pub tenant_id: String,
    pub decision_id: String,
    pub actions: Vec<WorkerAction>,
    pub state: Vec<u8>,
    pub span: SpanContext,
}

#[async_trait]
pub trait WorkerQueue: Send + Sync {
    async fn enqueue(&self, decision: WorkerDecisionRequest);
    async fn dequeue(&self, filter: &DequeueFilter) -> Option<WorkerDecisionRequest>;
}
