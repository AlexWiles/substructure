use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::protocol::{
    AgentConfig, DecisionResponse, DecisionTrigger, DraftMessage, Effect, ErrorInfo, Message,
    MessageTree, SessionOwner, WorkerRef, WorkerState,
};
use crate::runtime::session::decision::Action;
use crate::runtime::span::SpanContext;
use crate::runtime::Caller;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerDecisionRequest {
    pub session_id: String,
    pub decision_id: String,
    pub agent_id: String,
    pub identity: SessionOwner,
    pub trigger: DecisionTrigger,

    #[serde(default)]
    pub proposed: DecisionResponse,
    pub state: WorkerState,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<AgentConfig>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker: Option<WorkerRef>,
    #[serde(default)]
    pub calls: Vec<Effect>,
    #[serde(default)]
    pub pending_calls: usize,

    #[serde(default, rename = "messages")]
    pub transcript: Vec<Message>,
    #[serde(default)]
    pub message_tree: MessageTree,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ancestry: Vec<String>,

    #[serde(skip_serializing, default = "SpanContext::root")]
    pub span: SpanContext,
    pub attempts: u32,
    pub deadline: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
}

impl WorkerDecisionRequest {
    pub fn tenant_id(&self) -> &str {
        &self.identity.tenant_id
    }
}

pub struct DequeueFilter {
    pub tenant_id: String,
}

#[derive(Debug)]
pub struct SubmitDecision {
    pub session_id: String,
    pub caller: Caller,
    pub decision_id: String,
    pub transcript: Vec<DraftMessage>,
    pub actions: Vec<Action>,

    pub state: Option<WorkerState>,

    pub agent: Option<AgentConfig>,

    pub channels: std::collections::BTreeMap<String, serde_json::Value>,
    pub span: SpanContext,
}

#[derive(Debug)]
pub struct FailDecision {
    pub session_id: String,
    pub caller: Caller,
    pub decision_id: String,
    pub error: ErrorInfo,
    pub retryable: bool,
    pub span: SpanContext,
}

#[async_trait]
pub trait WorkerQueue: Send + Sync {
    async fn enqueue(&self, decision: WorkerDecisionRequest);
    async fn dequeue(&self, filter: &DequeueFilter) -> Option<WorkerDecisionRequest>;
}
