use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::protocol::{
    AgentConfig, DecisionResponse, DecisionTrigger, DraftMessage, Effect, ErrorInfo, Message,
    MessageTree, SessionOwner, WorkerState,
};
use crate::runtime::session::decision::Action;
use crate::runtime::span::SpanContext;
use crate::runtime::Caller;

/// Wire format sent to workers (via poll or push) when a decision is needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerDecisionRequest {
    pub session_id: String,
    pub decision_id: String,
    pub agent_id: String,
    pub identity: SessionOwner,
    pub trigger: DecisionTrigger,
    /// The engine-derived default continuation for `trigger`; empty when the
    /// trigger needs worker knowledge. Advisory — the worker accepts by echoing
    /// it (amended or verbatim) as its decision.
    #[serde(default)]
    pub proposed: DecisionResponse,
    pub state: WorkerState,
    /// The agent config resolved for the active path; `None` when none is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<AgentConfig>,
    #[serde(default)]
    pub calls: Vec<Effect>,
    /// Count of in-flight `tool_call`/`sub_agent` calls.
    #[serde(default)]
    pub pending_calls: usize,
    /// The active conversation as a flat list. Wire field: `messages`.
    #[serde(default, rename = "messages")]
    pub transcript: Vec<Message>,
    #[serde(default)]
    pub message_tree: MessageTree,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ancestry: Vec<String>,
    /// Engine tracing only; not sent in the body.
    #[serde(skip_serializing, default = "SpanContext::root")]
    pub span: SpanContext,
    pub attempts: u32,
    pub deadline: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
}

impl WorkerDecisionRequest {
    /// The tenant that owns this decision's session, for engine routing. The
    /// tenant lives on `identity` (serialized, so it survives a durable queue);
    /// this is the routing accessor over it, not a second copy that can drift.
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
    /// `None` = no opinion, keep the current state.
    pub state: Option<WorkerState>,
    /// `None` = no opinion, keep the current agent config.
    pub agent: Option<AgentConfig>,
    /// How each channel shows this decision, keyed by channel kind.
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
