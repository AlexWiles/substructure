use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::WorkerState;
use crate::runtime::aggregate::Caller;
use crate::runtime::owner::SessionOwner;
use crate::runtime::session::decision::Action;
use crate::runtime::session::events::MessageTree;
use crate::runtime::session::message::Message;
use crate::runtime::session::propose::Proposal;
use crate::runtime::session::state::Effect;
use crate::runtime::session::wire::{WireMessage, WireTrigger};
use crate::runtime::span::SpanContext;

/// Wire format sent to workers (via poll or push) when a decision is needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerDecisionRequest {
    pub session_id: String,
    pub decision_id: String,
    pub agent_id: String,
    pub identity: SessionOwner,
    pub trigger: WireTrigger,
    /// The engine-derived default continuation for `trigger`; `None` when the
    /// trigger needs worker knowledge. Advisory — the worker accepts by echoing
    /// it (amended or verbatim) as its decision.
    #[serde(default)]
    pub proposed: Option<Proposal>,
    pub state: WorkerState,
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
    pub transcript: Vec<WireMessage>,
    pub actions: Vec<Action>,
    /// `None` = no opinion, keep the current state.
    pub state: Option<WorkerState>,
    pub span: SpanContext,
}

#[derive(Debug)]
pub struct FailDecision {
    pub session_id: String,
    pub caller: Caller,
    pub decision_id: String,
    pub error: String,
    pub retryable: bool,
    pub span: SpanContext,
}

#[async_trait]
pub trait WorkerQueue: Send + Sync {
    async fn enqueue(&self, decision: WorkerDecisionRequest);
    async fn dequeue(&self, filter: &DequeueFilter) -> Option<WorkerDecisionRequest>;
}
