use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::WorkerState;
use crate::runtime::aggregate::Caller;
use crate::runtime::owner::SessionOwner;
use crate::runtime::session::decision::DecisionTrigger;
use crate::runtime::session::decision::WorkerAction;
use crate::runtime::session::events::MessageTree;
use crate::runtime::session::message::Message;
use crate::runtime::session::state::Effect;
use crate::runtime::span::SpanContext;

/// Wire format sent to workers (via poll or push) when a decision is needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerDecisionRequest {
    pub session_id: String,
    pub tenant_id: String,
    pub decision_id: String,
    pub agent_id: String,
    pub owner: SessionOwner,
    pub trigger: DecisionTrigger,
    pub worker_state: WorkerState,
    /// The in-flight effects as a flat, tagged list (each carries `kind`/`status`).
    /// The worker derives what it needs — e.g. prompt once no tool/sub-agent effect
    /// remains — rather than tracking steps itself. Open `kind` leaves room for new
    /// effect kinds without a wire change.
    #[serde(default)]
    pub effects: Vec<Effect>,
    /// The active conversation as a flat list (the tree's `head_id`-to-root
    /// path). A worker only needs this; the tree is provided for clients that
    /// want branch structure.
    #[serde(default)]
    pub transcript: Vec<Message>,
    /// The conversation tree as of this decision, for clients that need the
    /// full branch structure (e.g. the AG-UI snapshot).
    #[serde(default)]
    pub message_tree: MessageTree,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ancestry: Vec<String>,
    pub span: SpanContext,
    pub attempts: u32,
    pub deadline: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
}

pub struct DequeueFilter {
    pub tenant_id: String,
}

#[derive(Debug)]
pub struct SubmitDecision {
    pub session_id: String,
    pub caller: Caller,
    pub decision_id: String,
    pub transcript: Vec<Message>,
    pub actions: Vec<WorkerAction>,
    pub state: WorkerState,
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
