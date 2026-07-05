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
    /// The session owner, surfaced to workers as `identity`.
    pub identity: SessionOwner,
    pub trigger: DecisionTrigger,
    /// The session's current state, resolved from the active branch.
    pub state: WorkerState,
    /// The in-flight effects as a flat, tagged list (each carries `kind`/`status`).
    #[serde(default)]
    pub effects: Vec<Effect>,
    /// How many `tool_call`/`sub_agent` effects are still in flight: the step gate as a number.
    #[serde(default)]
    pub pending_effects: usize,
    /// The active conversation as a flat list (the tree's `head_id`-to-root path).
    #[serde(default)]
    pub transcript: Vec<Message>,
    /// The conversation tree, for clients that need the full branch structure.
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
