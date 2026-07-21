use std::collections::{BTreeMap, HashMap};

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use super::decision::{LlmHandler, ToolHandler, Trigger};
pub use crate::protocol::EffectKind;
use crate::protocol::{
    AgentConfig, DraftMessage, ErrorCode, InterruptOrigin, LlmFormat, LlmRequest, LlmResponse,
    Message, MessageTree, NewMessage, RetryPolicy, SessionOwner, WorkerState,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum EventPayload {
    #[serde(rename = "session.created")]
    SessionCreated(Box<SessionCreated>),
    #[serde(rename = "message.new")]
    NewMessage(NewMessage),
    #[serde(rename = "head.moved")]
    HeadMoved(HeadMoved),
    #[serde(rename = "llm.call.requested")]
    LlmCallRequested(LlmCallRequested),
    #[serde(rename = "llm.call.completed")]
    LlmCallCompleted(LlmCallCompleted),
    #[serde(rename = "llm.call.errored")]
    LlmCallErrored(LlmCallErrored),
    #[serde(rename = "tool.call.requested")]
    ToolCallRequested(ToolCallRequested),
    #[serde(rename = "tool.call.completed")]
    ToolCallCompleted(ToolCallCompleted),
    #[serde(rename = "tool.call.errored")]
    ToolCallErrored(ToolCallErrored),
    #[serde(rename = "sub_agent.requested")]
    SubAgentRequested(SubAgentRequested),
    #[serde(rename = "sub_agent.started")]
    SubAgentStarted(SubAgentStarted),
    #[serde(rename = "sub_agent.errored")]
    SubAgentErrored(SubAgentErrored),
    #[serde(rename = "session.interrupted")]
    SessionInterrupted(SessionInterrupted),
    #[serde(rename = "session.interrupt_resumed")]
    InterruptResumed(InterruptResumed),
    #[serde(rename = "worker.decision.requested")]
    WorkerDecisionRequested(WorkerDecisionRequested),
    #[serde(rename = "worker.decision.completed")]
    WorkerDecisionCompleted(WorkerDecisionCompleted),
    #[serde(rename = "worker.decision.errored")]
    WorkerDecisionErrored(WorkerDecisionErrored),
    #[serde(rename = "session.message_requested")]
    SessionMessageRequested(SessionMessageRequested),
    #[serde(rename = "worker.state.updated")]
    WorkerStateUpdated(WorkerStateUpdated),
    #[serde(rename = "agent.updated")]
    AgentConfigUpdated(AgentConfigUpdated),
    #[serde(rename = "sub_agent.turn_completed")]
    SubAgentTurnCompleted(SubAgentTurnCompleted),
    #[serde(rename = "decision_request.queued")]
    DecisionRequestQueued(DecisionRequestQueued),
    #[serde(rename = "decision_request.dropped")]
    DecisionRequestDropped(DecisionRequestDropped),
    #[serde(rename = "call.voided")]
    CallVoided(CallVoided),
    #[serde(rename = "session.cancelled")]
    SessionCancelled,
    #[serde(rename = "session.done")]
    SessionDone(SessionDone),
    #[serde(rename = "turn.started")]
    TurnStarted(TurnStarted),
    #[serde(rename = "turn.completed")]
    TurnCompleted(TurnCompleted),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCreated {
    pub agent_id: String,
    pub identity: SessionOwner,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ancestry: Vec<String>,
    pub worker_retry: RetryPolicy,
}

/// Pure lifecycle marker — no data. Turn output lives in TurnCompleted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionDone {}

/// The head moved to an existing node (truncation or branch switch); the only
/// non-append head writer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadMoved {
    pub head_id: String,
}

impl MessageTree {
    /// Root-to-`leaf` messages; empty if `leaf` is unknown.
    pub fn path_to(&self, leaf: &str) -> Vec<Message> {
        let mut by_id: HashMap<&str, &NewMessage> = self
            .nodes
            .iter()
            .map(|n| (n.message.id.as_str(), n))
            .collect();
        let mut path = Vec::new();
        let mut cursor = Some(leaf.to_string());
        // `remove` so a malformed parent cycle can't loop forever.
        while let Some(id) = cursor.take() {
            let Some(node) = by_id.remove(id.as_str()) else {
                break;
            };
            path.push(node.message.clone());
            cursor = node.parent_id.clone();
        }
        path.reverse();
        path
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCallRequested {
    pub call_id: String,
    pub attempt: u32,
    pub request: LlmRequest,
    pub stream: bool,
    pub retry: RetryPolicy,
    #[serde(default)]
    pub handler: LlmHandler,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<LlmFormat>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCallCompleted {
    pub call_id: String,
    pub attempt: u32,
    pub response: LlmResponse,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCallErrored {
    pub call_id: String,
    pub attempt: u32,
    pub error: String,
    #[serde(default = "default_true")]
    pub retryable: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<ErrorCode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubAgentRequested {
    pub session_id: String,
    pub agent_id: String,
    #[serde(default)]
    pub tool_call_id: String,
    pub retry: RetryPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubAgentStarted {
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubAgentErrored {
    pub session_id: String,
    pub error: String,
    #[serde(default)]
    pub retryable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRequested {
    pub tool_call_id: String,
    pub attempt: u32,
    pub name: String,
    pub arguments: String,
    #[serde(default)]
    pub handler: ToolHandler,
    pub retry: RetryPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallCompleted {
    pub tool_call_id: String,
    pub name: String,
    pub result: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallErrored {
    pub tool_call_id: String,
    pub name: String,
    pub error: String,
    #[serde(default)]
    pub retryable: bool,
}

impl InterruptOrigin {
    pub fn privilege(self) -> u8 {
        match self {
            InterruptOrigin::System => 2,
            InterruptOrigin::Machine => 1,
            InterruptOrigin::Frontend => 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInterrupted {
    pub interrupt_id: String,
    pub origin: InterruptOrigin,
    pub reason: String,
    pub payload: serde_json::Value,
    /// Head when raised; `None` parks every path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterruptResumed {
    pub interrupt_id: String,
    pub payload: serde_json::Value,
}

/// Promotion marker: the queued decision is now live. The trigger lives on
/// the decision's `DecisionRequestQueued` event, which always precedes this.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerDecisionRequested {
    pub decision_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerDecisionCompleted {
    pub decision_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerDecisionErrored {
    pub decision_id: String,
    pub error: String,
    #[serde(default = "default_true")]
    pub retryable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMessageRequested {
    pub target_session_id: String,
    pub message: DraftMessage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerStateUpdated {
    pub state: WorkerState,
    /// Active head when written; `None` if the tree was empty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfigUpdated {
    pub config: AgentConfig,
    /// Active head when written; `None` if the tree was empty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnStarted {
    pub turn_id: String,
}

/// The run terminal (pass 2): the turn's `turn.finished` finalizer has settled, so
/// the turn — output plus post-turn side effects — is done. `error` is set when the
/// finalizer failed terminally; the output is still durable but the run failed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnCompleted {
    pub turn_id: String,
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub data: serde_json::Value,
    #[serde(default)]
    pub turn_cost: Decimal,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub turn_token_usage: BTreeMap<String, u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRequestQueued {
    pub decision_id: String,
    pub trigger: Trigger,
}

/// A settle decision dropped when its branch was forked away.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRequestDropped {
    pub decision_id: String,
}

/// An abandoned in-flight call. For a sub-agent, `id` is the tool call and
/// `session_id` the child session to cancel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallVoided {
    pub kind: EffectKind,
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubAgentTurnCompleted {
    pub session_id: String,
    #[serde(default)]
    pub cost: Decimal,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub token_usage: BTreeMap<String, u64>,
    /// The child's turn result.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub data: serde_json::Value,
}
