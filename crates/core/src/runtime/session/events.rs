use std::collections::BTreeMap;

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use super::decision::DecisionTrigger;
use super::message::Message;
use crate::runtime::llm::{ErrorCode, LlmRequest, LlmResponse};
use crate::runtime::owner::SessionOwner;
use crate::runtime::retry::RetryPolicy;
use crate::runtime::worker::WorkerState;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum EventPayload {
    #[serde(rename = "session.created")]
    SessionCreated(Box<SessionCreated>),
    #[serde(rename = "message.new")]
    NewMessage(NewMessage),
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
    #[serde(rename = "sub_agent.turn_completed")]
    SubAgentTurnCompleted(SubAgentTurnCompleted),
    #[serde(rename = "decision_request.queued")]
    DecisionRequestQueued(DecisionRequestQueued),
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
    pub owner: SessionOwner,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ancestry: Vec<String>,
    pub worker_retry: RetryPolicy,
}

/// Pure lifecycle marker — no data. Turn output lives in TurnCompleted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionDone {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewMessage {
    pub message: Message,
    /// Stable node id, independent of the event's storage id.
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
}

/// Conversation nodes in append order plus the active leaf; the active
/// transcript is the `head_id`-to-root path.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MessageTree {
    #[serde(default)]
    pub nodes: Vec<NewMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_id: Option<String>,
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

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolHandler {
    /// Dispatched to the work queue for the worker to execute.
    #[default]
    Worker,
    /// Executed by the client. Session goes Idle while waiting.
    Client,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmHandler {
    /// Server-side executor resolves the provider and makes the call.
    #[default]
    Server,
    /// Dispatched to the worker via an `llm.request` decision trigger; the
    /// worker performs the call and replies with `return.llm.result`/`error`.
    Worker,
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

/// Privilege level of the caller that issued an interrupt. Derived from the
/// authenticated `Caller`, never from request data; resuming requires a
/// caller at or above the origin's privilege.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterruptOrigin {
    System,
    Machine,
    Frontend,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterruptResumed {
    pub interrupt_id: String,
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerDecisionRequested {
    pub decision_id: String,
    pub trigger: DecisionTrigger,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerDecisionCompleted {
    pub decision_id: String,
    /// Opaque worker state — session stores but never interprets.
    pub state: WorkerState,
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
    pub message: Message,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerStateUpdated {
    pub state: WorkerState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnStarted {
    pub turn_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnCompleted {
    pub turn_id: String,
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub data: serde_json::Value,
    #[serde(default)]
    pub turn_cost: Decimal,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub turn_token_usage: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRequestQueued {
    pub decision_id: String,
    pub trigger: DecisionTrigger,
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
