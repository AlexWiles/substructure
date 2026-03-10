use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::decision::DecisionTrigger;
use super::message::Message;
use crate::runtime::identity::ClientIdentity;
use crate::runtime::llm::types::{LlmRequest, LlmResponse};
use crate::runtime::retry::RetryPolicy;
use crate::runtime::serde_helpers::base64_bytes;
use crate::runtime::span::SpanContext;

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
    #[serde(rename = "worker.state.updated")]
    WorkerStateUpdated(WorkerStateUpdated),
    #[serde(rename = "session.cancelled")]
    SessionCancelled,
    #[serde(rename = "session.done")]
    SessionDone(SessionDone),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCreated {
    pub agent_id: String,
    pub auth: ClientIdentity,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on_done: Option<CompletionDelivery>,
    pub worker_retry: RetryPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionDelivery {
    /// Session ID of the parent session to deliver to
    pub parent_session_id: Uuid,
    /// Tool call ID that this result satisfies on the parent session
    pub tool_call_id: String,
    /// Tool name for the completion
    pub tool_name: String,
    /// Span context for tracing
    pub span: SpanContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionDone {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<Artifact>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parts: Vec<Part>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Part {
    Text { text: String },
    Data { data: serde_json::Value },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewMessage {
    pub message: Message,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCallRequested {
    pub call_id: String,
    pub request: LlmRequest,
    pub stream: bool,
    /// Which LLM provider to use (e.g. "openrouter", "mock").
    #[serde(default)]
    pub llm_client: String,
    pub retry: RetryPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCallCompleted {
    pub call_id: String,
    pub response: LlmResponse,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCallErrored {
    pub call_id: String,
    pub error: String,
    #[serde(default = "default_true")]
    pub retryable: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolHandler {
    /// Dispatched to the work queue for the worker to execute.
    #[default]
    Worker,
    /// Executed by the client. Session goes Idle while waiting.
    Client,
    /// Sub-agent: runtime spawns a child session.
    SubAgent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRequested {
    pub tool_call_id: String,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInterrupted {
    pub interrupt_id: String,
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
    #[serde(with = "base64_bytes")]
    pub state: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerDecisionErrored {
    pub decision_id: String,
    pub error: String,
    #[serde(default = "default_true")]
    pub retryable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerStateUpdated {
    #[serde(with = "base64_bytes")]
    pub state: Vec<u8>,
}
