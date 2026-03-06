use serde::{Deserialize, Serialize};

use crate::runtime::llm::{LlmCallCompleted, LlmCallErrored, LlmCallRequested};
use crate::runtime::session::types::{
    InterruptResumed, MessageAssistant, MessageTool, MessageUser, SessionCreated, SessionDone,
    SessionInterrupted, ToolCallCompleted, ToolCallErrored, ToolCallRequested, WorkerStateChanged,
};
use crate::runtime::session::worker::{WorkerDecisionCompleted, WorkerDecisionRequested};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum EventPayload {
    #[serde(rename = "session.created")]
    SessionCreated(Box<SessionCreated>),
    #[serde(rename = "message.user")]
    MessageUser(MessageUser),
    #[serde(rename = "message.assistant")]
    MessageAssistant(MessageAssistant),
    #[serde(rename = "llm.call.requested")]
    LlmCallRequested(LlmCallRequested),
    #[serde(rename = "llm.call.completed")]
    LlmCallCompleted(LlmCallCompleted),
    #[serde(rename = "llm.call.errored")]
    LlmCallErrored(LlmCallErrored),
    #[serde(rename = "message.tool")]
    MessageTool(MessageTool),
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
    #[serde(rename = "worker.state_changed")]
    WorkerStateChanged(WorkerStateChanged),
    #[serde(rename = "worker.decision.requested")]
    WorkerDecisionRequested(WorkerDecisionRequested),
    #[serde(rename = "worker.decision.completed")]
    WorkerDecisionCompleted(WorkerDecisionCompleted),
    #[serde(rename = "session.cancelled")]
    SessionCancelled,
    #[serde(rename = "session.done")]
    SessionDone(SessionDone),
}
