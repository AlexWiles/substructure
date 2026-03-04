pub mod llm;
pub mod message;
pub mod session;
pub mod tool;

pub use self::llm::*;
pub use self::message::*;
pub use self::session::*;
pub use self::tool::*;

pub use super::config::{AgentConfig, LlmConfig, RetryConfig, RetryPolicy};
pub use super::span::{SpanContext, SpanId, TraceId};

use serde::{Deserialize, Serialize};

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
    #[serde(rename = "strategy.state_changed")]
    StrategyStateChanged(StrategyStateChanged),
    #[serde(rename = "session.cancelled")]
    SessionCancelled,
    #[serde(rename = "session.done")]
    SessionDone(SessionDone),
}
