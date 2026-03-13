use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::events::{Artifact, ToolHandler};
use super::message::Message;
use crate::runtime::llm::LlmRequest;
use crate::runtime::retry::RetryPolicy;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub name: String,
    pub content: String,
    pub is_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionTrigger {
    UserMessage {
        stream: bool,
        message: Message,
    },
    LlmCompleted {
        call_id: String,
        message: Message,
        /// True when finish_reason was "length" (output truncated).
        truncated: bool,
    },
    LlmFailed {
        call_id: String,
        error: String,
    },
    ToolCallRequested {
        tool_call_id: String,
        name: String,
        arguments: String,
        attempt: u32,
        deadline: Option<DateTime<Utc>>,
    },
    ToolResolved {
        result: ToolResult,
    },
    SubAgentFailed {
        session_id: Uuid,
        agent_id: String,
        error: String,
    },
    InterruptResumed {
        interrupt_id: String,
    },
    Stall,
}

/// Actions a worker can request as part of a decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkerAction {
    RequestLlm {
        request: LlmRequest,
        stream: bool,
        llm_client: String,
        retry: RetryPolicy,
    },
    RequestToolCall {
        tool_call_id: String,
        name: String,
        arguments: String,
        handler: ToolHandler,
        retry: RetryPolicy,
    },
    CompleteToolCall {
        tool_call_id: String,
        result: String,
        attempt: u32,
    },
    FailToolCall {
        tool_call_id: String,
        error: String,
        retryable: bool,
        attempt: u32,
    },
    ResolveToolCall {
        session_id: Uuid,
        tool_call_id: String,
        result: String,
    },
    RequestSubAgent {
        session_id: Uuid,
        agent_id: String,
        retry: RetryPolicy,
    },
    SendSessionMessage {
        session_id: Uuid,
        message: Message,
    },
    Done {
        artifacts: Vec<Artifact>,
    },
}
