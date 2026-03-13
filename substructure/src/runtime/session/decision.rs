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
    LlmResponse {
        call_id: String,
        message: Message,
        /// True when finish_reason was "length" (output truncated).
        truncated: bool,
    },
    LlmError {
        call_id: String,
        error: String,
    },
    ToolExecute {
        tool_call_id: String,
        name: String,
        arguments: String,
        attempt: u32,
        deadline: Option<DateTime<Utc>>,
    },
    ToolResult {
        result: ToolResult,
    },
    SubAgentDone {
        session_id: Uuid,
        agent_id: String,
        artifacts: Vec<Artifact>,
    },
    SubAgentTurnComplete {
        session_id: Uuid,
        agent_id: String,
        turn_id: String,
        artifacts: Vec<Artifact>,
    },
    SubAgentError {
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
    CallLlm {
        request: LlmRequest,
        stream: bool,
        llm_client: String,
        retry: RetryPolicy,
    },
    CallTool {
        tool_call_id: String,
        name: String,
        arguments: String,
        handler: ToolHandler,
        retry: RetryPolicy,
    },
    ReturnToolResult {
        tool_call_id: String,
        result: String,
        attempt: u32,
    },
    ReturnToolError {
        tool_call_id: String,
        error: String,
        retryable: bool,
        attempt: u32,
    },
    ResolveRemoteTool {
        session_id: Uuid,
        tool_call_id: String,
        result: String,
    },
    SpawnSubAgent {
        session_id: Uuid,
        agent_id: String,
        retry: RetryPolicy,
    },
    SendMessage {
        session_id: Uuid,
        message: Message,
    },
    Done {
        artifacts: Vec<Artifact>,
    },
}
