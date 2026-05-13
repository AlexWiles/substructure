use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use super::events::ToolHandler;
use super::message::Message;
use crate::runtime::llm::{ErrorCode, LlmRequest};
use crate::runtime::retry::RetryPolicy;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientAction {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub args: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientPayload {
    Message {
        message: Message,
        #[serde(default)]
        stream: bool,
    },
    Action {
        #[serde(flatten)]
        action: ClientAction,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub name: String,
    pub content: String,
    pub is_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DecisionTrigger {
    #[serde(rename = "user.message")]
    UserMessage { stream: bool, message: Message },
    #[serde(rename = "client.action")]
    ClientAction {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        args: Option<serde_json::Value>,
    },
    #[serde(rename = "llm.response")]
    LlmResponse {
        call_id: String,
        message: Message,
        /// True when finish_reason was "length" (output truncated).
        truncated: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<serde_json::Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cost: Option<Decimal>,
    },
    #[serde(rename = "llm.error")]
    LlmError {
        call_id: String,
        error: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        code: Option<ErrorCode>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<serde_json::Value>,
    },
    #[serde(rename = "tool.execute")]
    ToolExecute {
        tool_call_id: String,
        name: String,
        arguments: String,
        attempt: u32,
        deadline: Option<DateTime<Utc>>,
    },
    #[serde(rename = "tool.result")]
    ToolResult { result: ToolResult },
    #[serde(rename = "sub_agent.done")]
    SubAgentDone {
        session_id: String,
        agent_id: String,
        data: serde_json::Value,
    },
    #[serde(rename = "sub_agent.turn.complete")]
    SubAgentTurnComplete {
        session_id: String,
        agent_id: String,
        turn_id: String,
        data: serde_json::Value,
    },
    #[serde(rename = "sub_agent.error")]
    SubAgentError {
        session_id: String,
        agent_id: String,
        error: String,
    },
    #[serde(rename = "interrupt.resumed")]
    InterruptResumed { interrupt_id: String },
    #[serde(rename = "stall")]
    Stall,
}

/// Actions a worker can request as part of a decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WorkerAction {
    #[serde(rename = "call.llm")]
    CallLlm {
        request: LlmRequest,
        stream: bool,
        retry: RetryPolicy,
    },
    #[serde(rename = "call.tool")]
    CallTool {
        tool_call_id: String,
        name: String,
        arguments: String,
        handler: ToolHandler,
        retry: RetryPolicy,
    },
    #[serde(rename = "return.tool.result")]
    ReturnToolResult {
        tool_call_id: String,
        result: String,
        attempt: u32,
    },
    #[serde(rename = "return.tool.error")]
    ReturnToolError {
        tool_call_id: String,
        error: String,
        retryable: bool,
        attempt: u32,
    },
    #[serde(rename = "spawn.sub_agent")]
    SpawnSubAgent {
        session_id: String,
        agent_id: String,
        retry: RetryPolicy,
    },
    #[serde(rename = "send.message")]
    SendMessage {
        session_id: String,
        message: Message,
    },
    #[serde(rename = "done")]
    Done {
        #[serde(default)]
        data: serde_json::Value,
    },
}
