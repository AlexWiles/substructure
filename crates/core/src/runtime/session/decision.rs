use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use super::events::{LlmHandler, NewMessage, ToolHandler};
use super::message::Message;
use crate::runtime::llm::{ErrorCode, LlmRequest, LlmResponse};
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
        /// When set, the message branches from here instead of the current head.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_id: Option<String>,
    },
    Action {
        #[serde(flatten)]
        action: ClientAction,
    },
}

/// A drained effect result, before it's recorded as a thread message. Internal
/// to the engine's undelivered-effects queue; the worker sees the recorded
/// `NewMessage` nodes on the `tool.results` trigger, not this.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub name: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DecisionTrigger {
    #[serde(rename = "user.message")]
    UserMessage {
        stream: bool,
        message: Message,
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_id: Option<String>,
    },
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
        /// Node id of the assistant message (distinct from `call_id`).
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_id: Option<String>,
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
    #[serde(rename = "llm.request")]
    LlmRequest {
        call_id: String,
        request: LlmRequest,
        stream: bool,
        attempt: u32,
    },
    #[serde(rename = "tool.execute")]
    ToolExecute {
        tool_call_id: String,
        name: String,
        arguments: String,
        attempt: u32,
        deadline: Option<DateTime<Utc>>,
    },
    /// A turn's tool and sub-agent results, recorded as thread messages and
    /// delivered as one batch once every effect is terminal.
    #[serde(rename = "tool.results")]
    ToolResults { messages: Vec<NewMessage> },
    #[serde(rename = "interrupt.resumed")]
    InterruptResumed {
        interrupt_id: String,
        #[serde(default)]
        payload: serde_json::Value,
    },
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
        #[serde(default)]
        handler: LlmHandler,
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
    #[serde(rename = "return.llm.result")]
    ReturnLlmResult {
        call_id: String,
        response: LlmResponse,
        attempt: u32,
    },
    #[serde(rename = "return.llm.error")]
    ReturnLlmError {
        call_id: String,
        error: String,
        retryable: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        code: Option<ErrorCode>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<serde_json::Value>,
        attempt: u32,
    },
    #[serde(rename = "spawn.sub_agent")]
    SpawnSubAgent {
        session_id: String,
        agent_id: String,
        /// The model tool-call id this delegation answers.
        #[serde(default)]
        tool_call_id: String,
        retry: RetryPolicy,
    },
    #[serde(rename = "send.message")]
    SendMessage {
        session_id: String,
        message: Message,
    },
    /// Pause the session awaiting external input. Recorded with
    /// `InterruptOrigin::Frontend` so the session owner can resume it.
    #[serde(rename = "interrupt")]
    Interrupt {
        #[serde(default)]
        interrupt_id: String,
        reason: String,
        #[serde(default)]
        payload: serde_json::Value,
    },
    #[serde(rename = "done")]
    Done {
        #[serde(default)]
        data: serde_json::Value,
    },
}
