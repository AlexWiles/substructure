use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use super::events::{LlmHandler, ToolHandler};
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
    },
    Messages {
        messages: Vec<Message>,
        #[serde(default)]
        stream: bool,
    },
    Action {
        #[serde(flatten)]
        action: ClientAction,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EffectWork {
    ToolCall { name: String, arguments: String },
    LlmCall { request: LlmRequest, stream: bool },
}

/// Uniform across kinds: `result` (or `message`) when `ok`, `error` when not.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EffectOutcome {
    ToolCall {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        result: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    SubAgent {
        session_id: String,
        agent_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        result: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    LlmCall {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<Message>,
        /// True when finish_reason was "length" (output truncated).
        #[serde(default)]
        truncated: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<serde_json::Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cost: Option<Decimal>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        code: Option<ErrorCode>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<serde_json::Value>,
    },
}

impl EffectOutcome {
    pub fn llm_ok(
        message: Message,
        truncated: bool,
        usage: Option<serde_json::Value>,
        cost: Option<Decimal>,
    ) -> Self {
        EffectOutcome::LlmCall {
            message: Some(message),
            truncated,
            usage,
            cost,
            error: None,
            code: None,
            detail: None,
        }
    }

    pub fn llm_err(
        error: String,
        code: Option<ErrorCode>,
        detail: Option<serde_json::Value>,
    ) -> Self {
        EffectOutcome::LlmCall {
            message: None,
            truncated: false,
            usage: None,
            cost: None,
            error: Some(error),
            code,
            detail,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DecisionTrigger {
    #[serde(rename = "client.message")]
    ClientMessage { message: Message },
    #[serde(rename = "client.transcript")]
    ClientTranscript { messages: Vec<Message> },
    #[serde(rename = "client.action")]
    ClientAction {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        args: Option<serde_json::Value>,
    },
    #[serde(rename = "effect.execute")]
    EffectExecute {
        id: String,
        attempt: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        deadline: Option<DateTime<Utc>>,
        #[serde(flatten)]
        work: EffectWork,
    },
    #[serde(rename = "effect.settled")]
    EffectSettled {
        id: String,
        ok: bool,
        #[serde(flatten)]
        outcome: EffectOutcome,
    },
    #[serde(rename = "interrupt.resumed")]
    InterruptResumed {
        interrupt_id: String,
        #[serde(default)]
        payload: serde_json::Value,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkKind {
    ToolCall,
    LlmCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EffectResultPayload {
    ToolCall { result: String },
    LlmCall { response: LlmResponse },
}

impl EffectResultPayload {
    pub fn kind(&self) -> WorkKind {
        match self {
            EffectResultPayload::ToolCall { .. } => WorkKind::ToolCall,
            EffectResultPayload::LlmCall { .. } => WorkKind::LlmCall,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WorkerAction {
    #[serde(rename = "call.llm")]
    CallLlm {
        id: String,
        request: LlmRequest,
        #[serde(default)]
        stream: bool,
        #[serde(default = "RetryPolicy::no_retry")]
        retry: RetryPolicy,
        handler: LlmHandler,
    },
    #[serde(rename = "call.tool")]
    CallTool {
        id: String,
        name: String,
        arguments: String,
        handler: ToolHandler,
        #[serde(default = "RetryPolicy::no_retry")]
        retry: RetryPolicy,
    },
    #[serde(rename = "effect.result")]
    EffectResult {
        id: String,
        /// Omitted = settle the current attempt; echo the trigger's `attempt`
        /// to fence out a stale executor instead.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        attempt: Option<u32>,
        #[serde(flatten)]
        result: EffectResultPayload,
    },
    #[serde(rename = "effect.error")]
    EffectError {
        kind: WorkKind,
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        attempt: Option<u32>,
        error: String,
        retryable: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        code: Option<ErrorCode>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<serde_json::Value>,
    },
    #[serde(rename = "spawn.sub_agent")]
    SpawnSubAgent {
        session_id: String,
        agent_id: String,
        /// The model tool-call id this delegation answers.
        #[serde(default)]
        tool_call_id: String,
        #[serde(default = "RetryPolicy::no_retry")]
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
