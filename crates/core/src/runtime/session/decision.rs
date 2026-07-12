use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::protocol::{
    ClientContext, DraftMessage, ErrorCode, Handler, LlmRequest, LlmResponse, RetryPolicy,
};

impl Handler {
    /// Serde default for `llm.call`.
    pub fn server() -> Self {
        Handler::Server
    }

    /// Serde default for `tool.call`.
    pub fn worker() -> Self {
        Handler::Worker
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Handler::Server => "server",
            Handler::Worker => "worker",
            Handler::Client => "client",
        }
    }
}

/// Where a tool call runs — the engine's strict form of the wire [`Handler`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolHandler {
    /// Dispatched to the work queue for the worker to execute.
    #[default]
    Worker,
    /// Executed by the client. Session goes Idle while waiting.
    Client,
}

/// Where an LLM call runs — the engine's strict form of the wire [`Handler`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmHandler {
    /// Server-side executor resolves the provider and makes the call.
    #[default]
    Server,
    /// The worker performs the call and replies with `effect.result`/`effect.error`.
    Worker,
}

impl From<ToolHandler> for Handler {
    fn from(h: ToolHandler) -> Self {
        match h {
            ToolHandler::Worker => Handler::Worker,
            ToolHandler::Client => Handler::Client,
        }
    }
}

impl From<LlmHandler> for Handler {
    fn from(h: LlmHandler) -> Self {
        match h {
            LlmHandler::Server => Handler::Server,
            LlmHandler::Worker => Handler::Worker,
        }
    }
}

impl TryFrom<Handler> for ToolHandler {
    type Error = Handler;

    fn try_from(h: Handler) -> Result<Self, Handler> {
        match h {
            Handler::Worker => Ok(ToolHandler::Worker),
            Handler::Client => Ok(ToolHandler::Client),
            Handler::Server => Err(h),
        }
    }
}

impl TryFrom<Handler> for LlmHandler {
    type Error = Handler;

    fn try_from(h: Handler) -> Result<Self, Handler> {
        match h {
            Handler::Server => Ok(LlmHandler::Server),
            Handler::Worker => Ok(LlmHandler::Worker),
            Handler::Client => Err(h),
        }
    }
}

/// Engine-sent trigger; `*.finished` carries the payload when ok, else error.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Trigger {
    /// The first decision of every session, before any client input. Carries no
    /// proposal — the worker declares its `agent` config (or returns `{}`).
    #[serde(rename = "session.start")]
    SessionStart,
    /// Internal only: materialized into `ClientTranscript` at delivery, never sent to workers.
    #[serde(rename = "client.message")]
    ClientMessage { message: DraftMessage },
    /// `messages` is the full proposed conversation; `messages[new_from..]` is
    /// unrecorded (recomputed at delivery against the tree). Wire tag: `client.messages`.
    #[serde(rename = "client.messages")]
    ClientTranscript {
        messages: Vec<DraftMessage>,
        #[serde(default)]
        new_from: usize,
        #[serde(default)]
        client: ClientContext,
    },
    #[serde(rename = "client.action")]
    ClientAction {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        args: Option<serde_json::Value>,
    },
    /// Answer with `tool.result`/`tool.error`.
    #[serde(rename = "tool.execute")]
    ToolExecute {
        id: String,
        name: String,
        arguments: String,
        attempt: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        deadline: Option<DateTime<Utc>>,
    },
    /// Answer with `llm.result`/`llm.error`.
    #[serde(rename = "llm.execute")]
    LlmExecute {
        id: String,
        request: LlmRequest,
        #[serde(default)]
        stream: bool,
        attempt: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        deadline: Option<DateTime<Utc>>,
    },
    #[serde(rename = "tool.finished")]
    ToolFinished {
        id: String,
        ok: bool,
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        result: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// `id` is the model tool call the delegation answers; `session_id` the child session.
    #[serde(rename = "sub_agent.finished")]
    SubAgentFinished {
        id: String,
        ok: bool,
        session_id: String,
        agent_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        result: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    #[serde(rename = "llm.finished")]
    LlmFinished {
        id: String,
        ok: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<DraftMessage>,
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
    #[serde(rename = "interrupt.resumed")]
    InterruptResumed {
        interrupt_id: String,
        #[serde(default)]
        payload: serde_json::Value,
    },
}

impl Trigger {
    pub fn llm_ok(
        id: String,
        message: DraftMessage,
        truncated: bool,
        usage: Option<serde_json::Value>,
        cost: Option<Decimal>,
    ) -> Self {
        Trigger::LlmFinished {
            id,
            ok: true,
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
        id: String,
        error: String,
        code: Option<ErrorCode>,
        detail: Option<serde_json::Value>,
    ) -> Self {
        Trigger::LlmFinished {
            id,
            ok: false,
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

/// Internal vocabulary for the settle APIs; not a wire discriminator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkKind {
    ToolCall,
    LlmCall,
}

/// Internal payload for the settle APIs; not a wire discriminator.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EffectResultPayload {
    ToolCall { result: String },
    LlmCall { response: LlmResponse },
}

/// Resolved worker action — the internal, engine-facing form. Every effect id is
/// present. Produced only by `resolve_response` from a `DecisionAction`; the core never
/// deserializes this from the wire. Omitting `attempt` on a result/error settles the
/// current attempt; echo it to fence a stale executor.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Action {
    #[serde(rename = "llm.call")]
    CallLlm {
        // Omit to have the engine mint one; it becomes the assistant node's id.
        id: String,
        request: LlmRequest,
        #[serde(default)]
        stream: bool,
        #[serde(default = "RetryPolicy::no_retry")]
        retry: RetryPolicy,
        handler: LlmHandler,
    },
    #[serde(rename = "tool.call")]
    CallTool {
        // Omit for an ad-hoc worker tool; the engine mints one.
        id: String,
        name: String,
        arguments: String,
        handler: ToolHandler,
        #[serde(default = "RetryPolicy::no_retry")]
        retry: RetryPolicy,
    },
    #[serde(rename = "tool.result")]
    ToolResult {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        attempt: Option<u32>,
        result: String,
    },
    #[serde(rename = "llm.result")]
    LlmResult {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        attempt: Option<u32>,
        response: LlmResponse,
    },
    #[serde(rename = "tool.error")]
    ToolError {
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
    #[serde(rename = "llm.error")]
    LlmError {
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
    #[serde(rename = "sub_agent.spawn")]
    SpawnSubAgent {
        session_id: String,
        agent_id: String,
        /// The model tool-call this delegation answers.
        tool_call_id: String,
        #[serde(default = "RetryPolicy::no_retry")]
        retry: RetryPolicy,
    },
    #[serde(rename = "message.send")]
    SendMessage {
        session_id: String,
        message: DraftMessage,
    },
    /// Pause the session awaiting external input.
    #[serde(rename = "interrupt")]
    Interrupt {
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
