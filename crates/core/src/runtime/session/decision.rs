use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::protocol::{
    ClientContext, DraftMessage, ErrorInfo, Handler, LlmFormat, LlmRequest, LlmResponse,
    RetryOverride, RetryPolicy,
};
use crate::runtime::retry::RetryTarget;

impl Handler {
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
    /// Executed by the engine against a connection. Only ever set by the engine
    /// when it expands a connector; a worker declaring it is rejected.
    Server,
}

impl ToolHandler {
    /// Which default bounds a call routed here. The three differ: a worker tool
    /// is bounded but never repeated, a client tool waits indefinitely because a
    /// human may be answering it, and a connector call is the engine's own.
    pub fn retry_target(self) -> RetryTarget {
        match self {
            ToolHandler::Worker => RetryTarget::WorkerTool,
            ToolHandler::Client => RetryTarget::ClientTool,
            ToolHandler::Server => RetryTarget::ConnectorTool,
        }
    }
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
            ToolHandler::Server => Handler::Server,
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
            Handler::Server => Ok(ToolHandler::Server),
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
    /// Internal only: materialized into `ClientTranscript` against the active
    /// path at delivery, never sent to workers. Late binding is the point — a
    /// queued append lands after whatever turn beat it instead of forking.
    #[serde(rename = "client.message")]
    ClientMessage {
        messages: Vec<DraftMessage>,
        #[serde(default)]
        client: ClientContext,
        /// The turn this message opens when it dispatches. `Some` = deferred:
        /// the decision parks until the phase is free, then its dispatch starts
        /// the turn. `None` = an ordinary message on the turn already running.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        turn_id: Option<String>,
    },
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
        #[serde(default, skip_serializing_if = "Option::is_none")]
        format: Option<LlmFormat>,
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
        error: Option<ErrorInfo>,
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
        error: Option<ErrorInfo>,
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
        error: Option<ErrorInfo>,
    },
    #[serde(rename = "interrupt.resumed")]
    InterruptResumed {
        interrupt_id: String,
        #[serde(default)]
        payload: serde_json::Value,
    },
    /// Fired after a turn completes, carrying its final output. Blocks `SessionDone`
    /// until answered; the turn completes when this decision settles, with or
    /// without the proposed `done` echoed.
    #[serde(rename = "turn.finished")]
    TurnFinished {
        turn_id: String,
        #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
        data: serde_json::Value,
        #[serde(default)]
        cost: Decimal,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        usage: BTreeMap<String, u64>,
    },
}

impl Trigger {
    /// The turn this trigger opens at dispatch, for the triggers that defer one.
    pub fn deferred_turn_id(&self) -> Option<&str> {
        match self {
            Trigger::ClientMessage { turn_id, .. } => turn_id.as_deref(),
            _ => None,
        }
    }

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
        }
    }

    pub fn llm_err(id: String, error: ErrorInfo) -> Self {
        Trigger::LlmFinished {
            id,
            ok: false,
            message: None,
            truncated: false,
            usage: None,
            cost: None,
            error: Some(error),
        }
    }

    pub fn turn_finished(
        turn_id: String,
        data: serde_json::Value,
        cost: Decimal,
        usage: BTreeMap<String, u64>,
    ) -> Self {
        Trigger::TurnFinished {
            turn_id,
            data,
            cost,
            usage,
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
/// deserializes this from the wire. An omitted `attempt` on a result/error is filled
/// from the `*.execute` trigger being answered, so it fences to the attempt that ran;
/// out-of-band (no trigger) it stays `None`, settling the current attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Action {
    #[serde(rename = "llm.call")]
    CallLlm {
        // Omit to have the engine mint one; it becomes the assistant node's id.
        id: String,
        /// The `[llm.*]` block the call runs on, resolved at the seam.
        llm: String,
        request: LlmRequest,
        #[serde(default)]
        stream: bool,
        #[serde(default = "RetryPolicy::no_retry")]
        retry: RetryPolicy,
        /// The block's venue, resolved at the seam.
        handler: LlmHandler,
        /// The block's `format`, resolved at the seam (worker calls only).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        format: Option<LlmFormat>,
    },
    #[serde(rename = "tool.call")]
    CallTool {
        // Omit for an ad-hoc worker tool; the engine mints one.
        id: String,
        name: String,
        arguments: String,
        /// Still unresolved here, unlike every other action's: which default
        /// the override lands on follows from where the tool runs, and the
        /// handler is not settled until the aggregate routes the call.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retry: Option<RetryOverride>,
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
        error: ErrorInfo,
        retryable: bool,
    },
    #[serde(rename = "llm.error")]
    LlmError {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        attempt: Option<u32>,
        error: ErrorInfo,
        retryable: bool,
    },
    #[serde(rename = "sub_agent.spawn")]
    SpawnSubAgent {
        session_id: String,
        agent_id: String,
        /// The model tool-call this delegation answers.
        tool_call_id: String,
        /// The child's opening message, delivered once its session exists.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<DraftMessage>,
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
