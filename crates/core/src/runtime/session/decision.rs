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
    /// A full transcript (e.g. an AG-UI client view); the worker reconciles it into the tree.
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

/// The work an `effect.execute` trigger delegates to the worker, tagged by
/// `kind` — the same discriminator the `effects` list uses.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EffectWork {
    ToolCall { name: String, arguments: String },
    LlmCall { request: LlmRequest, stream: bool },
}

/// How a settled effect landed, tagged by `kind`. For `tool_call` and
/// `sub_agent` the error text rides in `result` when the trigger's `ok` is
/// false — it folds into the transcript as the tool message either way. An
/// `llm_call` carries `message`/`truncated`/`usage`/`cost` on success and
/// `error`/`code`/`detail` on failure.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EffectOutcome {
    ToolCall {
        name: String,
        result: String,
    },
    /// `id` on the trigger is the sub-agent's session id (matching the
    /// `effects` list); `tool_call_id` is the model tool call it answers.
    SubAgent {
        tool_call_id: String,
        agent_id: String,
        result: String,
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
    /// A user message arrived; the worker appends it (rooting the branch with its
    /// system prompt on a fresh branch) and prompts.
    #[serde(rename = "user.message")]
    UserMessage { message: Message },
    /// A full client transcript arrived; the worker reconciles it into the tree.
    #[serde(rename = "user.transcript")]
    UserTranscript { messages: Vec<Message> },
    #[serde(rename = "client.action")]
    ClientAction {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        args: Option<serde_json::Value>,
    },
    /// The engine delegates effect work to the worker: run the `kind`-specific
    /// work for effect `id` and answer with `effect.result` or `effect.error`.
    #[serde(rename = "effect.execute")]
    EffectExecute {
        id: String,
        attempt: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        deadline: Option<DateTime<Utc>>,
        #[serde(flatten)]
        work: EffectWork,
    },
    /// An effect settled; fired as each one lands so the worker folds its
    /// outcome in and the tree fills incrementally. The worker prompts once no
    /// tool/sub-agent effect is in flight — a view derived from `effects` on
    /// the decision request.
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
    #[serde(rename = "stall")]
    Stall,
}

/// The effect kinds a worker can answer `effect.execute` for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkKind {
    ToolCall,
    LlmCall,
}

/// A successful effect result, tagged by `kind`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EffectResultPayload {
    ToolCall { result: String },
    LlmCall { response: LlmResponse },
}

/// Actions a worker can request as part of a decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WorkerAction {
    #[serde(rename = "call.llm")]
    CallLlm {
        request: LlmRequest,
        #[serde(default)]
        stream: bool,
        #[serde(default = "RetryPolicy::no_retry")]
        retry: RetryPolicy,
        handler: LlmHandler,
    },
    /// `id` names the effect (the model's tool call id); its outcome comes
    /// back as an `effect.settled` trigger with the same id.
    #[serde(rename = "call.tool")]
    CallTool {
        id: String,
        name: String,
        arguments: String,
        handler: ToolHandler,
        #[serde(default = "RetryPolicy::no_retry")]
        retry: RetryPolicy,
    },
    /// Successful answer to an `effect.execute` trigger.
    #[serde(rename = "effect.result")]
    EffectResult {
        id: String,
        attempt: u32,
        #[serde(flatten)]
        result: EffectResultPayload,
    },
    /// Failed answer to an `effect.execute` trigger; uniform across kinds.
    #[serde(rename = "effect.error")]
    EffectError {
        kind: WorkKind,
        id: String,
        attempt: u32,
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
