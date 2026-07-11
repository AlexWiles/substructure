//! The public protocol: every type that crosses the client or worker wire.
//! Types only — no logic. Conversions and seams live with the engine
//! (`runtime::session::wire`, `runtime::session::propose`, …). Every type
//! derives [`JsonSchema`]; `subs schema` emits the combined JSON Schema.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── Messages ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(inline)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolCallFunction,
}

// Multimodal content parts (OpenAI/OpenRouter wire format).

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(inline)]
pub struct ImageUrl {
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(inline)]
pub struct FileData {
    pub filename: String,
    pub file_data: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(inline)]
pub struct AudioData {
    pub data: String,
    pub format: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(inline)]
pub struct VideoUrl {
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    #[serde(rename = "image_url")]
    ImageUrl {
        image_url: ImageUrl,
    },
    File {
        file: FileData,
    },
    #[serde(rename = "input_audio")]
    InputAudio {
        input_audio: AudioData,
    },
    #[serde(rename = "video_url")]
    VideoUrl {
        video_url: VideoUrl,
    },
}

/// Message content: either a plain string or an array of typed parts.
/// Serializes as a raw string or array respectively (untagged).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
#[schemars(inline)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Message {
    pub id: String,
    pub role: Role,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Content>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// The wire form of a [`Message`]: `id` is optional because a client-submitted or
/// worker-authored message is not yet recorded. `record`/`rerecord`
/// (`runtime::session::wire`) are the seams that lower it to the internal
/// [`Message`] (id always present) at recording time.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DraftMessage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub role: Role,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Content>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

// ── Message tree ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(inline)]
pub struct NewMessage {
    pub message: Message,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(inline)]
pub struct NewControl {
    pub control: Control,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
}

/// A non-conversational tree marker (interrupt/resume); filtered out of LLM prompts.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Control {
    pub id: String,
    pub interrupt_id: String,
    pub kind: ControlKind,
    #[serde(default)]
    pub reason: String,
    #[serde(default)]
    pub payload: Value,
    pub origin: InterruptOrigin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ControlKind {
    Interrupt,
    Resume,
}

/// Privilege level of the caller that issued an interrupt. Derived from the
/// authenticated `Caller`, never from request data; resuming requires a
/// caller at or above the origin's privilege.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum InterruptOrigin {
    System,
    Machine,
    Frontend,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[schemars(inline)]
pub enum Node {
    Message(NewMessage),
    Control(NewControl),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct MessageTree {
    #[serde(default)]
    pub nodes: Vec<Node>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_id: Option<String>,
}

// ── Handlers ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ToolHandler {
    /// Dispatched to the work queue for the worker to execute.
    #[default]
    Worker,
    /// Executed by the client. Session goes Idle while waiting.
    Client,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LlmHandler {
    /// Server-side executor resolves the provider and makes the call.
    #[default]
    Server,
    /// The worker performs the call and replies with `effect.result`/`effect.error`.
    Worker,
}

// ── Retry ────────────────────────────────────────────────────────────────

/// Fully-resolved retry policy — no optional fields. Stored on call state and
/// read directly by retry logic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RetryPolicy {
    pub timeout_secs: Option<u32>,
    pub max_retries: u32,
    pub backoff_base_secs: u32,
    pub backoff_max_secs: u32,
}

// ── Identity ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SessionOwner {
    pub tenant_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
}

// ── Worker state ─────────────────────────────────────────────────────────

/// Opaque worker state: JSON the engine stores but never interprets.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(transparent)]
pub struct WorkerState(pub Value);

// ── Agent config ─────────────────────────────────────────────────────────

/// A declared agent identity. `model` is the only required field; everything else
/// refines the proposed LLM request the engine derives for `client.messages`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AgentConfig {
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry: Option<RetryPolicy>,
    /// Worker- or client-executed tools the model can call.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<AgentTool>,
    /// Sub-agents the model can delegate to. Presented to the model as tools (by
    /// id) alongside `tools`, but each call spawns a child session rather than
    /// executing a function.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sub_agents: Vec<SubAgent>,
}

/// A function tool the agent offers. The model-facing contract is
/// `name`/`description`/`input`/`output`; `handler` selects where a call runs —
/// `Some(Client)` ⇒ client-executed, absent ⇒ worker-executed (the default).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AgentTool {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handler: Option<ToolHandler>,
}

/// A sub-agent the model can delegate to. Named by `id` (the child agent to spawn,
/// and the tool name the model calls); its model-facing input is the conventional
/// single-`message` delegation schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SubAgent {
    pub id: String,
    #[serde(default)]
    pub description: String,
}

// ── LLM requests and responses ───────────────────────────────────────────

/// A tool's declared contract: flat on the wire. Providers that need
/// OpenAI-style `{"type": "function", "function": {…}}` nesting re-wrap at
/// their own boundary.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct LlmTool {
    pub name: String,
    pub description: String,
    /// JSON Schema for the tool's arguments; omitted declares a no-argument
    /// tool. The engine validates each call's arguments against it and hands
    /// providers their native form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    /// JSON Schema the settled result must satisfy; never sent to the model.
    /// A violating result settles as a terminal tool error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct LlmRequest {
    pub model: String,
    pub messages: Vec<DraftMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<LlmTool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ReasoningConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Xhigh,
    High,
    Medium,
    Low,
    Minimal,
    None,
}

/// An image returned by the model in the response.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ResponseImage {
    pub url: String,
}

/// Normalized LLM response. Provider adapters convert their raw responses
/// into this type at the boundary.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct LlmResponse {
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Value>,
    /// Cost in dollars for this call, if the provider reports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<Decimal>,
    /// Images generated by the model.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<ResponseImage>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    ProviderError,
    RateLimited,
    Refused,
    BudgetExceeded,
    DeadlineExceeded,
}

// ── Streaming ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ToolCallChunk {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct StreamDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallChunk>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TokenDelta {
    /// Tenant isolation key — subscribers must match.
    pub tenant_id: String,
    /// Transport routing key.
    pub root_session_id: String,
    /// May be a sub-agent of root.
    pub session_id: String,
    pub agent_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    pub call_id: String,
    pub attempt: u32,
    /// Per-call counter, distinct from event-store sequence.
    pub seq: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallChunk>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

// ── Effects ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EffectStatus {
    Pending,
    Completed,
    Failed,
    RetryScheduled,
    Queued,
}

/// An in-flight effect (Pending or RetryScheduled) surfaced on each worker decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Effect {
    pub id: String,
    pub status: EffectStatus,
    pub attempt: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deadline: Option<DateTime<Utc>>,
    /// The tree node the effect was requested at.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
    #[serde(flatten)]
    pub detail: EffectDetail,
}

/// Kind-specific fields, tagged by `kind` on the wire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EffectDetail {
    ToolCall {
        name: String,
        arguments: String,
        handler: ToolHandler,
    },
    SubAgent {
        agent_id: String,
        session_id: String,
    },
    LlmCall {
        handler: LlmHandler,
        stream: bool,
    },
}

// ── Tool contract ────────────────────────────────────────────────────────

/// The engine's classification of a tool call's arguments, delivered on the
/// `tool.execute` trigger alongside the raw `arguments` string. Always on the
/// wire — absence never carries meaning.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "status", rename_all = "lowercase")]
pub enum ToolInput {
    /// Parsed and, when the tool declares an `input` schema, conforming to it.
    /// `value` is exactly the parsed `arguments` — the engine never mutates it.
    Valid { value: Value },
    /// Parsed to an object that violates the declared `input` schema.
    Invalid { value: Value, error: String },
    /// Not a JSON object: malformed JSON or a non-object value.
    Malformed { error: String },
}

// ── Client → engine ──────────────────────────────────────────────────────

/// The body of a `client.message`: one message, optionally streamed.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(inline)]
pub struct ClientMessage {
    pub message: DraftMessage,
    #[serde(default)]
    pub stream: bool,
}

/// The body of a `client.messages`: the client's full conversation view, optionally streamed.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(inline)]
pub struct ClientMessages {
    pub messages: Vec<DraftMessage>,
    #[serde(default)]
    pub stream: bool,
}

/// The payload of a `client.action`: a named action with optional JSON args.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(inline)]
pub struct ClientAction {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub args: Option<Value>,
}

/// The client→engine inbound *submit* wire form: an untrusted client submits a message,
/// its full conversation view, or a named action. Lowered to domain events at the
/// `SubmitClientPayload` command seam (`runtime::session::command`); never persisted
/// as-is. Carried verbatim inside [`ClientInput`], which is the full client input
/// surface.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type")]
pub enum ClientPayload {
    #[serde(rename = "client.message")]
    Message(ClientMessage),
    #[serde(rename = "client.messages")]
    Messages(ClientMessages),
    #[serde(rename = "client.action")]
    Action(ClientAction),
}

/// Everything a client can send on the input surface: submit a message / a full view / a
/// named action, resume an interrupt, or settle a client tool. A flat, internally-tagged
/// union — its six tags produce serde's "unknown variant, expected one of …" error for
/// free. `Runtime::handle_client_input` is the single seam that dispatches it (mirroring
/// `resolve_response` on the worker side).
///
/// Addressing lives where it is meaningful, not in a shared envelope: `agent_id` (routes
/// the turn, creating the session if new) and the optional idempotency `turn_id` are
/// fields of the three submit variants only. A resume/settle addresses an interrupt/effect
/// id and continues whatever turn is active, so it carries neither — misplacing them is
/// unrepresentable rather than rejected. `session_id` is the one universal address and
/// rides the envelope. A submit's body rebuilds a [`ClientPayload`] at the seam.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type")]
pub enum ClientInput {
    #[serde(rename = "client.message")]
    Message {
        agent_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        turn_id: Option<String>,
        message: DraftMessage,
        #[serde(default)]
        stream: bool,
    },
    #[serde(rename = "client.messages")]
    Messages {
        agent_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        turn_id: Option<String>,
        messages: Vec<DraftMessage>,
        #[serde(default)]
        stream: bool,
    },
    #[serde(rename = "client.action")]
    Action {
        agent_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        turn_id: Option<String>,
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        args: Option<Value>,
    },
    #[serde(rename = "interrupt.resume")]
    InterruptResume {
        interrupt_id: String,
        #[serde(default)]
        payload: Value,
    },
    #[serde(rename = "tool.result")]
    ToolResult {
        id: String,
        #[serde(default)]
        attempt: Option<u32>,
        #[serde(default)]
        result: Value,
    },
    #[serde(rename = "tool.error")]
    ToolError {
        id: String,
        error: String,
        retryable: bool,
        #[serde(default)]
        attempt: Option<u32>,
    },
}

// ── Engine → worker ──────────────────────────────────────────────────────

/// The trigger a worker sees on the wire — the materialized projection of the
/// engine's internal decision trigger. It has no `ClientMessage`: a bare client
/// message is always materialized to `ClientTranscript` by `to_wire_trigger`
/// (`runtime::session::wire`) before delivery, so an unmaterialized message can
/// never reach a worker.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type")]
pub enum DecisionTrigger {
    /// The first decision of every session; carries no proposal.
    #[serde(rename = "session.start")]
    SessionStart,
    #[serde(rename = "client.messages")]
    ClientTranscript {
        messages: Vec<DraftMessage>,
        #[serde(default)]
        new_from: usize,
    },
    #[serde(rename = "client.action")]
    ClientAction {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        args: Option<Value>,
    },
    /// Answer with `tool.result`/`tool.error`.
    #[serde(rename = "tool.execute")]
    ToolExecute {
        id: String,
        name: String,
        arguments: String,
        /// The engine's classification of `arguments` against the tool's
        /// declared `input` schema: `valid` (with the parsed `value`),
        /// `invalid` (value plus the violation), or `malformed` (not a JSON
        /// object). Always on the wire.
        input: ToolInput,
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
    #[serde(rename = "llm.finished")]
    LlmFinished {
        id: String,
        ok: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<DraftMessage>,
        #[serde(default)]
        truncated: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cost: Option<Decimal>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        code: Option<ErrorCode>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<Value>,
    },
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
    #[serde(rename = "interrupt.resumed")]
    InterruptResumed {
        interrupt_id: String,
        #[serde(default)]
        payload: Value,
    },
}

// ── Worker → engine ──────────────────────────────────────────────────────

/// The action a worker authors on the wire. Mirrors the internal `Action`, but a
/// settle's effect id may be omitted: on the sync/pull paths the answered
/// `*.execute` trigger names it, so echoing it is redundant. `resolve_response`
/// (`runtime::session::wire`) turns this into the internal `Action` (id always
/// present) at the transport boundary.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type")]
pub enum DecisionAction {
    /// A flat, all-optional LLM request. `id` omitted ⇒ the engine mints one; it
    /// becomes the assistant node's id. Omitted fields are filled from the agent
    /// config (merge source), then engine defaults; `messages` omitted ⇒
    /// `[config.system?] + the decision's declared view`. Explicit `messages`
    /// suppress system injection. A bare `{"type":"llm.call"}` prompts per the
    /// agent's identity over the current view.
    #[serde(rename = "llm.call")]
    CallLlm {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        messages: Option<Vec<DraftMessage>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tools: Option<Vec<LlmTool>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        temperature: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_completion_tokens: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning: Option<ReasoningConfig>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stream: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retry: Option<RetryPolicy>,
        /// Omitted ⇒ `server`.
        #[serde(default)]
        handler: LlmHandler,
    },
    /// `id` omitted ⇒ the engine mints one (LLM-driven tools carry the model's id).
    #[serde(rename = "tool.call")]
    CallTool {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        name: String,
        // Any JSON value; non-strings are canonicalized to JSON text.
        arguments: Value,
        /// Omitted ⇒ `worker`.
        #[serde(default)]
        handler: ToolHandler,
        #[serde(default = "RetryPolicy::no_retry")]
        retry: RetryPolicy,
    },
    /// `id` omitted ⇒ the effect named by the answering `tool.execute` trigger.
    #[serde(rename = "tool.result")]
    ToolResult {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        attempt: Option<u32>,
        // Any JSON value; non-strings are canonicalized to JSON text.
        result: Value,
    },
    /// `id` omitted ⇒ the effect named by the answering `llm.execute` trigger.
    #[serde(rename = "llm.result")]
    LlmResult {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        attempt: Option<u32>,
        response: LlmResponse,
    },
    #[serde(rename = "tool.error")]
    ToolError {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        attempt: Option<u32>,
        error: String,
        /// Omitted ⇒ terminal.
        #[serde(default)]
        retryable: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        code: Option<ErrorCode>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<Value>,
    },
    #[serde(rename = "llm.error")]
    LlmError {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        attempt: Option<u32>,
        error: String,
        /// Omitted ⇒ terminal.
        #[serde(default)]
        retryable: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        code: Option<ErrorCode>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<Value>,
    },
    #[serde(rename = "sub_agent.spawn")]
    SpawnSubAgent {
        session_id: String,
        agent_id: String,
        /// The model tool-call this delegation answers — always required.
        tool_call_id: String,
        #[serde(default = "RetryPolicy::no_retry")]
        retry: RetryPolicy,
    },
    #[serde(rename = "message.send")]
    SendMessage {
        session_id: String,
        message: DraftMessage,
    },
    /// `interrupt_id` omitted ⇒ the engine mints one to correlate the later resume.
    #[serde(rename = "interrupt")]
    Interrupt {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        interrupt_id: Option<String>,
        reason: String,
        #[serde(default)]
        payload: Value,
    },
    #[serde(rename = "done")]
    Done {
        #[serde(default)]
        data: Value,
    },
}

/// A decision: the messages/actions to author, plus optional state/agent writes.
/// The worker returns one; the engine also proposes one as the default
/// continuation (`DecisionRequest::proposed`), which the worker echoes or amends.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct DecisionResponse {
    #[serde(default)]
    pub messages: Vec<DraftMessage>,
    #[serde(default)]
    pub actions: Vec<DecisionAction>,
    /// Omitted or `null` keeps the current state; clear with a non-null empty value.
    #[serde(default)]
    pub state: Option<WorkerState>,
    /// A new agent config write; omitted keeps the current config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<AgentConfig>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct DecisionRequest<'a> {
    pub session_id: &'a str,
    pub decision_id: &'a str,
    pub agent_id: &'a str,
    pub identity: &'a SessionOwner,
    pub trigger: &'a DecisionTrigger,
    /// The engine's default continuation for `trigger` (`null` when it needs
    /// worker knowledge). Advisory: accept by echoing it as the decision.
    pub proposed: &'a Option<DecisionResponse>,
    pub state: &'a WorkerState,
    /// The agent config resolved for the active path (`null` when none is set).
    pub agent: &'a Option<AgentConfig>,
    pub calls: &'a [Effect],
    /// Count of in-flight `tool_call`/`sub_agent` calls.
    pub pending_calls: usize,
    pub messages: &'a [Message],
    pub message_tree: &'a MessageTree,
    pub ancestry: &'a [String],
    pub attempts: u32,
    pub deadline: &'a Option<DateTime<Utc>>,
    pub turn_id: &'a Option<String>,
}
