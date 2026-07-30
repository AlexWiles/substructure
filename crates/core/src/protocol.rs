//! The public protocol: every type that crosses the client or worker wire.
//! Types only — no logic. Conversions and seams live with the engine
//! (`runtime::session::wire`, `runtime::session::propose`, …). Every type
//! derives [`JsonSchema`]; the schemas under `schemas/` are generated from them.

use std::collections::{BTreeMap, HashMap};

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// `Decimal` serializes as a string (`rust_decimal` default); pin the schema
/// to match instead of schemars' string-or-number union.
fn decimal_string_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
    schemars::json_schema!({
        "type": ["string", "null"],
        "pattern": r"^-?\d+(\.\d+)?$",
    })
}

// ── Messages ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
#[schemars(title = "Role")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "ToolCallFunction")]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "ToolCall")]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolCallFunction,
}

// Multimodal content parts (OpenAI/OpenRouter wire format).

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "ImageUrl")]
pub struct ImageUrl {
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "FileData")]
pub struct FileData {
    pub filename: String,
    pub file_data: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "AudioData")]
pub struct AudioData {
    pub data: String,
    pub format: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "VideoUrl")]
pub struct VideoUrl {
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
#[schemars(title = "ContentPart")]
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
#[schemars(title = "Content")]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "Message")]
pub struct Message {
    pub id: String,
    pub role: Role,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Content>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
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
#[schemars(title = "DraftMessage")]
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

/// Privilege level of the caller that issued an interrupt. Derived from the
/// authenticated `Caller`, never from request data; resuming requires a
/// caller at or above the origin's privilege.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "InterruptOrigin")]
pub enum InterruptOrigin {
    System,
    Machine,
    Frontend,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "MessageTree")]
pub struct MessageTree {
    pub nodes: Vec<NewMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_id: Option<String>,
}

// ── Handlers ─────────────────────────────────────────────────────────────

/// Where a call runs — one wire enum so `handler` has a single type on every
/// surface. Tool calls accept `worker` (default), `client`, or `server` (set by
/// the engine for connector tools, never declared by a worker); LLM calls accept
/// `server` (default) or `worker`. The invalid pairing (a `client` LLM call) is
/// rejected at the decision seam.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "Handler")]
pub enum Handler {
    /// Server-side executor resolves the provider or connection and makes the call.
    Server,
    /// Dispatched to the work queue for the worker to execute.
    Worker,
    /// Executed by the client. Session goes Idle while waiting (tools only).
    Client,
}

/// The wire shape of a worker-handled LLM call. Absent ⇒ the engine's neutral
/// format. Set ⇒ `llm.execute` carries the provider's native request body, and
/// `llm.result`/`llm.token.delta` accept the provider's native response and
/// stream events. Requires `handler: worker`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "LlmFormat")]
pub enum LlmFormat {
    /// OpenAI Chat Completions.
    Openai,
    /// Anthropic Messages API.
    Anthropic,
}

// ── Retry ────────────────────────────────────────────────────────────────

/// Fully-resolved retry policy — no optional fields. Stored on call state and
/// read directly by retry logic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "RetryPolicy")]
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

/// The owner as delivered to the worker on `DecisionRequest.identity`: the
/// subject and its metadata, without the tenant. The tenant scopes the session
/// internally but is not sent to the worker.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "WorkerIdentity")]
pub struct WorkerIdentity {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub metadata: HashMap<String, String>,
}

// ── Worker state ─────────────────────────────────────────────────────────

/// Opaque worker state: JSON the engine stores but never interprets.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(transparent)]
#[schemars(title = "WorkerState")]
pub struct WorkerState(pub Value);

// ── Agent config ─────────────────────────────────────────────────────────

/// A declared agent identity. `model` is the only required field; everything else
/// refines the proposed LLM request the engine derives for `client.messages`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "AgentConfig")]
pub struct AgentConfig {
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(default)]
    pub stream: bool,
    /// Where the proposed LLM call runs: `Some(Worker)` ⇒ the worker executes it
    /// (answering `llm.execute`); absent or `Some(Server)` ⇒ the engine's
    /// server-side provider. `client` is invalid and rejected at the decision seam.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handler: Option<Handler>,
    /// Provider wire format for worker-handled calls; requires `handler:
    /// worker`. Absent ⇒ the neutral format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<LlmFormat>,
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
    /// MCP servers the agent draws tools from. The engine resolves each against
    /// its connection registry into [`ConnectorTool`]s the model sees alongside
    /// `tools`. Like `sub_agents`, these are never merged into `tools` — the
    /// worker declares the server, not its tools.
    ///
    /// A second protocol gets its own field rather than a `type` tag here: its
    /// filter would not be this one (MCP annotations mean nothing to an A2A
    /// agent), and a union of conditionally-valid fields generates badly in the
    /// Go and Python bindings.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mcp: Vec<McpServer>,
}

/// A function tool the agent offers. The model-facing contract is
/// `name`/`description`/`input`/`output`; `handler` selects where a call runs —
/// `Some(Client)` ⇒ client-executed, absent ⇒ worker-executed (the default).
/// `server` is invalid for tools: engine-executed tools come from a connector,
/// which a worker declares by id rather than by tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "AgentTool")]
pub struct AgentTool {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handler: Option<Handler>,
}

/// One tool the engine resolved from a connector and will execute itself.
///
/// Derived, not stored: the session records what a connection *offered*
/// (`connector.sync.completed`) and re-derives this by filtering that offer
/// through the config in force. So replay is deterministic — a connection that
/// changes its tools underneath a live session cannot rewrite what already
/// happened — while a filter change costs no round trip.
/// `name` is what the model sees; `remote_name` is what the executor calls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "ConnectorTool")]
pub struct ConnectorTool {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,
    pub connector: String,
    pub via: ConnectorProtocol,
    pub remote_name: String,
}

/// How the engine reaches a connection. Internal: the config says which
/// protocol by which section a connection is declared under, and an agent names
/// a connection by id without knowing. Adding A2A is a variant here plus a
/// section, not a change to either surface.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "ConnectorProtocol")]
pub enum ConnectorProtocol {
    #[default]
    Mcp,
}

/// An MCP server the agent draws tools from. `id` resolves against the engine's
/// connection registry — locally from `[mcp]` in `substructure.toml`, in the
/// cloud from the connections an admin granted this app. The worker never names
/// a URL or a credential.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "McpServer")]
pub struct McpServer {
    pub id: String,
    /// Narrows what the model sees. Absent ⇒ every tool the connection grants.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<McpTools>,
}

/// An MCP server's tool filter. Applied in order — capability predicates, then
/// `include`, then `exclude` — and only ever narrowing, so a filter can never
/// widen what the connection grants.
///
/// `include`/`exclude` are globs matched against the tool's name on the
/// connection, the name its own documentation uses, not the prefixed name the
/// model sees. Capability predicates read the MCP annotations; a tool that
/// carries none fails the predicate, so an unannotated server yields nothing
/// under `read_only` rather than silently passing everything through.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "McpTools")]
pub struct McpTools {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub include: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exclude: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub read_only: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub non_destructive: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotent: Option<bool>,
}

/// A sub-agent the model can delegate to. Named by `id` (the child agent to spawn,
/// and the tool name the model calls); its model-facing input is the conventional
/// single-`message` delegation schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "SubAgent")]
pub struct SubAgent {
    pub id: String,
    #[serde(default)]
    pub description: String,
}

/// Inputs a client declares on its run (the AG-UI `tools`/`context`/`state`/
/// `forwardedProps`), forwarded to the worker on the `client.messages` decision.
/// `tools` are the browser's frontend tools, normalized to client-handled
/// [`AgentTool`]s; the engine layers them onto the proposed config by default, and
/// the worker may override (e.g. whitelist) by returning its own `agent`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "ClientContext")]
pub struct ClientContext {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<AgentTool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub context: Vec<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forwarded_props: Option<Value>,
}

// ── LLM requests and responses ───────────────────────────────────────────

/// A tool's declared contract: flat on the wire. Providers that need
/// OpenAI-style `{"type": "function", "function": {…}}` nesting re-wrap at
/// their own boundary.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "LlmTool")]
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
#[schemars(title = "LlmRequest")]
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
#[schemars(title = "ReasoningConfig")]
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
#[schemars(title = "ReasoningEffort")]
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
#[schemars(title = "ResponseImage")]
pub struct ResponseImage {
    pub url: String,
}

/// Normalized LLM response. Provider adapters convert their raw responses
/// into this type at the boundary.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "LlmResponse")]
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
    /// Cost in dollars for this call, if the provider reports it. A decimal
    /// string on the wire.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(schema_with = "decimal_string_schema")]
    pub cost: Option<Decimal>,
    /// Images generated by the model.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<ResponseImage>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "ErrorCode")]
pub enum ErrorCode {
    ProviderError,
    RateLimited,
    Refused,
    BudgetExceeded,
    DeadlineExceeded,
}

// ── Streaming ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "ToolCallChunk")]
pub struct ToolCallChunk {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "StreamDelta")]
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
#[schemars(title = "TokenDelta")]
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
    pub tool_calls: Vec<ToolCallChunk>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

// ── Effects ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "EffectStatus")]
pub enum EffectStatus {
    Pending,
    Completed,
    Failed,
    RetryScheduled,
    Queued,
}

/// What kind of work an effect is. One enum for the wire and for the engine's
/// own scheduling: a decision and a turn's end queue beside the calls and are
/// swept the same way, so they are kinds too. Neither ever appears on an
/// [`Effect`] — a decision rides the decision list, a turn end has no record.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "EffectKind")]
pub enum EffectKind {
    ToolCall,
    SubAgent,
    LlmCall,
    /// Fetching one connection's tool list. Its `id` is the connection id.
    ConnectorSync,
    /// A worker decision.
    Decision,
    /// The turn's completion, dependent on its `turn.finished` finalizer
    /// decision settling. Carries the turn id; the frozen output lives in the
    /// session's `finalizing`. Never swept: it has no deadline of its own.
    TurnEnd,
}

impl EffectKind {
    pub fn label(self) -> &'static str {
        match self {
            EffectKind::ToolCall => "tool_call",
            EffectKind::SubAgent => "sub_agent",
            EffectKind::LlmCall => "llm_call",
            EffectKind::ConnectorSync => "connector_sync",
            EffectKind::Decision => "decision",
            EffectKind::TurnEnd => "turn_end",
        }
    }
}

/// An in-flight effect (Pending or RetryScheduled) surfaced on each worker decision.
/// A flat envelope plus kind-specific fields: a tool call's
/// `name`/`arguments`/`handler`, an LLM call's `handler`/`stream`, a
/// sub-agent's `agent_id`/`session_id`. A connector sync carries none — its
/// `id` is the connection being fetched.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "Effect")]
pub struct Effect {
    pub id: String,
    pub kind: EffectKind,
    pub status: EffectStatus,
    pub attempt: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deadline: Option<DateTime<Utc>>,
    /// The tree node the effect was requested at.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handler: Option<Handler>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    /// The model tool call a delegation answers; its own `id` is the child session.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

// ── Tool contract ────────────────────────────────────────────────────────

/// The engine's classification of a tool call's arguments, delivered on the
/// `tool.execute` trigger alongside the raw `arguments` string. Always on the
/// wire — absence never carries meaning.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "status", rename_all = "lowercase")]
#[schemars(title = "ToolInput")]
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
    #[serde(default)]
    pub client: ClientContext,
}

/// The body of a `client.append`: messages appended at the session head. The
/// view is composed against the active path at delivery, so a queued append
/// lands after whatever turn beat it — it can never fork the tree. Messages
/// whose ids are already recorded are dropped.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(inline)]
pub struct ClientAppend {
    pub messages: Vec<DraftMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub client: ClientContext,
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
/// its full conversation view, an append batch, or a named action. Lowered to domain events at the
/// `SubmitClientPayload` command seam (`runtime::session::command`); never persisted
/// as-is. Carried verbatim inside [`ClientInput`], which is the full client input
/// surface.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type")]
#[schemars(title = "ClientPayload")]
pub enum ClientPayload {
    #[serde(rename = "client.message")]
    Message(ClientMessage),
    #[serde(rename = "client.messages")]
    Messages(ClientMessages),
    #[serde(rename = "client.append")]
    Append(ClientAppend),
    #[serde(rename = "client.action")]
    Action(ClientAction),
}

/// Everything a client can send on the input surface: submit a message / a full view / an
/// append batch / a named action, resume an interrupt, or settle a client tool. A flat,
/// internally-tagged union — its seven tags produce serde's "unknown variant, expected one
/// of …" error for free. `Runtime::handle_client_input` is the single seam that dispatches
/// it (mirroring `resolve_response` on the worker side).
///
/// Addressing lives where it is meaningful, not in a shared envelope: `agent_id` (routes
/// the turn, creating the session if new) and the optional idempotency `turn_id` are
/// fields of the four submit variants only. A resume/settle addresses an interrupt/effect
/// id and continues whatever turn is active, so it carries neither — misplacing them is
/// unrepresentable rather than rejected. `session_id` is the one universal address and
/// rides the envelope. A submit's body rebuilds a [`ClientPayload`] at the seam.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type")]
#[schemars(title = "ClientInput")]
pub enum ClientInput {
    #[serde(rename = "client.message")]
    Message {
        agent_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        turn_id: Option<String>,
        message: DraftMessage,
        #[serde(default)]
        stream: bool,
        /// Hold this message for the next turn instead of refusing it when one
        /// is already running. Off by default: rejection stays the contract for
        /// a plain submitter, and queuing is declared intent.
        #[serde(default)]
        queue: bool,
    },
    #[serde(rename = "client.messages")]
    Messages {
        agent_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        turn_id: Option<String>,
        messages: Vec<DraftMessage>,
        #[serde(default)]
        stream: bool,
        #[serde(default)]
        client: ClientContext,
    },
    #[serde(rename = "client.append")]
    Append {
        agent_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        turn_id: Option<String>,
        messages: Vec<DraftMessage>,
        #[serde(default)]
        stream: bool,
        #[serde(default)]
        client: ClientContext,
        /// Hold this batch for the next turn instead of refusing it when one is
        /// already running. Off by default: rejection stays the contract for a
        /// plain submitter, and queuing is declared intent.
        #[serde(default)]
        queue: bool,
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
        #[serde(flatten)]
        resumption: InterruptResumption,
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

/// The body of an interrupt resume: which interrupt, and the payload delivered
/// to the worker. Shared by the [`ClientInput::InterruptResume`] input and the
/// [`DecisionTrigger::InterruptResumed`] trigger.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct InterruptResumption {
    pub interrupt_id: String,
    #[serde(default)]
    pub payload: Value,
}

// ── Interrupt payload convention ─────────────────────────────────────────
//
// Offered vocabulary, never enforced: interrupt payloads stay opaque on the
// wire, but a payload shaped like the AG-UI Interrupt renders in every
// channel (Slack buttons, the AG-UI Interrupt object), and resumes authored
// by channels arrive as an [`InterruptResolution`]. Top-level keys come from
// the AG-UI spec; everything convention-specific rides `metadata`, which is
// client-visible by definition — anything private belongs in worker state,
// not the payload. Workers should treat an unrecognized resolution as their
// safe default — a resume can carry any payload.

/// An interrupt payload following the AG-UI Interrupt shape (spec spelling;
/// `id` and `reason` live on the interrupt itself).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[schemars(title = "InterruptPayload")]
pub struct InterruptPayload {
    /// Markdown; channels down-convert. Without it, channels fall back to
    /// the interrupt's `reason`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Binds the interrupt to a prior tool call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// JSON Schema for the expected resolution payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<Value>,
    /// RFC 3339; display only until engine TTLs land.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<String>,
    /// Free-form, delivered to clients verbatim. `metadata.options`
    /// ([`InterruptOption`] list) renders as Slack buttons.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// One enumerated response under `metadata.options` (Slack: a button). A
/// click resumes with the option's `value` as the resolution payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "InterruptOption")]
pub struct InterruptOption {
    pub label: String,
    /// Delivered verbatim as the resolution's `payload`; worker vocabulary.
    pub value: Value,
    /// `primary` or `danger`; anything else renders plain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
}

/// A channel-authored resume payload: the AG-UI resume shape
/// (`{status, payload}`) plus a provenance stamp.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "InterruptResolution")]
pub struct InterruptResolution {
    pub status: ResumeStatus,
    #[serde(default)]
    pub payload: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub responder: Option<InterruptResponder>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
#[schemars(title = "ResumeStatus")]
pub enum ResumeStatus {
    Resolved,
    Cancelled,
}

/// Who resolved it, stamped by the channel — never by the requester.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "InterruptResponder")]
pub struct InterruptResponder {
    /// The channel kind, e.g. `slack`, `ag-ui`.
    pub channel: String,
    /// Channel-native user id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// The chosen option's label, when the resolution was a pick.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

// ── Engine → worker ──────────────────────────────────────────────────────

/// The trigger a worker sees on the wire — the materialized projection of the
/// engine's internal decision trigger. It has no `ClientMessage`: a bare client
/// message is always materialized to `ClientTranscript` by `to_wire_trigger`
/// (`runtime::session::wire`) before delivery, so an unmaterialized message can
/// never reach a worker.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type")]
#[schemars(title = "DecisionTrigger")]
pub enum DecisionTrigger {
    /// The first decision of every session; carries no proposal.
    #[serde(rename = "session.start")]
    SessionStart,
    #[serde(rename = "client.messages")]
    ClientTranscript {
        messages: Vec<DraftMessage>,
        new_from: usize,
        /// Inputs the client declared on its run; the engine layers `client.tools`
        /// onto the proposed config by default.
        client: ClientContext,
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
        /// The neutral `LlmRequest` JSON, or the provider's native request body
        /// when `format` is set.
        request: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        format: Option<LlmFormat>,
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
        truncated: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[schemars(schema_with = "decimal_string_schema")]
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
        #[serde(flatten)]
        resumption: InterruptResumption,
    },
    /// Fired after a turn completes, carrying its final output; blocks the session
    /// going idle until answered. Echo the proposed `done` to finalize.
    #[serde(rename = "turn.finished")]
    TurnFinished {
        turn_id: String,
        #[serde(default, skip_serializing_if = "Value::is_null")]
        data: Value,
        #[serde(default)]
        #[schemars(schema_with = "decimal_string_schema")]
        cost: Decimal,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        usage: BTreeMap<String, u64>,
    },
}

impl DecisionTrigger {
    /// Compact wire tag for logs, without the payload.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::SessionStart => "session.start",
            Self::ClientTranscript { .. } => "client.messages",
            Self::ClientAction { .. } => "client.action",
            Self::ToolExecute { .. } => "tool.execute",
            Self::ToolFinished { .. } => "tool.finished",
            Self::LlmExecute { .. } => "llm.execute",
            Self::LlmFinished { .. } => "llm.finished",
            Self::SubAgentFinished { .. } => "sub_agent.finished",
            Self::InterruptResumed { .. } => "interrupt.resumed",
            Self::TurnFinished { .. } => "turn.finished",
        }
    }
}

// ── Worker → engine ──────────────────────────────────────────────────────

/// The action a worker authors on the wire. Mirrors the internal `Action`, but a
/// settle's effect id may be omitted: on the sync/pull paths the answered
/// `*.execute` trigger names it, so echoing it is redundant. `resolve_response`
/// (`runtime::session::wire`) turns this into the internal `Action` (id always
/// present) at the transport boundary.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type")]
#[schemars(title = "DecisionAction")]
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
        /// `server` or `worker`; omitted ⇒ `server`.
        #[serde(default = "Handler::server")]
        handler: Handler,
    },
    /// `id` omitted ⇒ the engine mints one (LLM-driven tools carry the model's id).
    ///
    /// There is no `handler`: where a call runs follows from its name. A tool
    /// resolved from a connector runs on the engine, a tool declared
    /// `handler: client` runs on the client, and anything else runs on the
    /// worker. The engine already knows all three, so asking the worker to
    /// restate it only creates a way for the two to disagree.
    #[serde(rename = "tool.call")]
    CallTool {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        name: String,
        // Any JSON value; non-strings are canonicalized to JSON text.
        arguments: Value,
        #[serde(default = "RetryPolicy::no_retry")]
        retry: RetryPolicy,
    },
    /// `id`/`attempt` omitted ⇒ taken from the answering `tool.execute` trigger,
    /// fencing the result to the attempt that ran.
    #[serde(rename = "tool.result")]
    ToolResult {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        attempt: Option<u32>,
        // Any JSON value; non-strings are canonicalized to JSON text.
        result: Value,
    },
    /// `id`/`attempt` omitted ⇒ taken from the answering `llm.execute` trigger,
    /// fencing the result to the attempt that ran.
    #[serde(rename = "llm.result")]
    LlmResult {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        attempt: Option<u32>,
        /// A neutral `LlmResponse`, or the provider's native response when the
        /// answered `llm.execute` carried a `format`.
        response: Value,
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
#[schemars(title = "DecisionResponse")]
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
#[schemars(title = "DecisionRequest")]
pub struct DecisionRequest<'a> {
    pub session_id: &'a str,
    pub decision_id: &'a str,
    pub agent_id: &'a str,
    pub identity: WorkerIdentity,
    pub trigger: &'a DecisionTrigger,
    /// The engine's default continuation for `trigger` (empty when it needs
    /// worker knowledge). Advisory: accept by echoing it as the decision.
    pub proposed: &'a DecisionResponse,
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
