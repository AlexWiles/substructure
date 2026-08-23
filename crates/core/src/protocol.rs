//! The public protocol: every type that crosses the client or worker wire.
//! Types only — no logic. Conversions and seams live with the engine
//! (`runtime::session::wire`, `runtime::session::propose`, …). Every type
//! derives [`JsonSchema`]; the schemas under `schemas/` are generated from them.

use std::collections::{BTreeMap, HashMap};
use std::num::NonZeroUsize;

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Where a connection is declared, which is what a file, the CLI, and the wire
/// name it by.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConnectionPath {
    Mcp(String),
    PluginServer { plugin: String, server: String },
}

impl ConnectionPath {
    pub fn parse(path: &str) -> Option<Self> {
        if let Some(id) = path.strip_prefix("mcp.") {
            return (!id.is_empty() && !id.contains('.')).then(|| Self::Mcp(id.to_string()));
        }
        let rest = path.strip_prefix("plugin.")?;
        let (plugin, server) = rest.split_once(".mcp.")?;
        let named = !plugin.is_empty() && !server.is_empty();
        let flat = !plugin.contains('.') && !server.contains('.');
        (named && flat).then(|| Self::PluginServer {
            plugin: plugin.to_string(),
            server: server.to_string(),
        })
    }

    /// The model-facing prefix. Anything a provider would reject in a tool
    /// name is flattened to `_`, here rather than where names are built, so
    /// validation and expansion cannot disagree about a collision.
    pub fn tool_prefix(&self) -> String {
        let raw = match self {
            Self::Mcp(id) => id.clone(),
            Self::PluginServer { plugin, server } => format!("{plugin}_{server}"),
        };
        raw.chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect()
    }
}

impl Serialize for ConnectionPath {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for ConnectionPath {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let written = String::deserialize(d)?;
        Self::parse(&written).ok_or_else(|| {
            serde::de::Error::custom(format!(
                "`{written}` is not a connection path: `mcp.<id>` or `plugin.<id>.mcp.<server>`"
            ))
        })
    }
}

impl JsonSchema for ConnectionPath {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "ConnectionPath".into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        let mut schema = <String as JsonSchema>::json_schema(generator);
        schema.insert(
            "description".to_string(),
            "Where a connection is declared: `mcp.<id>` or `plugin.<id>.mcp.<server>`".into(),
        );
        schema
    }
}

impl std::fmt::Display for ConnectionPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mcp(id) => write!(f, "mcp.{id}"),
            Self::PluginServer { plugin, server } => write!(f, "plugin.{plugin}.mcp.{server}"),
        }
    }
}

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

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
#[schemars(title = "ResourceContents")]
pub struct ResourceContents {
    pub uri: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob: Option<String>,
}

impl ToolResult {
    pub fn from_action(
        result: Option<Value>,
        content: Option<Vec<ToolContent>>,
        structured_content: Option<Value>,
        is_error: bool,
    ) -> Result<Self, &'static str> {
        let content = match (result, content) {
            (Some(_), Some(_)) => return Err("a tool result names both `result` and `content`"),
            (Some(value), None) => Self::from_value(value).content,
            (None, Some(content)) => content,
            (None, None) => Vec::new(),
        };
        Ok(Self {
            content,
            structured_content,
            is_error,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
#[schemars(title = "ToolContent")]
pub enum ToolContent {
    Text {
        text: String,
    },
    #[serde(rename_all = "camelCase")]
    Image {
        data: String,
        mime_type: String,
    },
    #[serde(rename_all = "camelCase")]
    Audio {
        data: String,
        mime_type: String,
    },
    Resource {
        resource: ResourceContents,
    },
    #[serde(rename_all = "camelCase")]
    ResourceLink {
        uri: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
    },
}

impl ToolContent {
    pub fn inline(&self) -> Option<(&str, &str)> {
        match self {
            Self::Image { data, mime_type } | Self::Audio { data, mime_type } => {
                Some((data, mime_type))
            }
            Self::Resource { resource } => Some((
                resource.blob.as_deref()?,
                resource.mime_type.as_deref().unwrap_or(OCTET_STREAM),
            )),
            Self::Text { .. } | Self::ResourceLink { .. } => None,
        }
    }

    pub fn name(&self) -> Option<&str> {
        match self {
            Self::Resource { resource } => {
                Some(resource.uri.rsplit('/').next().unwrap_or(&resource.uri))
            }
            Self::ResourceLink { name, .. } => name.as_deref(),
            _ => None,
        }
    }
}

pub const OCTET_STREAM: &str = "application/octet-stream";

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
#[schemars(title = "StoredContent")]
pub enum StoredContent {
    Text {
        text: String,
    },
    Blob {
        uri: String,
    },
    #[serde(rename_all = "camelCase")]
    Link {
        uri: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
#[schemars(title = "StoredResult")]
pub struct StoredResult {
    #[serde(default)]
    pub content: Vec<StoredContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured_content: Option<Value>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub is_error: bool,
}

impl StoredResult {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![StoredContent::Text { text: text.into() }],
            ..Default::default()
        }
    }

    /// A tool that ran and reported failure.
    pub fn error(text: impl Into<String>) -> Self {
        Self {
            is_error: true,
            ..Self::text(text)
        }
    }

    pub fn as_text(&self) -> String {
        self.content
            .iter()
            .filter_map(|c| match c {
                StoredContent::Text { text } => Some(text.clone()),
                StoredContent::Link { uri, .. } => Some(uri.clone()),
                StoredContent::Blob { .. } => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn rendered(&self) -> String {
        match &self.structured_content {
            Some(value) => value.to_string(),
            None => self.as_text(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
#[schemars(title = "ToolResult")]
pub struct ToolResult {
    #[serde(default)]
    pub content: Vec<ToolContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured_content: Option<Value>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub is_error: bool,
}

impl ToolResult {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ToolContent::Text { text: text.into() }],
            ..Default::default()
        }
    }

    pub fn from_value(value: Value) -> Self {
        Self::text(match value {
            Value::String(s) => s,
            Value::Null => String::new(),
            other => other.to_string(),
        })
    }

    pub fn as_text(&self) -> String {
        self.content
            .iter()
            .filter_map(|c| match c {
                ToolContent::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn rendered(&self) -> String {
        match &self.structured_content {
            Some(value) => value.to_string(),
            None => self.as_text(),
        }
    }
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
    Parts(Vec<StoredContent>),
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
#[schemars(title = "PromptContent")]
pub enum PromptContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl PromptContent {
    pub fn text_owned(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "PromptMessage")]
pub struct PromptMessage {
    pub role: Role,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<PromptContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Box<Reasoning>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "PromptRequest")]
pub struct PromptRequest {
    pub model: String,
    pub messages: Vec<PromptMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<LlmTool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
}

impl PromptRequest {
    pub fn offered_tools(&self, search: DeferToolsStrategy) -> Option<Vec<&LlmTool>> {
        Some(search.offered(self.tools.as_ref()?))
    }
}

/// Which provider wrote a [`Reasoning`]'s blocks. They ride back only to it:
/// another provider reads them as noise, and Anthropic rejects blocks it did
/// not sign.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "ReasoningProvider")]
pub enum ReasoningProvider {
    Anthropic,
    Openai,
    Openrouter,
}

/// What the model thought before it answered. `text` is for a reader; `blocks`
/// are the provider's own, held verbatim because Anthropic requires the
/// thinking that precedes a tool call back unmodified, signature included.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "Reasoning")]
pub struct Reasoning {
    pub provider: ReasoningProvider,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blocks: Vec<serde_json::Value>,
}

impl Reasoning {
    /// The reasoning of one response, or nothing where it thought in the open
    /// and left no blocks to return.
    /// Boxed: a response carrying one is passed by value through every effect
    /// and event, and the blocks are dead weight on the paths that never read
    /// them.
    pub fn new(
        provider: ReasoningProvider,
        text: Option<String>,
        blocks: Vec<serde_json::Value>,
    ) -> Option<Box<Self>> {
        (text.is_some() || !blocks.is_empty()).then(|| {
            Box::new(Self {
                provider,
                text,
                blocks,
            })
        })
    }

    /// The blocks, for the provider that wrote them and no other.
    pub fn blocks_for(&self, provider: ReasoningProvider) -> &[serde_json::Value] {
        if self.provider == provider {
            &self.blocks
        } else {
            &[]
        }
    }
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Box<Reasoning>>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Box<Reasoning>>,
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
    Operator,
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
/// the engine for connector tools, never declared by a worker). An LLM call has
/// no `handler`: where it runs follows from the `[llm.*]` block it names.
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

/// The wire shape of a worker-run LLM call, declared on a `type = "worker"`
/// `[llm.*]` block. Absent ⇒ the engine's neutral format. Set ⇒ `llm.execute`
/// carries the provider's native request body, and `llm.result`/
/// `llm.token.delta` accept the provider's native response and stream events.
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

/// Fully-resolved retry policy. Stored on call state and read directly by retry
/// logic.
///
/// Two timeouts, because one span cannot express both bounds: `attempt` covers a
/// single dispatch-to-settle, `total` covers the whole effect — every attempt,
/// the backoff between them, and any time spent `Running`. An effect with only
/// an attempt timeout can still stall forever once it stops being `Pending`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "RetryPolicy")]
pub struct RetryPolicy {
    pub attempt_timeout_secs: Option<u32>,
    pub total_timeout_secs: Option<u32>,
    /// Cap on total attempts, not on retries: `1` allows one try and no retry.
    pub max_attempts: u32,
    pub backoff_base_secs: u32,
    pub backoff_max_secs: u32,
}

/// A partial retry policy: only the fields it names change, and the rest are
/// inherited. Every override is a layer over the engine's default for the effect
/// kind, so tuning one knob does not mean restating the other four — and leaving
/// a timeout out keeps the default bound rather than removing it.
///
/// An override cannot set a timeout back to unbounded. Waiting effectively
/// forever is a large number, which is also the honest way to say it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "RetryOverride")]
pub struct RetryOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attempt_timeout_secs: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_timeout_secs: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_attempts: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backoff_base_secs: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backoff_max_secs: Option<u32>,
}

/// An agent's retry overrides, one per effect kind. `default` covers the kinds
/// that name nothing; a kind is layered on top of it, so the two compose.
///
/// Per kind because the kinds are not alike: an LLM call is idempotent and worth
/// retrying, a tool call may not be, and a connector fetch holds up every
/// decision behind it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "RetryConfig")]
pub struct RetryConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<RetryOverride>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm: Option<RetryOverride>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool: Option<RetryOverride>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sub_agent: Option<RetryOverride>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connector: Option<RetryOverride>,
}

// ── Identity ─────────────────────────────────────────────────────────────

/// Where a person's name comes from — `slack`, `app`, `cli`, or whatever a
/// deployment registers. Stamped by whatever authenticated the request, and
/// never read out of one: a caller free to name its own issuer could name
/// another source's people.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(transparent)]
#[schemars(title = "Issuer")]
pub struct Issuer(String);

impl Issuer {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// The one person at an installation nothing authenticates.
    pub fn cli() -> Self {
        Self("cli".into())
    }

    pub fn slack() -> Self {
        Self("slack".into())
    }

    /// A login on this deployment: whoever configures it.
    pub fn operator() -> Self {
        Self("operator".into())
    }

    /// An end user of the project's own application, named in a client token.
    pub fn app() -> Self {
        Self("app".into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Issuer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// One identity, as the source that authenticated it named it: OIDC's
/// `(iss, sub)`. An id means nothing without its issuer, because it is only
/// unique within one.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "Subject")]
pub struct Subject {
    pub issuer: Issuer,
    pub id: String,
}

impl Subject {
    pub fn new(issuer: Issuer, id: impl Into<String>) -> Self {
        Self {
            issuer,
            id: id.into(),
        }
    }
}

impl std::fmt::Display for Subject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.issuer, self.id)
    }
}

/// Who can read what a session says. The transport sets it once, at the
/// session's start; everything absent or unknown reads as `shared`, because
/// `shared` is the value that never selects a personal credential.
///
/// Not OAuth's `aud`, which names a resource server rather than a readership.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "Visibility")]
pub enum Visibility {
    /// More than one person can read the answer.
    #[default]
    Shared,
    /// One person only.
    Private,
}

/// Who a session runs for. No subject is a schedule, a key, or the engine
/// itself — nobody whose own credential could apply.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "Requester")]
pub struct Requester {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subject: Option<Subject>,
    #[serde(default)]
    pub visibility: Visibility,
}

impl Requester {
    /// Nobody in particular.
    pub fn machine() -> Self {
        Self::default()
    }

    /// The requester a session runs for. A session with no owner is nobody's.
    pub fn of_owner(owner: Option<&SessionOwner>) -> Self {
        owner.map(|o| o.requester.clone()).unwrap_or_default()
    }

    /// One person, in a conversation only they can read.
    pub fn private(subject: Subject) -> Self {
        Self {
            subject: Some(subject),
            visibility: Visibility::Private,
        }
    }

    pub fn new(subject: Subject, visibility: Visibility) -> Self {
        Self {
            subject: Some(subject),
            visibility,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct SessionOwner {
    pub tenant_id: String,
    #[serde(flatten)]
    pub requester: Requester,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
}

/// The owner as delivered to the worker on `DecisionRequest.identity`, without
/// the tenant. Read `kind` with `id`: only `frontend` is an end user.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "WorkerIdentity")]
pub struct WorkerIdentity {
    #[serde(flatten)]
    pub requester: Requester,
    pub metadata: HashMap<String, String>,
}

// ── Worker state ─────────────────────────────────────────────────────────

/// Opaque worker state: JSON the engine stores but never interprets.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(transparent)]
#[schemars(title = "WorkerState")]
pub struct WorkerState(pub Value);

// ── Agent config ─────────────────────────────────────────────────────────

/// A declared agent identity — the same shape whether it is written in an
/// `[agent.<id>]` section or returned by a worker.
///
/// `llm` names the `[llm.*]` block every proposed call runs on, and so decides
/// both the venue (the engine with a vendor key, or the agent's own worker) and
/// the wire shape of a worker-run call. It is effectively required: a config
/// that names none fails when the engine resolves a call against it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "AgentConfig")]
pub struct AgentConfig {
    /// The `[llm.*]` block this agent's calls run on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm: Option<String>,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// How hard the model thinks, carried on the agent because it pairs with
    /// the model. Unset sends no reasoning config and leaves the provider its
    /// own default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
    /// Boxed: five per-kind overrides is a lot of bytes to carry inline
    /// through every command that holds a config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry: Option<Box<RetryConfig>>,
    /// Worker- or client-executed tools the model can call.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<AgentTool>,
    /// Sub-agents the model can delegate to. Presented to the model as tools (by
    /// id) alongside `tools`, but each call spawns a child session rather than
    /// executing a function.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sub_agents: Vec<SubAgent>,
    /// MCP servers
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mcp: Vec<McpServer>,
    /// Plugins this agent uses.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub plugins: Vec<AgentPlugin>,
    /// Defer every tool this agent offers, from any source, unless the tool or
    /// the connection says otherwise. Absent ⇒ the agent defers nothing of its
    /// own; a connection may still defer on its own account.
    ///
    /// Presence is the switch, so an agent cannot carry settings that do
    /// nothing. Declared on the agent because an agent can hold this opinion
    /// before it names a connection: one that sets it gets the search tools
    /// from its first turn, so a connection added later costs no cache.
    #[serde(
        default,
        deserialize_with = "de_defer_tools",
        skip_serializing_if = "Option::is_none"
    )]
    #[schemars(with = "Option<DeferToolsWire>")]
    pub defer_tools: Option<DeferTools>,
    /// Where the engine tells the model that an MCP server is available, and
    /// what that server says it is for.
    ///
    /// Separate from `defer_tools`: a server exists whether or not its tools
    /// are deferred, and where a notice lands is a fact about this agent's
    /// prompt rather than about any server.
    #[serde(default, skip_serializing_if = "Announce::is_default")]
    pub announce_mcp: Announce,
}

/// Where an MCP announcement lands.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "Announce")]
pub enum Announce {
    /// The system prompt while no call has dispatched; then a block on the
    /// last user message; then a message of its own. The engine takes the
    /// first place it can use, so the order is not a setting.
    #[default]
    Auto,
    /// Nowhere. For a server whose own words help nobody.
    Never,
}

impl Announce {
    fn is_default(&self) -> bool {
        *self == Self::Auto
    }
}

impl AgentConfig {
    /// The reasoning this agent's calls carry, or nothing where it named no
    /// effort and the provider's default stands.
    pub fn reasoning(&self) -> Option<ReasoningConfig> {
        self.effort.map(|effort| ReasoningConfig {
            effort: Some(effort),
            ..Default::default()
        })
    }

    /// Whether this agent defers its own tools. A connection's own `defer`
    /// still overrides this either way.
    pub fn defers_tools(&self) -> bool {
        self.defer_tools.is_some()
    }

    /// The settings for the tools this agent defers. An agent that holds no
    /// opinion still needs them, because a connection can defer on its own
    /// account.
    pub fn defer_settings(&self) -> DeferTools {
        self.defer_tools.unwrap_or_default()
    }

    /// Which tools the agent gets to reach the ones it defers.
    pub fn defer_strategy(&self) -> DeferToolsStrategy {
        self.defer_settings().strategy
    }
}

/// How many tools one search answers with, when the agent does not say.
///
/// A match carries a whole definition, so an answer of many is the tool list
/// the search replaced. The engine reports what it left out, so a model that
/// wanted more can narrow the query and ask again.
pub const DEFAULT_MAX_MATCHES: usize = 5;

fn default_max_matches() -> NonZeroUsize {
    NonZeroUsize::new(DEFAULT_MAX_MATCHES).expect("the default is not zero")
}

fn is_default_max_matches(n: &NonZeroUsize) -> bool {
    *n == default_max_matches()
}

/// How an agent's deferred tools reach the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
#[schemars(title = "DeferTools")]
pub struct DeferTools {
    /// Which tools the agent gets to reach the ones it defers.
    #[serde(default, skip_serializing_if = "DeferToolsStrategy::is_default")]
    pub strategy: DeferToolsStrategy,
    /// The most matches one search answers with. Never zero: a search that can
    /// answer with nothing is a search the model cannot use.
    #[serde(
        default = "default_max_matches",
        skip_serializing_if = "is_default_max_matches"
    )]
    pub max_matches: NonZeroUsize,
}

impl Default for DeferTools {
    fn default() -> Self {
        Self {
            strategy: DeferToolsStrategy::default(),
            max_matches: default_max_matches(),
        }
    }
}

/// The two forms `defer_tools` accepts: `true` for the defaults, or a table.
/// `false` reads the same as absent, so a config can turn off what it inherits.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
#[schemars(title = "DeferToolsWire")]
pub enum DeferToolsWire {
    Flag(bool),
    Config(DeferTools),
}

pub(crate) fn de_defer_tools<'de, D>(d: D) -> Result<Option<DeferTools>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(match Option::<DeferToolsWire>::deserialize(d)? {
        None | Some(DeferToolsWire::Flag(false)) => None,
        Some(DeferToolsWire::Flag(true)) => Some(DeferTools::default()),
        Some(DeferToolsWire::Config(c)) => Some(c),
    })
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
    /// Keep this tool out of the request. See [`LlmTool::defer`]. Absent ⇒
    /// the agent's `defer_tools`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub defer: Option<bool>,
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
    /// The connection this tool dials. `None` for the engine's own tools,
    /// which reach no connection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connector: Option<ConnectionPath>,
    /// The protocol of the connection, not of this tool: the engine's own tools
    /// carry `Mcp` whether or not they dial it.
    pub via: ConnectorProtocol,
    pub remote_name: String,
    #[serde(default, skip_serializing_if = "ConnectorToolKind::is_remote")]
    pub kind: ConnectorToolKind,
    /// Keep this tool out of the request. See [`LlmTool::defer`]. Resolved
    /// from the connection's `defer` and the agent's `defer_tools`.
    #[serde(default, skip_serializing_if = "is_false")]
    pub defer: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub approve: bool,
}

/// What the engine does with a call. Every value but `Remote` is one of the
/// engine's own tools, and none of those has a `remote_name`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "ConnectorToolKind")]
pub enum ConnectorToolKind {
    /// Call `remote_name` on the connection.
    #[default]
    Remote,
    /// Search the recorded offer. This reaches nothing.
    Find,
    /// Run the tool the arguments name.
    Call,
    /// Load a skill's instructions from a plugin bundle.
    Skill,
}

impl ConnectorToolKind {
    pub fn is_remote(&self) -> bool {
        matches!(self, Self::Remote)
    }
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

/// An MCP server the agent draws tools from. `path` resolves against the
/// engine's connection registry — locally from `substructure.toml`, in the
/// cloud from the connections an admin granted this app. The worker never names
/// a URL or a credential.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "McpServer")]
pub struct McpServer {
    pub path: ConnectionPath,
    /// Narrows what the model sees. Absent ⇒ every tool the connection grants.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<McpTools>,
    #[serde(default, skip_serializing_if = "AuthFailure::is_default")]
    pub auth_failure: AuthFailure,
    #[serde(default, skip_serializing_if = "Approve::is_default")]
    pub approve: Approve,
}

/// Which of a connection's calls stop for a person.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "Approve")]
pub enum Approve {
    #[default]
    Never,
    /// A tool that the connection marks `destructiveHint`.
    Destructive,
    Always,
}

impl Approve {
    fn is_default(&self) -> bool {
        *self == Self::Never
    }
}

/// What a session does when a connection needs a person to authorize it. It
/// belongs to the pair: one credential serves an agent that stops and asks, and
/// an agent that has nobody to ask.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "AuthFailure")]
pub enum AuthFailure {
    /// Stop and ask. A channel that cannot show the question degrades instead.
    #[default]
    Interrupt,
    /// Go on without this connection's tools.
    Degrade,
}

impl AuthFailure {
    fn is_default(&self) -> bool {
        *self == Self::Interrupt
    }
}

/// What the model sees of one connection, for one agent: which tools, and how
/// they reach the model.
///
/// The filter is applied in order — capability predicates, then `include`, then
/// `exclude` — and only ever narrowing, so a filter can never widen what the
/// connection grants. `defer` runs after it and removes nothing.
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
    /// Keep every surviving tool out of the request. See [`LlmTool::defer`].
    /// Absent ⇒ the agent's `defer_tools`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub defer: Option<bool>,
}

/// A plugin an agent uses. The skills and servers are stamped from the bundle
/// when the config loads. To enable a plugin, write it into the config.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "AgentPlugin")]
pub struct AgentPlugin {
    pub id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skills: Vec<SkillMeta>,
    /// Where each of this plugin's servers is declared.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub servers: Vec<ConnectionPath>,
    /// Applied to each of the plugin's servers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<McpTools>,
    #[serde(default, skip_serializing_if = "AuthFailure::is_default")]
    pub auth_failure: AuthFailure,
    #[serde(default, skip_serializing_if = "Approve::is_default")]
    pub approve: Approve,
}

impl AgentPlugin {
    /// One of the plugin's servers, with the plugin's policy on it.
    pub fn server(&self, path: &ConnectionPath) -> McpServer {
        McpServer {
            path: path.clone(),
            tools: self.tools.clone(),
            auth_failure: self.auth_failure,
            approve: self.approve,
        }
    }
}

/// What the model sees of a skill before it loads it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "SkillMeta")]
pub struct SkillMeta {
    pub name: String,
    #[serde(default)]
    pub description: String,
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
    /// Keep this definition out of the request.
    ///
    /// The engine still records it, still routes a call to it, and still finds
    /// it in a search. Only the request omits it, which is what keeps a large
    /// tool set out of the model's context and out of the cached prefix.
    ///
    /// Any source can set it: a tool the config declares, a connection, or
    /// whatever comes next. Deferral is a property of a tool, not of where it
    /// came from.
    #[serde(default, skip_serializing_if = "is_false")]
    pub defer: bool,
}

fn is_false(value: &bool) -> bool {
    !*value
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

impl LlmRequest {
    /// The definitions this request carries under `search`.
    ///
    /// `None` when the request declares no tool at all, which is not the same
    /// as one whose tools all defer — that one still offers the search.
    pub fn offered_tools(&self, search: DeferToolsStrategy) -> Option<Vec<&LlmTool>> {
        Some(search.offered(self.tools.as_ref()?))
    }
}

/// How the tools an agent defers reach the model.
///
/// The engine holds every deferred definition whatever this says, and answers
/// its own tools whatever this says. This chooses two things: which of those
/// tools the request advertises, and whether the request carries the deferred
/// definitions.
///
/// Declared on the agent, beside `defer_tools`: which tools an agent gets is
/// the agent's business, the same way as whether it defers at all.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "DeferToolsStrategy")]
pub enum DeferToolsStrategy {
    /// `tool_search` and `call_tool`. A search answers with the schema, so one
    /// search is the whole distance to a call, and nothing hands the model a
    /// name it cannot then reach.
    #[default]
    Search,
}

impl DeferToolsStrategy {
    /// The definitions the request carries.
    ///
    /// A strategy the engine answers leaves each deferred tool out: the engine
    /// finds it and routes to it from state, and the model never reads it. A
    /// strategy the provider answers keeps them, and the serializer marks each
    /// one with the provider's own flag.
    pub fn offered(self, tools: &[LlmTool]) -> Vec<&LlmTool> {
        match self {
            Self::Search => tools.iter().filter(|t| !t.defer).collect(),
        }
    }

    fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
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

/// What one call read and wrote, in counts every provider means the same way.
///
/// Each vendor names and scopes these differently: Anthropic reports the part
/// of the prompt it did not read from the cache, OpenAI reports the whole
/// prompt including that part. A session that changes model, and a tree whose
/// agents name different blocks, add these counts together, so the adapter
/// normalizes them rather than the reader.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "Usage")]
pub struct Usage {
    /// Every input token of the call, cached or not.
    pub input: u64,
    pub output: u64,
    /// The part of `input` the provider read fresh.
    pub uncached_input: u64,
    /// The part of `input` the provider read from the cache.
    pub cache_read: u64,
    /// The part of `input` the provider wrote to the cache.
    pub cache_write: u64,
    /// `input` and `output` together.
    pub total: u64,
    /// The counts as the provider reported them, for a reader that wants a
    /// number this type does not name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<Value>,
}

impl Usage {
    /// The counts of one call, from the parts a provider reports. `input` and
    /// `total` follow from the rest, so no caller states them.
    pub fn new(uncached_input: u64, cache_read: u64, cache_write: u64, output: u64) -> Self {
        let input = uncached_input + cache_read + cache_write;
        Self {
            input,
            output,
            uncached_input,
            cache_read,
            cache_write,
            total: input + output,
            provider: None,
        }
    }

    pub fn with_provider(mut self, raw: Value) -> Self {
        self.provider = Some(raw);
        self
    }

    /// Add the counts of another call. The raw report belongs to one call, so
    /// a sum carries none.
    pub fn add(&mut self, other: &Usage) {
        self.input += other.input;
        self.output += other.output;
        self.uncached_input += other.uncached_input;
        self.cache_read += other.cache_read;
        self.cache_write += other.cache_write;
        self.total += other.total;
        self.provider = None;
    }
}

/// Normalized LLM response. Provider adapters convert their raw responses
/// into this type at the boundary.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "LlmResponse")]
pub struct LlmResponse {
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Box<Reasoning>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Cost in dollars for this call, if the provider reports it. A decimal
    /// string on the wire.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(schema_with = "decimal_string_schema")]
    pub cost: Option<Decimal>,
    /// Images generated by the model.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<ResponseImage>,
}

/// What kind of failure, for a consumer that branches instead of reading the
/// sentence. A closed set, and required on every [`ErrorInfo`]: an optional
/// code is one nobody fills in, which leaves every consumer handling a `None`
/// that should not exist.
///
/// `provider_error`, `rate_limited`, `refused`, `budget_exceeded` and
/// `deadline_exceeded` describe a call that ran and went wrong.
/// `invalid_response` — a document did not parse, or parsed into something
/// unusable. `handler_error` — whoever was asked to do the work (a worker, a
/// client) reported a failure of its own. `worker_unreachable` — it was never
/// reached. `unroutable` — nothing could decide. `internal` — the engine's own
/// fault, and the honest answer when nothing else fits.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "ErrorCode")]
pub enum ErrorCode {
    ProviderError,
    RateLimited,
    Refused,
    BudgetExceeded,
    DeadlineExceeded,
    InvalidResponse,
    HandlerError,
    WorkerUnreachable,
    Unroutable,
    Internal,
}

/// Why something failed. One shape on every event, on the wire, and in the
/// internal carriers that produce them — shaped after a Stripe API error.
///
/// `retryable` is deliberately absent: whether to try again is a decision the
/// engine makes about one attempt, not a fact about the failure, and it is
/// meaningless on a terminal like `turn.completed`. It rides on the events
/// that settle an attempt instead.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "ErrorInfo")]
pub struct ErrorInfo {
    /// One engine-authored sentence, safe to show a human. Never a raw
    /// document — an unbounded body belongs in the log.
    pub message: String,
    pub code: ErrorCode,
    /// The one input to go and fix, when the failure names one: `agent.llm`,
    /// `actions[0].type`. Stripe's `param`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    /// Small structured particulars: a status, the llm blocks that exist.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<Value>,
}

impl ErrorInfo {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            code,
            param: None,
            detail: None,
        }
    }

    /// The engine's own fault, or a failure nothing else describes.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Internal, message)
    }

    /// Whoever ran the work reported a failure — the default for an error a
    /// worker or client authored without classifying it.
    pub fn handler(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::HandlerError, message)
    }

    pub fn with_param(mut self, param: impl Into<String>) -> Self {
        self.param = Some(param.into());
        self
    }

    pub fn with_detail(mut self, detail: Value) -> Self {
        self.detail = Some(detail);
        self
    }

    /// Take what an author supplied, keeping the default where they said
    /// nothing. The seam for a worker- or client-authored error, which arrives
    /// flat and may classify itself or not.
    pub fn or_code(mut self, code: Option<ErrorCode>) -> Self {
        if let Some(code) = code {
            self.code = code;
        }
        self
    }

    pub fn or_detail(mut self, detail: Option<Value>) -> Self {
        if detail.is_some() {
            self.detail = detail;
        }
        self
    }
}

impl std::fmt::Display for ErrorInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
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
    /// Dispatched and alive, awaiting its result. Off the deadline clock: the
    /// work succeeded in starting, and how long it then runs is its own
    /// business. A delegation sits here for as long as its child turn takes.
    Running,
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
        #[serde(default, skip_serializing_if = "Option::is_none")]
        result: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<Vec<ToolContent>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        structured_content: Option<Value>,
        #[serde(default, skip_serializing_if = "is_false")]
        is_error: bool,
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
    /// The chosen option's `style`, when the resolution was a pick.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
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
        result: Option<StoredResult>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<ErrorInfo>,
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
        /// True when the model declined the request rather than answering it.
        /// A refusal reads as a turn that stopped well and said nothing, so
        /// without this the run continues from a blank answer.
        #[serde(default)]
        refused: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[schemars(schema_with = "decimal_string_schema")]
        cost: Option<Decimal>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<ErrorInfo>,
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
        error: Option<ErrorInfo>,
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
        #[serde(default)]
        usage: Usage,
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
        /// The `[llm.*]` block this call runs on; omitted ⇒ the merge source
        /// config's `llm`. Naming a different block moves one call to another
        /// venue or vendor.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        llm: Option<String>,
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
        /// Layered over the agent config's `llm` policy, else over the engine's
        /// default.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retry: Option<RetryOverride>,
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
        /// Layered over the agent config's policy for this kind, else over the
        /// engine's default for where the tool runs.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retry: Option<RetryOverride>,
    },
    /// `id`/`attempt` omitted ⇒ taken from the answering `tool.execute` trigger,
    /// fencing the result to the attempt that ran.
    #[serde(rename = "tool.result")]
    ToolResult {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        attempt: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        result: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<Vec<ToolContent>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        structured_content: Option<Value>,
        #[serde(default, skip_serializing_if = "is_false")]
        is_error: bool,
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
        /// The child's opening message. It travels with the spawn, so it
        /// cannot race the creation of the session it opens.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<DraftMessage>,
        /// Layered over the agent config's `sub_agent` policy, else over the
        /// engine's default.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retry: Option<RetryOverride>,
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
    /// Resolve an open interrupt and resume the session.
    #[serde(rename = "interrupt.resolve")]
    ResolveInterrupt {
        interrupt_id: String,
        #[serde(default)]
        payload: Value,
    },
    /// Fetch a connection's tools again, after a person replaced its
    /// credential.
    #[serde(rename = "connector.sync")]
    SyncConnector { path: ConnectionPath },
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
    /// How each channel shows this decision, keyed by channel kind (e.g.
    /// `slack`). Opaque to the engine.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub channels: BTreeMap<String, Value>,
}

impl DecisionResponse {
    /// The response writes nothing, so settling with it changes nothing.
    pub fn authors_nothing(&self) -> bool {
        self.messages.is_empty()
            && self.actions.is_empty()
            && self.state.is_none()
            && self.agent.is_none()
            && self.channels.is_empty()
    }
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
