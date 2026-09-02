use std::collections::{BTreeMap, HashMap};
use std::num::NonZeroUsize;

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::attachments::{Attachment, Attachments};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthNeed {
    NeverAuthorized,
    Reauthorize,
    TokenRejected,
}

pub const TOOL_SEARCH: &str = "tool_search";
pub const CALL_TOOL: &str = "call_tool";
pub const SKILL: &str = "skill";
pub const SUBAGENT: &str = "subagent";
pub const SUBAGENT_WAIT: &str = "subagent_wait";

/// How a file, the CLI, and the wire name a connection.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConnectionPath {
    Mcp(String),
    PluginServer { plugin: String, server: String },
    Agent(String),
}

impl ConnectionPath {
    pub fn parse(path: &str) -> Option<Self> {
        if let Some(id) = path.strip_prefix("mcp.") {
            return (!id.is_empty() && !id.contains('.')).then(|| Self::Mcp(id.to_string()));
        }
        if let Some(id) = path.strip_prefix("agent.") {
            return (!id.is_empty() && !id.contains('.')).then(|| Self::Agent(id.to_string()));
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

    /// The prefix the model sees. Any character a provider rejects becomes `_`.
    pub fn tool_prefix(&self) -> String {
        let raw = match self {
            Self::Mcp(id) | Self::Agent(id) => id.clone(),
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
                "`{written}` is not a connection path: `mcp.<id>`, `plugin.<id>.mcp.<server>`, \
                 or `agent.<id>`"
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
            Self::Agent(id) => write!(f, "agent.{id}"),
        }
    }
}

/// `Decimal` serializes as a string, so the schema says string.
fn decimal_string_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
    schemars::json_schema!({
        "type": ["string", "null"],
        "pattern": r"^-?\d+(\.\d+)?$",
    })
}

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
        let value = match (result, content) {
            (Some(_), Some(_)) => return Err("a tool result names both `result` and `content`"),
            (Some(value), None) => value,
            (None, content) => {
                return Ok(Self {
                    content: content.unwrap_or_default(),
                    structured_content,
                    is_error,
                })
            }
        };

        if let Ok(lifted) = serde_json::from_value::<Self>(value.clone()) {
            if !lifted.content.is_empty() || lifted.structured_content.is_some() {
                return Ok(Self {
                    content: lifted.content,
                    structured_content: structured_content.or(lifted.structured_content),
                    is_error: is_error || lifted.is_error,
                });
            }
        }
        Ok(Self {
            content: Self::from_value(value).content,
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
    Attachment(Attachment),
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
                StoredContent::Attachment(attachment) => Some(attachment.line()),
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

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(untagged, from = "ContentWire")]
#[schemars(title = "Content")]
pub enum Content {
    Text(String),
    Parts(Vec<StoredContent>),
}

#[derive(Deserialize, JsonSchema)]
#[serde(untagged)]
#[schemars(title = "Content")]
pub enum ContentWire {
    Text(String),
    Parts(Vec<ContentPartWire>),
}

#[derive(Deserialize, JsonSchema)]
#[serde(untagged)]
#[schemars(title = "ContentPartWire")]
pub enum ContentPartWire {
    Stored(StoredContent),
    Prompt(ContentPart),
}

impl From<ContentWire> for Content {
    fn from(wire: ContentWire) -> Self {
        match wire {
            ContentWire::Text(text) => Self::Text(text),
            ContentWire::Parts(parts) => Self::Parts(
                parts
                    .into_iter()
                    .map(|p| match p {
                        ContentPartWire::Stored(part) => part,
                        ContentPartWire::Prompt(part) => part.into(),
                    })
                    .collect(),
            ),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum PromptContent {
    Text(String),
    Parts(Vec<PromptPart>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum PromptPart {
    Text {
        text: String,
    },
    Media {
        mime: String,
        name: Option<String>,
        bytes: Vec<u8>,
    },
    Link {
        uri: String,
        name: Option<String>,
        mime_type: Option<String>,
    },
}

impl PromptPart {
    pub fn link_text(uri: &str, name: Option<&str>) -> String {
        match name {
            Some(name) => format!("{name}: {uri}"),
            None => uri.to_string(),
        }
    }
}

impl Serialize for PromptPart {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        ContentPart::from(self).serialize(s)
    }
}

impl From<&PromptPart> for ContentPart {
    fn from(part: &PromptPart) -> Self {
        use crate::mime;
        match part {
            PromptPart::Text { text } => Self::Text { text: text.clone() },
            PromptPart::Link {
                uri,
                name,
                mime_type,
            } => match mime_type.as_deref().map(mime::essence) {
                Some("image") => Self::ImageUrl {
                    image_url: ImageUrl { url: uri.clone() },
                },
                Some("video") => Self::VideoUrl {
                    video_url: VideoUrl { url: uri.clone() },
                },
                _ => Self::Text {
                    text: PromptPart::link_text(uri, name.as_deref()),
                },
            },
            PromptPart::Media { mime, name, bytes } => match mime::essence(mime) {
                "image" => Self::ImageUrl {
                    image_url: ImageUrl {
                        url: mime::data_uri(mime, bytes),
                    },
                },
                "audio" => Self::InputAudio {
                    input_audio: AudioData {
                        data: mime::base64(bytes),
                        format: mime::parts(mime).1.to_string(),
                    },
                },
                "video" => Self::VideoUrl {
                    video_url: VideoUrl {
                        url: mime::data_uri(mime, bytes),
                    },
                },
                _ => Self::File {
                    file: FileData {
                        filename: name.clone().unwrap_or_else(|| "file".to_string()),
                        file_data: mime::data_uri(mime, bytes),
                    },
                },
            },
        }
    }
}

impl PromptContent {
    pub fn text_owned(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    PromptPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
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

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
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

/// Which provider wrote the blocks. They go back only to that provider.
/// Anthropic rejects blocks it did not sign.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "ReasoningProvider")]
pub enum ReasoningProvider {
    Anthropic,
    Openai,
    Openrouter,
}

/// What the model thought before it answered. `text` is for a reader.
/// `blocks` are the provider's own and stay unchanged. Anthropic requires the
/// thinking before a tool call back with its signature.
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
    /// The reasoning of one response. Absent if it left no blocks.
    /// Boxed to keep it off the paths that do not read it.
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

    /// The blocks. Only the provider that wrote them can read them.
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

/// The wire form of a [`Message`]. `id` is absent until the message is
/// recorded.
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

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(inline)]
pub struct NewMessage {
    pub message: Message,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
}

/// The privilege of the caller that raised an interrupt. Read from the
/// authenticated caller, never from the request. To resume it, a caller needs
/// this privilege or more.
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

/// Where a call runs. A tool call takes `worker` (the default), `client`, or
/// `server`. The engine sets `server` for connector tools; a worker cannot.
/// An LLM call has no handler. It runs where its `[llm.*]` block says.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "Handler")]
pub enum Handler {
    /// The engine makes the call.
    Server,
    /// The worker executes it.
    Worker,
    /// The client executes it. The session goes idle until it answers.
    /// Tools only.
    Client,
}

/// The shape of a worker-run LLM call. Absent gives the engine's own format.
/// Set gives the provider's own request, response, and stream events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "LlmFormat")]
pub enum LlmFormat {
    /// OpenAI Chat Completions.
    Openai,
    /// Anthropic Messages API.
    Anthropic,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "RetryPolicy")]
pub struct RetryPolicy {
    pub queue_timeout_secs: Option<u32>,
    pub run_timeout_secs: Option<u32>,
    pub total_timeout_secs: Option<u32>,
    /// Total attempts, not retries. `1` gives one try.
    pub max_attempts: u32,
    pub backoff_base_secs: u32,
    pub backoff_max_secs: u32,
}

/// Only the fields it names change. An override cannot make a timeout
/// unbounded.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "RetryOverride")]
pub struct RetryOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_timeout_secs: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_timeout_secs: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_timeout_secs: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_attempts: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backoff_base_secs: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backoff_max_secs: Option<u32>,
}

/// Retry overrides, one for each effect kind. `default` covers the kinds that
/// name nothing. A kind layers on top of `default`.
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
    pub subagent: Option<RetryOverride>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connector: Option<RetryOverride>,
}

/// Where a person's name comes from: `slack`, `app`, `cli`, or another source
/// a deployment registers. Set by whatever authenticated the request, never
/// read from the request.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(transparent)]
#[schemars(title = "Issuer")]
pub struct Issuer(String);

impl Issuer {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// The one person at an installation with no authentication.
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

/// One identity, as the source that authenticated it named it. An id is
/// unique only within its issuer.
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

/// Who can read what a session says. The transport sets it once, when the
/// session starts. Absent or unknown reads as `shared`. `shared` never selects
/// a personal credential.
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

/// Who a session runs for. No subject means a schedule, a key, or the engine.
/// None of them has a credential of its own.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChannelKind(&'static str);

impl ChannelKind {
    pub const SLACK: Self = Self("slack");
    pub const AG_UI: Self = Self("ag-ui");
    pub const CLI: Self = Self("cli");

    pub const fn as_str(&self) -> &'static str {
        self.0
    }

    pub fn owning(owner: &SessionOwner) -> Option<Self> {
        let issuer = owner.requester.subject.as_ref()?.issuer.as_str();
        [Self::SLACK, Self::AG_UI, Self::CLI]
            .into_iter()
            .find(|kind| kind.as_str() == issuer)
    }
}

impl std::fmt::Display for ChannelKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
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

/// The owner as the worker receives it, without the tenant. Read `kind` with
/// `id`. Only `frontend` is an end user.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "WorkerIdentity")]
pub struct WorkerIdentity {
    #[serde(flatten)]
    pub requester: Requester,
    pub metadata: HashMap<String, String>,
}

/// Opaque worker state: JSON the engine stores but never interprets.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(transparent)]
#[schemars(title = "WorkerState")]
pub struct WorkerState(pub Value);

/// A declared agent. The same shape whether a file writes it or a worker
/// returns it.
///
/// `llm` names the `[llm.*]` block every call runs on. That block decides where
/// the call runs and what shape it takes. A config that names no block fails
/// when the engine resolves a call.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "AgentConfig")]
pub struct AgentConfig {
    /// The `[llm.*]` block this agent's calls run on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm: Option<String>,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// How hard the model thinks. Unset leaves the provider's own default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
    /// Boxed. Five per-kind overrides are too many bytes to carry inline.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry: Option<Box<RetryConfig>>,
    /// Worker- or client-executed tools the model can call.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<AgentTool>,
    /// Subagents the model can delegate to. The model sees them as tools.
    /// Each call starts a child session.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub subagents: Vec<Subagent>,
    /// What shape the subagents take as tools. Absent ⇒ one tool per agent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subagent_tools: Option<SubagentTools>,
    /// How deep this agent's subagents may nest. A session whose depth
    /// reaches it may not delegate. `0` never delegates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_subagent_depth: Option<u32>,
    /// MCP servers this agent draws tools from.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mcp: Vec<McpServer>,
    /// Plugins this agent uses.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub plugins: Vec<AgentPlugin>,
    /// Defer every tool this agent offers, whatever its source. A tool or a
    /// connection overrides this with its own `defer`. Absent, the agent defers
    /// nothing; a connection can still defer on its own.
    #[serde(
        default,
        deserialize_with = "de_defer_tools",
        skip_serializing_if = "Option::is_none"
    )]
    #[schemars(with = "Option<DeferToolsWire>")]
    pub defer_tools: Option<DeferTools>,
    /// Whether the engine tells the model that an MCP server is available, and
    /// what that server says it is for.
    #[serde(default, skip_serializing_if = "McpAnnounce::is_default")]
    pub mcp_announce: McpAnnounce,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attachments: Option<Attachments>,
}

/// Where an MCP announcement lands.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "McpAnnounce")]
pub enum McpAnnounce {
    /// The system prompt while no call has run. Then a block on the last user
    /// message. Then a message of its own. The engine takes the first place it
    /// can use.
    #[default]
    Auto,
    /// Nowhere. For a server whose own description does not help.
    Never,
}

impl McpAnnounce {
    fn is_default(&self) -> bool {
        *self == Self::Auto
    }
}

impl AgentConfig {
    /// The reasoning this agent's calls carry. Absent if it named no effort.
    pub fn reasoning(&self) -> Option<ReasoningConfig> {
        self.effort.map(|effort| ReasoningConfig {
            effort: Some(effort),
            ..Default::default()
        })
    }

    /// Whether this agent defers its own tools. A connection's `defer`
    /// overrides it.
    pub fn defers_tools(&self) -> bool {
        self.defer_tools.is_some()
    }

    /// The settings for deferred tools. An agent that defers nothing still
    /// needs them, because a connection can defer on its own.
    pub fn defer_settings(&self) -> DeferTools {
        self.defer_tools.unwrap_or_default()
    }

    /// Which tools the agent gets to reach the ones it defers.
    pub fn defer_strategy(&self) -> DeferToolsStrategy {
        self.defer_settings().strategy
    }

    pub fn depth_limit(&self) -> u32 {
        self.max_subagent_depth
            .unwrap_or(DEFAULT_MAX_SUBAGENT_DEPTH)
    }

    pub fn may_spawn_subagent(&self, depth: u32) -> bool {
        depth < self.depth_limit()
    }

    pub fn subagent_strategy(&self) -> SubagentToolsStrategy {
        SubagentTools::strategy_of(self.subagent_tools)
    }

    pub fn subagent_mode(&self, agent_id: &str) -> Option<SubagentMode> {
        self.subagents.iter().find(|s| s.id == agent_id)?.mode
    }
}

/// What shape an agent's subagents take as tools.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
#[schemars(title = "SubagentTools")]
pub struct SubagentTools {
    #[serde(default, skip_serializing_if = "SubagentToolsStrategy::is_default")]
    pub strategy: SubagentToolsStrategy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wait: Option<bool>,
}

/// How the model reaches an agent's subagents.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "SubagentToolsStrategy")]
pub enum SubagentToolsStrategy {
    /// One tool per subagent, named by the agent.
    #[default]
    PerAgent,
    /// One `subagent` tool for all of them. The call names the agent.
    Single,
}

impl SubagentTools {
    pub fn strategy_of(tools: Option<Self>) -> SubagentToolsStrategy {
        tools.unwrap_or_default().strategy
    }

    pub fn wait_of(tools: Option<Self>) -> bool {
        tools.and_then(|t| t.wait).unwrap_or(true)
    }
}

impl SubagentToolsStrategy {
    pub fn is_default(&self) -> bool {
        matches!(self, Self::PerAgent)
    }
}

/// How calls to a subagent return. Configuration; each call carries a
/// [`SpawnMode`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "SubagentMode")]
pub enum SubagentMode {
    Blocking,
    Detached,
    #[default]
    Any,
}

impl SubagentMode {
    pub fn offered(self) -> &'static [SpawnMode] {
        match self {
            Self::Blocking => &[],
            Self::Detached => &[SpawnMode::Detached],
            Self::Any => &[SpawnMode::Blocking, SpawnMode::Detached],
        }
    }
}

/// How one subagent call returns.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "SpawnMode")]
pub enum SpawnMode {
    #[default]
    Blocking,
    Detached,
    Wait,
}

pub const DEFAULT_MAX_SUBAGENT_DEPTH: u32 = 5;

/// How many tools one search answers with, when the agent does not say.
///
/// A match carries a whole definition, so a large answer is as big as the tool
/// list. The engine says what it left out.
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

/// A function tool the agent offers. `handler` says where a call runs:
/// `client` for the client, absent for the worker. `server` is invalid here.
/// The engine runs only connector tools, and a worker declares those by
/// connection id.
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

/// One tool the engine resolved from a connector and runs itself.
///
/// Derived, not stored. The session records what a connection offered, then
/// filters that offer through the config in force. A connection that changes
/// its tools cannot rewrite what already happened, and a filter change costs no
/// round trip.
///
/// `name` is what the model sees. `remote_name` is what the engine calls.
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
    /// The connection this tool calls. Absent for the engine's own tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connector: Option<ConnectionPath>,
    /// The protocol of the connection. The engine's own tools carry `Mcp` too.
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
    Subagent,
    Attachment,
}

impl ConnectorToolKind {
    pub fn is_remote(&self) -> bool {
        matches!(self, Self::Remote)
    }
}

/// How the engine reaches a connection. Internal. A connection's section says
/// which protocol it uses, and an agent names it by id without knowing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "ConnectorProtocol")]
pub enum ConnectorProtocol {
    #[default]
    Mcp,
    Agent,
}

/// An MCP server the agent draws tools from. `id` names an `[mcp.*]`
/// connection the engine holds. A worker never writes a URL or a credential.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "McpServer")]
pub struct McpServer {
    pub id: String,
    /// Narrows what the model sees. Absent ⇒ every tool the connection grants.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<McpTools>,
    #[serde(default, skip_serializing_if = "McpAuthFailure::is_default")]
    pub auth_failure: McpAuthFailure,
    #[serde(default, skip_serializing_if = "McpToolSyncFailure::is_default")]
    pub tool_sync_failure: McpToolSyncFailure,
    #[serde(default, skip_serializing_if = "Approve::is_default")]
    pub approve: Approve,
}

impl McpServer {
    pub fn bind(&self) -> BoundServer {
        BoundServer {
            path: ConnectionPath::Mcp(self.id.clone()),
            tools: self.tools.clone(),
            auth_failure: self.auth_failure,
            tool_sync_failure: self.tool_sync_failure,
            approve: self.approve,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundServer {
    pub path: ConnectionPath,
    pub tools: Option<McpTools>,
    pub auth_failure: McpAuthFailure,
    pub tool_sync_failure: McpToolSyncFailure,
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

/// What a session does when a connection needs a person to authorize it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "McpAuthFailure")]
pub enum McpAuthFailure {
    /// Stop and ask. A channel that cannot show the question degrades.
    #[default]
    Interrupt,
    /// Go on without this connection's tools.
    Degrade,
}

impl McpAuthFailure {
    fn is_default(&self) -> bool {
        *self == Self::Interrupt
    }
}

/// Whether the model is told that a connection's tool fetch failed. The turn
/// goes ahead without those tools either way.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "McpToolSyncFailure")]
pub enum McpToolSyncFailure {
    /// Name the connection wherever its tools would have been.
    #[default]
    Warn,
    /// Say nothing. For a connection the agent does not need.
    Silent,
}

impl McpToolSyncFailure {
    fn is_default(&self) -> bool {
        *self == Self::Warn
    }

    pub fn warns(&self) -> bool {
        *self == Self::Warn
    }
}

/// Which of a connection's tools the model sees, and how they reach it.
///
/// The filter runs in order: capability predicates, then `include`, then
/// `exclude`. Each step only removes, so a filter cannot widen what the
/// connection grants. `defer` runs last and removes nothing.
///
/// `include` and `exclude` are globs over the tool's name on the connection,
/// not the prefixed name the model sees.
///
/// Capability predicates read the MCP annotations. A tool with no annotation
/// fails the predicate, so a server that annotates nothing yields nothing under
/// `read_only`.
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

/// A plugin an agent uses. The skills and servers come from the bundle when the
/// config loads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "AgentPlugin")]
pub struct AgentPlugin {
    pub id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skills: Vec<SkillMeta>,
    /// The plugin's server names, from its bundle.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub servers: Vec<String>,
    /// Applied to each of the plugin's servers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<McpTools>,
    #[serde(default, skip_serializing_if = "McpAuthFailure::is_default")]
    pub auth_failure: McpAuthFailure,
    #[serde(default, skip_serializing_if = "McpToolSyncFailure::is_default")]
    pub tool_sync_failure: McpToolSyncFailure,
    #[serde(default, skip_serializing_if = "Approve::is_default")]
    pub approve: Approve,
}

impl AgentPlugin {
    /// One of the plugin's servers, with the plugin's policy on it.
    pub fn server(&self, name: &str) -> BoundServer {
        BoundServer {
            path: ConnectionPath::PluginServer {
                plugin: self.id.clone(),
                server: name.to_string(),
            },
            tools: self.tools.clone(),
            auth_failure: self.auth_failure,
            tool_sync_failure: self.tool_sync_failure,
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

/// A subagent the model can delegate to. `id` names the child agent; the model
/// calls the tool [`Subagent::offered_name`] gives. Its input is one `message`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "Subagent")]
pub struct Subagent {
    pub id: String,
    #[serde(default)]
    pub description: String,
    /// Keep this tool out of the request. See [`LlmTool::defer`]. Absent ⇒
    /// the agent's `defer_tools`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub defer: Option<bool>,
    /// Offer the tool as `agent__<id>` instead of `<id>`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<SubagentMode>,
}

impl Subagent {
    pub fn offered_name(&self) -> String {
        match self.prefix {
            Some(true) => format!("agent__{}", self.id),
            _ => self.id.clone(),
        }
    }

    pub fn defers(&self, default: bool) -> bool {
        self.defer.unwrap_or(default)
    }

    pub fn resolved_mode(&self) -> SubagentMode {
        self.mode.unwrap_or_default()
    }
}

/// Inputs a client declares on its run, passed to the worker on the
/// `client.messages` decision. `tools` are the browser's own tools, read as
/// client-handled [`AgentTool`]s. The engine adds them to the proposed config.
/// A worker can override that by returning its own `agent`.
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

/// A tool's declared contract. Flat on the wire. A provider that needs it
/// nested re-wraps it at its own boundary.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "LlmTool")]
pub struct LlmTool {
    pub name: String,
    pub description: String,
    /// JSON Schema for the arguments. Absent declares a tool with no
    /// arguments. The engine checks every call against it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    /// JSON Schema the result must satisfy. The model never sees it. A result
    /// that breaks it becomes a terminal tool error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,
    /// Keep this definition out of the request.
    ///
    /// The engine still records it, still routes a call to it, and still finds
    /// it in a search. Only the request leaves it out. That keeps a large tool
    /// set out of the model's context and out of the cached prefix.
    ///
    /// Any source can set it. Deferral belongs to a tool, not to where the tool
    /// came from.
    #[serde(default, skip_serializing_if = "is_false")]
    pub defer: bool,
}

pub(crate) fn is_false(value: &bool) -> bool {
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
    /// Absent when the request declares no tool at all. That is not the same as
    /// a request whose tools all defer, which still offers the search.
    pub fn offered_tools(&self, search: DeferToolsStrategy) -> Option<Vec<&LlmTool>> {
        Some(search.offered(self.tools.as_ref()?))
    }
}

/// How the tools an agent defers reach the model.
///
/// The engine holds every deferred definition whatever this says. This chooses
/// which tools the request advertises, and whether the request carries the
/// deferred definitions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "DeferToolsStrategy")]
pub enum DeferToolsStrategy {
    /// `tool_search` and `call_tool`. A search answers with the schema, so one
    /// search is enough to make a call.
    #[default]
    Search,
}

impl DeferToolsStrategy {
    /// The definitions the request carries.
    ///
    /// A strategy the engine answers leaves each deferred tool out. The engine
    /// finds it and routes to it from state. A strategy the provider answers
    /// keeps them and marks each one with the provider's flag.
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

/// What one call read and wrote. Every provider means these counts the same
/// way.
///
/// Vendors report different things. Anthropic gives the part of the prompt it
/// did not read from the cache. OpenAI gives the whole prompt. Each adapter
/// converts to this shape, because these counts get added together.
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
    /// The counts as the provider reported them, for a number this type does
    /// not name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<Value>,
}

impl Usage {
    /// The counts of one call. `input` and `total` follow from the rest.
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

    /// Add the counts of another call. A raw report belongs to one call, so a
    /// sum carries none.
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
    /// Cost in dollars, if the provider reports it. A decimal string on the
    /// wire.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(schema_with = "decimal_string_schema")]
    pub cost: Option<Decimal>,
    /// Images generated by the model.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<ResponseImage>,
}

/// What kind of failure, so a consumer can branch on it instead of reading the
/// sentence. A closed set, required on every [`ErrorInfo`].
///
/// `provider_error`, `rate_limited`, `refused`, `budget_exceeded`, and
/// `deadline_exceeded` mean a call ran and went wrong.
/// `invalid_response` means a document did not parse, or parsed into something
/// unusable. `handler_error` means the worker or client reported its own
/// failure. `worker_unreachable` means it was never reached. `unroutable`
/// means nothing could decide. `internal` means the engine's own fault.
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

/// Why something failed. One shape on every event and on the wire.
///
/// There is no `retryable` field. Whether to try again is a decision about one
/// attempt, not a fact about the failure. The events that settle an attempt
/// carry it instead.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "ErrorInfo")]
pub struct ErrorInfo {
    /// One sentence the engine wrote, safe to show a human. Never a raw
    /// document. An unbounded body belongs in the log.
    pub message: String,
    pub code: ErrorCode,
    /// The one input to fix, when the failure names one. For example
    /// `agent.llm` or `actions[0].type`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    /// Small structured details, such as a status or the llm blocks that
    /// exist.
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

    /// The engine's own fault, or a failure no other code describes.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Internal, message)
    }

    /// Whoever ran the work reported a failure. The default for an error a
    /// worker or client wrote without a code.
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

    /// Take what the author supplied and keep the default where they said
    /// nothing. For an error a worker or client wrote.
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
    /// May be a subagent of root.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "EffectStatus")]
pub enum EffectStatus {
    Pending,
    /// Running, waiting for its result. Off the deadline clock. A subagent
    /// stays here for as long as its child turn takes.
    Running,
    Completed,
    Failed,
    RetryScheduled,
    Queued,
}

/// What kind of work an effect is. One enum for the wire and for scheduling. A
/// decision and a turn's end queue beside the calls and are swept the same way,
/// so they are kinds too. Neither appears on an [`Effect`].
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "EffectKind")]
pub enum EffectKind {
    ToolCall,
    Subagent,
    LlmCall,
    /// Fetching one connection's tool list. Its `id` is the connection id.
    ConnectorSync,
    /// A worker decision.
    Decision,
    /// The turn's completion, which waits for the `turn.finished` decision to
    /// settle. Carries the turn id. Never swept, because it has no deadline.
    TurnEnd,
}

impl EffectKind {
    pub fn label(self) -> &'static str {
        match self {
            EffectKind::ToolCall => "tool_call",
            EffectKind::Subagent => "subagent",
            EffectKind::LlmCall => "llm_call",
            EffectKind::ConnectorSync => "connector_sync",
            EffectKind::Decision => "decision",
            EffectKind::TurnEnd => "turn_end",
        }
    }
}

/// An effect still running, shown on each worker decision. A flat envelope
/// plus the fields of its kind. A connector sync carries none. Its `id` is the
/// connection being fetched.
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
    /// The child session a subagent runs in. Its `id` is the model tool call
    /// the subagent answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

/// What the engine made of a tool call's arguments, sent with the raw
/// `arguments` string. Always on the wire.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "status", rename_all = "lowercase")]
#[schemars(title = "ToolInput")]
pub enum ToolInput {
    /// Parsed, and valid against the `input` schema if the tool declares one.
    /// `value` is the parsed `arguments`. The engine never changes it.
    Valid { value: Value },
    /// Parsed to an object that violates the declared `input` schema.
    Invalid { value: Value, error: String },
    /// Not a JSON object. Either malformed JSON or another type.
    Malformed { error: String },
}

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

/// The body of a `client.append`. Messages are added at the session head,
/// against the active path at delivery. A queued append lands after whatever
/// turn beat it, so it cannot fork the tree. A message whose id is already
/// recorded is dropped.
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

/// What a client submits: a message, its full conversation view, an append
/// batch, or a named action. The engine turns it into events and never stores
/// it as it arrived.
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

/// Everything a client can send: submit a message, a full view, an append
/// batch, or a named action; resume an interrupt; or settle a client tool.
///
/// Each variant carries only the addressing it needs. The four submit variants
/// carry `agent_id`, which routes the turn and starts the session if it is new,
/// and an optional `turn_id`. A resume or settle names an interrupt or effect
/// and continues whatever turn is running, so it carries neither.
/// `session_id` is on the envelope.
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
        /// Hold this message for the next turn instead of refusing it while a
        /// turn is running. Off by default.
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
        /// Hold this batch for the next turn instead of refusing it while a
        /// turn is running. Off by default.
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

/// Which worker identity decides for a session, and optionally at what
/// address. `url` fills in when the declared block has none, or overrides it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "WorkerRef")]
pub struct WorkerRef {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

impl ClientInput {
    /// The submit variants route on an agent; a resume or settle continues
    /// whatever is running.
    pub fn agent_id(&self) -> Option<&str> {
        match self {
            ClientInput::Message { agent_id, .. }
            | ClientInput::Messages { agent_id, .. }
            | ClientInput::Append { agent_id, .. }
            | ClientInput::Action { agent_id, .. } => Some(agent_id),
            _ => None,
        }
    }
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

/// An interrupt payload in the AG-UI shape. `id` and `reason` live on the
/// interrupt itself.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[schemars(title = "InterruptPayload")]
pub struct InterruptPayload {
    /// Markdown. A channel converts it as it needs. Without it, a channel
    /// shows the interrupt's `reason`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Binds the interrupt to a prior tool call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// JSON Schema for the expected resolution payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<Value>,
    /// RFC 3339. Display only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<String>,
    /// Free-form, delivered to clients unchanged. `metadata.options` is a
    /// list of [`InterruptOption`], which Slack shows as buttons.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// One answer under `metadata.options`. Slack shows it as a button. A click
/// resumes with the option's `value`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "InterruptOption")]
pub struct InterruptOption {
    pub label: String,
    /// Delivered unchanged as the resolution's `payload`. The worker chooses
    /// what it means.
    pub value: Value,
    /// `primary` or `danger`. Anything else shows plain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
}

/// A resume payload a channel wrote: the AG-UI shape plus who resolved it.
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

/// Who resolved it. The channel sets this, never the requester.
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

/// The trigger a worker sees on the wire. There is no `ClientMessage`: the
/// engine turns a bare client message into `ClientTranscript` before it sends
/// it.
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
        /// Inputs the client declared on its run. The engine adds
        /// `client.tools` to the proposed config.
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
        /// What the engine made of `arguments` against the tool's `input`
        /// schema: `valid`, `invalid`, or `malformed`. Always on the wire.
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
        /// True when the model declined the request. Without it, a refusal
        /// looks like a turn that ended well and said nothing.
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
    #[serde(rename = "subagent.finished")]
    SubagentFinished {
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
    /// Sent after a turn completes, with its final output. The session stays
    /// busy until it is answered. Echo the proposed `done` to finish.
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
            Self::SubagentFinished { .. } => "subagent.finished",
            Self::InterruptResumed { .. } => "interrupt.resumed",
            Self::TurnFinished { .. } => "turn.finished",
        }
    }
}

/// The action a worker writes on the wire. A settle can leave out the effect
/// id, because the `*.execute` trigger it answers already names it.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type")]
#[schemars(title = "DecisionAction")]
pub enum DecisionAction {
    /// A flat LLM request. Every field is optional.
    ///
    /// Without `id`, the engine mints one, and it becomes the assistant node's
    /// id. A field left out comes from the agent config, then from the engine's
    /// default. Without `messages`, the request carries the config's system
    /// message and the decision's view. Given `messages`, no system message is
    /// added.
    #[serde(rename = "llm.call")]
    CallLlm {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// The `[llm.*]` block this call runs on. Absent uses the config's
        /// `llm`. Naming another block moves this one call elsewhere.
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
        /// Layered over the agent config's `llm` policy, or over the engine's
        /// default.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retry: Option<RetryOverride>,
    },
    /// Without `id`, the engine mints one. A tool the model called carries the
    /// model's id.
    ///
    /// There is no `handler`. The name says where the call runs: a connector
    /// tool on the engine, a `handler: client` tool on the client, anything
    /// else on the worker. The engine knows all three already.
    #[serde(rename = "tool.call")]
    CallTool {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        name: String,
        arguments: Value,
        /// Layered over the agent config's policy for this kind, or over the
        /// engine's default for where the tool runs.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retry: Option<RetryOverride>,
    },
    /// Without `id` and `attempt`, both come from the `tool.execute` trigger
    /// this answers. That ties the result to the attempt that ran.
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
    /// Without `id` and `attempt`, both come from the `llm.execute` trigger
    /// this answers. That ties the result to the attempt that ran.
    #[serde(rename = "llm.result")]
    LlmResult {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        attempt: Option<u32>,
        /// An `LlmResponse`, or the provider's own response when the
        /// `llm.execute` this answers carried a `format`.
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
    #[serde(rename = "subagent.spawn")]
    SpawnSubagent {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        session_id: Option<String>,
        agent_id: String,
        /// The model tool call this subagent answers. Required.
        tool_call_id: String,
        /// The child's opening message. It travels with the spawn, so it
        /// cannot arrive before the session exists.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<DraftMessage>,
        /// Layered over the agent config's `subagent` policy, or over the
        /// engine's default.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retry: Option<RetryOverride>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mode: Option<SpawnMode>,
    },
    #[serde(rename = "message.send")]
    SendMessage {
        session_id: String,
        message: DraftMessage,
    },
    /// Without `interrupt_id`, the engine mints one to match the later
    /// resume.
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

/// The messages and actions to author, plus optional state and agent writes.
/// A worker returns one. The engine proposes one too, which the worker echoes
/// or changes.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "DecisionResponse")]
pub struct DecisionResponse {
    #[serde(default)]
    pub messages: Vec<DraftMessage>,
    #[serde(default)]
    pub actions: Vec<DecisionAction>,
    /// Absent or `null` keeps the current state. Send an empty value to
    /// clear it.
    #[serde(default)]
    pub state: Option<WorkerState>,
    /// A new agent config write; omitted keeps the current config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<AgentConfig>,
    /// How each channel shows this decision, keyed by channel kind. The engine
    /// does not read it.
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
    /// The engine's default continuation for `trigger`. Empty when only the
    /// worker can decide. Accept it by echoing it back.
    pub proposed: &'a DecisionResponse,
    pub state: &'a WorkerState,
    /// The agent config for the active path. `null` when none is set.
    pub agent: &'a Option<AgentConfig>,
    /// The worker this session is pinned to. `null` when the file's own
    /// routing decides.
    pub worker: &'a Option<WorkerRef>,
    pub calls: &'a [Effect],
    /// Count of in-flight `tool_call`/`subagent` calls.
    pub pending_calls: usize,
    pub messages: &'a [Message],
    pub message_tree: &'a MessageTree,
    pub ancestry: &'a [String],
    pub attempts: u32,
    pub deadline: &'a Option<DateTime<Utc>>,
    pub turn_id: &'a Option<String>,
}

#[cfg(test)]
mod channel_kind_tests {
    use super::*;

    fn owned_by(issuer: Issuer) -> SessionOwner {
        SessionOwner {
            tenant_id: "t1".into(),
            requester: Requester::new(Subject::new(issuer, "u1"), Visibility::Private),
            metadata: Default::default(),
        }
    }

    #[test]
    fn a_session_belongs_to_the_channel_that_opened_it() {
        assert_eq!(
            ChannelKind::owning(&owned_by(Issuer::slack())),
            Some(ChannelKind::SLACK)
        );
        assert_eq!(
            ChannelKind::owning(&owned_by(Issuer::cli())),
            Some(ChannelKind::CLI)
        );
    }

    #[test]
    fn a_browser_frontend_names_no_channel() {
        assert_eq!(ChannelKind::owning(&owned_by(Issuer::app())), None);
        assert_ne!(Issuer::app().as_str(), ChannelKind::AG_UI.as_str());
    }

    #[test]
    fn a_session_no_channel_opened_belongs_to_none() {
        assert_eq!(ChannelKind::owning(&owned_by(Issuer::operator())), None);
        assert_eq!(ChannelKind::owning(&owned_by(Issuer::app())), None);

        let anonymous = SessionOwner {
            tenant_id: "t1".into(),
            requester: Requester::machine(),
            metadata: Default::default(),
        };
        assert_eq!(ChannelKind::owning(&anonymous), None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn from_result(value: Value) -> ToolResult {
        ToolResult::from_action(Some(value), None, None, false).expect("a readable result")
    }

    /// `result` is a whole `ToolResult`. Reading it as a plain value quoted
    /// its JSON into a text block.
    #[test]
    fn a_result_written_as_a_tool_result_is_read_as_one() {
        let result = from_result(serde_json::json!({
            "content": [{ "type": "text", "text": "sent" }]
        }));
        assert_eq!(result.as_text(), "sent");
        assert_eq!(result.content.len(), 1);
    }

    #[test]
    fn a_lifted_result_keeps_what_it_says_about_itself() {
        let result = from_result(serde_json::json!({
            "content": [{ "type": "text", "text": "nope" }],
            "isError": true,
        }));
        assert!(result.is_error);

        let structured = from_result(serde_json::json!({
            "structuredContent": { "temp": 62 },
        }));
        assert_eq!(
            structured.structured_content,
            Some(serde_json::json!({ "temp": 62 }))
        );
    }

    /// The outer flag wins. An action that says the call failed says so
    /// whatever its result claims.
    #[test]
    fn the_actions_own_error_flag_is_kept() {
        let result = ToolResult::from_action(
            Some(serde_json::json!({ "content": [{ "type": "text", "text": "x" }] })),
            None,
            None,
            true,
        )
        .unwrap();
        assert!(result.is_error);
    }

    /// `deny_unknown_fields` tells a result apart from data that looks like
    /// one.
    #[test]
    fn data_that_is_not_a_tool_result_is_left_alone() {
        assert_eq!(from_result(serde_json::json!("Lisbon")).as_text(), "Lisbon");
        assert_eq!(
            from_result(serde_json::json!({ "temp": 62 })).as_text(),
            r#"{"temp":62}"#
        );
        assert_eq!(
            from_result(serde_json::json!({ "content": 3 })).as_text(),
            r#"{"content":3}"#
        );
        assert_eq!(from_result(serde_json::json!(null)).as_text(), "");
    }

    #[test]
    fn content_and_result_together_are_refused() {
        let both = ToolResult::from_action(
            Some(serde_json::json!("a")),
            Some(vec![ToolContent::Text { text: "b".into() }]),
            None,
            false,
        );
        assert!(both.is_err());
    }
}
