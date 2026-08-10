//! Anthropic Messages API (`POST /v1/messages`) provider.
//!
//! The engine's internal request/response types are OpenAI-Chat-shaped, so this
//! module translates in both directions: the flat `messages` array with a
//! `system` role and OpenAI-style `tool_calls`/`tool` messages is mapped onto
//! Anthropic's top-level `system`, content-block messages, and
//! `tool_use`/`tool_result` blocks — and the block-structured response and SSE
//! event stream are mapped back.

use std::sync::Arc;

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::UnboundedSender;
use tokio_stream::StreamExt;

use crate::llm::{CallContext, LlmCallError, LlmCallable, LlmProviderTrait};
use crate::protocol::{
    Content, ContentPart, DeferToolsStrategy, ErrorCode, LlmRequest, LlmResponse, ReasoningEffort,
    Role, SessionOwner, StreamDelta, ToolCall, ToolCallChunk, ToolCallFunction,
};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const DEFAULT_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u64 = 4096;

// ── Request wire format ──────────────────────────────────────────────────

/// A cache breakpoint. Anthropic caches nothing without one, so a request that
/// carries none is re-read in full on every turn.
#[derive(Serialize, Clone, Copy)]
struct CacheControl {
    #[serde(rename = "type")]
    kind: &'static str,
}

impl CacheControl {
    const EPHEMERAL: Self = Self { kind: "ephemeral" };
}

#[derive(Serialize)]
struct SystemBlock {
    #[serde(rename = "type")]
    kind: &'static str,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

#[derive(Serialize)]
struct AnthropicBody {
    model: String,
    max_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<Vec<SystemBlock>>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<Thinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_config: Option<OutputConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
    /// Never sent: a strategy the engine answers drops each deferred tool
    /// before this point. The field is what a strategy the provider answers
    /// serializes as its own flag.
    #[serde(skip)]
    defer: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: &'static str,
    content: Vec<RequestBlock>,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RequestBlock {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Image {
        source: ImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

impl RequestBlock {
    fn cache(&mut self) {
        match self {
            RequestBlock::Text { cache_control, .. }
            | RequestBlock::Image { cache_control, .. }
            | RequestBlock::ToolUse { cache_control, .. }
            | RequestBlock::ToolResult { cache_control, .. } => {
                *cache_control = Some(CacheControl::EPHEMERAL)
            }
        }
    }
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ImageSource {
    Url { url: String },
    Base64 { media_type: String, data: String },
}

#[derive(Serialize)]
struct Thinking {
    #[serde(rename = "type")]
    thinking_type: &'static str,
}

#[derive(Serialize)]
struct OutputConfig {
    effort: &'static str,
}

fn anthropic_effort(e: ReasoningEffort) -> &'static str {
    match e {
        ReasoningEffort::Xhigh => "xhigh",
        ReasoningEffort::High => "high",
        ReasoningEffort::Medium => "medium",
        // Anthropic has no `minimal`/`none`; `low` is the nearest floor.
        ReasoningEffort::Low | ReasoningEffort::Minimal | ReasoningEffort::None => "low",
    }
}

/// Map an OpenAI-style role message's content into Anthropic content blocks.
fn content_to_blocks(content: Option<&Content>) -> Vec<RequestBlock> {
    match content {
        None => Vec::new(),
        Some(Content::Text(s)) => {
            if s.is_empty() {
                Vec::new()
            } else {
                vec![RequestBlock::Text {
                    text: s.clone(),
                    cache_control: None,
                }]
            }
        }
        Some(Content::Parts(parts)) => parts.iter().filter_map(part_to_block).collect(),
    }
}

fn part_to_block(part: &ContentPart) -> Option<RequestBlock> {
    match part {
        ContentPart::Text { text } => Some(RequestBlock::Text {
            text: text.clone(),
            cache_control: None,
        }),
        ContentPart::ImageUrl { image_url } => Some(image_block(&image_url.url)),
        // Audio/video/file parts have no direct Messages API equivalent here.
        _ => None,
    }
}

fn image_block(url: &str) -> RequestBlock {
    if let Some(rest) = url.strip_prefix("data:") {
        if let Some((meta, data)) = rest.split_once(',') {
            let media_type = meta.split(';').next().unwrap_or("image/png").to_string();
            return RequestBlock::Image {
                source: ImageSource::Base64 {
                    media_type,
                    data: data.to_string(),
                },
                cache_control: None,
            };
        }
    }
    RequestBlock::Image {
        source: ImageSource::Url {
            url: url.to_string(),
        },
        cache_control: None,
    }
}

/// Append blocks to the last turn if it shares the role, else open a new turn.
/// Anthropic requires strictly alternating user/assistant turns.
fn push_turn(turns: &mut Vec<AnthropicMessage>, role: &'static str, blocks: Vec<RequestBlock>) {
    if blocks.is_empty() {
        return;
    }
    if let Some(last) = turns.last_mut() {
        if last.role == role {
            last.content.extend(blocks);
            return;
        }
    }
    turns.push(AnthropicMessage {
        role,
        content: blocks,
    });
}

/// `stream: None` omits the field so the body is valid input for both the
/// create and stream calls a worker might make.
fn build_body(
    request: &LlmRequest,
    default_max_tokens: u64,
    search: DeferToolsStrategy,
    stream: Option<bool>,
) -> AnthropicBody {
    let mut system_parts: Vec<String> = Vec::new();
    let mut turns: Vec<AnthropicMessage> = Vec::new();

    for msg in &request.messages {
        match msg.role {
            Role::System => {
                if let Some(c) = &msg.content {
                    let text = c.text_owned();
                    if !text.is_empty() {
                        system_parts.push(text);
                    }
                }
            }
            Role::User => {
                push_turn(&mut turns, "user", content_to_blocks(msg.content.as_ref()));
            }
            Role::Assistant => {
                let mut blocks = Vec::new();
                if let Some(c) = &msg.content {
                    let text = c.text_owned();
                    if !text.is_empty() {
                        blocks.push(RequestBlock::Text {
                            text,
                            cache_control: None,
                        });
                    }
                }
                if let Some(tcs) = &msg.tool_calls {
                    for tc in tcs {
                        let input = serde_json::from_str(&tc.function.arguments)
                            .unwrap_or_else(|_| serde_json::json!({}));
                        blocks.push(RequestBlock::ToolUse {
                            id: tc.id.clone(),
                            name: tc.function.name.clone(),
                            input,
                            cache_control: None,
                        });
                    }
                }
                push_turn(&mut turns, "assistant", blocks);
            }
            Role::Tool => {
                let tool_use_id = msg.tool_call_id.clone().unwrap_or_default();
                let content = msg
                    .content
                    .as_ref()
                    .map(|c| c.text_owned())
                    .unwrap_or_default();
                push_turn(
                    &mut turns,
                    "user",
                    vec![RequestBlock::ToolResult {
                        tool_use_id,
                        content,
                        cache_control: None,
                    }],
                );
            }
        }
    }

    // Three breakpoints, in the order the provider reads the request: after the
    // tools, after the system prompt, and after the last block of the last
    // message. The first two hold for the session. The third grows the cached
    // part by one turn each turn, which is where the transcript is.
    let system = (!system_parts.is_empty()).then(|| {
        vec![SystemBlock {
            kind: "text",
            text: system_parts.join("\n\n"),
            cache_control: Some(CacheControl::EPHEMERAL),
        }]
    });

    let tools = request.offered_tools(search).map(|ts| {
        let mut ts: Vec<AnthropicTool> = ts
            .into_iter()
            .map(|t| AnthropicTool {
                name: t.name.clone(),
                description: t.description.clone(),
                input_schema: t.input_schema(),
                defer: t.defer,
                cache_control: None,
            })
            .collect();
        // The last tool the provider reads for itself. A deferred one cannot
        // carry a breakpoint, so the mark goes on the last that is not.
        if let Some(last) = ts.iter_mut().rev().find(|t| !t.defer) {
            last.cache_control = Some(CacheControl::EPHEMERAL);
        }
        ts
    });
    if let Some(block) = turns.last_mut().and_then(|t| t.content.last_mut()) {
        block.cache();
    }

    let (thinking, output_config) = match request.reasoning.as_ref().and_then(|r| r.effort) {
        None | Some(ReasoningEffort::None) => (None, None),
        Some(e) => (
            Some(Thinking {
                thinking_type: "adaptive",
            }),
            Some(OutputConfig {
                effort: anthropic_effort(e),
            }),
        ),
    };

    AnthropicBody {
        model: request.model.clone(),
        max_tokens: request.max_completion_tokens.unwrap_or(default_max_tokens),
        system,
        messages: turns,
        tools,
        thinking,
        output_config,
        stream,
    }
}

/// Map an Anthropic `stop_reason` onto the OpenAI-style `finish_reason` vocab
/// the rest of the engine uses.
fn map_stop_reason(reason: &str) -> String {
    match reason {
        "tool_use" => "tool_calls",
        "end_turn" | "stop_sequence" => "stop",
        "max_tokens" => "length",
        other => other,
    }
    .to_string()
}

// ── Non-streaming response ───────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct MessagesResponse {
    model: String,
    #[serde(default)]
    content: Vec<ResponseBlock>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    usage: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ResponseBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(other)]
    Other,
}

impl MessagesResponse {
    fn into_llm_response(self) -> LlmResponse {
        let mut content = String::new();
        let mut tool_calls = Vec::new();
        for block in self.content {
            match block {
                ResponseBlock::Text { text } => content.push_str(&text),
                ResponseBlock::ToolUse { id, name, input } => tool_calls.push(ToolCall {
                    id,
                    call_type: "function".to_string(),
                    function: ToolCallFunction {
                        name,
                        arguments: input.to_string(),
                    },
                }),
                ResponseBlock::Other => {}
            }
        }
        LlmResponse {
            model: self.model,
            content: (!content.is_empty()).then_some(content),
            tool_calls,
            finish_reason: self.stop_reason.as_deref().map(map_stop_reason),
            usage: self.usage,
            cost: None,
            images: Vec::new(),
        }
    }
}

// ── Worker-format seam ───────────────────────────────────────────────────

/// The Messages API body for `request`, `stream` omitted.
pub(crate) fn request_to_wire(
    request: &LlmRequest,
    search: DeferToolsStrategy,
) -> serde_json::Value {
    serde_json::to_value(build_body(request, DEFAULT_MAX_TOKENS, search, None)).unwrap_or_default()
}

/// A raw Messages API response → the neutral `LlmResponse`.
pub(crate) fn response_from_wire(value: serde_json::Value) -> Result<LlmResponse, String> {
    crate::json::from_value::<MessagesResponse>("anthropic response", value)
        .map(MessagesResponse::into_llm_response)
        .map_err(|e| e.to_string())
}

// ── Streaming events ─────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamEvent {
    MessageStart {
        message: StreamMessageStart,
    },
    ContentBlockStart {
        index: usize,
        content_block: StreamContentBlock,
    },
    ContentBlockDelta {
        index: usize,
        delta: StreamBlockDelta,
    },
    ContentBlockStop {},
    MessageDelta {
        delta: StreamMessageDelta,
        #[serde(default)]
        usage: Option<StreamUsage>,
    },
    MessageStop {},
    Ping {},
    Error {
        error: StreamError,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
struct StreamMessageStart {
    #[serde(default)]
    model: String,
    #[serde(default)]
    usage: Option<StreamUsage>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamContentBlock {
    ToolUse {
        id: String,
        name: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamBlockDelta {
    TextDelta {
        text: String,
    },
    ThinkingDelta {
        thinking: String,
    },
    InputJsonDelta {
        partial_json: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
struct StreamMessageDelta {
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StreamUsage {
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default)]
    output_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct StreamError {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

/// Per-content-block streaming accumulator.
enum BlockAccum {
    /// Text or thinking — streamed out, not retained past the delta.
    Passthrough,
    ToolUse {
        id: String,
        name: String,
        arguments: String,
    },
}

/// Folds Messages API stream events into `StreamDelta`s and the final
/// response. Shared by the server-side SSE loop and the worker-format delta
/// seam (`providers::format`).
#[derive(Default)]
pub(crate) struct StreamParser {
    content: String,
    blocks: Vec<BlockAccum>,
    finish_reason: Option<String>,
    model: Option<String>,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
}

impl StreamParser {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Fold one raw event payload; unknown or non-streaming events yield nothing.
    /// Errors are the worker's to surface, so an error event yields nothing too.
    pub(crate) fn parse_data(&mut self, data: &str) -> Vec<StreamDelta> {
        serde_json::from_str(data)
            .ok()
            .and_then(|event| self.on_event(event).ok().flatten())
            .into_iter()
            .collect()
    }

    fn on_event(&mut self, event: StreamEvent) -> Result<Option<StreamDelta>, LlmCallError> {
        match event {
            StreamEvent::MessageStart { message } => {
                if !message.model.is_empty() {
                    self.model = Some(message.model);
                }
                if let Some(u) = message.usage {
                    self.input_tokens = u.input_tokens.or(self.input_tokens);
                }
                Ok(None)
            }
            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                while self.blocks.len() <= index {
                    self.blocks.push(BlockAccum::Passthrough);
                }
                self.blocks[index] = match content_block {
                    StreamContentBlock::ToolUse { id, name } => BlockAccum::ToolUse {
                        id,
                        name,
                        arguments: String::new(),
                    },
                    StreamContentBlock::Other => BlockAccum::Passthrough,
                };
                Ok(None)
            }
            StreamEvent::ContentBlockDelta { index, delta } => Ok(match delta {
                StreamBlockDelta::TextDelta { text } => {
                    self.content.push_str(&text);
                    Some(StreamDelta {
                        text: Some(text),
                        ..Default::default()
                    })
                }
                StreamBlockDelta::ThinkingDelta { thinking } => {
                    (!thinking.is_empty()).then(|| StreamDelta {
                        reasoning: Some(thinking),
                        ..Default::default()
                    })
                }
                StreamBlockDelta::InputJsonDelta { partial_json } => {
                    if let Some(BlockAccum::ToolUse {
                        id,
                        name,
                        arguments,
                    }) = self.blocks.get_mut(index)
                    {
                        arguments.push_str(&partial_json);
                        Some(StreamDelta {
                            tool_calls: vec![ToolCallChunk {
                                id: id.clone(),
                                name: (!name.is_empty()).then(|| name.clone()),
                                arguments: Some(partial_json),
                            }],
                            ..Default::default()
                        })
                    } else {
                        None
                    }
                }
                StreamBlockDelta::Other => None,
            }),
            StreamEvent::MessageDelta { delta, usage } => {
                if let Some(u) = usage {
                    self.output_tokens = u.output_tokens.or(self.output_tokens);
                }
                Ok(delta.stop_reason.map(|reason| {
                    let mapped = map_stop_reason(&reason);
                    self.finish_reason = Some(mapped.clone());
                    StreamDelta {
                        finish_reason: Some(mapped),
                        ..Default::default()
                    }
                }))
            }
            StreamEvent::Error { error } => Err(LlmCallError::new(
                ErrorCode::ProviderError,
                format!(
                    "Anthropic stream error ({}): {}",
                    error.error_type, error.message
                ),
                matches!(error.error_type.as_str(), "overloaded_error" | "api_error"),
            )),
            StreamEvent::ContentBlockStop {}
            | StreamEvent::MessageStop {}
            | StreamEvent::Ping {}
            | StreamEvent::Unknown => Ok(None),
        }
    }

    fn into_response(self, fallback_model: &str) -> LlmResponse {
        let tool_calls = self
            .blocks
            .into_iter()
            .filter_map(|b| match b {
                BlockAccum::ToolUse {
                    id,
                    name,
                    arguments,
                } if !id.is_empty() => Some(ToolCall {
                    id,
                    call_type: "function".to_string(),
                    function: ToolCallFunction { name, arguments },
                }),
                _ => None,
            })
            .collect();
        LlmResponse {
            model: self.model.unwrap_or_else(|| fallback_model.to_string()),
            content: (!self.content.is_empty()).then_some(self.content),
            tool_calls,
            finish_reason: self.finish_reason,
            usage: build_usage(self.input_tokens, self.output_tokens),
            cost: None,
            images: Vec::new(),
        }
    }
}

// ── Config / client ──────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct AnthropicConfig {
    pub base_url: String,
    pub api_key: String,
    #[serde(default = "default_version")]
    pub version: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u64,
}

fn default_version() -> String {
    DEFAULT_VERSION.to_string()
}

fn default_max_tokens() -> u64 {
    DEFAULT_MAX_TOKENS
}

impl AnthropicConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: api_key.into(),
            version: DEFAULT_VERSION.to_string(),
            max_tokens: DEFAULT_MAX_TOKENS,
        }
    }
}

pub struct AnthropicClient {
    http: Client,
    base_url: String,
    headers: HeaderMap,
    default_max_tokens: u64,
}

impl AnthropicClient {
    pub fn from_config(config: AnthropicConfig) -> Self {
        let mut headers = HeaderMap::new();
        if let Ok(v) = HeaderValue::from_str(&config.api_key) {
            headers.insert(HeaderName::from_static("x-api-key"), v);
        }
        if let Ok(v) = HeaderValue::from_str(&config.version) {
            headers.insert(HeaderName::from_static("anthropic-version"), v);
        }
        Self {
            http: crate::providers::http_client(),
            base_url: config.base_url,
            headers,
            default_max_tokens: config.max_tokens,
        }
    }

    async fn post_messages(
        &self,
        request: &LlmRequest,
        search: DeferToolsStrategy,
        stream: bool,
    ) -> Result<reqwest::Response, LlmCallError> {
        let body = build_body(request, self.default_max_tokens, search, Some(stream));
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));

        self.http
            .post(&url)
            .headers(self.headers.clone())
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                LlmCallError::new(
                    ErrorCode::ProviderError,
                    format!("HTTP request failed: {e}"),
                    e.is_timeout() || e.is_connect(),
                )
            })
    }
}

fn classify_error(status: reqwest::StatusCode, body: &str) -> LlmCallError {
    let status_code = status.as_u16();
    let retryable = status.is_server_error() || status_code == 408 || status_code == 429;
    let code = if status_code == 429 {
        ErrorCode::RateLimited
    } else {
        ErrorCode::ProviderError
    };
    // The provider says why in `error.message`; the rest of the body is a
    // request id and echoed request, which the log keeps and the reader does
    // not need.
    let message = match crate::json::error_message(body.as_bytes()) {
        Some(reported) => format!("Anthropic API error {status}: {reported}"),
        None => format!("Anthropic API error {status}"),
    };
    LlmCallError::new(code, message, retryable)
}

fn build_usage(input_tokens: Option<u64>, output_tokens: Option<u64>) -> Option<serde_json::Value> {
    if input_tokens.is_none() && output_tokens.is_none() {
        return None;
    }
    Some(serde_json::json!({
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }))
}

#[async_trait]
impl LlmCallable for AnthropicClient {
    async fn call(
        &self,
        request: &LlmRequest,
        ctx: &CallContext<'_>,
    ) -> Result<LlmResponse, LlmCallError> {
        let resp = self
            .post_messages(request, ctx.defer_tools_strategy, false)
            .await?;
        let status = resp.status();
        let body = resp.text().await.map_err(|e| {
            LlmCallError::new(ErrorCode::ProviderError, format!("read body: {e}"), true)
        })?;

        if !status.is_success() {
            tracing::warn!(status = status.as_u16(), body = %crate::json::excerpt(body.as_bytes()), "anthropic api call failed");
            return Err(classify_error(status, &body));
        }

        let parsed: MessagesResponse =
            crate::json::from_str("anthropic response", &body).map_err(LlmCallError::from)?;

        Ok(parsed.into_llm_response())
    }

    async fn call_streaming(
        &self,
        request: &LlmRequest,
        ctx: &CallContext<'_>,
        chunk_tx: UnboundedSender<StreamDelta>,
    ) -> Result<LlmResponse, LlmCallError> {
        let resp = self
            .post_messages(request, ctx.defer_tools_strategy, true)
            .await?;
        let status = resp.status();

        if !status.is_success() {
            let body = resp.text().await.map_err(|e| {
                LlmCallError::new(ErrorCode::ProviderError, format!("read body: {e}"), true)
            })?;
            return Err(classify_error(status, &body));
        }

        let mut parser = StreamParser::new();

        let byte_stream = resp.bytes_stream();
        let mut stream =
            tokio_stream::StreamExt::map(byte_stream, |chunk| chunk.map_err(std::io::Error::other));
        let mut line_buf = String::new();

        while let Some(chunk_result) = stream.next().await {
            let bytes = chunk_result.map_err(|e| {
                LlmCallError::new(ErrorCode::ProviderError, format!("stream read: {e}"), true)
            })?;
            line_buf.push_str(&String::from_utf8_lossy(&bytes));

            while let Some(newline_pos) = line_buf.find('\n') {
                let line = line_buf[..newline_pos].trim_end_matches('\r').to_string();
                line_buf = line_buf[newline_pos + 1..].to_string();

                // Anthropic sends both `event:` and `data:` lines; the `data:`
                // JSON carries its own `type`, so we only parse those.
                let data = match line.strip_prefix("data: ") {
                    Some(d) => d,
                    None => continue,
                };

                let event: StreamEvent = match serde_json::from_str(data) {
                    Ok(e) => e,
                    Err(_) => continue,
                };

                if let Some(delta) = parser.on_event(event)? {
                    let _ = chunk_tx.send(delta);
                }
            }
        }

        Ok(parser.into_response(&request.model))
    }
}

pub struct AnthropicProvider {
    client: Arc<AnthropicClient>,
}

impl AnthropicProvider {
    pub fn new(config: AnthropicConfig) -> Self {
        Self {
            client: Arc::new(AnthropicClient::from_config(config)),
        }
    }
}

#[async_trait]
impl LlmProviderTrait for AnthropicProvider {
    async fn resolve(&self, _owner: &SessionOwner) -> Result<Arc<dyn LlmCallable>, String> {
        Ok(self.client.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{DraftMessage, LlmTool, ReasoningConfig};
    use serde_json::json;

    fn msg(role: Role, content: Option<&str>) -> DraftMessage {
        DraftMessage {
            id: None,
            role,
            content: content.map(|c| Content::Text(c.to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    fn tool_call(id: &str, name: &str, args: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            call_type: "function".to_string(),
            function: ToolCallFunction {
                name: name.to_string(),
                arguments: args.to_string(),
            },
        }
    }

    fn req(messages: Vec<DraftMessage>) -> LlmRequest {
        LlmRequest {
            model: "claude-opus-4-8".to_string(),
            messages,
            tools: None,
            temperature: None,
            max_completion_tokens: None,
            reasoning: None,
        }
    }

    #[test]
    fn extracts_system_and_maps_user_text() {
        let body = build_body(
            &req(vec![
                msg(Role::System, Some("be nice")),
                msg(Role::User, Some("hi")),
            ]),
            4096,
            DeferToolsStrategy::Search,
            Some(false),
        );
        let v = serde_json::to_value(&body).unwrap();
        assert_eq!(v["system"][0]["text"], "be nice");
        assert_eq!(v["max_tokens"], 4096);
        assert_eq!(v["stream"], false);
        assert_eq!(v["messages"][0]["role"], "user");
        assert_eq!(v["messages"][0]["content"][0]["type"], "text");
        assert_eq!(v["messages"][0]["content"][0]["text"], "hi");
    }

    #[test]
    fn folds_tool_results_into_one_user_turn_and_maps_tool_calls() {
        let mut assistant = msg(Role::Assistant, None);
        assistant.tool_calls = Some(vec![
            tool_call("call_1", "get_weather", r#"{"city":"NYC"}"#),
            tool_call("call_2", "get_time", "{}"),
        ]);
        let mut result_1 = msg(Role::Tool, Some("72F"));
        result_1.tool_call_id = Some("call_1".to_string());
        let mut result_2 = msg(Role::Tool, Some("noon"));
        result_2.tool_call_id = Some("call_2".to_string());

        let body = build_body(
            &req(vec![
                msg(Role::User, Some("weather?")),
                assistant,
                result_1,
                result_2,
            ]),
            4096,
            DeferToolsStrategy::Search,
            Some(false),
        );
        let v = serde_json::to_value(&body).unwrap();

        // user(weather?), assistant(2 tool_use), user(2 tool_result) — coalesced
        assert_eq!(v["messages"].as_array().unwrap().len(), 3);
        assert_eq!(v["messages"][1]["role"], "assistant");
        assert_eq!(v["messages"][1]["content"][0]["type"], "tool_use");
        assert_eq!(v["messages"][1]["content"][0]["id"], "call_1");
        assert_eq!(v["messages"][1]["content"][0]["name"], "get_weather");
        assert_eq!(v["messages"][1]["content"][0]["input"]["city"], "NYC");
        assert_eq!(v["messages"][2]["role"], "user");
        assert_eq!(v["messages"][2]["content"].as_array().unwrap().len(), 2);
        assert_eq!(v["messages"][2]["content"][0]["type"], "tool_result");
        assert_eq!(v["messages"][2]["content"][0]["tool_use_id"], "call_1");
        assert_eq!(v["messages"][2]["content"][0]["content"], "72F");
        assert_eq!(v["messages"][2]["content"][1]["tool_use_id"], "call_2");
    }

    #[test]
    fn every_cacheable_part_of_the_request_carries_a_breakpoint() {
        let mut r = req(vec![
            msg(Role::System, Some("be nice")),
            msg(Role::User, Some("hello")),
        ]);
        r.tools = Some(vec![
            LlmTool {
                name: "a".into(),
                description: String::new(),
                input: None,
                output: None,
                defer: false,
            },
            LlmTool {
                name: "b".into(),
                description: String::new(),
                input: None,
                output: None,
                defer: false,
            },
        ]);
        let v = request_to_wire(&r, DeferToolsStrategy::Search);
        let mark = serde_json::json!({ "type": "ephemeral" });

        assert_eq!(v["tools"][1]["cache_control"], mark, "after the tools");
        assert!(
            v["tools"][0].get("cache_control").is_none(),
            "one breakpoint for the block, not one for each tool"
        );
        assert_eq!(v["system"][0]["cache_control"], mark, "after the system");
        let turns = v["messages"].as_array().expect("messages");
        let last = turns.last().expect("a turn");
        let block = last["content"].as_array().unwrap().last().unwrap();
        assert_eq!(
            block["cache_control"], mark,
            "after the last block, so the transcript is cached as it grows"
        );
    }

    #[test]
    fn a_request_with_no_tools_and_no_system_still_caches_the_transcript() {
        let r = req(vec![msg(Role::User, Some("hello"))]);
        let v = request_to_wire(&r, DeferToolsStrategy::Search);
        assert!(v.get("system").is_none());
        assert!(v.get("tools").is_none());
        let block = v["messages"][0]["content"][0].clone();
        assert_eq!(
            block["cache_control"],
            serde_json::json!({ "type": "ephemeral" })
        );
    }

    #[test]
    fn maps_tools_effort_and_max_tokens() {
        let mut r = req(vec![msg(Role::User, Some("hi"))]);
        r.tools = Some(vec![
            LlmTool {
                name: "f".to_string(),
                description: "d".to_string(),
                input: Some(json!({"type": "object"})),
                output: None,
                defer: false,
            },
            LlmTool {
                name: "no_args".to_string(),
                description: "d".to_string(),
                input: None,
                output: None,
                defer: false,
            },
        ]);
        r.max_completion_tokens = Some(1000);
        r.reasoning = Some(ReasoningConfig {
            effort: Some(ReasoningEffort::High),
            max_tokens: None,
            exclude: None,
            enabled: None,
        });
        let v = serde_json::to_value(build_body(&r, 4096, DeferToolsStrategy::Search, Some(true)))
            .unwrap();
        assert_eq!(v["max_tokens"], 1000);
        assert_eq!(v["tools"][0]["name"], "f");
        assert_eq!(v["tools"][0]["input_schema"]["type"], "object");
        assert_eq!(
            v["tools"][1]["input_schema"],
            json!({"type": "object", "properties": {}}),
            "a no-input tool gets the empty object schema"
        );
        assert_eq!(v["thinking"]["type"], "adaptive");
        assert_eq!(v["output_config"]["effort"], "high");
        assert_eq!(v["stream"], true);
        // system omitted when absent
        assert!(v.get("system").is_none());
    }

    #[test]
    fn parses_response_blocks_into_content_and_tool_calls() {
        let raw = json!({
            "model": "claude-opus-4-8",
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"},
                {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "NYC"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        })
        .to_string();

        let resp = serde_json::from_str::<MessagesResponse>(&raw)
            .unwrap()
            .into_llm_response();

        assert_eq!(resp.content.as_deref(), Some("Hello world"));
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].id, "tu_1");
        assert_eq!(resp.tool_calls[0].function.name, "get_weather");
        let args: serde_json::Value =
            serde_json::from_str(&resp.tool_calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "NYC");
        assert_eq!(resp.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(resp.usage.unwrap()["input_tokens"], 10);
    }

    #[test]
    fn stop_reason_mapping() {
        assert_eq!(map_stop_reason("end_turn"), "stop");
        assert_eq!(map_stop_reason("stop_sequence"), "stop");
        assert_eq!(map_stop_reason("tool_use"), "tool_calls");
        assert_eq!(map_stop_reason("max_tokens"), "length");
        assert_eq!(map_stop_reason("refusal"), "refusal");
    }

    #[test]
    fn deserializes_stream_events() {
        let start: StreamEvent = serde_json::from_str(
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tu_1","name":"get_weather","input":{}}}"#,
        )
        .unwrap();
        match start {
            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                assert_eq!(index, 1);
                assert!(matches!(content_block, StreamContentBlock::ToolUse { .. }));
            }
            _ => panic!("expected content_block_start"),
        }

        let delta: StreamEvent = serde_json::from_str(
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}"#,
        )
        .unwrap();
        assert!(matches!(
            delta,
            StreamEvent::ContentBlockDelta {
                delta: StreamBlockDelta::InputJsonDelta { .. },
                ..
            }
        ));

        let msg_delta: StreamEvent = serde_json::from_str(
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":7}}"#,
        )
        .unwrap();
        match msg_delta {
            StreamEvent::MessageDelta { delta, usage } => {
                assert_eq!(delta.stop_reason.as_deref(), Some("tool_use"));
                assert_eq!(usage.unwrap().output_tokens, Some(7));
            }
            _ => panic!("expected message_delta"),
        }

        assert!(matches!(
            serde_json::from_str::<StreamEvent>(r#"{"type":"ping"}"#).unwrap(),
            StreamEvent::Ping {}
        ));
        assert!(matches!(
            serde_json::from_str::<StreamEvent>(r#"{"type":"future_event","x":1}"#).unwrap(),
            StreamEvent::Unknown
        ));
    }
}
