use std::sync::Arc;

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::UnboundedSender;
use tokio_stream::StreamExt;

use crate::llm::{CallContext, LlmCallError, LlmCallable, LlmProviderTrait};
use crate::mime;
use crate::protocol::{
    AudioData, ContentPart, DeferToolsStrategy, ErrorCode, LlmResponse, LlmTool, PromptContent,
    PromptMessage, PromptPart, PromptRequest, Reasoning, ReasoningEffort, ReasoningProvider, Role,
    SessionOwner, StreamDelta, ToolCall, ToolCallChunk, ToolCallFunction, Usage,
};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

#[derive(Serialize)]
struct WireTool {
    #[serde(rename = "type")]
    tool_type: &'static str,
    function: WireToolFunction,
}

#[derive(Serialize)]
struct WireToolFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

impl From<&LlmTool> for WireTool {
    fn from(t: &LlmTool) -> Self {
        WireTool {
            tool_type: "function",
            function: WireToolFunction {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.input_schema(),
            },
        }
    }
}

#[derive(Serialize)]
struct WireBody<'a> {
    model: &'a str,
    messages: Vec<WireMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<WireTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_retention: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
}

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Serialize)]
struct WireMessage<'a> {
    role: Role,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<WireContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<&'a Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<&'a String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<&'a String>,
}

#[derive(Serialize)]
#[serde(untagged)]
pub(super) enum WireContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

pub(super) type PartPolicy = fn(&PromptPart) -> ContentPart;

pub(super) fn wire_content(
    content: Option<&PromptContent>,
    part: PartPolicy,
) -> Option<WireContent> {
    match content {
        None => None,
        Some(PromptContent::Text(text)) => Some(WireContent::Text(text.clone())),
        Some(PromptContent::Parts(parts)) => {
            Some(WireContent::Parts(parts.iter().map(part).collect()))
        }
    }
}

fn part(p: &PromptPart) -> ContentPart {
    match p {
        PromptPart::Media { mime, bytes, .. } => match (mime::essence(mime), mime::parts(mime).1) {
            ("image", _) | ("application", "pdf") => ContentPart::from(p),
            ("audio", "wav" | "x-wav" | "wave") => audio(bytes, "wav"),
            ("audio", "mpeg" | "mp3") => audio(bytes, "mp3"),
            (kind, _) => ContentPart::Text {
                text: format!("[{kind} content]"),
            },
        },
        PromptPart::Link {
            uri,
            name,
            mime_type,
        } if mime_type.as_deref().map(mime::essence) != Some("image") => ContentPart::Text {
            text: PromptPart::link_text(uri, name.as_deref()),
        },
        _ => ContentPart::from(p),
    }
}

fn audio(bytes: &[u8], format: &str) -> ContentPart {
    ContentPart::InputAudio {
        input_audio: AudioData {
            data: mime::base64(bytes),
            format: format.to_string(),
        },
    }
}

fn wire_message<'a>(m: &'a PromptMessage, part: PartPolicy) -> WireMessage<'a> {
    WireMessage {
        role: m.role.clone(),
        content: wire_content(m.content.as_ref(), part),
        tool_calls: m.tool_calls.as_ref(),
        tool_call_id: m.tool_call_id.as_ref(),
        name: m.name.as_ref(),
    }
}

pub(super) enum Turn<'a> {
    Message(&'a PromptMessage),
    ToolText(&'a PromptMessage, String),
    Media(Vec<ContentPart>),
}

pub(super) fn turns(messages: &[PromptMessage], part: PartPolicy) -> Vec<Turn<'_>> {
    let mut out = Vec::with_capacity(messages.len());
    let mut carried: Vec<ContentPart> = Vec::new();
    for message in messages {
        if message.role != Role::Tool {
            flush_media(&mut carried, &mut out);
            out.push(Turn::Message(message));
            continue;
        }
        let (text, media) = split_media(message.content.as_ref(), part);
        if media.is_empty() {
            out.push(Turn::Message(message));
            continue;
        }
        let said = match text.is_empty() {
            true => {
                "This tool answered with attachments. They follow in the next message.".to_string()
            }
            false => format!("{text}\n\nAttachments follow in the next message."),
        };
        out.push(Turn::ToolText(message, said));
        carried.push(ContentPart::Text {
            text: format!("Attachment from {}:", names(message)),
        });
        carried.extend(media);
    }
    flush_media(&mut carried, &mut out);
    out
}

fn flush_media<'a>(carried: &mut Vec<ContentPart>, out: &mut Vec<Turn<'a>>) {
    if !carried.is_empty() {
        out.push(Turn::Media(std::mem::take(carried)));
    }
}

fn wire_messages(messages: &[PromptMessage]) -> Vec<WireMessage<'_>> {
    turns(messages, part)
        .into_iter()
        .map(|turn| match turn {
            Turn::Message(m) => wire_message(m, part),
            Turn::ToolText(m, text) => WireMessage {
                content: Some(WireContent::Text(text)),
                ..wire_message(m, part)
            },
            Turn::Media(parts) => WireMessage {
                role: Role::User,
                content: Some(WireContent::Parts(parts)),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        })
        .collect()
}

fn names(message: &PromptMessage) -> String {
    match (&message.name, &message.tool_call_id) {
        (Some(name), Some(id)) => format!("the `{name}` tool ({id})"),
        (Some(name), None) => format!("the `{name}` tool"),
        (None, Some(id)) => format!("tool call {id}"),
        (None, None) => "the tool result above".to_string(),
    }
}

fn split_media(content: Option<&PromptContent>, part: PartPolicy) -> (String, Vec<ContentPart>) {
    match content {
        None => (String::new(), Vec::new()),
        Some(PromptContent::Text(text)) => (text.clone(), Vec::new()),
        Some(PromptContent::Parts(parts)) => {
            let (text, media): (Vec<ContentPart>, Vec<ContentPart>) = parts
                .iter()
                .map(part)
                .partition(|p| matches!(p, ContentPart::Text { .. }));
            let text = text
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            (text, media)
        }
    }
}

#[derive(Default, Clone, Copy)]
struct CacheOpts<'a> {
    key: Option<&'a str>,
    retention: Option<&'a str>,
}

const CACHE_KEY_MAX_CHARS: usize = 64;

fn cache_key(session_id: &str) -> Option<&str> {
    if session_id.is_empty() {
        return None;
    }
    Some(match session_id.char_indices().nth(CACHE_KEY_MAX_CHARS) {
        Some((end, _)) => &session_id[..end],
        None => session_id,
    })
}

fn effort_str(e: ReasoningEffort) -> &'static str {
    match e {
        ReasoningEffort::Xhigh => "xhigh",
        ReasoningEffort::High => "high",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::Low => "low",
        ReasoningEffort::Minimal => "minimal",
        ReasoningEffort::None => "none",
    }
}

fn digits(s: &str) -> (&str, &str) {
    s.split_at(s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len()))
}

fn o_series(model: &str) -> Option<u32> {
    let (major, rest) = digits(model.strip_prefix('o')?);
    if !rest.is_empty() && !rest.starts_with('-') {
        return None;
    }
    major.parse().ok()
}

struct GptName<'a> {
    major: u32,
    minor: Option<u32>,
    variant: Option<&'a str>,
}

fn gpt_name(model: &str) -> Option<GptName<'_>> {
    let (major, rest) = digits(model.strip_prefix("gpt-")?);
    let major = major.parse().ok()?;
    let (minor, rest) = match rest.strip_prefix('.') {
        Some(after_dot) => {
            let (minor, rest) = digits(after_dot);
            (Some(minor.parse().ok()?), rest)
        }
        None => (None, rest),
    };
    let variant = match rest {
        "" => None,
        rest => Some(rest.strip_prefix('-')?),
    };
    Some(GptName {
        major,
        minor,
        variant,
    })
}

fn is_reasoning_model(model: &str) -> bool {
    if o_series(model).is_some() {
        return true;
    }
    gpt_name(model).is_some_and(|g| {
        let chat = g.minor.is_none() && g.variant.is_some_and(|v| v.starts_with("chat"));
        g.major >= 5 && !chat
    })
}

fn reads_sampling_at_no_effort(model: &str) -> bool {
    gpt_name(model).is_some_and(|g| (g.major, g.minor.unwrap_or(0)) >= (5, 1))
}

impl<'a> WireBody<'a> {
    fn build(
        request: &'a PromptRequest,
        search: DeferToolsStrategy,
        stream: Option<bool>,
        cache: CacheOpts<'a>,
    ) -> Self {
        let reasoning_effort = request
            .reasoning
            .as_ref()
            .and_then(|r| r.effort)
            .map(effort_str);

        let reasoning = reasoning_effort.is_some() || is_reasoning_model(&request.model);
        let sampling_back =
            reasoning_effort == Some("none") && reads_sampling_at_no_effort(&request.model);
        let temperature = if reasoning && !sampling_back {
            None
        } else {
            request.temperature
        };

        WireBody {
            model: &request.model,
            messages: wire_messages(&request.messages),
            tools: request
                .offered_tools(search)
                .map(|ts| ts.into_iter().map(WireTool::from).collect()),
            temperature,
            max_completion_tokens: request.max_completion_tokens,
            reasoning_effort,
            stream,
            prompt_cache_key: cache.key,
            prompt_cache_retention: cache.retention,
            stream_options: (stream == Some(true)).then_some(StreamOptions {
                include_usage: true,
            }),
        }
    }
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    model: String,
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<serde_json::Value>,
}

#[derive(Debug, Default, Deserialize)]
struct ChatUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    prompt_tokens_details: PromptTokensDetails,
}

#[derive(Debug, Default, Deserialize)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: u64,
    #[serde(default)]
    cache_write_tokens: u64,
}

pub(crate) fn usage_from_value(raw: Option<serde_json::Value>) -> Option<Usage> {
    let raw = raw?;
    let counts: ChatUsage = serde_json::from_value(raw.clone()).unwrap_or_default();
    let cache_read = counts.prompt_tokens_details.cached_tokens;
    let cache_write = counts.prompt_tokens_details.cache_write_tokens;
    Some(
        Usage::new(
            counts
                .prompt_tokens
                .saturating_sub(cache_read + cache_write),
            cache_read,
            cache_write,
            counts.completion_tokens,
        )
        .with_provider(raw),
    )
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ChoiceMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChoiceMessage {
    content: Option<String>,
    #[serde(default, alias = "reasoning_content")]
    reasoning: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<WireToolCall>>,
}

#[derive(Debug, Clone, Deserialize)]
struct WireToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: WireFunctionCall,
}

#[derive(Debug, Clone, Deserialize)]
struct WireFunctionCall {
    name: String,
    arguments: String,
}

impl From<WireToolCall> for ToolCall {
    fn from(tc: WireToolCall) -> Self {
        ToolCall {
            id: tc.id,
            call_type: tc.call_type,
            function: ToolCallFunction {
                name: tc.function.name,
                arguments: tc.function.arguments,
            },
        }
    }
}

impl ChatCompletionResponse {
    fn into_llm_response(self) -> LlmResponse {
        let choice = self.choices.into_iter().next();
        LlmResponse {
            model: self.model,
            content: choice.as_ref().and_then(|c| c.message.content.clone()),
            reasoning: Reasoning::new(
                ReasoningProvider::Openai,
                choice.as_ref().and_then(|c| c.message.reasoning.clone()),
                Vec::new(),
            ),
            tool_calls: choice
                .as_ref()
                .and_then(|c| c.message.tool_calls.as_ref())
                .map(|tcs| tcs.iter().map(|tc| tc.clone().into()).collect())
                .unwrap_or_default(),
            finish_reason: choice.and_then(|c| c.finish_reason),
            usage: usage_from_value(self.usage),
            cost: None,
            images: Vec::new(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct StreamChunkResponse {
    #[serde(default)]
    model: String,
    #[serde(default)]
    choices: Vec<StreamChunkChoice>,
    #[serde(default)]
    usage: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct StreamChunkChoice {
    delta: StreamChunkDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StreamChunkDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default, alias = "reasoning_content")]
    reasoning: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct ToolCallDelta {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<FunctionCallDelta>,
}

#[derive(Debug, Deserialize)]
struct FunctionCallDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Default)]
struct ToolCallAccum {
    id: String,
    name: String,
    arguments: String,
}

pub(crate) fn request_to_wire(
    request: &PromptRequest,
    search: DeferToolsStrategy,
) -> serde_json::Value {
    serde_json::to_value(WireBody::build(request, search, None, CacheOpts::default()))
        .unwrap_or_default()
}

pub(crate) fn response_from_wire(value: serde_json::Value) -> Result<LlmResponse, String> {
    crate::json::from_value::<ChatCompletionResponse>("openai response", value)
        .map(ChatCompletionResponse::into_llm_response)
        .map_err(|e| e.to_string())
}

#[derive(Default)]
pub(crate) struct StreamParser {
    content: String,
    reasoning: String,
    tool_calls: Vec<ToolCallAccum>,
    finish_reason: Option<String>,
    model: Option<String>,
    usage: Option<serde_json::Value>,
}

impl StreamParser {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn parse_data(&mut self, data: &str) -> Vec<StreamDelta> {
        if data == "[DONE]" {
            return Vec::new();
        }
        match serde_json::from_str(data) {
            Ok(chunk) => self.on_chunk(chunk),
            Err(_) => Vec::new(),
        }
    }

    fn on_chunk(&mut self, chunk: StreamChunkResponse) -> Vec<StreamDelta> {
        let mut deltas = Vec::new();
        if !chunk.model.is_empty() {
            self.model = Some(chunk.model);
        }
        if let Some(u) = chunk.usage {
            self.usage = Some(u);
        }

        for choice in chunk.choices {
            if let Some(ref text) = choice.delta.content {
                self.content.push_str(text);
                deltas.push(StreamDelta {
                    text: Some(text.clone()),
                    ..Default::default()
                });
            }

            if let Some(ref thinking) = choice.delta.reasoning {
                if !thinking.is_empty() {
                    self.reasoning.push_str(thinking);
                    deltas.push(StreamDelta {
                        reasoning: Some(thinking.clone()),
                        ..Default::default()
                    });
                }
            }

            if let Some(tc_deltas) = choice.delta.tool_calls {
                for tc_delta in tc_deltas {
                    while self.tool_calls.len() <= tc_delta.index {
                        self.tool_calls.push(ToolCallAccum::default());
                    }
                    let accum = &mut self.tool_calls[tc_delta.index];
                    if let Some(id) = tc_delta.id {
                        accum.id = id;
                    }
                    let mut args_fragment = None;
                    if let Some(f) = tc_delta.function {
                        if let Some(name) = f.name {
                            accum.name = name;
                        }
                        if let Some(args) = f.arguments {
                            accum.arguments.push_str(&args);
                            args_fragment = Some(args);
                        }
                    }
                    if !accum.id.is_empty() {
                        deltas.push(StreamDelta {
                            tool_calls: vec![ToolCallChunk {
                                id: accum.id.clone(),
                                name: (!accum.name.is_empty()).then(|| accum.name.clone()),
                                arguments: args_fragment,
                            }],
                            ..Default::default()
                        });
                    }
                }
            }

            if let Some(reason) = choice.finish_reason {
                self.finish_reason = Some(reason.clone());
                deltas.push(StreamDelta {
                    finish_reason: Some(reason),
                    ..Default::default()
                });
            }
        }
        deltas
    }

    fn into_response(self, fallback_model: &str) -> LlmResponse {
        LlmResponse {
            model: self.model.unwrap_or_else(|| fallback_model.to_string()),
            content: (!self.content.is_empty()).then_some(self.content),
            reasoning: Reasoning::new(
                ReasoningProvider::Openai,
                (!self.reasoning.is_empty()).then_some(self.reasoning),
                Vec::new(),
            ),
            tool_calls: self
                .tool_calls
                .into_iter()
                .filter(|tc| !tc.id.is_empty())
                .map(|tc| ToolCall {
                    id: tc.id,
                    call_type: "function".to_string(),
                    function: ToolCallFunction {
                        name: tc.name,
                        arguments: tc.arguments,
                    },
                })
                .collect(),
            finish_reason: self.finish_reason,
            usage: usage_from_value(self.usage),
            cost: None,
            images: Vec::new(),
        }
    }
}

#[derive(Deserialize)]
pub struct OpenAiConfig {
    pub base_url: String,
    pub api_key: String,
    #[serde(default)]
    pub organization: Option<String>,
    #[serde(default)]
    pub project: Option<String>,
    #[serde(default)]
    pub cache_retention: Option<String>,
}

impl OpenAiConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: api_key.into(),
            organization: None,
            project: None,
            cache_retention: None,
        }
    }
}

pub struct OpenAiClient {
    http: Client,
    base_url: String,
    api_key: String,
    extra_headers: HeaderMap,
    cache_retention: Option<String>,
}

impl OpenAiClient {
    pub fn from_config(config: OpenAiConfig) -> Self {
        let mut extra_headers = HeaderMap::new();
        if let Some(org) = config.organization.as_deref() {
            if let Ok(v) = HeaderValue::from_str(org) {
                extra_headers.insert(HeaderName::from_static("openai-organization"), v);
            }
        }
        if let Some(project) = config.project.as_deref() {
            if let Ok(v) = HeaderValue::from_str(project) {
                extra_headers.insert(HeaderName::from_static("openai-project"), v);
            }
        }
        Self {
            http: crate::providers::http_client(),
            base_url: config.base_url,
            api_key: config.api_key,
            extra_headers,
            cache_retention: config.cache_retention,
        }
    }

    async fn post_chat_completion(
        &self,
        request: &PromptRequest,
        search: DeferToolsStrategy,
        stream: bool,
        session_id: &str,
    ) -> Result<reqwest::Response, LlmCallError> {
        let cache = CacheOpts {
            key: cache_key(session_id),
            retention: self.cache_retention.as_deref(),
        };
        let wire = WireBody::build(request, search, Some(stream), cache);
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));

        self.http
            .post(&url)
            .bearer_auth(&self.api_key)
            .headers(self.extra_headers.clone())
            .json(&wire)
            .send()
            .await
            .map_err(|e| {
                LlmCallError::new(
                    ErrorCode::ProviderError,
                    format!("HTTP request failed: {e}"),
                    !e.is_builder(),
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
    let message = match crate::json::error_message(body.as_bytes()) {
        Some(reported) => format!("OpenAI API error {status}: {reported}"),
        None => format!("OpenAI API error {status}"),
    };
    LlmCallError::new(code, message, retryable)
}

#[async_trait]
impl LlmCallable for OpenAiClient {
    async fn call(
        &self,
        request: &PromptRequest,
        ctx: &CallContext<'_>,
    ) -> Result<LlmResponse, LlmCallError> {
        let resp = self
            .post_chat_completion(request, ctx.defer_tools_strategy, false, ctx.session_id)
            .await?;
        let status = resp.status();
        let body = resp.text().await.map_err(|e| {
            LlmCallError::new(ErrorCode::ProviderError, format!("read body: {e}"), true)
        })?;

        if !status.is_success() {
            tracing::warn!(status = status.as_u16(), body = %crate::json::excerpt(body.as_bytes()), "openai api call failed");
            return Err(classify_error(status, &body));
        }

        let parsed: ChatCompletionResponse =
            crate::json::from_str("openai response", &body).map_err(LlmCallError::from)?;

        Ok(parsed.into_llm_response())
    }

    async fn call_streaming(
        &self,
        request: &PromptRequest,
        ctx: &CallContext<'_>,
        chunk_tx: UnboundedSender<StreamDelta>,
    ) -> Result<LlmResponse, LlmCallError> {
        let resp = self
            .post_chat_completion(request, ctx.defer_tools_strategy, true, ctx.session_id)
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

                if line.is_empty() || line.starts_with(':') {
                    continue;
                }

                let data = match crate::providers::sse_data(&line) {
                    Some(d) => d,
                    None => continue,
                };

                if data == "[DONE]" {
                    break;
                }

                for delta in parser.parse_data(data) {
                    let _ = chunk_tx.send(delta);
                }
            }
        }

        Ok(parser.into_response(&request.model))
    }
}

pub struct OpenAiProvider {
    client: Arc<OpenAiClient>,
}

impl OpenAiProvider {
    pub fn new(config: OpenAiConfig) -> Self {
        Self {
            client: Arc::new(OpenAiClient::from_config(config)),
        }
    }
}

#[async_trait]
impl LlmProviderTrait for OpenAiProvider {
    async fn resolve(&self, _owner: &SessionOwner) -> Result<Arc<dyn LlmCallable>, String> {
        Ok(self.client.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{PromptContent, PromptMessage, ReasoningConfig, Role};

    fn req(model: &str) -> PromptRequest {
        PromptRequest {
            model: model.to_string(),
            messages: vec![PromptMessage {
                role: Role::User,
                content: Some(PromptContent::Text("hi".to_string())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
                reasoning: None,
            }],
            tools: None,
            temperature: Some(0.7),
            max_completion_tokens: Some(500),
            reasoning: None,
        }
    }

    #[test]
    fn a_thought_response_reads_under_either_field_name() {
        for field in ["reasoning", "reasoning_content"] {
            let parsed = response_from_wire(serde_json::json!({
                "model": "m",
                "choices": [{
                    "message": { "role": "assistant", "content": "hi", field: "thought" },
                    "finish_reason": "stop",
                }],
            }))
            .expect("parses");
            let reasoning = parsed.reasoning.expect(field);
            assert_eq!(reasoning.provider, ReasoningProvider::Openai);
            assert_eq!(reasoning.text.as_deref(), Some("thought"));
            assert!(reasoning.blocks.is_empty());
        }
    }

    #[test]
    fn a_streamed_thought_accumulates_and_streams_out() {
        let mut parser = StreamParser::new();
        let deltas: Vec<_> = ["think", "ing"]
            .iter()
            .flat_map(|piece| {
                parser.parse_data(
                    &serde_json::json!({
                        "choices": [{ "delta": { "reasoning": piece } }],
                    })
                    .to_string(),
                )
            })
            .collect();

        assert_eq!(
            deltas
                .iter()
                .filter_map(|d| d.reasoning.as_deref())
                .collect::<Vec<_>>(),
            vec!["think", "ing"]
        );
        let response = parser.into_response("m");
        assert_eq!(
            response.reasoning.expect("reasoning").text.as_deref(),
            Some("thinking")
        );
    }

    #[test]
    fn the_engines_own_message_fields_stay_out_of_the_request() {
        let mut request = req("gpt-4o");
        request.messages[0].reasoning = Reasoning::new(
            ReasoningProvider::Openai,
            Some("thought".to_string()),
            Vec::new(),
        );
        let body = request_to_wire(&request, DeferToolsStrategy::Search);

        let sent = &body["messages"][0];
        assert!(sent.get("reasoning").is_none(), "internal field leaked");
        assert!(sent.get("id").is_none(), "node id leaked");
        assert_eq!(sent["content"], "hi");
    }

    #[test]
    fn the_openai_policy_sends_what_openai_takes_and_notes_the_rest() {
        let media = |mime: &str| PromptPart::Media {
            mime: mime.into(),
            name: Some("f".into()),
            bytes: vec![1, 2, 3],
        };
        let kind = |p: &PromptPart| serde_json::to_value(part(p)).unwrap();
        assert_eq!(kind(&media("image/jpeg"))["type"], "image_url");
        assert_eq!(kind(&media("application/pdf"))["type"], "file");
        let wav = kind(&media("audio/x-wav"));
        assert_eq!(wav["type"], "input_audio");
        assert_eq!(wav["input_audio"]["format"], "wav");
        assert_eq!(kind(&media("audio/mpeg"))["input_audio"]["format"], "mp3");
        assert_eq!(kind(&media("audio/ogg"))["text"], "[audio content]");
        assert_eq!(kind(&media("video/mp4"))["text"], "[video content]");
        assert_eq!(kind(&media("text/csv"))["text"], "[text content]");
        let link = |mime: &str| PromptPart::Link {
            uri: "https://x.example/f".into(),
            name: None,
            mime_type: Some(mime.into()),
        };
        assert_eq!(kind(&link("image/*"))["type"], "image_url");
        assert_eq!(kind(&link("video/*"))["text"], "https://x.example/f");
    }

    #[test]
    fn image_parts_serialize_in_the_openai_wire_shape() {
        let mut request = req("gpt-4o");
        request.messages[0].content = Some(PromptContent::Parts(vec![
            PromptPart::Text {
                text: "look".into(),
            },
            PromptPart::Media {
                mime: "image/png".into(),
                name: None,
                bytes: vec![1, 2, 3],
            },
        ]));
        let body = request_to_wire(&request, DeferToolsStrategy::Search);

        let content = &body["messages"][0]["content"];
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "look");
        assert_eq!(content[1]["type"], "image_url");
        assert_eq!(content[1]["image_url"]["url"], "data:image/png;base64,AQID");
    }

    #[test]
    fn reasoning_model_strips_temperature_and_maps_effort() {
        let mut r = req("gpt-5.5");
        r.reasoning = Some(ReasoningConfig {
            effort: Some(ReasoningEffort::High),
            max_tokens: None,
            exclude: None,
            enabled: None,
        });
        let v = serde_json::to_value(WireBody::build(
            &r,
            DeferToolsStrategy::Search,
            Some(false),
            CacheOpts::default(),
        ))
        .unwrap();
        assert_eq!(v["reasoning_effort"], "high");
        assert!(v.get("temperature").is_none());
        assert_eq!(v["max_completion_tokens"], 500);
        assert!(v.get("max_tokens").is_none());
    }

    #[test]
    fn reasoning_model_without_explicit_effort_still_strips_temperature() {
        let v = serde_json::to_value(WireBody::build(
            &req("gpt-5.4-mini"),
            DeferToolsStrategy::Search,
            Some(false),
            CacheOpts::default(),
        ))
        .unwrap();
        assert!(v.get("temperature").is_none());
        assert!(v.get("reasoning_effort").is_none());
    }

    #[test]
    fn a_reasoning_name_is_read_whole_and_not_by_its_first_letters() {
        for model in [
            "o1",
            "o3-mini",
            "o4-mini-2025-04-16",
            "o9",
            "o99-2099-01-01",
        ] {
            assert!(is_reasoning_model(model), "{model}");
        }
        for model in ["gpt-5", "gpt-5-mini", "gpt-5.4-nano-2026-03-17", "gpt-9"] {
            assert!(is_reasoning_model(model), "{model}");
        }

        assert!(!is_reasoning_model("gpt-5-chat-latest"));
        assert!(is_reasoning_model("gpt-5.4-chat-latest"));

        for model in ["gpt-4o", "gpt-4.1-mini", "gpt-3.5-turbo", "o"] {
            assert!(!is_reasoning_model(model), "{model}");
        }

        for model in ["ft:gpt-9:acme:custom:abc123", "acme-gpt-9-proxy"] {
            assert!(!is_reasoning_model(model), "{model}");
        }
    }

    #[test]
    fn a_chat_model_named_for_gpt_5_keeps_its_temperature() {
        let v = serde_json::to_value(WireBody::build(
            &req("gpt-5-chat-latest"),
            DeferToolsStrategy::Search,
            Some(false),
            CacheOpts::default(),
        ))
        .unwrap();
        assert_eq!(v["temperature"], 0.7);
    }

    #[test]
    fn no_effort_hands_temperature_back_where_the_model_reads_it() {
        let mut r = req("gpt-5.1");
        r.reasoning = Some(ReasoningConfig {
            effort: Some(ReasoningEffort::None),
            max_tokens: None,
            exclude: None,
            enabled: None,
        });
        let v = serde_json::to_value(WireBody::build(
            &r,
            DeferToolsStrategy::Search,
            Some(false),
            CacheOpts::default(),
        ))
        .unwrap();
        assert_eq!(v["reasoning_effort"], "none");
        assert_eq!(v["temperature"], 0.7, "GPT-5.1 reads it again at no effort");

        for model in ["gpt-5", "o3"] {
            let mut r = req(model);
            r.reasoning = Some(ReasoningConfig {
                effort: Some(ReasoningEffort::None),
                max_tokens: None,
                exclude: None,
                enabled: None,
            });
            let v = serde_json::to_value(WireBody::build(
                &r,
                DeferToolsStrategy::Search,
                Some(false),
                CacheOpts::default(),
            ))
            .unwrap();
            assert!(v.get("temperature").is_none(), "{model}");
        }
    }

    #[test]
    fn non_reasoning_model_keeps_temperature() {
        let v = serde_json::to_value(WireBody::build(
            &req("gpt-4o"),
            DeferToolsStrategy::Search,
            Some(false),
            CacheOpts::default(),
        ))
        .unwrap();
        assert_eq!(v["temperature"], 0.7);
        assert!(v.get("reasoning_effort").is_none());
    }

    #[test]
    fn a_streamed_call_names_its_session_and_asks_for_the_token_counts() {
        let cache = CacheOpts {
            key: Some("sess_1"),
            retention: Some("24h"),
        };
        let v = serde_json::to_value(WireBody::build(
            &req("gpt-4o"),
            DeferToolsStrategy::Search,
            Some(true),
            cache,
        ))
        .unwrap();
        assert_eq!(v["prompt_cache_key"], "sess_1");
        assert_eq!(v["prompt_cache_retention"], "24h");
        assert_eq!(v["stream_options"]["include_usage"], true);
    }

    #[test]
    fn the_counts_of_a_response_read_the_same_as_any_other_provider() {
        let usage = usage_from_value(Some(serde_json::json!({
            "prompt_tokens": 10000,
            "completion_tokens": 200,
            "prompt_tokens_details": { "cached_tokens": 9000 }
        })))
        .expect("usage");
        assert_eq!(usage.input, 10000);
        assert_eq!(usage.uncached_input, 1000);
        assert_eq!(usage.cache_read, 9000);
        assert_eq!(usage.cache_write, 0);
        assert_eq!(usage.output, 200);
        assert_eq!(usage.total, 10200);
    }

    #[test]
    fn a_response_that_counts_nothing_still_reads() {
        assert_eq!(usage_from_value(None), None);
        let usage = usage_from_value(Some(serde_json::json!({}))).expect("usage");
        assert_eq!(usage.total, 0);
    }

    #[test]
    fn a_long_session_name_is_cut_to_what_the_api_takes() {
        let long = "sé".repeat(80);
        let key = cache_key(&long).expect("a key");
        assert_eq!(key.chars().count(), 64);
        assert!(long.starts_with(key));
        assert_eq!(cache_key("sess_1"), Some("sess_1"));
        assert_eq!(cache_key(""), None);
    }

    fn tool_message(id: &str, content: PromptContent) -> PromptMessage {
        PromptMessage {
            role: Role::Tool,
            content: Some(content),
            tool_calls: None,
            tool_call_id: Some(id.to_string()),
            name: Some("skill".to_string()),
            reasoning: None,
        }
    }

    fn image() -> PromptPart {
        PromptPart::Media {
            mime: "image/png".into(),
            name: None,
            bytes: vec![0, 0, 0],
        }
    }

    #[test]
    fn a_tool_that_answered_with_media_sends_it_on_a_user_turn() {
        let mut r = req("gpt-4o");
        r.messages.push(tool_message(
            "call_1",
            PromptContent::Parts(vec![
                PromptPart::Text {
                    text: "the diagram".to_string(),
                },
                image(),
            ]),
        ));
        let v = serde_json::to_value(WireBody::build(
            &r,
            DeferToolsStrategy::Search,
            Some(false),
            CacheOpts::default(),
        ))
        .unwrap();
        let m = v["messages"].as_array().unwrap();

        assert_eq!(m.len(), 3, "the user turn is added, not substituted");
        assert_eq!(m[1]["role"], "tool");
        assert_eq!(m[1]["tool_call_id"], "call_1");
        let said = m[1]["content"].as_str().expect("a tool answers in text");
        assert!(said.starts_with("the diagram"), "{said}");
        assert!(said.contains("next message"), "{said}");

        assert_eq!(m[2]["role"], "user");
        assert!(m[2].get("tool_call_id").is_none());
        assert_eq!(m[2]["content"][0]["type"], "text");
        assert!(m[2]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("call_1"));
        assert_eq!(m[2]["content"][1]["type"], "image_url");
        assert_eq!(
            m[2]["content"][1]["image_url"]["url"],
            "data:image/png;base64,AAAA"
        );
    }

    #[test]
    fn parallel_tool_answers_stay_together_and_share_one_user_turn() {
        let mut r = req("gpt-4o");
        r.messages
            .push(tool_message("call_1", PromptContent::Parts(vec![image()])));
        r.messages.push(tool_message(
            "call_2",
            PromptContent::Text("plain".to_string()),
        ));
        r.messages
            .push(tool_message("call_3", PromptContent::Parts(vec![image()])));
        let v = serde_json::to_value(WireBody::build(
            &r,
            DeferToolsStrategy::Search,
            Some(false),
            CacheOpts::default(),
        ))
        .unwrap();
        let m = v["messages"].as_array().unwrap();

        let roles: Vec<&str> = m.iter().map(|x| x["role"].as_str().unwrap()).collect();
        assert_eq!(roles, ["user", "tool", "tool", "tool", "user"]);
        assert_eq!(m[2]["content"], "plain", "a text answer is untouched");
        let parts = m[4]["content"].as_array().unwrap();
        assert_eq!(parts.len(), 4, "a label and an image for each of the two");
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(parts[3]["type"], "image_url");
    }

    #[test]
    fn a_text_only_transcript_is_unchanged() {
        let mut r = req("gpt-4o");
        r.messages.push(tool_message(
            "call_1",
            PromptContent::Text("72F".to_string()),
        ));
        let v = serde_json::to_value(WireBody::build(
            &r,
            DeferToolsStrategy::Search,
            Some(false),
            CacheOpts::default(),
        ))
        .unwrap();
        let m = v["messages"].as_array().unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(m[1]["content"], "72F");
    }

    #[test]
    fn a_body_that_does_not_stream_asks_for_no_stream_options() {
        let v = serde_json::to_value(WireBody::build(
            &req("gpt-4o"),
            DeferToolsStrategy::Search,
            Some(false),
            CacheOpts::default(),
        ))
        .unwrap();
        assert!(v.get("stream_options").is_none());
        assert!(v.get("prompt_cache_key").is_none());
        assert!(v.get("prompt_cache_retention").is_none());
    }
}
