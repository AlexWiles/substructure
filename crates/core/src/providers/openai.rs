use std::sync::Arc;

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::UnboundedSender;
use tokio_stream::StreamExt;

use crate::llm::{CallContext, LlmCallError, LlmCallable, LlmProviderTrait};
use crate::protocol::{
    DeferToolsStrategy, ErrorCode, LlmRequest, LlmResponse, LlmTool, ReasoningEffort, SessionOwner,
    StreamDelta, ToolCall, ToolCallChunk, ToolCallFunction,
};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Wraps our normalized `LlmTool` with the `"type": "function"` field
/// that the OpenAI API expects.
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

/// Wire-format request body.
#[derive(Serialize)]
struct WireBody<'a> {
    model: &'a str,
    messages: &'a [crate::protocol::DraftMessage],
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
    /// Which cache machine the prompt routes to. Sessions of one agent open
    /// alike, so without this they crowd onto one machine and spill.
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_retention: Option<&'a str>,
    /// A streamed call reports no tokens at all without this.
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
}

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

/// What the caller knows about caching this call: the session the prompt
/// belongs to, and how long the vendor holds it.
#[derive(Default, Clone, Copy)]
struct CacheOpts<'a> {
    key: Option<&'a str>,
    retention: Option<&'a str>,
}

/// The API refuses a longer key, and a session id is the caller's own string:
/// a Slack thread names one, and a client can send any name it likes.
const CACHE_KEY_MAX_CHARS: usize = 64;

/// The first 64 characters of `session_id`, or nothing for an unnamed session.
/// Two sessions sharing those characters share a machine, and nothing worse.
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

/// Reasoning models (GPT-5 family, o-series) reject sampling params like
/// `temperature`. Detect them by id prefix so we strip it even when the
/// request doesn't set an explicit effort.
fn is_reasoning_model(model: &str) -> bool {
    const PREFIXES: &[&str] = &["gpt-5", "o1", "o3", "o4"];
    PREFIXES.iter().any(|p| model.starts_with(p))
}

impl<'a> WireBody<'a> {
    /// `stream: None` omits the field so the body is valid input for both the
    /// create and stream calls a worker might make.
    fn build(
        request: &'a LlmRequest,
        search: DeferToolsStrategy,
        stream: Option<bool>,
        cache: CacheOpts<'a>,
    ) -> Self {
        let reasoning_effort = request
            .reasoning
            .as_ref()
            .and_then(|r| r.effort)
            .map(effort_str);

        // Omit temperature for reasoning models — they 400 on it.
        let temperature = if reasoning_effort.is_some() || is_reasoning_model(&request.model) {
            None
        } else {
            request.temperature
        };

        WireBody {
            model: &request.model,
            messages: &request.messages,
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

#[derive(Debug, Deserialize)]
struct Choice {
    message: ChoiceMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChoiceMessage {
    content: Option<String>,
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
            tool_calls: choice
                .as_ref()
                .and_then(|c| c.message.tool_calls.as_ref())
                .map(|tcs| tcs.iter().map(|tc| tc.clone().into()).collect())
                .unwrap_or_default(),
            finish_reason: choice.and_then(|c| c.finish_reason),
            usage: self.usage,
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

// ── Worker-format seam ───────────────────────────────────────────────────

/// The Chat Completions body for `request`, `stream` omitted. The worker owns
/// its own call, so the caching fields are the worker's to add.
pub(crate) fn request_to_wire(
    request: &LlmRequest,
    search: DeferToolsStrategy,
) -> serde_json::Value {
    serde_json::to_value(WireBody::build(request, search, None, CacheOpts::default()))
        .unwrap_or_default()
}

/// A raw Chat Completions response → the neutral `LlmResponse`.
pub(crate) fn response_from_wire(value: serde_json::Value) -> Result<LlmResponse, String> {
    crate::json::from_value::<ChatCompletionResponse>("openai response", value)
        .map(ChatCompletionResponse::into_llm_response)
        .map_err(|e| e.to_string())
}

/// Folds Chat Completions stream chunks into `StreamDelta`s and the final
/// response. Shared by the server-side SSE loop and the worker-format delta
/// seam (`providers::format`).
#[derive(Default)]
pub(crate) struct StreamParser {
    content: String,
    tool_calls: Vec<ToolCallAccum>,
    finish_reason: Option<String>,
    model: Option<String>,
    usage: Option<serde_json::Value>,
}

impl StreamParser {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Fold one raw chunk payload; `[DONE]` and unknown payloads yield nothing.
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
            usage: self.usage,
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
    /// How long a cached prefix lives: `24h`, or `in_memory` for the default
    /// few minutes. Absent sends nothing, which the newer models want.
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
        request: &LlmRequest,
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
        Some(reported) => format!("OpenAI API error {status}: {reported}"),
        None => format!("OpenAI API error {status}"),
    };
    LlmCallError::new(code, message, retryable)
}

#[async_trait]
impl LlmCallable for OpenAiClient {
    async fn call(
        &self,
        request: &LlmRequest,
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
        request: &LlmRequest,
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

                let data = match line.strip_prefix("data: ") {
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
    use crate::protocol::{Content, DraftMessage, ReasoningConfig, Role};

    fn req(model: &str) -> LlmRequest {
        LlmRequest {
            model: model.to_string(),
            messages: vec![DraftMessage {
                id: None,
                role: Role::User,
                content: Some(Content::Text("hi".to_string())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            tools: None,
            temperature: Some(0.7),
            max_completion_tokens: Some(500),
            reasoning: None,
        }
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
    fn a_long_session_name_is_cut_to_what_the_api_takes() {
        let long = "sé".repeat(80);
        let key = cache_key(&long).expect("a key");
        assert_eq!(key.chars().count(), 64);
        assert!(long.starts_with(key));
        assert_eq!(cache_key("sess_1"), Some("sess_1"));
        assert_eq!(cache_key(""), None);
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
