use std::sync::Arc;

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::UnboundedSender;
use tokio_stream::StreamExt;

use rust_decimal::Decimal;

use super::openai::{wire_content, WireContent};
use crate::llm::{CallContext, LlmCallError, LlmCallable, LlmProviderTrait};
use crate::mime;
use crate::protocol::{
    AudioData, ContentPart, DeferToolsStrategy, ErrorCode, FileData, LlmResponse, LlmTool,
    PromptMessage, PromptPart, PromptRequest, Reasoning, ReasoningConfig, ReasoningProvider,
    ResponseImage, Role, SessionOwner, StreamDelta, ToolCall, ToolCallChunk, ToolCallFunction,
};

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
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    reasoning_details: &'a [serde_json::Value],
}

fn wire_message(m: &PromptMessage) -> WireMessage<'_> {
    {
        WireMessage {
            role: m.role.clone(),
            content: wire_content(m.content.as_ref(), part),
            tool_calls: m.tool_calls.as_ref(),
            tool_call_id: m.tool_call_id.as_ref(),
            name: m.name.as_ref(),
            reasoning_details: m
                .reasoning
                .as_ref()
                .map(|r| r.blocks_for(ReasoningProvider::Openrouter))
                .unwrap_or_default(),
        }
    }
}

fn part(p: &PromptPart) -> ContentPart {
    match p {
        PromptPart::Media { mime, name, bytes } => match mime::essence(mime) {
            "audio" => match audio_format(mime) {
                Some(format) => ContentPart::InputAudio {
                    input_audio: AudioData {
                        data: mime::base64(bytes),
                        format: format.to_string(),
                    },
                },
                None => file(mime, name.as_deref(), bytes),
            },
            "video" if !video_playable(mime) => file(mime, name.as_deref(), bytes),
            _ => ContentPart::from(p),
        },
        _ => ContentPart::from(p),
    }
}

fn file(mime: &str, name: Option<&str>, bytes: &[u8]) -> ContentPart {
    ContentPart::File {
        file: FileData {
            filename: name.unwrap_or("file").to_string(),
            file_data: mime::data_uri(mime, bytes),
        },
    }
}

fn audio_format(mime: &str) -> Option<&'static str> {
    let (kind, sub) = mime::parts(mime);
    if kind != "audio" {
        return None;
    }
    match sub {
        "mpeg" | "mp3" => Some("mp3"),
        "wav" | "x-wav" | "wave" | "vnd.wave" => Some("wav"),
        "aiff" | "x-aiff" => Some("aiff"),
        "aac" => Some("aac"),
        "ogg" => Some("ogg"),
        "flac" | "x-flac" => Some("flac"),
        "mp4" | "m4a" | "x-m4a" => Some("m4a"),
        _ => None,
    }
}

fn video_playable(mime: &str) -> bool {
    let (kind, sub) = mime::parts(mime);
    kind == "video" && matches!(sub, "mp4" | "mpeg" | "mov" | "quicktime" | "webm")
}

fn wire_messages(messages: &[PromptMessage]) -> Vec<WireMessage<'_>> {
    super::openai::turns(messages, part)
        .into_iter()
        .map(|turn| match turn {
            super::openai::Turn::Message(m) => wire_message(m),
            super::openai::Turn::ToolText(m, text) => WireMessage {
                content: Some(WireContent::Text(text)),
                ..wire_message(m)
            },
            super::openai::Turn::Media(parts) => WireMessage {
                role: Role::User,
                content: Some(WireContent::Parts(parts)),
                tool_calls: None,
                tool_call_id: None,
                name: None,
                reasoning_details: &[],
            },
        })
        .collect()
}

#[derive(Default)]
struct ReasoningAccum {
    blocks: Vec<serde_json::Value>,
}

impl ReasoningAccum {
    fn merge(&mut self, detail: serde_json::Value) {
        let Some(fields) = detail.as_object() else {
            return;
        };
        let key = (detail.get("index").cloned(), detail.get("type").cloned());
        let found = self
            .blocks
            .iter_mut()
            .find(|b| (b.get("index").cloned(), b.get("type").cloned()) == key);
        let Some(block) = found else {
            self.blocks.push(detail.clone());
            return;
        };
        let Some(target) = block.as_object_mut() else {
            return;
        };
        for (name, value) in fields {
            let piecewise = matches!(name.as_str(), "text" | "summary" | "data");
            match target.get_mut(name) {
                Some(serde_json::Value::String(held)) if piecewise => {
                    if let serde_json::Value::String(next) = value {
                        held.push_str(next);
                    }
                }
                _ if !value.is_null() => {
                    target.insert(name.clone(), value.clone());
                }
                _ => {}
            }
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
    #[serde(rename = "max_tokens", skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<&'a ReasoningConfig>,
    stream: bool,
    cache_control: CacheControl,
    #[serde(skip_serializing_if = "str::is_empty")]
    session_id: &'a str,
}

#[derive(Serialize, Clone, Copy)]
struct CacheControl {
    #[serde(rename = "type")]
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    ttl: Option<&'static str>,
}

impl CacheControl {
    const EPHEMERAL: Self = Self {
        kind: "ephemeral",
        ttl: None,
    };

    fn with_ttl(ttl: Option<&str>) -> Self {
        match ttl {
            Some("1h") => Self {
                kind: "ephemeral",
                ttl: Some("1h"),
            },
            _ => Self::EPHEMERAL,
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
struct WireResponseImage {
    image_url: WireImageUrl,
}

#[derive(Debug, Deserialize)]
struct WireImageUrl {
    url: String,
}

#[derive(Debug, Deserialize)]
struct ChoiceMessage {
    content: Option<String>,
    #[serde(default, alias = "reasoning_content")]
    reasoning: Option<String>,
    #[serde(default)]
    reasoning_details: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    tool_calls: Option<Vec<WireToolCall>>,
    #[serde(default)]
    images: Option<Vec<WireResponseImage>>,
}

#[derive(Debug, Deserialize)]
struct WireToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: WireFunctionCall,
}

#[derive(Debug, Deserialize)]
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

fn extract_cost(usage: &Option<serde_json::Value>) -> Option<Decimal> {
    let cost_value = usage.as_ref()?.get("cost")?;
    cost_value
        .as_number()
        .and_then(|n| n.to_string().parse::<Decimal>().ok())
}

impl ChatCompletionResponse {
    fn into_llm_response(self) -> LlmResponse {
        let choice = self.choices.into_iter().next();
        let cost = extract_cost(&self.usage);
        let images = choice
            .as_ref()
            .and_then(|c| c.message.images.as_ref())
            .map(|imgs| {
                imgs.iter()
                    .map(|img| ResponseImage {
                        url: img.image_url.url.clone(),
                    })
                    .collect()
            })
            .unwrap_or_default();
        LlmResponse {
            model: self.model,
            content: choice.as_ref().and_then(|c| c.message.content.clone()),
            reasoning: Reasoning::new(
                ReasoningProvider::Openrouter,
                choice.as_ref().and_then(|c| c.message.reasoning.clone()),
                choice
                    .as_ref()
                    .and_then(|c| c.message.reasoning_details.clone())
                    .unwrap_or_default(),
            ),
            tool_calls: choice
                .as_ref()
                .and_then(|c| c.message.tool_calls.as_ref())
                .map(|tcs| tcs.iter().map(|tc| tc.clone().into()).collect())
                .unwrap_or_default(),
            finish_reason: choice.and_then(|c| c.finish_reason),
            usage: super::openai::usage_from_value(self.usage),
            cost,
            images,
        }
    }
}

impl Clone for WireToolCall {
    fn clone(&self) -> Self {
        WireToolCall {
            id: self.id.clone(),
            call_type: self.call_type.clone(),
            function: WireFunctionCall {
                name: self.function.name.clone(),
                arguments: self.function.arguments.clone(),
            },
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
    reasoning_details: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallDelta>>,
    #[serde(default)]
    images: Option<Vec<WireResponseImage>>,
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

#[derive(Deserialize)]
pub struct OpenRouterConfig {
    pub base_url: String,
    pub api_key: String,
    #[serde(default)]
    pub cache_ttl: Option<String>,
}

pub struct OpenRouterClient {
    http: Client,
    config: OpenRouterConfig,
    cache: CacheControl,
}

impl OpenRouterClient {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self::from_config(OpenRouterConfig {
            base_url: base_url.into(),
            api_key: api_key.into(),
            cache_ttl: None,
        })
    }

    pub fn from_config(config: OpenRouterConfig) -> Self {
        Self {
            http: crate::providers::http_client(),
            cache: CacheControl::with_ttl(config.cache_ttl.as_deref()),
            config,
        }
    }

    const SESSION_ID_MAX_CHARS: usize = 256;

    fn body<'a>(
        &self,
        request: &'a PromptRequest,
        search: DeferToolsStrategy,
        stream: bool,
        session_id: &'a str,
    ) -> WireBody<'a> {
        let session_id = match session_id.char_indices().nth(Self::SESSION_ID_MAX_CHARS) {
            Some((end, _)) => &session_id[..end],
            None => session_id,
        };
        WireBody {
            model: &request.model,
            messages: wire_messages(&request.messages),
            tools: request
                .offered_tools(search)
                .map(|ts| ts.into_iter().map(WireTool::from).collect()),
            temperature: request.temperature,
            max_completion_tokens: request.max_completion_tokens,
            reasoning: request.reasoning.as_ref(),
            stream,
            cache_control: self.cache,
            session_id,
        }
    }

    async fn post_chat_completion(
        &self,
        request: &PromptRequest,
        search: DeferToolsStrategy,
        stream: bool,
        session_id: &str,
    ) -> Result<reqwest::Response, LlmCallError> {
        let wire = self.body(request, search, stream, session_id);

        let url = format!(
            "{}/v1/chat/completions",
            self.config.base_url.trim_end_matches('/')
        );

        self.http
            .post(&url)
            .bearer_auth(&self.config.api_key)
            .header("HTTP-Referer", "https://app.substructure.ai")
            .header("X-OpenRouter-Title", "substructure.ai")
            .header("X-OpenRouter-Categories", "cloud-agent")
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
        Some(reported) => format!("OpenRouter API error {status}: {reported}"),
        None => format!("OpenRouter API error {status}"),
    };
    LlmCallError::new(code, message, retryable)
}

#[async_trait]
impl LlmCallable for OpenRouterClient {
    async fn call(
        &self,
        request: &PromptRequest,
        ctx: &CallContext<'_>,
    ) -> Result<LlmResponse, LlmCallError> {
        let resp = self
            .post_chat_completion(
                request,
                ctx.defer_tools_strategy,
                false,
                ctx.root_session_id(),
            )
            .await?;
        let status = resp.status();
        let body = resp.text().await.map_err(|e| {
            LlmCallError::new(ErrorCode::ProviderError, format!("read body: {e}"), true)
        })?;

        if !status.is_success() {
            tracing::warn!(status = status.as_u16(), body = %crate::json::excerpt(body.as_bytes()), "openrouter api call failed");
            return Err(classify_error(status, &body));
        }

        let parsed: ChatCompletionResponse =
            crate::json::from_str("openrouter response", &body).map_err(LlmCallError::from)?;

        Ok(parsed.into_llm_response())
    }

    async fn call_streaming(
        &self,
        request: &PromptRequest,
        ctx: &CallContext<'_>,
        chunk_tx: UnboundedSender<StreamDelta>,
    ) -> Result<LlmResponse, LlmCallError> {
        let resp = self
            .post_chat_completion(
                request,
                ctx.defer_tools_strategy,
                true,
                ctx.root_session_id(),
            )
            .await?;
        let status = resp.status();

        if !status.is_success() {
            let body = resp.text().await.map_err(|e| {
                LlmCallError::new(ErrorCode::ProviderError, format!("read body: {e}"), true)
            })?;
            return Err(classify_error(status, &body));
        }

        let mut content = String::new();
        let mut thought = String::new();
        let mut reasoning_details = ReasoningAccum::default();
        let mut tool_calls: Vec<ToolCallAccum> = Vec::new();
        let mut images: Vec<ResponseImage> = Vec::new();
        let mut finish_reason: Option<String> = None;
        let mut model = request.model.clone();
        let mut usage: Option<serde_json::Value> = None;

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

                let chunk: StreamChunkResponse = match serde_json::from_str(data) {
                    Ok(c) => c,
                    Err(_) => continue,
                };

                if !chunk.model.is_empty() {
                    model = chunk.model;
                }
                if let Some(u) = chunk.usage {
                    usage = Some(u);
                }

                for choice in chunk.choices {
                    if let Some(ref text) = choice.delta.content {
                        content.push_str(text);
                        let _ = chunk_tx.send(StreamDelta {
                            text: Some(text.clone()),
                            ..Default::default()
                        });
                    }

                    if let Some(details) = choice.delta.reasoning_details {
                        for detail in details {
                            reasoning_details.merge(detail);
                        }
                    }

                    if let Some(ref reasoning) = choice.delta.reasoning {
                        if !reasoning.is_empty() {
                            thought.push_str(reasoning);
                            let _ = chunk_tx.send(StreamDelta {
                                reasoning: Some(reasoning.clone()),
                                ..Default::default()
                            });
                        }
                    }

                    if let Some(tc_deltas) = choice.delta.tool_calls {
                        for tc_delta in tc_deltas {
                            while tool_calls.len() <= tc_delta.index {
                                tool_calls.push(ToolCallAccum::default());
                            }
                            let accum = &mut tool_calls[tc_delta.index];
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
                                let _ = chunk_tx.send(StreamDelta {
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

                    if let Some(imgs) = choice.delta.images {
                        images.extend(imgs.into_iter().map(|img| ResponseImage {
                            url: img.image_url.url,
                        }));
                    }

                    if let Some(reason) = choice.finish_reason {
                        finish_reason = Some(reason.clone());
                        let _ = chunk_tx.send(StreamDelta {
                            finish_reason: Some(reason),
                            ..Default::default()
                        });
                    }
                }
            }
        }

        let cost = extract_cost(&usage);

        Ok(LlmResponse {
            model,
            content: if content.is_empty() {
                None
            } else {
                Some(content)
            },
            tool_calls: tool_calls
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
            finish_reason,
            usage: super::openai::usage_from_value(usage),
            cost,
            images,
            reasoning: Reasoning::new(
                ReasoningProvider::Openrouter,
                (!thought.is_empty()).then_some(thought),
                reasoning_details.blocks,
            ),
        })
    }
}

pub struct OpenRouterProvider {
    client: Arc<OpenRouterClient>,
}

impl OpenRouterProvider {
    pub fn new(config: OpenRouterConfig) -> Self {
        Self {
            client: Arc::new(OpenRouterClient::from_config(config)),
        }
    }
}

#[async_trait]
impl LlmProviderTrait for OpenRouterProvider {
    async fn resolve(&self, _owner: &SessionOwner) -> Result<Arc<dyn LlmCallable>, String> {
        Ok(self.client.clone())
    }
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn a_dropped_connection_is_worth_another_attempt() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                drop(stream);
            }
        });

        let client = OpenRouterClient::new(format!("http://{addr}"), "k");
        let owner = SessionOwner::default();
        let err = client
            .call(
                &req(),
                &CallContext {
                    session_id: "s1",
                    tenant_id: "t1",
                    agent_id: "a1",
                    call_id: "c1",
                    attempt: 0,
                    owner: &owner,
                    ancestry: &[],
                    defer_tools_strategy: Default::default(),
                },
            )
            .await
            .expect_err("the server hung up");

        assert!(
            err.retryable,
            "a request that got no answer must be tried again: {}",
            err.error.message
        );
    }

    use super::*;
    use crate::protocol::{PromptContent, PromptMessage, Role};

    #[test]
    fn audio_formats_cover_what_openrouter_names() {
        for (mime, want) in [
            ("audio/mpeg", Some("mp3")),
            ("audio/mp4", Some("m4a")),
            ("audio/x-m4a", Some("m4a")),
            ("audio/ogg", Some("ogg")),
            ("audio/flac", Some("flac")),
            ("audio/aac", Some("aac")),
            ("audio/wav; rate=44100", Some("wav")),
            ("audio/basic", None),
            ("video/mp4", None),
            ("video/mpeg", None),
        ] {
            assert_eq!(audio_format(mime), want, "{mime}");
        }
        assert!(video_playable("video/mp4"));
        assert!(video_playable("video/quicktime"));
        assert!(!video_playable("video/x-matroska"));
        assert!(
            !video_playable("audio/mpeg"),
            "the type decides, not the subtype"
        );
    }

    #[test]
    fn the_openrouter_policy_sends_named_media_and_files_the_rest() {
        let media = |mime: &str| PromptPart::Media {
            mime: mime.into(),
            name: Some("f".into()),
            bytes: vec![1, 2, 3],
        };
        let kind = |p: &PromptPart| serde_json::to_value(part(p)).unwrap();
        let ogg = kind(&media("audio/ogg"));
        assert_eq!(ogg["type"], "input_audio");
        assert_eq!(ogg["input_audio"]["format"], "ogg");
        assert_eq!(kind(&media("audio/mpeg"))["input_audio"]["format"], "mp3");
        assert_eq!(kind(&media("audio/basic"))["type"], "file");
        let mov = kind(&media("video/quicktime"));
        assert_eq!(mov["type"], "video_url");
        assert!(mov["video_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:video/quicktime;base64,"));
        assert_eq!(kind(&media("video/x-matroska"))["type"], "file");
        assert_eq!(kind(&media("image/png"))["type"], "image_url");
        assert_eq!(kind(&media("text/csv"))["type"], "file");
        let zip = kind(&media("application/zip"));
        assert_eq!(zip["type"], "file");
        assert_eq!(zip["file"]["filename"], "f");
    }

    fn req() -> PromptRequest {
        PromptRequest {
            model: "anthropic/claude-opus-4-8".to_string(),
            messages: vec![PromptMessage {
                role: Role::User,
                content: Some(PromptContent::Text("hi".to_string())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
                reasoning: None,
            }],
            tools: None,
            temperature: None,
            max_completion_tokens: None,
            reasoning: None,
        }
    }

    #[test]
    fn streamed_reasoning_fragments_fold_back_into_one_block() {
        let mut accum = ReasoningAccum::default();
        for fragment in [
            serde_json::json!({"type": "reasoning.text", "index": 0,
                               "format": "anthropic-claude-v1", "text": "Let"}),
            serde_json::json!({"type": "reasoning.text", "index": 0,
                               "format": "anthropic-claude-v1", "text": " me check."}),
            serde_json::json!({"type": "reasoning.text", "index": 0,
                               "format": "anthropic-claude-v1", "signature": "sig-abc"}),
        ] {
            accum.merge(fragment);
        }

        assert_eq!(
            accum.blocks,
            vec![serde_json::json!({
                "type": "reasoning.text",
                "index": 0,
                "format": "anthropic-claude-v1",
                "text": "Let me check.",
                "signature": "sig-abc",
            })]
        );
    }

    #[test]
    fn reasoning_blocks_at_different_indexes_stay_apart() {
        let mut accum = ReasoningAccum::default();
        for fragment in [
            serde_json::json!({"type": "reasoning.text", "index": 0, "text": "first"}),
            serde_json::json!({"type": "reasoning.text", "index": 1, "text": "second"}),
            serde_json::json!({"type": "reasoning.text", "index": 0, "text": "!"}),
        ] {
            accum.merge(fragment);
        }

        assert_eq!(accum.blocks.len(), 2);
        assert_eq!(accum.blocks[0]["text"], "first!");
        assert_eq!(accum.blocks[1]["text"], "second");
    }

    #[test]
    fn the_routers_reasoning_record_goes_back_to_the_router() {
        let detail = serde_json::json!({ "type": "reasoning.text", "text": "thought" });
        let mut request = req();
        request.messages.push(PromptMessage {
            role: Role::Assistant,
            content: Some(PromptContent::Text("hi".to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning: Reasoning::new(
                ReasoningProvider::Openrouter,
                Some("thought".to_string()),
                vec![detail.clone()],
            ),
        });
        let client = OpenRouterClient::from_config(OpenRouterConfig {
            base_url: "https://openrouter.ai/api".to_string(),
            api_key: "k".to_string(),
            cache_ttl: None,
        });
        let body =
            serde_json::to_value(client.body(&request, DeferToolsStrategy::Search, false, "s"))
                .unwrap();

        let sent = &body["messages"][1];
        assert_eq!(sent["reasoning_details"], serde_json::json!([detail]));
        assert!(sent.get("reasoning").is_none(), "internal field leaked");
        assert!(sent.get("id").is_none(), "node id leaked");
    }

    #[test]
    fn thinking_from_another_provider_is_not_offered_to_the_router() {
        let mut request = req();
        request.messages.push(PromptMessage {
            role: Role::Assistant,
            content: Some(PromptContent::Text("hi".to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning: Reasoning::new(
                ReasoningProvider::Anthropic,
                Some("thought".to_string()),
                vec![serde_json::json!({ "type": "thinking", "signature": "s" })],
            ),
        });
        let client = OpenRouterClient::from_config(OpenRouterConfig {
            base_url: "https://openrouter.ai/api".to_string(),
            api_key: "k".to_string(),
            cache_ttl: None,
        });
        let body =
            serde_json::to_value(client.body(&request, DeferToolsStrategy::Search, false, "s"))
                .unwrap();

        assert!(body["messages"][1].get("reasoning_details").is_none());
    }

    fn body(cache_ttl: Option<&str>, session_id: &str) -> serde_json::Value {
        let client = OpenRouterClient::from_config(OpenRouterConfig {
            base_url: "https://openrouter.ai/api".to_string(),
            api_key: "k".to_string(),
            cache_ttl: cache_ttl.map(str::to_string),
        });
        serde_json::to_value(client.body(&req(), DeferToolsStrategy::Search, false, session_id))
            .unwrap()
    }

    #[test]
    fn every_call_asks_for_a_breakpoint_and_names_its_session() {
        let v = body(None, "sess_1");
        assert_eq!(
            v["cache_control"],
            serde_json::json!({ "type": "ephemeral" })
        );
        assert_eq!(v["session_id"], "sess_1");
    }

    #[test]
    fn the_configured_life_rides_the_breakpoint() {
        let v = body(Some("1h"), "sess_1");
        assert_eq!(
            v["cache_control"],
            serde_json::json!({ "type": "ephemeral", "ttl": "1h" })
        );
        assert!(body(Some("5m"), "sess_1")["cache_control"]
            .get("ttl")
            .is_none());
    }

    #[test]
    fn a_deferred_tool_is_not_offered_to_the_router() {
        let mut request = req();
        request.tools = Some(vec![
            LlmTool {
                name: "open".to_string(),
                description: "d".to_string(),
                input: None,
                output: None,
                defer: false,
            },
            LlmTool {
                name: "hidden".to_string(),
                description: "d".to_string(),
                input: None,
                output: None,
                defer: true,
            },
        ]);
        let client = OpenRouterClient::from_config(OpenRouterConfig {
            base_url: "https://openrouter.ai/api".to_string(),
            api_key: "k".to_string(),
            cache_ttl: None,
        });
        let v = serde_json::to_value(client.body(&request, DeferToolsStrategy::Search, false, "s"))
            .unwrap();
        let offered: Vec<&str> = v["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["function"]["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            offered,
            vec!["open"],
            "an agent that defers a tool asked for it to stay out of the request"
        );
    }

    #[test]
    fn a_call_with_no_session_names_none() {
        assert!(body(None, "").get("session_id").is_none());
    }

    #[test]
    fn a_long_session_name_is_cut_to_what_the_router_takes() {
        let long = "sé".repeat(200);
        let sent = body(None, &long)["session_id"]
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(sent.chars().count(), 256);
        assert!(long.starts_with(&sent));
    }
}
