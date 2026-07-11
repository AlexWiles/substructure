use std::sync::Arc;

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::UnboundedSender;
use tokio_stream::StreamExt;

use crate::llm::{CallContext, LlmCallError, LlmCallable, LlmProviderTrait};
use crate::protocol::{
    ErrorCode, LlmRequest, LlmResponse, LlmTool, ReasoningEffort, SessionOwner, StreamDelta,
    ToolCall, ToolCallChunk, ToolCallFunction,
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
    stream: bool,
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
    fn build(request: &'a LlmRequest, stream: bool) -> Self {
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
                .tools
                .as_ref()
                .map(|ts| ts.iter().map(WireTool::from).collect()),
            temperature,
            max_completion_tokens: request.max_completion_tokens,
            reasoning_effort,
            stream,
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

#[derive(Deserialize)]
pub struct OpenAiConfig {
    pub base_url: String,
    pub api_key: String,
    #[serde(default)]
    pub organization: Option<String>,
    #[serde(default)]
    pub project: Option<String>,
}

impl OpenAiConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: api_key.into(),
            organization: None,
            project: None,
        }
    }
}

pub struct OpenAiClient {
    http: Client,
    base_url: String,
    api_key: String,
    extra_headers: HeaderMap,
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
            http: Client::new(),
            base_url: config.base_url,
            api_key: config.api_key,
            extra_headers,
        }
    }

    async fn post_chat_completion(
        &self,
        request: &LlmRequest,
        stream: bool,
    ) -> Result<reqwest::Response, LlmCallError> {
        let wire = WireBody::build(request, stream);
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));

        self.http
            .post(&url)
            .bearer_auth(&self.api_key)
            .headers(self.extra_headers.clone())
            .json(&wire)
            .send()
            .await
            .map_err(|e| LlmCallError {
                message: format!("HTTP request failed: {e}"),
                retryable: e.is_timeout() || e.is_connect(),
                code: Some(ErrorCode::ProviderError),
                detail: None,
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
    LlmCallError {
        message: format!("OpenAI API error {status}: {body}"),
        retryable,
        code: Some(code),
        detail: None,
    }
}

#[async_trait]
impl LlmCallable for OpenAiClient {
    async fn call(
        &self,
        request: &LlmRequest,
        _ctx: &CallContext<'_>,
    ) -> Result<LlmResponse, LlmCallError> {
        let resp = self.post_chat_completion(request, false).await?;
        let status = resp.status();
        let body = resp.text().await.map_err(|e| LlmCallError {
            message: format!("read body: {e}"),
            retryable: true,
            code: Some(ErrorCode::ProviderError),
            detail: None,
        })?;

        if !status.is_success() {
            return Err(classify_error(status, &body));
        }

        let parsed: ChatCompletionResponse =
            serde_json::from_str(&body).map_err(|e| LlmCallError {
                message: format!("parse response: {e}"),
                retryable: false,
                code: Some(ErrorCode::ProviderError),
                detail: None,
            })?;

        Ok(parsed.into_llm_response())
    }

    async fn call_streaming(
        &self,
        request: &LlmRequest,
        _ctx: &CallContext<'_>,
        chunk_tx: UnboundedSender<StreamDelta>,
    ) -> Result<LlmResponse, LlmCallError> {
        let resp = self.post_chat_completion(request, true).await?;
        let status = resp.status();

        if !status.is_success() {
            let body = resp.text().await.map_err(|e| LlmCallError {
                message: format!("read body: {e}"),
                retryable: true,
                code: Some(ErrorCode::ProviderError),
                detail: None,
            })?;
            return Err(classify_error(status, &body));
        }

        let mut content = String::new();
        let mut tool_calls: Vec<ToolCallAccum> = Vec::new();
        let mut finish_reason: Option<String> = None;
        let mut model = request.model.clone();
        let mut usage: Option<serde_json::Value> = None;

        let byte_stream = resp.bytes_stream();
        let mut stream =
            tokio_stream::StreamExt::map(byte_stream, |chunk| chunk.map_err(std::io::Error::other));
        let mut line_buf = String::new();

        while let Some(chunk_result) = stream.next().await {
            let bytes = chunk_result.map_err(|e| LlmCallError {
                message: format!("stream read: {e}"),
                retryable: true,
                code: Some(ErrorCode::ProviderError),
                detail: None,
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
            usage,
            cost: None,
            images: Vec::new(),
        })
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
        let v = serde_json::to_value(WireBody::build(&r, false)).unwrap();
        assert_eq!(v["reasoning_effort"], "high");
        assert!(v.get("temperature").is_none());
        assert_eq!(v["max_completion_tokens"], 500);
        assert!(v.get("max_tokens").is_none());
    }

    #[test]
    fn reasoning_model_without_explicit_effort_still_strips_temperature() {
        let v = serde_json::to_value(WireBody::build(&req("gpt-5.4-mini"), false)).unwrap();
        assert!(v.get("temperature").is_none());
        assert!(v.get("reasoning_effort").is_none());
    }

    #[test]
    fn non_reasoning_model_keeps_temperature() {
        let v = serde_json::to_value(WireBody::build(&req("gpt-4o"), false)).unwrap();
        assert_eq!(v["temperature"], 0.7);
        assert!(v.get("reasoning_effort").is_none());
    }
}
