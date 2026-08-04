use std::sync::Arc;

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::UnboundedSender;
use tokio_stream::StreamExt;

use rust_decimal::Decimal;

use crate::llm::{CallContext, LlmCallError, LlmCallable, LlmProviderTrait};
use crate::protocol::{
    ErrorCode, LlmRequest, LlmResponse, LlmTool, ReasoningConfig, ResponseImage, SessionOwner,
    StreamDelta, ToolCall, ToolCallChunk, ToolCallFunction,
};

/// Wraps our normalized `LlmTool` with the `"type": "function"` field
/// that the OpenAI/OpenRouter API expects.
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
    #[serde(rename = "max_tokens", skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<&'a ReasoningConfig>,
    stream: bool,
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

/// Extract `cost` from OpenRouter's usage object as a Decimal.
/// Parses from the raw JSON number string to avoid f64 precision loss.
fn extract_cost(usage: &Option<serde_json::Value>) -> Option<Decimal> {
    let cost_value = usage.as_ref()?.get("cost")?;
    // serde_json::Value::Number preserves the original string representation
    // when using Number::to_string(), so we parse that directly into Decimal.
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
            tool_calls: choice
                .as_ref()
                .and_then(|c| c.message.tool_calls.as_ref())
                .map(|tcs| tcs.iter().map(|tc| tc.clone().into()).collect())
                .unwrap_or_default(),
            finish_reason: choice.and_then(|c| c.finish_reason),
            usage: self.usage,
            cost,
            images,
        }
    }
}

// Cloning needed for the into_llm_response borrow pattern
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
    #[serde(default)]
    reasoning: Option<String>,
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
}

pub struct OpenRouterClient {
    http: Client,
    config: OpenRouterConfig,
}

impl OpenRouterClient {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            http: crate::providers::http_client(),
            config: OpenRouterConfig {
                base_url: base_url.into(),
                api_key: api_key.into(),
            },
        }
    }

    pub fn from_config(config: OpenRouterConfig) -> Self {
        Self {
            http: crate::providers::http_client(),
            config,
        }
    }

    async fn post_chat_completion(
        &self,
        request: &LlmRequest,
        stream: bool,
    ) -> Result<reqwest::Response, LlmCallError> {
        let wire = WireBody {
            model: &request.model,
            messages: &request.messages,
            tools: request
                .tools
                .as_ref()
                .map(|ts| ts.iter().map(WireTool::from).collect()),
            temperature: request.temperature,
            max_completion_tokens: request.max_completion_tokens,
            reasoning: request.reasoning.as_ref(),
            stream,
        };

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
        message: format!("OpenRouter API error {status}: {body}"),
        retryable,
        code: Some(code),
        detail: None,
    }
}

#[async_trait]
impl LlmCallable for OpenRouterClient {
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
        let mut images: Vec<ResponseImage> = Vec::new();
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

                    if let Some(ref reasoning) = choice.delta.reasoning {
                        if !reasoning.is_empty() {
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
            usage,
            cost,
            images,
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
