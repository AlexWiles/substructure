use std::sync::Arc;

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::UnboundedSender;
use tokio_stream::StreamExt;

use crate::runtime::event::{LlmRequest, LlmResponse, Message};
use super::types::{
    ChatCompletionResponse, ChatMessage, Choice, FunctionCall, Role, ToolCall,
};

/// Convert our internal Message to the OpenAI wire format.
fn to_wire_message(msg: &Message) -> ChatMessage {
    ChatMessage {
        role: match msg.role {
            crate::runtime::event::Role::System => Role::System,
            crate::runtime::event::Role::User => Role::User,
            crate::runtime::event::Role::Assistant => Role::Assistant,
            crate::runtime::event::Role::Tool => Role::Tool,
        },
        content: msg.content.clone(),
        tool_calls: if msg.tool_calls.is_empty() {
            None
        } else {
            Some(
                msg.tool_calls
                    .iter()
                    .map(|tc| ToolCall {
                        id: tc.id.clone(),
                        call_type: "function".to_string(),
                        function: FunctionCall {
                            name: tc.name.clone(),
                            arguments: tc.arguments.clone(),
                        },
                    })
                    .collect(),
            )
        },
        tool_call_id: msg.tool_call_id.clone(),
    }
}

use super::{LlmCallError, LlmCallable, StreamDelta};

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct OpenAiClientConfig {
    pub base_url: String,
    pub api_key: String,
}

pub struct OpenAiClient {
    http: Client,
    config: OpenAiClientConfig,
}

impl OpenAiClient {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            http: Client::new(),
            config: OpenAiClientConfig {
                base_url: base_url.into(),
                api_key: api_key.into(),
            },
        }
    }

    pub fn from_config(
        settings: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<Arc<dyn LlmCallable>, String> {
        let config: OpenAiClientConfig =
            serde_json::from_value(serde_json::Value::Object(settings.clone()))
                .map_err(|e| format!("openrouter config: {e}"))?;
        Ok(Arc::new(Self {
            http: Client::new(),
            config,
        }))
    }

    /// Build and send the POST to `/v1/chat/completions`.
    async fn post_chat_completion(
        &self,
        request: &LlmRequest,
        stream: bool,
    ) -> Result<reqwest::Response, LlmCallError> {
        /// Wire-format body with ChatMessage (OpenAI format) instead of our internal Message.
        #[derive(Serialize)]
        struct WireBody {
            model: String,
            messages: Vec<ChatMessage>,
            #[serde(skip_serializing_if = "Option::is_none")]
            tools: Option<Vec<super::types::Tool>>,
            #[serde(skip_serializing_if = "Option::is_none")]
            temperature: Option<f64>,
            #[serde(rename = "max_tokens", skip_serializing_if = "Option::is_none")]
            max_completion_tokens: Option<u64>,
            stream: bool,
        }

        let wire = WireBody {
            model: request.model.clone(),
            messages: request.messages.iter().map(to_wire_message).collect(),
            tools: request.tools.clone(),
            temperature: request.temperature,
            max_completion_tokens: request.max_completion_tokens,
            stream,
        };

        let url = format!(
            "{}/v1/chat/completions",
            self.config.base_url.trim_end_matches('/')
        );
        self.http
            .post(&url)
            .bearer_auth(&self.config.api_key)
            .json(&wire)
            .send()
            .await
            .map_err(|e| LlmCallError {
                message: format!("HTTP request failed: {e}"),
                retryable: true,
                source: Some(serde_json::json!({"kind": "network"})),
            })
    }
}

// ---------------------------------------------------------------------------
// LlmCallable impl
// ---------------------------------------------------------------------------

#[async_trait]
impl LlmCallable for OpenAiClient {
    #[tracing::instrument(skip(self, request), fields(base_url = %self.config.base_url))]
    async fn call(&self, request: &LlmRequest) -> Result<LlmResponse, LlmCallError> {
        let resp = self.post_chat_completion(request, false).await?;
        let status = resp.status();
        let body = resp.text().await.map_err(|e| LlmCallError {
            message: format!("read body: {e}"),
            retryable: true,
            source: Some(serde_json::json!({"kind": "network"})),
        })?;
        if !status.is_success() {
            let status_code = status.as_u16();
            let body_json =
                serde_json::from_str(&body).unwrap_or(serde_json::Value::String(body.clone()));
            let retryable = status.is_server_error() || status_code == 408 || status_code == 429;
            return Err(LlmCallError {
                message: format!("OpenAI API error {status}: {body}"),
                retryable,
                source: Some(serde_json::json!({"kind": "openai", "status": status_code, "body": body_json})),
            });
        }
        let parsed: ChatCompletionResponse = serde_json::from_str(&body).map_err(|e| LlmCallError {
            message: format!("parse response: {e}"),
            retryable: false,
            source: Some(serde_json::json!({"kind": "openai"})),
        })?;
        Ok(LlmResponse::OpenAi(parsed))
    }

    #[tracing::instrument(skip(self, request, chunk_tx), fields(base_url = %self.config.base_url))]
    async fn call_streaming(
        &self,
        request: &LlmRequest,
        chunk_tx: UnboundedSender<StreamDelta>,
    ) -> Result<LlmResponse, LlmCallError> {
        let resp = self.post_chat_completion(request, true).await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.map_err(|e| LlmCallError {
                message: format!("read body: {e}"),
                retryable: true,
                source: Some(serde_json::json!({"kind": "network"})),
            })?;
            let status_code = status.as_u16();
            let body_json =
                serde_json::from_str(&body).unwrap_or(serde_json::Value::String(body.clone()));
            let retryable = status.is_server_error() || status_code == 408 || status_code == 429;
            return Err(LlmCallError {
                message: format!("OpenAI API error {status}: {body}"),
                retryable,
                source: Some(serde_json::json!({"kind": "openai", "status": status_code, "body": body_json})),
            });
        }

        // Accumulator for the final response
        let mut content = String::new();
        let mut tool_calls: Vec<ToolCallAccum> = Vec::new();
        let mut finish_reason: Option<String> = None;
        let mut model = request.model.clone();
        let mut id = String::new();
        let mut usage: Option<serde_json::Value> = None;

        // SSE line-based parser over the byte stream
        let byte_stream = resp.bytes_stream();
        let mut stream =
            tokio_stream::StreamExt::map(byte_stream, |chunk| chunk.map_err(std::io::Error::other));

        let mut line_buf = String::new();

        while let Some(chunk_result) = stream.next().await {
            let bytes = chunk_result.map_err(|e| LlmCallError {
                message: format!("stream read: {e}"),
                retryable: true,
                source: Some(serde_json::json!({"kind": "network"})),
            })?;
            let text = String::from_utf8_lossy(&bytes);
            line_buf.push_str(&text);

            // Process complete lines
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

                if !chunk.id.is_empty() {
                    id = chunk.id;
                }
                if !chunk.model.is_empty() {
                    model = chunk.model;
                }
                if let Some(u) = chunk.usage {
                    usage = Some(u);
                }

                for choice in chunk.choices {
                    let delta = choice.delta;

                    // Text content
                    if let Some(ref text) = delta.content {
                        content.push_str(text);
                        let _ = chunk_tx.send(StreamDelta {
                            text: Some(text.clone()),
                            finish_reason: None,
                        });
                    }

                    // Tool call deltas — accumulate fragments
                    if let Some(tc_deltas) = delta.tool_calls {
                        for tc_delta in tc_deltas {
                            let idx = tc_delta.index;
                            // Grow the vec if needed
                            while tool_calls.len() <= idx {
                                tool_calls.push(ToolCallAccum::default());
                            }
                            let accum = &mut tool_calls[idx];
                            if let Some(id) = tc_delta.id {
                                accum.id = id;
                            }
                            if let Some(f) = tc_delta.function {
                                if let Some(name) = f.name {
                                    accum.name = name;
                                }
                                if let Some(args) = f.arguments {
                                    accum.arguments.push_str(&args);
                                }
                            }
                        }
                    }

                    if let Some(reason) = choice.finish_reason {
                        finish_reason = Some(reason.clone());
                        let _ = chunk_tx.send(StreamDelta {
                            text: None,
                            finish_reason: Some(reason),
                        });
                    }
                }
            }
        }

        // Assemble final response
        let assembled_tool_calls: Vec<ToolCall> = tool_calls
            .into_iter()
            .filter(|tc| !tc.id.is_empty())
            .map(|tc| ToolCall {
                id: tc.id,
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: tc.name,
                    arguments: tc.arguments,
                },
            })
            .collect();

        let message = ChatMessage {
            role: Role::Assistant,
            content: if content.is_empty() {
                None
            } else {
                Some(content)
            },
            tool_calls: if assembled_tool_calls.is_empty() {
                None
            } else {
                Some(assembled_tool_calls)
            },
            tool_call_id: None,
        };

        let response = ChatCompletionResponse {
            id,
            model,
            choices: vec![Choice {
                index: 0,
                message,
                finish_reason,
            }],
            usage,
        };

        Ok(LlmResponse::OpenAi(response))
    }
}

// ---------------------------------------------------------------------------
// Private serde types for SSE streaming chunks
// ---------------------------------------------------------------------------

#[derive(Default)]
struct ToolCallAccum {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct StreamChunkResponse {
    #[serde(default)]
    id: String,
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
