use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::UnboundedSender;
use tokio_stream::StreamExt;

use crate::domain::event::{LlmRequest, LlmResponse};
use crate::domain::openai::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Choice, FunctionCall, Role,
    ToolCall, Usage,
};

use std::sync::Arc;

use super::client::{LlmClient, StreamDelta};

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
    ) -> Result<Arc<dyn LlmClient>, String> {
        let config: OpenAiClientConfig =
            serde_json::from_value(serde_json::Value::Object(settings.clone()))
                .map_err(|e| format!("openai_compatible config: {e}"))?;
        Ok(Arc::new(Self { http: Client::new(), config }))
    }

    /// Build and send the POST to `/v1/chat/completions`.
    async fn post_chat_completion(
        &self,
        request: &ChatCompletionRequest,
        stream: bool,
    ) -> Result<reqwest::Response, String> {
        #[derive(Serialize)]
        struct Body<'a> {
            #[serde(flatten)]
            inner: &'a ChatCompletionRequest,
            stream: bool,
        }

        let url = format!("{}/v1/chat/completions", self.config.base_url.trim_end_matches('/'));
        self.http
            .post(&url)
            .bearer_auth(&self.config.api_key)
            .json(&Body {
                inner: request,
                stream,
            })
            .send()
            .await
            .map_err(|e| format!("HTTP request failed: {e}"))
    }
}

// ---------------------------------------------------------------------------
// LlmClient impl
// ---------------------------------------------------------------------------

#[async_trait]
impl LlmClient for OpenAiClient {
    async fn call(&self, request: &LlmRequest) -> Result<LlmResponse, String> {
        let LlmRequest::OpenAi(req) = request;
        let resp = self.post_chat_completion(req, false).await?;
        let status = resp.status();
        let body = resp.text().await.map_err(|e| format!("read body: {e}"))?;
        if !status.is_success() {
            return Err(format!("OpenAI API error {status}: {body}"));
        }
        let parsed: ChatCompletionResponse =
            serde_json::from_str(&body).map_err(|e| format!("parse response: {e}"))?;
        Ok(LlmResponse::OpenAi(parsed))
    }

    async fn call_streaming(
        &self,
        request: &LlmRequest,
        chunk_tx: UnboundedSender<StreamDelta>,
    ) -> Result<LlmResponse, String> {
        let LlmRequest::OpenAi(req) = request;
        let resp = self.post_chat_completion(req, true).await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.map_err(|e| format!("read body: {e}"))?;
            return Err(format!("OpenAI API error {status}: {body}"));
        }

        // Accumulator for the final response
        let mut content = String::new();
        let mut tool_calls: Vec<ToolCallAccum> = Vec::new();
        let mut finish_reason: Option<String> = None;
        let mut model = req.model.clone();
        let mut id = String::new();
        let mut usage: Option<Usage> = None;

        // SSE line-based parser over the byte stream
        let byte_stream = resp.bytes_stream();
        let mut stream = tokio_stream::StreamExt::map(byte_stream, |chunk| {
            chunk.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
        });

        let mut line_buf = String::new();

        while let Some(chunk_result) = stream.next().await {
            let bytes = chunk_result.map_err(|e| format!("stream read: {e}"))?;
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
                            let idx = tc_delta.index as usize;
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
    usage: Option<Usage>,
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
    index: u32,
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
