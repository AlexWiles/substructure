mod mock;
pub mod openai;
mod provider;

pub use mock::MockLlmClient;
pub use openai::OpenAiClient;
pub use provider::{LlmClientFactory, StaticLlmClientProvider};

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::config::ClientIdentity;
use super::message::{Message, ToolCall};

// ---------------------------------------------------------------------------
// Tool definitions (provider-agnostic)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: LlmToolFunction,
}

// ---------------------------------------------------------------------------
// LLM request / response (provider-agnostic)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<LlmTool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u64>,
}

impl LlmRequest {
    /// Inject tools into the request.
    pub fn with_tools(mut self, tools: Option<Vec<LlmTool>>) -> Self {
        self.tools = tools;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "provider")]
pub enum LlmResponse {
    #[serde(rename = "openai")]
    OpenAi(openai::ChatCompletionResponse),
}

impl LlmResponse {
    pub fn as_parts(&self) -> (Option<String>, Vec<ToolCall>, Option<serde_json::Value>) {
        match self {
            LlmResponse::OpenAi(resp) => {
                let choice = &resp.choices[0];
                let content = choice.message.content.clone();
                let tool_calls = choice
                    .message
                    .tool_calls
                    .as_ref()
                    .map(|tcs| {
                        tcs.iter()
                            .map(|tc| ToolCall {
                                id: tc.id.clone(),
                                name: tc.function.name.clone(),
                                arguments: tc.function.arguments.clone(),
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let usage = resp.usage.clone();
                (content, tool_calls, usage)
            }
        }
    }

    /// Extract raw usage JSON without exposing provider internals.
    pub fn usage(&self) -> Option<&serde_json::Value> {
        match self {
            LlmResponse::OpenAi(r) => r.usage.as_ref(),
        }
    }
}

// ---------------------------------------------------------------------------
// LLM call events
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCallRequested {
    pub call_id: String,
    pub request: LlmRequest,
    pub stream: bool,
    pub deadline: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCallCompleted {
    pub call_id: String,
    pub response: LlmResponse,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCallErrored {
    pub call_id: String,
    pub error: String,
    #[serde(default = "default_true")]
    pub retryable: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// LLM traits
// ---------------------------------------------------------------------------

#[async_trait]
/// Trait for LLM client providers (resolved by the runtime).
pub trait LlmProviderTrait: Send + Sync {
    async fn resolve(&self, client_id: &str, auth: &ClientIdentity) -> Result<Arc<dyn LlmCallable>, String>;
}

#[async_trait]
/// Trait for calling an LLM (single call or streaming).
pub trait LlmCallable: Send + Sync + 'static {
    async fn call(&self, request: &LlmRequest) -> Result<LlmResponse, LlmCallError>;

    /// Streaming variant — sends deltas through `tx` while the call is
    /// in progress, then returns the final assembled response.
    /// Default implementation ignores the channel and delegates to `call()`.
    async fn call_streaming(
        &self,
        request: &LlmRequest,
        _tx: tokio::sync::mpsc::UnboundedSender<StreamDelta>,
    ) -> Result<LlmResponse, LlmCallError> {
        self.call(request).await
    }
}

#[derive(Debug, Clone)]
pub struct LlmCallError {
    pub message: String,
    pub retryable: bool,
    pub source: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct StreamDelta {
    pub text: Option<String>,
    pub finish_reason: Option<String>,
}
