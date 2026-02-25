use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::UnboundedSender;

use crate::domain::event::{LlmRequest, LlmResponse};

#[derive(Debug, Clone)]
pub struct StreamDelta {
    pub text: Option<String>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmError {
    pub message: String,
    pub retryable: bool,
    pub source: ErrorSource,
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// Provider-specific error data. Open structure — each provider
/// puts whatever detail it has into `detail` without modifying a shared enum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorSource {
    /// Provider/error kind identifier, e.g. "openai", "anthropic", "network"
    pub kind: String,
    /// Provider-specific error data (status, body, metadata, etc.)
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub detail: serde_json::Value,
}

#[async_trait]
pub trait LlmClient: Send + Sync + 'static {
    async fn call(&self, request: &LlmRequest) -> Result<LlmResponse, LlmError>;

    /// Streaming variant — sends deltas through `chunk_tx` while the call is
    /// in progress, then returns the final assembled response.
    /// Default implementation ignores the channel and delegates to `call()`.
    async fn call_streaming(
        &self,
        request: &LlmRequest,
        _chunk_tx: UnboundedSender<StreamDelta>,
    ) -> Result<LlmResponse, LlmError> {
        self.call(request).await
    }
}
