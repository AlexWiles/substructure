mod mock;
pub mod openai;
mod provider;
pub mod types;

pub use mock::MockLlmClient;
pub use openai::OpenAiClient;
pub use provider::{LlmClientFactory, StaticLlmClientProvider};

use std::sync::Arc;

use async_trait::async_trait;

use super::event::{ClientIdentity, LlmRequest, LlmResponse};

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
