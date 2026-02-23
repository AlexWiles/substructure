use async_trait::async_trait;
use tokio::sync::mpsc::UnboundedSender;

use crate::domain::event::{LlmRequest, LlmResponse};

#[derive(Debug, Clone)]
pub struct StreamDelta {
    pub text: Option<String>,
    pub finish_reason: Option<String>,
}

#[async_trait]
pub trait LlmClient: Send + Sync + 'static {
    async fn call(&self, request: &LlmRequest) -> Result<LlmResponse, String>;

    /// Streaming variant — sends deltas through `chunk_tx` while the call is
    /// in progress, then returns the final assembled response.
    /// Default implementation ignores the channel and delegates to `call()`.
    async fn call_streaming(
        &self,
        request: &LlmRequest,
        _chunk_tx: UnboundedSender<StreamDelta>,
    ) -> Result<LlmResponse, String> {
        self.call(request).await
    }
}
