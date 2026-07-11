use std::sync::Arc;

use async_trait::async_trait;

use crate::protocol::{ErrorCode, LlmRequest, LlmResponse, LlmTool, SessionOwner, StreamDelta};

impl LlmTool {
    /// The schema providers receive: the declared `input`, or the empty
    /// object schema for a no-argument tool.
    pub fn input_schema(&self) -> serde_json::Value {
        self.input
            .clone()
            .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}))
    }
}

/// Trait for LLM client providers (resolved by the runtime).
#[async_trait]
pub trait LlmProviderTrait: Send + Sync {
    async fn resolve(&self, owner: &SessionOwner) -> Result<Arc<dyn LlmCallable>, String>;
}

#[derive(Debug, Clone)]
pub struct CallContext<'a> {
    pub session_id: &'a str,
    pub tenant_id: &'a str,
    pub agent_id: &'a str,
    pub call_id: &'a str,
    pub attempt: u32,
    pub owner: &'a SessionOwner,
    /// Parent chain, root-last. Empty for top-level sessions.
    pub ancestry: &'a [String],
}

/// Trait for calling an LLM (single call or streaming).
#[async_trait]
pub trait LlmCallable: Send + Sync + 'static {
    async fn call(
        &self,
        request: &LlmRequest,
        ctx: &CallContext<'_>,
    ) -> Result<LlmResponse, LlmCallError>;

    /// Streaming variant — sends deltas through `tx` while the call is
    /// in progress, then returns the final assembled response.
    /// Default implementation ignores the channel and delegates to `call()`.
    async fn call_streaming(
        &self,
        request: &LlmRequest,
        ctx: &CallContext<'_>,
        _tx: tokio::sync::mpsc::UnboundedSender<StreamDelta>,
    ) -> Result<LlmResponse, LlmCallError> {
        self.call(request, ctx).await
    }
}

#[derive(Debug, Clone)]
pub struct LlmCallError {
    pub message: String,
    pub retryable: bool,
    pub code: Option<ErrorCode>,
    pub detail: Option<serde_json::Value>,
}
