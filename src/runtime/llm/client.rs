use async_trait::async_trait;

use crate::domain::event::{LlmRequest, LlmResponse};

#[async_trait]
pub trait LlmClient: Send + Sync + 'static {
    async fn call(&self, request: &LlmRequest) -> Result<LlmResponse, String>;
}
