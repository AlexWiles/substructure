use async_trait::async_trait;

use crate::domain::event::{LlmRequest, LlmResponse};
use crate::domain::openai;

use std::sync::Arc;

use super::client::LlmClient;

/// A mock LLM client that returns a static text response.
pub struct MockLlmClient {
    call_count: std::sync::atomic::AtomicU32,
}

impl Default for MockLlmClient {
    fn default() -> Self {
        Self {
            call_count: std::sync::atomic::AtomicU32::new(0),
        }
    }
}

impl MockLlmClient {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_config(
        _settings: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<Arc<dyn LlmClient>, String> {
        Ok(Arc::new(Self::new()))
    }
}

#[async_trait]
impl LlmClient for MockLlmClient {
    async fn call(&self, request: &LlmRequest) -> Result<LlmResponse, String> {
        let n = self
            .call_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let model = match request {
            LlmRequest::OpenAi(req) => req.model.clone(),
        };
        Ok(LlmResponse::OpenAi(openai::ChatCompletionResponse {
            id: format!("mock-{}", n + 1),
            model,
            choices: vec![openai::Choice {
                index: 0,
                message: openai::ChatMessage {
                    role: openai::Role::Assistant,
                    content: Some(format!("[mock response #{}]", n + 1)),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: Some("stop".into()),
            }],
            usage: Some(openai::Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            }),
        }))
    }
}
