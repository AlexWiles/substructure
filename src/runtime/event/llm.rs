use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::runtime::llm::types as openai;

use super::message::{Message, ToolCall};

// --- Tool definitions (provider-agnostic) ---

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

// --- Request / Response ---

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

// --- LLM call events ---

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
