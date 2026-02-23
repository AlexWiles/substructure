use uuid::Uuid;

use serde::{Deserialize, Serialize};

use crate::domain::event::{
    AgentConfig, Event, EventPayload, LlmRequest, Message, Role, SessionAuth,
};
use crate::domain::openai;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenBudget {
    pub used: u64,
    pub limit: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub session_id: Uuid,
    pub agent: Option<AgentConfig>,
    pub auth: Option<SessionAuth>,
    pub messages: Vec<Message>,
    pub tokens: TokenBudget,
    pub stream_version: u64,
    pub last_applied: Option<u64>,
    pub last_reacted: Option<u64>,
}

impl AgentState {
    pub fn new(session_id: Uuid) -> Self {
        AgentState {
            session_id,
            agent: None,
            auth: None,
            messages: Vec::new(),
            tokens: TokenBudget {
                used: 0,
                limit: None,
            },
            stream_version: 0,
            last_applied: None,
            last_reacted: None,
        }
    }

    /// Apply shared state from an event. Returns `true` if applied (not a duplicate).
    pub fn apply_core(&mut self, event: &Event) -> bool {
        if self.last_applied.is_some_and(|seq| event.sequence <= seq) {
            return false;
        }
        self.last_applied = Some(event.sequence);
        self.stream_version += 1;
        match &event.payload {
            EventPayload::SessionCreated(payload) => {
                self.session_id = event.session_id;
                self.agent = Some(payload.agent.clone());
                self.auth = Some(payload.auth.clone());
            }
            EventPayload::MessageUser(payload) => {
                self.track_tokens(&payload.message);
                self.messages.push(payload.message.clone());
            }
            EventPayload::MessageAssistant(payload) => {
                self.track_tokens(&payload.message);
                self.messages.push(payload.message.clone());
            }
            EventPayload::MessageTool(payload) => {
                self.track_tokens(&payload.message);
                self.messages.push(payload.message.clone());
            }
            _ => {}
        }
        true
    }

    pub fn build_llm_request(&self, tools: Option<Vec<openai::Tool>>) -> Option<LlmRequest> {
        let agent = self.agent.as_ref()?;

        let mut messages = vec![openai::ChatMessage {
            role: openai::Role::System,
            content: Some(agent.system_prompt.clone()),
            tool_calls: None,
            tool_call_id: None,
        }];

        for msg in &self.messages {
            messages.push(to_openai_message(msg));
        }

        let model = agent
            .llm
            .params
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let temperature = agent
            .llm
            .params
            .get("temperature")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32);
        let max_tokens = agent
            .llm
            .params
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);

        Some(LlmRequest::OpenAi(openai::ChatCompletionRequest {
            model,
            messages,
            tools,
            tool_choice: None,
            temperature,
            max_tokens,
        }))
    }

    fn track_tokens(&mut self, message: &Message) {
        if let Some(count) = message.token_count {
            self.tokens.used += count as u64;
        }
    }
}

fn to_openai_message(msg: &Message) -> openai::ChatMessage {
    openai::ChatMessage {
        role: match msg.role {
            Role::System => openai::Role::System,
            Role::User => openai::Role::User,
            Role::Assistant => openai::Role::Assistant,
            Role::Tool => openai::Role::Tool,
        },
        content: msg.content.clone(),
        tool_calls: if msg.tool_calls.is_empty() {
            None
        } else {
            Some(
                msg.tool_calls
                    .iter()
                    .map(|tc| openai::ToolCall {
                        id: tc.id.clone(),
                        call_type: "function".to_string(),
                        function: openai::FunctionCall {
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
