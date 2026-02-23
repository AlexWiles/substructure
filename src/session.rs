use std::collections::HashMap;

use uuid::Uuid;

use crate::event::{AgentConfig, Event, EventPayload, LlmRequest, Message, Role, SessionAuth};
use crate::openai;

#[derive(Debug, Clone)]
pub struct TokenBudget {
    pub used: u64,
    pub limit: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LlmCallStatus {
    Pending,
    Completed,
    Failed,
}

#[derive(Debug, Clone)]
pub struct LlmCall {
    pub call_id: String,
    pub request: LlmRequest,
    pub status: LlmCallStatus,
}

#[derive(Debug, Clone)]
pub struct SessionState {
    pub session_id: Uuid,
    pub agent: Option<AgentConfig>,
    pub auth: Option<SessionAuth>,
    pub messages: Vec<Message>,
    pub tokens: TokenBudget,
    pub llm_calls: HashMap<String, LlmCall>,
    pub dirty: bool,
    pub stream_version: u64,
    pub last_applied: Option<u64>,
    pub last_reacted: Option<u64>,
}

impl SessionState {
    pub fn new(session_id: Uuid) -> Self {
        SessionState {
            session_id,
            agent: None,
            auth: None,
            messages: Vec::new(),
            tokens: TokenBudget {
                used: 0,
                limit: None,
            },
            llm_calls: HashMap::new(),
            dirty: false,
            stream_version: 0,
            last_applied: None,
            last_reacted: None,
        }
    }

    pub fn active_llm_call(&self) -> Option<&LlmCall> {
        self.llm_calls
            .values()
            .find(|c| c.status == LlmCallStatus::Pending)
    }

    pub fn apply(&mut self, event: &Event) {
        if self
            .last_applied
            .is_some_and(|seq| event.sequence <= seq)
        {
            return;
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
                if self.active_llm_call().is_some() {
                    self.dirty = true;
                }
            }
            EventPayload::MessageAssistant(payload) => {
                self.track_tokens(&payload.message);
                self.messages.push(payload.message.clone());
            }
            EventPayload::MessageTool(payload) => {
                self.track_tokens(&payload.message);
                self.messages.push(payload.message.clone());
            }
            EventPayload::LlmCallRequested(payload) => {
                self.llm_calls.insert(
                    payload.call_id.clone(),
                    LlmCall {
                        call_id: payload.call_id.clone(),
                        request: payload.request.clone(),
                        status: LlmCallStatus::Pending,
                    },
                );
            }
            EventPayload::LlmCallCompleted(payload) => {
                if let Some(call) = self.llm_calls.get_mut(&payload.call_id) {
                    call.status = LlmCallStatus::Completed;
                }
            }
            EventPayload::LlmCallErrored(payload) => {
                if let Some(call) = self.llm_calls.get_mut(&payload.call_id) {
                    call.status = LlmCallStatus::Failed;
                }
            }
            _ => {}
        }
    }

    pub fn build_llm_request(&self) -> Option<LlmRequest> {
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

        Some(LlmRequest::OpenAi(openai::ChatCompletionRequest {
            model: agent.model.clone(),
            messages,
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
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
