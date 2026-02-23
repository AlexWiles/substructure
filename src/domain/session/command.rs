use chrono::{DateTime, Utc};

use crate::domain::event::*;
use super::state::{LlmCallStatus, SessionState, ToolCallStatus};

#[derive(Debug, Clone)]
pub struct SessionCommand {
    pub span: SpanContext,
    pub occurred_at: DateTime<Utc>,
    pub payload: CommandPayload,
}

#[derive(Debug, Clone)]
pub enum CommandPayload {
    CreateSession {
        agent: AgentConfig,
        auth: SessionAuth,
    },
    SendUserMessage {
        content: String,
        stream: bool,
    },
    SendAssistantMessage {
        call_id: String,
        content: Option<String>,
        tool_calls: Vec<ToolCall>,
        token_count: Option<u32>,
    },
    SendToolMessage {
        tool_call_id: String,
        content: String,
        token_count: Option<u32>,
    },
    RequestLlmCall {
        call_id: String,
        request: LlmRequest,
        stream: bool,
    },
    CompleteLlmCall {
        call_id: String,
        response: LlmResponse,
    },
    FailLlmCall {
        call_id: String,
        error: String,
    },
    StreamLlmChunk {
        call_id: String,
        chunk_index: u32,
        text: String,
    },
    RequestToolCall {
        tool_call_id: String,
        name: String,
        arguments: String,
    },
    CompleteToolCall {
        tool_call_id: String,
        name: String,
        result: String,
    },
    FailToolCall {
        tool_call_id: String,
        name: String,
        error: String,
    },
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum SessionError {
    #[error("session has not been created")]
    SessionNotCreated,
    #[error("session already exists")]
    SessionAlreadyCreated,
}

impl SessionState {
    pub fn handle(&self, cmd: CommandPayload) -> Result<Vec<EventPayload>, SessionError> {
        match (&self.agent, cmd) {
            // Uncreated session: only CreateSession is valid
            (None, CommandPayload::CreateSession { agent, auth }) => {
                Ok(vec![EventPayload::SessionCreated(SessionCreated { agent, auth })])
            }
            (Some(_), CommandPayload::CreateSession { .. }) => {
                Err(SessionError::SessionAlreadyCreated)
            }
            (None, _) => {
                Err(SessionError::SessionNotCreated)
            }

            // Active session
            (Some(_), CommandPayload::SendUserMessage { content, stream }) => {
                Ok(vec![EventPayload::MessageUser(MessageUser {
                    message: Message {
                        role: Role::User,
                        content: Some(content),
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                        call_id: None,
                        token_count: None,
                    },
                    stream,
                })])
            }
            // SendAssistantMessage guard: skip if message already extracted for this call_id
            (Some(_), CommandPayload::SendAssistantMessage { ref call_id, .. })
                if self.llm_calls.get(call_id).is_some_and(|c| c.response_processed) =>
            {
                Ok(vec![])
            }
            (Some(_), CommandPayload::SendAssistantMessage { call_id, content, tool_calls, token_count }) => {
                Ok(vec![EventPayload::MessageAssistant(MessageAssistant {
                    call_id: call_id.clone(),
                    message: Message {
                        role: Role::Assistant,
                        content,
                        tool_calls,
                        tool_call_id: None,
                        call_id: Some(call_id),
                        token_count,
                    },
                })])
            }
            // SendToolMessage guard: skip if a tool message with this tool_call_id already exists
            (Some(_), CommandPayload::SendToolMessage { ref tool_call_id, .. })
                if self.messages.iter().any(|m| m.tool_call_id.as_ref() == Some(tool_call_id)) =>
            {
                Ok(vec![])
            }
            (Some(_), CommandPayload::SendToolMessage { tool_call_id, content, token_count }) => {
                Ok(vec![EventPayload::MessageTool(MessageTool {
                    message: Message {
                        role: Role::Tool,
                        content: Some(content),
                        tool_calls: Vec::new(),
                        tool_call_id: Some(tool_call_id),
                        call_id: None,
                        token_count,
                    },
                })])
            }
            // Command guards — use LLM call lifecycle to decide.
            //
            // RequestLlmCall: skip if call_id already known (duplicate) or
            // another call is already in flight.
            (Some(_), CommandPayload::RequestLlmCall { ref call_id, .. })
                if self.llm_calls.contains_key(call_id)
                    || self.active_llm_call().is_some() =>
            {
                Ok(vec![])
            }
            // CompleteLlmCall: only valid if this call_id is currently pending.
            (Some(_), CommandPayload::CompleteLlmCall { ref call_id, .. })
                if !matches!(
                    self.llm_calls.get(call_id).map(|c| &c.status),
                    Some(&LlmCallStatus::Pending)
                ) =>
            {
                Ok(vec![])
            }
            // FailLlmCall: same — only valid if this call_id is currently pending.
            (Some(_), CommandPayload::FailLlmCall { ref call_id, .. })
                if !matches!(
                    self.llm_calls.get(call_id).map(|c| &c.status),
                    Some(&LlmCallStatus::Pending)
                ) =>
            {
                Ok(vec![])
            }
            (Some(_), CommandPayload::RequestLlmCall { call_id, request, stream }) => {
                Ok(vec![EventPayload::LlmCallRequested(LlmCallRequested { call_id, request, stream })])
            }
            (Some(_), CommandPayload::CompleteLlmCall { call_id, response }) => {
                Ok(vec![EventPayload::LlmCallCompleted(LlmCallCompleted { call_id, response })])
            }
            (Some(_), CommandPayload::FailLlmCall { call_id, error }) => {
                Ok(vec![EventPayload::LlmCallErrored(LlmCallErrored { call_id, error })])
            }
            // StreamLlmChunk guard: skip if this call_id is not Pending
            (Some(_), CommandPayload::StreamLlmChunk { ref call_id, .. })
                if !matches!(
                    self.llm_calls.get(call_id).map(|c| &c.status),
                    Some(&LlmCallStatus::Pending)
                ) =>
            {
                Ok(vec![])
            }
            (Some(_), CommandPayload::StreamLlmChunk { call_id, chunk_index, text }) => {
                Ok(vec![EventPayload::LlmStreamChunk(LlmStreamChunk { call_id, chunk_index, text })])
            }
            // Tool call guards — skip if tool_call_id already known (duplicate).
            (Some(_), CommandPayload::RequestToolCall { ref tool_call_id, .. })
                if self.tool_calls.contains_key(tool_call_id) =>
            {
                Ok(vec![])
            }
            // CompleteToolCall: only valid if this tool_call_id is currently Pending.
            (Some(_), CommandPayload::CompleteToolCall { ref tool_call_id, .. })
                if !matches!(
                    self.tool_calls.get(tool_call_id).map(|tc| &tc.status),
                    Some(&ToolCallStatus::Pending)
                ) =>
            {
                Ok(vec![])
            }
            // FailToolCall: only valid if this tool_call_id is currently Pending.
            (Some(_), CommandPayload::FailToolCall { ref tool_call_id, .. })
                if !matches!(
                    self.tool_calls.get(tool_call_id).map(|tc| &tc.status),
                    Some(&ToolCallStatus::Pending)
                ) =>
            {
                Ok(vec![])
            }
            (Some(_), CommandPayload::RequestToolCall { tool_call_id, name, arguments }) => {
                Ok(vec![EventPayload::ToolCallRequested(ToolCallRequested {
                    tool_call_id, name, arguments,
                })])
            }
            (Some(_), CommandPayload::CompleteToolCall { tool_call_id, name, result }) => {
                Ok(vec![EventPayload::ToolCallCompleted(ToolCallCompleted {
                    tool_call_id, name, result,
                })])
            }
            (Some(_), CommandPayload::FailToolCall { tool_call_id, name, error }) => {
                Ok(vec![EventPayload::ToolCallErrored(ToolCallErrored {
                    tool_call_id, name, error,
                })])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::agent::{AgentConfig, LlmConfig};
    use crate::domain::openai;
    use uuid::Uuid;

    fn test_agent() -> AgentConfig {
        AgentConfig {
            id: Uuid::new_v4(),
            name: "test".into(),
            llm: LlmConfig { client: "mock".into(), params: Default::default() },
            system_prompt: "test".into(),
            mcp_servers: vec![],
        }
    }

    fn test_auth() -> SessionAuth {
        SessionAuth {
            tenant_id: "t".into(),
            client_id: "c".into(),
            sub: None,
        }
    }

    fn mock_llm_request() -> LlmRequest {
        LlmRequest::OpenAi(openai::ChatCompletionRequest {
            model: "mock".into(),
            messages: vec![],
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
        })
    }

    fn mock_llm_response() -> LlmResponse {
        LlmResponse::OpenAi(openai::ChatCompletionResponse {
            id: "resp-1".into(),
            model: "mock".into(),
            choices: vec![openai::Choice {
                index: 0,
                message: openai::ChatMessage {
                    role: openai::Role::Assistant,
                    content: Some("hello".into()),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: Some("stop".into()),
            }],
            usage: None,
        })
    }

    fn created_state() -> SessionState {
        let mut state = SessionState::new(Uuid::new_v4());
        let event = Event {
            id: Uuid::new_v4(),
            tenant_id: "t".into(),
            session_id: state.session_id,
            sequence: 0,
            span: SpanContext::root(),
            occurred_at: chrono::Utc::now(),
            payload: EventPayload::SessionCreated(SessionCreated {
                agent: test_agent(),
                auth: test_auth(),
            }),
        };
        state.apply(&event);
        state
    }

    fn apply_events(state: &mut SessionState, payloads: Vec<EventPayload>) {
        let seq = state.last_applied.unwrap_or(0);
        for (i, payload) in payloads.into_iter().enumerate() {
            let event = Event {
                id: Uuid::new_v4(),
                tenant_id: "t".into(),
                session_id: state.session_id,
                sequence: seq + 1 + i as u64,
                span: SpanContext::root(),
                occurred_at: chrono::Utc::now(),
                payload,
            };
            state.apply(&event);
        }
    }

    #[test]
    fn duplicate_send_assistant_message_is_skipped() {
        let mut state = created_state();
        let call_id = "call-1".to_string();

        // Request and complete an LLM call
        let payloads = state.handle(CommandPayload::RequestLlmCall {
            call_id: call_id.clone(),
            request: mock_llm_request(),
            stream: false,
        }).unwrap();
        apply_events(&mut state, payloads);

        let payloads = state.handle(CommandPayload::CompleteLlmCall {
            call_id: call_id.clone(),
            response: mock_llm_response(),
        }).unwrap();
        apply_events(&mut state, payloads);

        // First SendAssistantMessage succeeds
        let payloads = state.handle(CommandPayload::SendAssistantMessage {
            call_id: call_id.clone(),
            content: Some("hello".into()),
            tool_calls: vec![],
            token_count: None,
        }).unwrap();
        assert!(!payloads.is_empty());
        apply_events(&mut state, payloads);

        // Duplicate SendAssistantMessage is skipped
        let payloads = state.handle(CommandPayload::SendAssistantMessage {
            call_id: call_id.clone(),
            content: Some("hello".into()),
            tool_calls: vec![],
            token_count: None,
        }).unwrap();
        assert!(payloads.is_empty(), "duplicate SendAssistantMessage should produce no events");
    }

    #[test]
    fn duplicate_send_tool_message_is_skipped() {
        let mut state = created_state();
        let call_id = "call-1".to_string();
        let tool_call_id = "tc-1".to_string();

        // Set up: LLM call -> assistant message with tool call -> tool call requested -> tool call completed
        let payloads = state.handle(CommandPayload::RequestLlmCall {
            call_id: call_id.clone(),
            request: mock_llm_request(),
            stream: false,
        }).unwrap();
        apply_events(&mut state, payloads);

        let payloads = state.handle(CommandPayload::CompleteLlmCall {
            call_id: call_id.clone(),
            response: mock_llm_response(),
        }).unwrap();
        apply_events(&mut state, payloads);

        let payloads = state.handle(CommandPayload::SendAssistantMessage {
            call_id: call_id.clone(),
            content: None,
            tool_calls: vec![ToolCall { id: tool_call_id.clone(), name: "test".into(), arguments: "{}".into() }],
            token_count: None,
        }).unwrap();
        apply_events(&mut state, payloads);

        let payloads = state.handle(CommandPayload::RequestToolCall {
            tool_call_id: tool_call_id.clone(),
            name: "test".into(),
            arguments: "{}".into(),
        }).unwrap();
        apply_events(&mut state, payloads);

        let payloads = state.handle(CommandPayload::CompleteToolCall {
            tool_call_id: tool_call_id.clone(),
            name: "test".into(),
            result: "ok".into(),
        }).unwrap();
        apply_events(&mut state, payloads);

        // First SendToolMessage succeeds
        let payloads = state.handle(CommandPayload::SendToolMessage {
            tool_call_id: tool_call_id.clone(),
            content: "ok".into(),
            token_count: None,
        }).unwrap();
        assert!(!payloads.is_empty());
        apply_events(&mut state, payloads);

        // Duplicate SendToolMessage is skipped
        let payloads = state.handle(CommandPayload::SendToolMessage {
            tool_call_id: tool_call_id.clone(),
            content: "ok".into(),
            token_count: None,
        }).unwrap();
        assert!(payloads.is_empty(), "duplicate SendToolMessage should produce no events");
    }

    #[test]
    fn stream_chunk_for_completed_call_is_skipped() {
        let mut state = created_state();
        let call_id = "call-1".to_string();

        // Request and complete an LLM call
        let payloads = state.handle(CommandPayload::RequestLlmCall {
            call_id: call_id.clone(),
            request: mock_llm_request(),
            stream: true,
        }).unwrap();
        apply_events(&mut state, payloads);

        let payloads = state.handle(CommandPayload::CompleteLlmCall {
            call_id: call_id.clone(),
            response: mock_llm_response(),
        }).unwrap();
        apply_events(&mut state, payloads);

        // StreamLlmChunk for completed call is skipped
        let payloads = state.handle(CommandPayload::StreamLlmChunk {
            call_id: call_id.clone(),
            chunk_index: 0,
            text: "late chunk".into(),
        }).unwrap();
        assert!(payloads.is_empty(), "StreamLlmChunk for completed call should produce no events");
    }
}
