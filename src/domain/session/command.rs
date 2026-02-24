use chrono::{DateTime, Utc};

use crate::domain::event::*;

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
        deadline: DateTime<Utc>,
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
        deadline: DateTime<Utc>,
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
    Interrupt {
        interrupt_id: String,
        reason: String,
        payload: serde_json::Value,
    },
    ResumeInterrupt {
        interrupt_id: String,
        payload: serde_json::Value,
    },
    UpdateStrategyState {
        state: serde_json::Value,
    },
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum SessionError {
    #[error("session has not been created")]
    SessionNotCreated,
    #[error("session already exists")]
    SessionAlreadyCreated,
    #[error("session is interrupted")]
    SessionInterrupted,
    #[error("session is busy")]
    SessionBusy,
}

#[cfg(test)]
mod tests {
    use super::super::agent_session::AgentSession;
    use super::super::strategy::ReactStrategy;
    use super::*;
    use crate::domain::agent::{AgentConfig, LlmConfig};
    use crate::domain::openai;
    use chrono::Utc;
    use uuid::Uuid;

    fn far_future() -> DateTime<Utc> {
        Utc::now() + chrono::Duration::hours(1)
    }

    fn test_agent() -> AgentConfig {
        AgentConfig {
            id: Uuid::new_v4(),
            name: "test".into(),
            llm: LlmConfig {
                client: "mock".into(),
                params: Default::default(),
            },
            system_prompt: "test".into(),
            mcp_servers: vec![],
            strategy: Default::default(),
            retry: Default::default(),
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

    fn created_state() -> AgentSession {
        let mut state = AgentSession::new(
            Uuid::new_v4(),
            Box::new(ReactStrategy::new()),
        );
        let event = Event {
            id: Uuid::new_v4(),
            tenant_id: "t".into(),
            session_id: state.agent_state.session_id,
            sequence: 0,
            span: SpanContext::root(),
            occurred_at: chrono::Utc::now(),
            payload: EventPayload::SessionCreated(SessionCreated {
                agent: test_agent(),
                auth: test_auth(),
            }),
            derived: None,
        };
        state.apply(&event);
        state
    }

    fn apply_events(state: &mut AgentSession, payloads: Vec<EventPayload>) {
        let seq = state.agent_state.last_applied.unwrap_or(0);
        for (i, payload) in payloads.into_iter().enumerate() {
            let event = Event {
                id: Uuid::new_v4(),
                tenant_id: "t".into(),
                session_id: state.agent_state.session_id,
                sequence: seq + 1 + i as u64,
                span: SpanContext::root(),
                occurred_at: chrono::Utc::now(),
                payload,
                derived: None,
            };
            state.apply(&event);
        }
    }

    #[test]
    fn duplicate_send_assistant_message_is_skipped() {
        let mut state = created_state();
        let call_id = "call-1".to_string();

        // Request and complete an LLM call
        let payloads = state
            .handle(CommandPayload::RequestLlmCall {
                call_id: call_id.clone(),
                request: mock_llm_request(),
                stream: false,
                deadline: far_future(),
            })
            .unwrap();
        apply_events(&mut state, payloads);

        let payloads = state
            .handle(CommandPayload::CompleteLlmCall {
                call_id: call_id.clone(),
                response: mock_llm_response(),
            })
            .unwrap();
        apply_events(&mut state, payloads);

        // First SendAssistantMessage succeeds
        let payloads = state
            .handle(CommandPayload::SendAssistantMessage {
                call_id: call_id.clone(),
                content: Some("hello".into()),
                tool_calls: vec![],
                token_count: None,
            })
            .unwrap();
        assert!(!payloads.is_empty());
        apply_events(&mut state, payloads);

        // Duplicate SendAssistantMessage is skipped
        let payloads = state
            .handle(CommandPayload::SendAssistantMessage {
                call_id: call_id.clone(),
                content: Some("hello".into()),
                tool_calls: vec![],
                token_count: None,
            })
            .unwrap();
        assert!(
            payloads.is_empty(),
            "duplicate SendAssistantMessage should produce no events"
        );
    }

    #[test]
    fn duplicate_send_tool_message_is_skipped() {
        let mut state = created_state();
        let call_id = "call-1".to_string();
        let tool_call_id = "tc-1".to_string();

        // Set up: LLM call -> assistant message with tool call -> tool call requested -> tool call completed
        let payloads = state
            .handle(CommandPayload::RequestLlmCall {
                call_id: call_id.clone(),
                request: mock_llm_request(),
                stream: false,
                deadline: far_future(),
            })
            .unwrap();
        apply_events(&mut state, payloads);

        let payloads = state
            .handle(CommandPayload::CompleteLlmCall {
                call_id: call_id.clone(),
                response: mock_llm_response(),
            })
            .unwrap();
        apply_events(&mut state, payloads);

        let payloads = state
            .handle(CommandPayload::SendAssistantMessage {
                call_id: call_id.clone(),
                content: None,
                tool_calls: vec![ToolCall {
                    id: tool_call_id.clone(),
                    name: "test".into(),
                    arguments: "{}".into(),
                }],
                token_count: None,
            })
            .unwrap();
        apply_events(&mut state, payloads);

        let payloads = state
            .handle(CommandPayload::RequestToolCall {
                tool_call_id: tool_call_id.clone(),
                name: "test".into(),
                arguments: "{}".into(),
                deadline: far_future(),
            })
            .unwrap();
        apply_events(&mut state, payloads);

        let payloads = state
            .handle(CommandPayload::CompleteToolCall {
                tool_call_id: tool_call_id.clone(),
                name: "test".into(),
                result: "ok".into(),
            })
            .unwrap();
        apply_events(&mut state, payloads);

        // First SendToolMessage succeeds
        let payloads = state
            .handle(CommandPayload::SendToolMessage {
                tool_call_id: tool_call_id.clone(),
                content: "ok".into(),
                token_count: None,
            })
            .unwrap();
        assert!(!payloads.is_empty());
        apply_events(&mut state, payloads);

        // Duplicate SendToolMessage is skipped
        let payloads = state
            .handle(CommandPayload::SendToolMessage {
                tool_call_id: tool_call_id.clone(),
                content: "ok".into(),
                token_count: None,
            })
            .unwrap();
        assert!(
            payloads.is_empty(),
            "duplicate SendToolMessage should produce no events"
        );
    }

    #[test]
    fn stream_chunk_for_completed_call_is_skipped() {
        let mut state = created_state();
        let call_id = "call-1".to_string();

        // Request and complete an LLM call
        let payloads = state
            .handle(CommandPayload::RequestLlmCall {
                call_id: call_id.clone(),
                request: mock_llm_request(),
                stream: true,
                deadline: far_future(),
            })
            .unwrap();
        apply_events(&mut state, payloads);

        let payloads = state
            .handle(CommandPayload::CompleteLlmCall {
                call_id: call_id.clone(),
                response: mock_llm_response(),
            })
            .unwrap();
        apply_events(&mut state, payloads);

        // StreamLlmChunk for completed call is skipped
        let payloads = state
            .handle(CommandPayload::StreamLlmChunk {
                call_id: call_id.clone(),
                chunk_index: 0,
                text: "late chunk".into(),
            })
            .unwrap();
        assert!(
            payloads.is_empty(),
            "StreamLlmChunk for completed call should produce no events"
        );
    }

    #[test]
    fn interrupt_command_produces_session_interrupted_event() {
        let state = created_state();

        let payloads = state
            .handle(CommandPayload::Interrupt {
                interrupt_id: "int-1".into(),
                reason: "approval_needed".into(),
                payload: serde_json::json!({"tool": "delete_file"}),
            })
            .unwrap();
        assert_eq!(payloads.len(), 1);
        assert!(
            matches!(&payloads[0], EventPayload::SessionInterrupted(p) if p.interrupt_id == "int-1")
        );
    }

    #[test]
    fn resume_interrupt_with_matching_id_produces_event() {
        let mut state = created_state();

        // Interrupt first
        let payloads = state
            .handle(CommandPayload::Interrupt {
                interrupt_id: "int-1".into(),
                reason: "approval_needed".into(),
                payload: serde_json::json!({}),
            })
            .unwrap();
        apply_events(&mut state, payloads);

        // Resume with matching ID
        let payloads = state
            .handle(CommandPayload::ResumeInterrupt {
                interrupt_id: "int-1".into(),
                payload: serde_json::json!({"approved": true}),
            })
            .unwrap();
        assert_eq!(payloads.len(), 1);
        assert!(
            matches!(&payloads[0], EventPayload::InterruptResumed(p) if p.interrupt_id == "int-1")
        );
    }

    #[test]
    fn resume_interrupt_with_wrong_id_is_skipped() {
        let mut state = created_state();

        // Interrupt first
        let payloads = state
            .handle(CommandPayload::Interrupt {
                interrupt_id: "int-1".into(),
                reason: "approval_needed".into(),
                payload: serde_json::json!({}),
            })
            .unwrap();
        apply_events(&mut state, payloads);

        // Resume with wrong ID
        let payloads = state
            .handle(CommandPayload::ResumeInterrupt {
                interrupt_id: "int-WRONG".into(),
                payload: serde_json::json!({}),
            })
            .unwrap();
        assert!(
            payloads.is_empty(),
            "ResumeInterrupt with wrong ID should produce no events"
        );
    }

    #[test]
    fn send_user_message_during_interrupt_returns_error() {
        let mut state = created_state();

        // Interrupt first
        let payloads = state
            .handle(CommandPayload::Interrupt {
                interrupt_id: "int-1".into(),
                reason: "approval_needed".into(),
                payload: serde_json::json!({}),
            })
            .unwrap();
        apply_events(&mut state, payloads);

        // SendUserMessage should fail
        let result = state.handle(CommandPayload::SendUserMessage {
            content: "hello".into(),
            stream: true,
        });
        assert!(matches!(result, Err(SessionError::SessionInterrupted)));
    }
}
