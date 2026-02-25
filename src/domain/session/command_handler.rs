use chrono::{DateTime, Utc};

use crate::domain::event::*;

use super::agent_session::{new_call_id, AgentSession};
use super::agent_state::{LlmCallStatus, SessionStatus, ToolCallStatus};
use super::event_handler::extract_assistant_message;

// ---------------------------------------------------------------------------
// Command types
// ---------------------------------------------------------------------------

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
        retryable: bool,
        source: Option<serde_json::Value>,
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
    MarkDone,
    Wake,
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

// ---------------------------------------------------------------------------
// Command handling
// ---------------------------------------------------------------------------

impl AgentSession {
    pub fn handle(&self, cmd: CommandPayload) -> Result<Vec<EventPayload>, SessionError> {
        match (&self.agent_state.agent, cmd) {
            // Uncreated session: only CreateSession is valid
            (None, CommandPayload::CreateSession { agent, auth }) => {
                Ok(vec![EventPayload::SessionCreated(SessionCreated {
                    agent,
                    auth,
                })])
            }
            (Some(_), CommandPayload::CreateSession { .. }) => {
                Err(SessionError::SessionAlreadyCreated)
            }
            (None, _) => Err(SessionError::SessionNotCreated),
            // Active session
            (Some(_), cmd) => self.handle_active(cmd),
        }
    }

    /// Command validation using AgentState for idempotency guards.
    fn handle_active(&self, cmd: CommandPayload) -> Result<Vec<EventPayload>, SessionError> {
        let state = &self.agent_state;
        match cmd {
            CommandPayload::CreateSession { .. } => {
                unreachable!("CreateSession is handled by SessionState::handle")
            }
            CommandPayload::SendUserMessage { content, stream } => {
                if matches!(state.status, SessionStatus::Interrupted { .. }) {
                    return Err(SessionError::SessionInterrupted);
                }
                if matches!(state.status, SessionStatus::Active) {
                    return Err(SessionError::SessionBusy);
                }
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
            CommandPayload::SendAssistantMessage {
                call_id,
                content,
                tool_calls,
                token_count,
            } => {
                if state
                    .llm_calls
                    .get(&call_id)
                    .is_some_and(|c| c.response_processed)
                {
                    return Ok(vec![]);
                }
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
            CommandPayload::SendToolMessage {
                tool_call_id,
                content,
                token_count,
            } => {
                if state
                    .messages
                    .iter()
                    .any(|m| m.tool_call_id.as_ref() == Some(&tool_call_id))
                {
                    return Ok(vec![]);
                }
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
            CommandPayload::RequestLlmCall {
                call_id,
                request,
                stream,
                deadline,
            } => {
                if state.is_over_budget() {
                    return Ok(vec![EventPayload::BudgetExceeded]);
                }
                let existing = state.llm_calls.get(&call_id);
                if existing.is_some_and(|c| {
                    c.status != LlmCallStatus::Failed && c.status != LlmCallStatus::RetryScheduled
                }) || state
                    .llm_calls
                    .values()
                    .any(|c| c.status == LlmCallStatus::Pending)
                {
                    return Ok(vec![]);
                }
                Ok(vec![EventPayload::LlmCallRequested(LlmCallRequested {
                    call_id,
                    request,
                    stream,
                    deadline,
                })])
            }
            CommandPayload::CompleteLlmCall { call_id, response } => {
                if !matches!(
                    state.llm_calls.get(&call_id).map(|c| &c.status),
                    Some(&LlmCallStatus::Pending)
                ) {
                    return Ok(vec![]);
                }
                Ok(vec![EventPayload::LlmCallCompleted(LlmCallCompleted {
                    call_id,
                    response,
                })])
            }
            CommandPayload::FailLlmCall {
                call_id,
                error,
                retryable,
                source,
            } => {
                if !matches!(
                    state.llm_calls.get(&call_id).map(|c| &c.status),
                    Some(&LlmCallStatus::Pending)
                ) {
                    return Ok(vec![]);
                }
                Ok(vec![EventPayload::LlmCallErrored(LlmCallErrored {
                    call_id,
                    error,
                    retryable,
                    source,
                })])
            }
            CommandPayload::StreamLlmChunk {
                call_id,
                chunk_index,
                text,
            } => {
                if !matches!(
                    state.llm_calls.get(&call_id).map(|c| &c.status),
                    Some(&LlmCallStatus::Pending)
                ) {
                    return Ok(vec![]);
                }
                Ok(vec![EventPayload::LlmStreamChunk(LlmStreamChunk {
                    call_id,
                    chunk_index,
                    text,
                })])
            }
            CommandPayload::RequestToolCall {
                tool_call_id,
                name,
                arguments,
                deadline,
            } => {
                if state.tool_calls.contains_key(&tool_call_id) {
                    return Ok(vec![]);
                }
                Ok(vec![EventPayload::ToolCallRequested(ToolCallRequested {
                    tool_call_id,
                    name,
                    arguments,
                    deadline,
                })])
            }
            CommandPayload::CompleteToolCall {
                tool_call_id,
                name,
                result,
            } => {
                if !matches!(
                    state.tool_calls.get(&tool_call_id).map(|tc| &tc.status),
                    Some(&ToolCallStatus::Pending)
                ) {
                    return Ok(vec![]);
                }
                Ok(vec![EventPayload::ToolCallCompleted(ToolCallCompleted {
                    tool_call_id,
                    name,
                    result,
                })])
            }
            CommandPayload::FailToolCall {
                tool_call_id,
                name,
                error,
            } => {
                if !matches!(
                    state.tool_calls.get(&tool_call_id).map(|tc| &tc.status),
                    Some(&ToolCallStatus::Pending)
                ) {
                    return Ok(vec![]);
                }
                Ok(vec![EventPayload::ToolCallErrored(ToolCallErrored {
                    tool_call_id,
                    name,
                    error,
                })])
            }
            CommandPayload::Interrupt {
                interrupt_id,
                reason,
                payload,
            } => {
                if matches!(state.status, SessionStatus::Interrupted { .. }) {
                    return Ok(vec![]);
                }
                Ok(vec![EventPayload::SessionInterrupted(SessionInterrupted {
                    interrupt_id,
                    reason,
                    payload,
                })])
            }
            CommandPayload::ResumeInterrupt {
                interrupt_id,
                payload,
            } => {
                if state.active_interrupt() != Some(interrupt_id.as_str()) {
                    return Ok(vec![]);
                }
                Ok(vec![EventPayload::InterruptResumed(InterruptResumed {
                    interrupt_id,
                    payload,
                })])
            }
            CommandPayload::UpdateStrategyState { state } => {
                Ok(vec![EventPayload::StrategyStateChanged(
                    StrategyStateChanged { state },
                )])
            }
            CommandPayload::MarkDone => Ok(vec![EventPayload::SessionDone]),
            CommandPayload::Wake => self.handle_wake(),
        }
    }

    // -----------------------------------------------------------------------
    // Wake — inspects state and emits events for timeouts, retries, recovery
    // -----------------------------------------------------------------------

    fn handle_wake(&self) -> Result<Vec<EventPayload>, SessionError> {
        let now = Utc::now();
        let state = &self.agent_state;

        // 1. Timed-out pending LLM calls → fail
        for call in state.llm_calls.values() {
            if call.status == LlmCallStatus::Pending && call.deadline <= now {
                return Ok(vec![EventPayload::LlmCallErrored(LlmCallErrored {
                    call_id: call.call_id.clone(),
                    error: "deadline exceeded".to_string(),
                    retryable: true,
                    source: None,
                })]);
            }
        }

        // 2. Timed-out pending tool calls → fail
        for tc in state.tool_calls.values() {
            if tc.status == ToolCallStatus::Pending && tc.deadline <= now {
                return Ok(vec![EventPayload::ToolCallErrored(ToolCallErrored {
                    tool_call_id: tc.tool_call_id.clone(),
                    name: tc.name.clone(),
                    error: "deadline exceeded".to_string(),
                })]);
            }
        }

        // 3. RetryScheduled LLM call with next_at passed → re-issue
        for call in state.llm_calls.values() {
            if call.status == LlmCallStatus::RetryScheduled {
                if let Some(next_at) = call.retry.next_at {
                    if next_at <= now {
                        if state.is_over_budget() {
                            return Ok(vec![EventPayload::BudgetExceeded]);
                        }
                        if let Some(request) = state.build_llm_request(None) {
                            return Ok(vec![EventPayload::LlmCallRequested(LlmCallRequested {
                                call_id: call.call_id.clone(),
                                request,
                                stream: true,
                                deadline: self.llm_deadline(),
                            })]);
                        }
                    }
                }
            }
        }

        // 4. Completed LLM call, response not processed → emit message + tool calls
        for call in state.llm_calls.values() {
            if call.status == LlmCallStatus::Completed && !call.response_processed {
                if let Some(ref response) = call.response {
                    let (content, tool_calls, token_count) = extract_assistant_message(response);
                    let mut events = vec![EventPayload::MessageAssistant(MessageAssistant {
                        call_id: call.call_id.clone(),
                        message: Message {
                            role: Role::Assistant,
                            content,
                            tool_calls: tool_calls.clone(),
                            tool_call_id: None,
                            call_id: Some(call.call_id.clone()),
                            token_count,
                        },
                    })];
                    for tc in &tool_calls {
                        events.push(EventPayload::ToolCallRequested(ToolCallRequested {
                            tool_call_id: tc.id.clone(),
                            name: tc.name.clone(),
                            arguments: tc.arguments.clone(),
                            deadline: self.tool_deadline(),
                        }));
                    }
                    return Ok(events);
                }
            }
        }

        // 5. Pending tool calls still in flight → re-emit (crash recovery)
        for tc in state.tool_calls.values() {
            if tc.status == ToolCallStatus::Pending && tc.deadline > now {
                return Ok(vec![EventPayload::ToolCallRequested(ToolCallRequested {
                    tool_call_id: tc.tool_call_id.clone(),
                    name: tc.name.clone(),
                    arguments: state
                        .messages
                        .iter()
                        .flat_map(|m| m.tool_calls.iter())
                        .find(|t| t.id == tc.tool_call_id)
                        .map(|t| t.arguments.clone())
                        .unwrap_or_default(),
                    deadline: tc.deadline,
                })]);
            }
        }

        // 6. Pending LLM calls still in flight → re-emit (crash recovery)
        for call in state.llm_calls.values() {
            if call.status == LlmCallStatus::Pending && call.deadline > now {
                if state.is_over_budget() {
                    return Ok(vec![EventPayload::BudgetExceeded]);
                }
                return Ok(vec![EventPayload::LlmCallRequested(LlmCallRequested {
                    call_id: call.call_id.clone(),
                    request: call.request.clone(),
                    stream: true,
                    deadline: call.deadline,
                })]);
            }
        }

        // 7. All tools done, no next step → request next LLM call
        let last_is_tool = state.messages.last().is_some_and(|m| m.role == Role::Tool);
        if state.pending_tool_results() == 0
            && last_is_tool
            && !state
                .llm_calls
                .values()
                .any(|c| c.status == LlmCallStatus::Pending)
        {
            if state.is_over_budget() {
                return Ok(vec![EventPayload::BudgetExceeded]);
            }
            if let Some(request) = state.build_llm_request(None) {
                return Ok(vec![EventPayload::LlmCallRequested(LlmCallRequested {
                    call_id: new_call_id(),
                    request,
                    stream: true,
                    deadline: self.llm_deadline(),
                })]);
            }
        }

        Ok(vec![])
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::super::agent_session::AgentSession;
    use super::super::strategy::DefaultStrategy;
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
            token_budget: None,
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
        let mut state = AgentSession::new(Uuid::new_v4(), Arc::new(DefaultStrategy::default()));
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
    fn request_llm_call_over_budget_emits_budget_exceeded() {
        let mut agent = test_agent();
        agent.token_budget = Some(100);

        let mut state = AgentSession::new(Uuid::new_v4(), Arc::new(DefaultStrategy::default()));
        let event = Event {
            id: Uuid::new_v4(),
            tenant_id: "t".into(),
            session_id: state.agent_state.session_id,
            sequence: 0,
            span: SpanContext::root(),
            occurred_at: chrono::Utc::now(),
            payload: EventPayload::SessionCreated(SessionCreated {
                agent,
                auth: test_auth(),
            }),
            derived: None,
        };
        state.apply(&event);

        // Simulate token usage exceeding budget
        state.agent_state.token_usage.total_tokens = 200;

        let payloads = state
            .handle(CommandPayload::RequestLlmCall {
                call_id: "call-1".into(),
                request: mock_llm_request(),
                stream: false,
                deadline: far_future(),
            })
            .unwrap();

        assert_eq!(payloads.len(), 1);
        assert!(
            matches!(&payloads[0], EventPayload::BudgetExceeded),
            "expected BudgetExceeded, got {:?}",
            payloads[0],
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
