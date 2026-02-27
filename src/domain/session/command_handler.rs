use chrono::{DateTime, Utc};

use crate::domain::event::*;
use crate::domain::session::agent_state::ToolCallState;

use super::agent_session::{new_call_id, AgentSession};
use super::agent_state::{LlmCallStatus, SessionStatus, ToolCallStatus};
use super::event_handler::extract_assistant_message;

// ---------------------------------------------------------------------------
// Command types
// ---------------------------------------------------------------------------

/// A message from an external client (AG-UI, HTTP, etc.).
#[derive(Debug, Clone)]
pub enum IncomingMessage {
    User {
        content: String,
    },
    ToolResult {
        tool_call_id: String,
        content: String,
        error: Option<String>,
    },
}

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
        on_done: Option<CompletionDelivery>,
    },
    SendMessage {
        message: IncomingMessage,
        stream: bool,
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
        #[allow(dead_code)]
        handler: ToolHandler,
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
    SyncConversation {
        messages: Vec<Message>,
        stream: bool,
    },
    CancelSession,
    MarkDone {
        artifacts: Vec<Artifact>,
    },
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
    #[error("conversation has diverged")]
    ConversationDiverged,
}

// ---------------------------------------------------------------------------
// Command handling
// ---------------------------------------------------------------------------

impl AgentSession {
    pub fn handle(&self, cmd: CommandPayload) -> Result<Vec<EventPayload>, SessionError> {
        match (&self.agent_state.agent, cmd) {
            (
                None,
                CommandPayload::CreateSession {
                    agent,
                    auth,
                    on_done,
                },
            ) => Ok(vec![EventPayload::SessionCreated(SessionCreated {
                agent,
                auth,
                on_done,
            })]),
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
            CommandPayload::SendMessage { message, stream } => match message {
                IncomingMessage::User { content } => match state.status {
                    SessionStatus::Interrupted { .. } => Err(SessionError::SessionInterrupted),
                    SessionStatus::Active => Err(SessionError::SessionBusy),
                    _ => Ok(vec![EventPayload::MessageUser(MessageUser {
                        message: Message {
                            role: Role::User,
                            content: Some(content),
                            tool_calls: Vec::new(),
                            tool_call_id: None,
                            call_id: None,
                            token_count: None,
                        },
                        stream,
                    })]),
                },
                IncomingMessage::ToolResult {
                    tool_call_id,
                    content,
                    error,
                } => {
                    let tc = state.tool_calls.get(&tool_call_id);

                    match tc {
                        // Case where we havent seen the tool call before. We ignore it.
                        None => Ok(vec![]),
                        // Have seen a terminal state for this toolcall already
                        Some(ToolCallState {
                            status: ToolCallStatus::Completed | ToolCallStatus::Failed,
                            ..
                        }) => Ok(vec![]),
                        // We are expecting a result
                        Some(ToolCallState {
                            status: ToolCallStatus::Pending,
                            name,
                            ..
                        }) => {
                            if let Some(err) = error {
                                Ok(vec![EventPayload::ToolCallErrored(ToolCallErrored {
                                    tool_call_id,
                                    name: name.clone(),
                                    error: err,
                                })])
                            } else {
                                Ok(vec![EventPayload::ToolCallCompleted(ToolCallCompleted {
                                    tool_call_id,
                                    name: name.clone(),
                                    result: content,
                                })])
                            }
                        }
                    }
                }
            },
            CommandPayload::RequestLlmCall {
                call_id,
                request,
                stream,
                deadline,
            } => {
                if state.is_over_budget() {
                    return Ok(vec![EventPayload::BudgetExceeded]);
                }
                let has_pending = state
                    .llm_calls
                    .values()
                    .any(|c| c.status == LlmCallStatus::Pending);

                if has_pending {
                    return Ok(vec![]);
                }

                let issue = match state.llm_calls.get(&call_id).map(|c| &c.status) {
                    // New call
                    None => true,
                    // Previously failed — retry
                    Some(&LlmCallStatus::Failed) => true,
                    // Retry scheduled but not yet fired — short-circuit the backoff
                    Some(&LlmCallStatus::RetryScheduled) => true,
                    // Already in flight or completed — skip
                    _ => false,
                };
                if issue {
                    Ok(vec![EventPayload::LlmCallRequested(LlmCallRequested {
                        call_id,
                        request,
                        stream,
                        deadline,
                    })])
                } else {
                    Ok(vec![])
                }
            }
            CommandPayload::CompleteLlmCall { call_id, response } => {
                match state.llm_calls.get(&call_id).map(|c| &c.status) {
                    // Pending call — complete it
                    Some(&LlmCallStatus::Pending) => {
                        let (content, tool_calls, token_count) =
                            extract_assistant_message(&response);

                        let mut events = vec![
                            EventPayload::LlmCallCompleted(LlmCallCompleted {
                                call_id: call_id.clone(),
                                response,
                            }),
                            EventPayload::MessageAssistant(MessageAssistant {
                                call_id: call_id.clone(),
                                message: Message {
                                    role: Role::Assistant,
                                    content,
                                    tool_calls: tool_calls.clone(),
                                    tool_call_id: None,
                                    call_id: Some(call_id),
                                    token_count,
                                },
                            }),
                        ];
                        for tc in &tool_calls {
                            events.push(EventPayload::ToolCallRequested(ToolCallRequested {
                                tool_call_id: tc.id.clone(),
                                name: tc.name.clone(),
                                arguments: tc.arguments.clone(),
                                deadline: self.tool_deadline(),
                                handler: Default::default(),
                            }));
                        }
                        Ok(events)
                    }
                    // Not pending or unknown — skip
                    _ => Ok(vec![]),
                }
            }
            CommandPayload::FailLlmCall {
                call_id,
                error,
                retryable,
                source,
            } => match state.llm_calls.get(&call_id).map(|c| &c.status) {
                // Pending call — fail it
                Some(&LlmCallStatus::Pending) => {
                    Ok(vec![EventPayload::LlmCallErrored(LlmCallErrored {
                        call_id,
                        error,
                        retryable,
                        source,
                    })])
                }
                // Not pending or unknown — skip
                _ => Ok(vec![]),
            },
            CommandPayload::StreamLlmChunk {
                call_id,
                chunk_index,
                text,
            } => match state.llm_calls.get(&call_id).map(|c| &c.status) {
                // Pending call — forward chunk
                Some(&LlmCallStatus::Pending) => {
                    Ok(vec![EventPayload::LlmStreamChunk(LlmStreamChunk {
                        call_id,
                        chunk_index,
                        text,
                    })])
                }
                // Not pending or unknown — skip
                _ => Ok(vec![]),
            },
            CommandPayload::RequestToolCall {
                tool_call_id,
                name,
                arguments,
                deadline,
                handler,
            } => match state.tool_calls.get(&tool_call_id) {
                // Already tracked — skip
                Some(_) => Ok(vec![]),
                // New tool call
                None => Ok(vec![EventPayload::ToolCallRequested(ToolCallRequested {
                    tool_call_id,
                    name,
                    arguments,
                    deadline,
                    handler,
                })]),
            },
            CommandPayload::CompleteToolCall {
                tool_call_id,
                name,
                result,
            } => match state.tool_calls.get(&tool_call_id).map(|tc| &tc.status) {
                // Pending — complete and emit tool message
                Some(&ToolCallStatus::Pending) => Ok(vec![
                    EventPayload::ToolCallCompleted(ToolCallCompleted {
                        tool_call_id: tool_call_id.clone(),
                        name,
                        result: result.clone(),
                    }),
                    EventPayload::MessageTool(MessageTool {
                        message: Message {
                            role: Role::Tool,
                            content: Some(result),
                            tool_calls: Vec::new(),
                            tool_call_id: Some(tool_call_id),
                            call_id: None,
                            token_count: None,
                        },
                    }),
                ]),
                // Not pending or unknown — skip
                _ => Ok(vec![]),
            },
            CommandPayload::FailToolCall {
                tool_call_id,
                name,
                error,
            } => match state.tool_calls.get(&tool_call_id).map(|tc| &tc.status) {
                // Pending — fail and emit error tool message
                Some(&ToolCallStatus::Pending) => {
                    let error_content = format!("Error: {}", error);
                    Ok(vec![
                        EventPayload::ToolCallErrored(ToolCallErrored {
                            tool_call_id: tool_call_id.clone(),
                            name,
                            error,
                        }),
                        EventPayload::MessageTool(MessageTool {
                            message: Message {
                                role: Role::Tool,
                                content: Some(error_content),
                                tool_calls: Vec::new(),
                                tool_call_id: Some(tool_call_id),
                                call_id: None,
                                token_count: None,
                            },
                        }),
                    ])
                }
                // Not pending or unknown — skip
                _ => Ok(vec![]),
            },
            CommandPayload::Interrupt {
                interrupt_id,
                reason,
                payload,
            } => match state.status {
                // Already interrupted — skip
                SessionStatus::Interrupted { .. } => Ok(vec![]),
                _ => Ok(vec![EventPayload::SessionInterrupted(SessionInterrupted {
                    interrupt_id,
                    reason,
                    payload,
                })]),
            },
            CommandPayload::ResumeInterrupt {
                interrupt_id,
                payload,
            } => match state.active_interrupt() {
                // Matching active interrupt — resume
                Some(id) if id == interrupt_id => {
                    Ok(vec![EventPayload::InterruptResumed(InterruptResumed {
                        interrupt_id,
                        payload,
                    })])
                }
                // No active interrupt or wrong ID — skip
                _ => Ok(vec![]),
            },
            CommandPayload::UpdateStrategyState { state } => {
                Ok(vec![EventPayload::StrategyStateChanged(
                    StrategyStateChanged { state },
                )])
            }
            CommandPayload::SyncConversation { messages, stream } => {
                self.handle_sync_conversation(messages, stream)
            }
            CommandPayload::CancelSession => Ok(vec![EventPayload::SessionCancelled]),
            CommandPayload::MarkDone { artifacts } => {
                Ok(vec![EventPayload::SessionDone(SessionDone { artifacts })])
            }
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

        // 2. Timed-out pending tool calls → fail (or re-issue for sub-agents)
        for tc in state.tool_calls.values() {
            if tc.status == ToolCallStatus::Pending
                && tc.deadline <= now
                && tc.handler != ToolHandler::Client
            {
                if tc.child_session_id.is_some() {
                    // Sub-agent: re-arm with fresh deadline (parent-driven retry)
                    let arguments = state
                        .messages
                        .iter()
                        .flat_map(|m| m.tool_calls.iter())
                        .find(|t| t.id == tc.tool_call_id)
                        .map(|t| t.arguments.clone())
                        .unwrap_or_default();
                    return Ok(vec![EventPayload::ToolCallRequested(ToolCallRequested {
                        tool_call_id: tc.tool_call_id.clone(),
                        name: tc.name.clone(),
                        arguments,
                        deadline: self.tool_deadline(),
                        handler: tc.handler.clone(),
                    })]);
                }
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

        // 4. Pending tool calls still in flight → re-emit (crash recovery)
        for tc in state.tool_calls.values() {
            if tc.status == ToolCallStatus::Pending
                && tc.deadline > now
                && tc.handler != ToolHandler::Client
            {
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
                    handler: tc.handler.clone(),
                })]);
            }
        }

        // 5. Pending LLM calls still in flight → re-emit (crash recovery)
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

        // 6. All tools done, no next step → request next LLM call
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

    // -----------------------------------------------------------------------
    // SyncConversation — diff incoming history and emit events for the delta
    // -----------------------------------------------------------------------

    fn handle_sync_conversation(
        &self,
        incoming: Vec<Message>,
        stream: bool,
    ) -> Result<Vec<EventPayload>, SessionError> {
        let state = &self.agent_state;

        // Verify prefix matches current state
        let prefix_len = state.messages.len();
        if incoming.len() < prefix_len {
            return Err(SessionError::ConversationDiverged);
        }
        for (existing, incoming_msg) in state.messages.iter().zip(incoming.iter()) {
            if !messages_match(existing, incoming_msg) {
                return Err(SessionError::ConversationDiverged);
            }
        }

        // Delta = incoming messages beyond what we already have
        let delta = &incoming[prefix_len..];
        if delta.is_empty() {
            return Ok(vec![]);
        }

        let mut events = Vec::new();
        for msg in delta {
            match msg.role {
                Role::User => {
                    events.push(EventPayload::MessageUser(MessageUser {
                        message: msg.clone(),
                        stream,
                    }));
                }
                Role::Assistant => {
                    let call_id = new_call_id();
                    events.push(EventPayload::MessageAssistant(MessageAssistant {
                        call_id: call_id.clone(),
                        message: msg.clone(),
                    }));
                    for tc in &msg.tool_calls {
                        events.push(EventPayload::ToolCallRequested(ToolCallRequested {
                            tool_call_id: tc.id.clone(),
                            name: tc.name.clone(),
                            arguments: tc.arguments.clone(),
                            deadline: self.tool_deadline(),
                            handler: Default::default(),
                        }));
                    }
                }
                Role::Tool => {
                    if let Some(ref tc_id) = msg.tool_call_id {
                        let name = state
                            .tool_calls
                            .get(tc_id)
                            .map(|tc| tc.name.clone())
                            .or_else(|| {
                                delta
                                    .iter()
                                    .flat_map(|m| m.tool_calls.iter())
                                    .find(|t| t.id == *tc_id)
                                    .map(|t| t.name.clone())
                            })
                            .unwrap_or_default();

                        events.push(EventPayload::ToolCallCompleted(ToolCallCompleted {
                            tool_call_id: tc_id.clone(),
                            name,
                            result: msg.content.clone().unwrap_or_default(),
                        }));
                    }
                    events.push(EventPayload::MessageTool(MessageTool {
                        message: msg.clone(),
                    }));
                }
                Role::System => {}
            }
        }

        Ok(events)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compare two messages for structural equality (role, content, tool_call_id,
/// and tool call IDs). Used by SyncConversation to verify the prefix.
fn messages_match(a: &Message, b: &Message) -> bool {
    if a.role != b.role {
        return false;
    }
    if a.content != b.content {
        return false;
    }
    if a.tool_call_id != b.tool_call_id {
        return false;
    }
    if a.tool_calls.len() != b.tool_calls.len() {
        return false;
    }
    a.tool_calls
        .iter()
        .zip(b.tool_calls.iter())
        .all(|(ta, tb)| ta.id == tb.id && ta.name == tb.name)
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
            description: None,
            llm: LlmConfig {
                client: "mock".into(),
                params: Default::default(),
            },
            system_prompt: "test".into(),
            mcp_servers: vec![],
            strategy: Default::default(),
            retry: Default::default(),
            token_budget: None,
            sub_agents: vec![],
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
        state.apply(
            &EventPayload::SessionCreated(SessionCreated {
                agent: test_agent(),
                auth: test_auth(),
                on_done: None,
            }),
            1,
        );
        state
    }

    fn apply_events(state: &mut AgentSession, payloads: Vec<EventPayload>) {
        let seq = state.agent_state.last_applied.unwrap_or(0);
        for (i, payload) in payloads.iter().enumerate() {
            state.apply(payload, seq + 1 + i as u64);
        }
    }

    #[test]
    fn complete_llm_call_emits_message_and_tool_calls() {
        let mut state = created_state();
        let call_id = "call-1".to_string();

        // Request an LLM call
        let payloads = state
            .handle(CommandPayload::RequestLlmCall {
                call_id: call_id.clone(),
                request: mock_llm_request(),
                stream: false,
                deadline: far_future(),
            })
            .unwrap();
        apply_events(&mut state, payloads);

        // Complete with a response that has tool calls
        let response = LlmResponse::OpenAi(openai::ChatCompletionResponse {
            id: "resp-1".into(),
            model: "mock".into(),
            choices: vec![openai::Choice {
                index: 0,
                message: openai::ChatMessage {
                    role: openai::Role::Assistant,
                    content: Some("thinking...".into()),
                    tool_calls: Some(vec![openai::ToolCall {
                        id: "tc-1".into(),
                        call_type: "function".into(),
                        function: openai::FunctionCall {
                            name: "read_file".into(),
                            arguments: r#"{"path":"foo.rs"}"#.into(),
                        },
                    }]),
                    tool_call_id: None,
                },
                finish_reason: Some("tool_calls".into()),
            }],
            usage: None,
        });

        let payloads = state
            .handle(CommandPayload::CompleteLlmCall {
                call_id: call_id.clone(),
                response,
            })
            .unwrap();

        assert_eq!(
            payloads.len(),
            3,
            "expected LlmCallCompleted + MessageAssistant + ToolCallRequested"
        );
        assert!(matches!(&payloads[0], EventPayload::LlmCallCompleted(_)));
        assert!(
            matches!(&payloads[1], EventPayload::MessageAssistant(m) if m.message.content == Some("thinking...".into()))
        );
        assert!(
            matches!(&payloads[2], EventPayload::ToolCallRequested(t) if t.tool_call_id == "tc-1" && t.name == "read_file")
        );
    }

    #[test]
    fn complete_tool_call_emits_message_tool() {
        let mut state = created_state();
        let call_id = "call-1".to_string();
        let tool_call_id = "tc-1".to_string();

        // Set up: LLM call -> complete (emits assistant message + tool call requested)
        let payloads = state
            .handle(CommandPayload::RequestLlmCall {
                call_id: call_id.clone(),
                request: mock_llm_request(),
                stream: false,
                deadline: far_future(),
            })
            .unwrap();
        apply_events(&mut state, payloads);

        let response = LlmResponse::OpenAi(openai::ChatCompletionResponse {
            id: "resp-1".into(),
            model: "mock".into(),
            choices: vec![openai::Choice {
                index: 0,
                message: openai::ChatMessage {
                    role: openai::Role::Assistant,
                    content: None,
                    tool_calls: Some(vec![openai::ToolCall {
                        id: tool_call_id.clone(),
                        call_type: "function".into(),
                        function: openai::FunctionCall {
                            name: "test".into(),
                            arguments: "{}".into(),
                        },
                    }]),
                    tool_call_id: None,
                },
                finish_reason: Some("tool_calls".into()),
            }],
            usage: None,
        });

        let payloads = state
            .handle(CommandPayload::CompleteLlmCall {
                call_id: call_id.clone(),
                response,
            })
            .unwrap();
        apply_events(&mut state, payloads);

        // Complete the tool call
        let payloads = state
            .handle(CommandPayload::CompleteToolCall {
                tool_call_id: tool_call_id.clone(),
                name: "test".into(),
                result: "ok".into(),
            })
            .unwrap();

        assert_eq!(
            payloads.len(),
            2,
            "expected ToolCallCompleted + MessageTool"
        );
        assert!(
            matches!(&payloads[0], EventPayload::ToolCallCompleted(t) if t.tool_call_id == "tc-1")
        );
        assert!(
            matches!(&payloads[1], EventPayload::MessageTool(m) if m.message.content == Some("ok".into()))
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
        state.apply(
            &EventPayload::SessionCreated(SessionCreated {
                agent,
                auth: test_auth(),
                on_done: None,
            }),
            1,
        );

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

        // SendMessage with user content should fail
        let result = state.handle(CommandPayload::SendMessage {
            message: IncomingMessage::User {
                content: "hello".into(),
            },
            stream: true,
        });
        assert!(matches!(result, Err(SessionError::SessionInterrupted)));
    }
}
