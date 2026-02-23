use std::sync::Arc;

use ractor::{Actor, ActorProcessingErr, ActorRef, RpcReplyPort};
use uuid::Uuid;

use chrono::Utc;

use super::event_store::{EventStore, StoreError, Version};
use super::llm::LlmClientProvider;
use super::llm::StreamDelta;
use super::mcp::McpClientProvider;
use super::mcp::{Content, McpClient};
use super::strategy::StrategyProvider;
use crate::domain::event::*;
use crate::domain::session::{
    AgentSession, AgentState, CommandPayload, Effect, SessionCommand, SessionError, SessionStatus,
    Strategy,
};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error(transparent)]
    Session(#[from] SessionError),
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("actor call failed: {0}")]
    ActorCall(String),
    #[error("unknown LLM client: {0}")]
    UnknownLlmClient(String),
    #[error("unknown agent: {0}")]
    UnknownAgent(String),
    #[error("session not found")]
    SessionNotFound,
    #[error("strategy resolution failed: {0}")]
    StrategyResolution(String),
}

// ---------------------------------------------------------------------------
// SessionMessage — commands from API + events from dispatcher
// ---------------------------------------------------------------------------

pub enum SessionMessage {
    Execute(
        SessionCommand,
        RpcReplyPort<Result<Vec<Event>, RuntimeError>>,
    ),
    Cast(SessionCommand),
    GetState(RpcReplyPort<AgentState>),
    Events(Vec<Arc<Event>>),
    /// Resume effects to execute after MCP servers are ready.
    ResumeRecovery(Vec<Effect>),
    /// Timer-triggered wake from Idle state.
    Wake,
}

// ---------------------------------------------------------------------------
// SessionActor — merged session + reactor
// ---------------------------------------------------------------------------

pub struct SessionActor;

pub struct SessionActorState {
    pub session_id: Uuid,
    pub session: AgentSession,
    pub store: Arc<dyn EventStore>,
    pub auth: SessionAuth,
    pub llm_provider: Arc<dyn LlmClientProvider>,
    pub mcp_provider: Arc<dyn McpClientProvider>,
    pub strategy_provider: Arc<dyn StrategyProvider>,
    pub mcp_clients: Vec<Arc<dyn McpClient>>,
}

impl SessionActorState {
    pub fn new(
        session_id: Uuid,
        store: Arc<dyn EventStore>,
        auth: SessionAuth,
        llm_provider: Arc<dyn LlmClientProvider>,
        mcp_provider: Arc<dyn McpClientProvider>,
        strategy_provider: Arc<dyn StrategyProvider>,
        strategy: Box<dyn Strategy>,
        strategy_state: serde_json::Value,
    ) -> Self {
        SessionActorState {
            session_id,
            session: AgentSession::new(session_id, strategy, strategy_state),
            store,
            auth,
            llm_provider,
            mcp_provider,
            strategy_provider,
            mcp_clients: Vec::new(),
        }
    }

    /// Build from a pre-constructed `AgentSession` (e.g. after resuming from
    /// events). Avoids the caller having to know about `SessionActorState`
    /// field layout.
    pub fn from_session(
        session: AgentSession,
        store: Arc<dyn EventStore>,
        auth: SessionAuth,
        llm_provider: Arc<dyn LlmClientProvider>,
        mcp_provider: Arc<dyn McpClientProvider>,
        strategy_provider: Arc<dyn StrategyProvider>,
    ) -> Self {
        SessionActorState {
            session_id: session.agent_state.session_id,
            session,
            store,
            auth,
            llm_provider,
            mcp_provider,
            strategy_provider,
            mcp_clients: Vec::new(),
        }
    }

    pub fn with_mcp_clients(mut self, mcp_clients: Vec<Arc<dyn McpClient>>) -> Self {
        self.mcp_clients = mcp_clients;
        self
    }

    /// Collect all tool definitions from all MCP clients as OpenAI tools.
    pub fn all_tools(&self) -> Option<Vec<crate::domain::openai::Tool>> {
        let tools: Vec<crate::domain::openai::Tool> = self
            .mcp_clients
            .iter()
            .flat_map(|c| c.tools().iter().map(|t| t.to_openai_tool()))
            .collect();
        if tools.is_empty() {
            None
        } else {
            Some(tools)
        }
    }

    /// Find the MCP client that owns a given tool name.
    pub fn find_mcp_client(&self, tool_name: &str) -> Option<&Arc<dyn McpClient>> {
        self.mcp_clients
            .iter()
            .find(|c| c.tools().iter().any(|t| t.name == tool_name))
    }
}

async fn execute(
    state: &mut SessionActorState,
    cmd: SessionCommand,
) -> Result<Vec<Event>, RuntimeError> {
    let payloads = state.session.handle(cmd.payload)?;
    if payloads.is_empty() {
        return Ok(vec![]);
    }

    let version = Version(state.session.agent_state.stream_version);

    // Aggregate builds events (assigns session-local sequences)
    let base_seq = state.session.agent_state.stream_version + 1;
    let events: Vec<Event> = payloads
        .into_iter()
        .enumerate()
        .map(|(i, payload)| Event {
            id: Uuid::new_v4(),
            tenant_id: state.auth.tenant_id.clone(),
            session_id: state.session_id,
            sequence: base_seq + i as u64,
            span: cmd.span.clone(),
            occurred_at: cmd.occurred_at,
            payload,
        })
        .collect();

    // Apply locally, then snapshot
    for event in &events {
        state.session.apply(event);
    }
    let snapshot = state.session.snapshot();

    // Persist events + snapshot atomically
    state
        .store
        .append(
            state.session_id,
            &state.auth,
            version,
            events.clone(),
            snapshot,
        )
        .await?;

    Ok(events)
}

// ---------------------------------------------------------------------------
// Effect delivery — sends Cast to self
// ---------------------------------------------------------------------------

async fn handle_effect(
    state: &mut SessionActorState,
    myself: &ActorRef<SessionMessage>,
    span: &SpanContext,
    effect: Effect,
) {
    match effect {
        Effect::Command(payload) => {
            let _ = myself.send_message(SessionMessage::Cast(SessionCommand {
                span: span.child(),
                occurred_at: Utc::now(),
                payload,
            }));
        }
        Effect::StartMcpServers(configs) => {
            for config in &configs {
                match state.mcp_provider.start_server(config, &state.auth).await {
                    Ok(client) => state.mcp_clients.push(client),
                    Err(e) => eprintln!("MCP '{}' failed: {}", config.name, e),
                }
            }
        }
        Effect::CallLlm {
            call_id,
            request,
            stream,
        } => {
            let client_id = state
                .session
                .agent_state
                .agent
                .as_ref()
                .map(|a| a.llm.client.clone())
                .unwrap_or_default();
            let client = match state.llm_provider.resolve(&client_id, &state.auth).await {
                Ok(c) => c,
                Err(e) => {
                    let _ = execute(
                        state,
                        SessionCommand {
                            span: span.child(),
                            occurred_at: Utc::now(),
                            payload: CommandPayload::FailLlmCall {
                                call_id,
                                error: e.to_string(),
                            },
                        },
                    )
                    .await;
                    return;
                }
            };

            let result = if stream {
                let (chunk_tx, mut chunk_rx) =
                    tokio::sync::mpsc::unbounded_channel::<StreamDelta>();

                let call_id_fwd = call_id.clone();
                let span_fwd = span.clone();

                let (result, _) = tokio::join!(client.call_streaming(&request, chunk_tx), async {
                    let mut chunk_index: u32 = 0;
                    while let Some(delta) = chunk_rx.recv().await {
                        if let Some(text) = delta.text {
                            let _ = execute(
                                state,
                                SessionCommand {
                                    span: span_fwd.child(),
                                    occurred_at: Utc::now(),
                                    payload: CommandPayload::StreamLlmChunk {
                                        call_id: call_id_fwd.clone(),
                                        chunk_index,
                                        text,
                                    },
                                },
                            )
                            .await;
                            chunk_index += 1;
                        }
                    }
                });
                result
            } else {
                client.call(&request).await
            };

            let payload = match result {
                Ok(response) => CommandPayload::CompleteLlmCall { call_id, response },
                Err(error) => CommandPayload::FailLlmCall { call_id, error },
            };
            let _ = execute(
                state,
                SessionCommand {
                    span: span.child(),
                    occurred_at: Utc::now(),
                    payload,
                },
            )
            .await;
        }
        Effect::CallMcpTool {
            tool_call_id,
            name,
            arguments,
        } => {
            let mcp = state.find_mcp_client(&name).cloned();
            if let Some(mcp) = mcp {
                let payload = match mcp.call_tool(&name, arguments).await {
                    Ok(result) => {
                        let text = result
                            .content
                            .iter()
                            .filter_map(|c| match c {
                                Content::Text { text } => Some(text.as_str()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        if result.is_error {
                            CommandPayload::FailToolCall {
                                tool_call_id,
                                name,
                                error: text,
                            }
                        } else {
                            CommandPayload::CompleteToolCall {
                                tool_call_id,
                                name,
                                result: text,
                            }
                        }
                    }
                    Err(e) => CommandPayload::FailToolCall {
                        tool_call_id,
                        name,
                        error: e.to_string(),
                    },
                };
                let _ = execute(
                    state,
                    SessionCommand {
                        span: span.child(),
                        occurred_at: Utc::now(),
                        payload,
                    },
                )
                .await;
            }
            // else: no MCP client — client-side tool, handled via event fan-out
        }
        Effect::ScheduleWake { wake_at } => {
            let delay = (wake_at - Utc::now())
                .to_std()
                .unwrap_or(std::time::Duration::ZERO);
            let myself = myself.clone();
            tokio::spawn(async move {
                tokio::time::sleep(delay).await;
                let _ = myself.send_message(SessionMessage::Wake);
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Actor implementation
// ---------------------------------------------------------------------------

impl Actor for SessionActor {
    type Msg = SessionMessage;
    type State = SessionActorState;
    type Arguments = SessionActorState;

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        Ok(args)
    }

    async fn handle(
        &self,
        myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match message {
            SessionMessage::Execute(cmd, reply) => {
                let result = execute(state, cmd).await;
                let _ = reply.send(result);
            }
            SessionMessage::Cast(cmd) => {
                if let Err(e) = execute(state, cmd).await {
                    eprintln!("session cast error: {e}");
                }
            }
            SessionMessage::Events(events) => {
                for event in &events {
                    state.session.apply(event);
                    let reacted = state
                        .session
                        .agent_state
                        .last_reacted
                        .is_some_and(|seq| event.sequence <= seq);
                    if !reacted {
                        let tools = state.all_tools();
                        let effects = state.session.react(tools, event);
                        for effect in effects {
                            handle_effect(state, &myself, &event.span, effect).await;
                        }
                        state.session.agent_state.last_reacted = Some(event.sequence);
                    }
                }
            }
            SessionMessage::GetState(reply) => {
                let _ = reply.send(state.session.agent_state.clone());
            }
            SessionMessage::ResumeRecovery(effects) => {
                let span = SpanContext::root();
                for effect in effects {
                    handle_effect(state, &myself, &span, effect).await;
                }
            }
            SessionMessage::Wake => {
                if matches!(state.session.agent_state.status, SessionStatus::Idle { .. }) {
                    let tools = state.all_tools();
                    let effects = state.session.wake(tools);
                    let span = SpanContext::root();
                    for effect in effects {
                        handle_effect(state, &myself, &span, effect).await;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::agent::{AgentConfig, LlmConfig};
    use crate::domain::openai;
    use crate::domain::session::ReactStrategy;

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
            serde_json::Value::Null,
        );
        let event = Event {
            id: Uuid::new_v4(),
            tenant_id: "t".into(),
            session_id: state.agent_state.session_id,
            sequence: 1,
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
            };
            state.apply(&event);
        }
    }

    #[test]
    fn recover_pending_llm_call() {
        let mut state = created_state();

        // Add a user message and a pending LLM call
        apply_events(
            &mut state,
            vec![
                EventPayload::MessageUser(MessageUser {
                    message: Message {
                        role: Role::User,
                        content: Some("hi".into()),
                        tool_calls: vec![],
                        tool_call_id: None,
                        call_id: None,
                        token_count: None,
                    },
                    stream: false,
                }),
                EventPayload::LlmCallRequested(LlmCallRequested {
                    call_id: "call-1".into(),
                    request: mock_llm_request(),
                    stream: false,
                }),
            ],
        );

        let recovery = state.recover(None);
        assert!(recovery.setup.is_empty());
        assert!(
            recovery
                .resume
                .iter()
                .any(|e| matches!(e, Effect::CallLlm { call_id, .. } if call_id == "call-1")),
            "should recover pending LLM call"
        );
    }

    #[test]
    fn recover_completed_llm_no_assistant_message() {
        let mut state = created_state();

        // User message, LLM call requested, LLM call completed — but no assistant message
        apply_events(
            &mut state,
            vec![
                EventPayload::MessageUser(MessageUser {
                    message: Message {
                        role: Role::User,
                        content: Some("hi".into()),
                        tool_calls: vec![],
                        tool_call_id: None,
                        call_id: None,
                        token_count: None,
                    },
                    stream: false,
                }),
                EventPayload::LlmCallRequested(LlmCallRequested {
                    call_id: "call-1".into(),
                    request: mock_llm_request(),
                    stream: false,
                }),
                EventPayload::LlmCallCompleted(LlmCallCompleted {
                    call_id: "call-1".into(),
                    response: mock_llm_response(),
                }),
            ],
        );

        let recovery = state.recover(None);
        assert!(recovery.resume.iter().any(|e| matches!(e, Effect::Command(CommandPayload::SendAssistantMessage { call_id, .. }) if call_id == "call-1")),
            "should recover completed LLM call with no assistant message");
    }

    #[test]
    fn recover_pending_tool_call() {
        let mut state = created_state();

        // User msg, LLM call, LLM completed, assistant message with tool call, tool call requested
        let call_id = "call-1".to_string();
        let tool_call_id = "tc-1".to_string();

        apply_events(
            &mut state,
            vec![
                EventPayload::MessageUser(MessageUser {
                    message: Message {
                        role: Role::User,
                        content: Some("hi".into()),
                        tool_calls: vec![],
                        tool_call_id: None,
                        call_id: None,
                        token_count: None,
                    },
                    stream: false,
                }),
                EventPayload::LlmCallRequested(LlmCallRequested {
                    call_id: call_id.clone(),
                    request: mock_llm_request(),
                    stream: false,
                }),
                EventPayload::LlmCallCompleted(LlmCallCompleted {
                    call_id: call_id.clone(),
                    response: mock_llm_response(),
                }),
                EventPayload::MessageAssistant(MessageAssistant {
                    call_id: call_id.clone(),
                    message: Message {
                        role: Role::Assistant,
                        content: None,
                        tool_calls: vec![ToolCall {
                            id: tool_call_id.clone(),
                            name: "test_tool".into(),
                            arguments: "{}".into(),
                        }],
                        tool_call_id: None,
                        call_id: Some(call_id.clone()),
                        token_count: None,
                    },
                }),
                EventPayload::ToolCallRequested(ToolCallRequested {
                    tool_call_id: tool_call_id.clone(),
                    name: "test_tool".into(),
                    arguments: "{}".into(),
                }),
            ],
        );

        let recovery = state.recover(None);
        assert!(
            recovery
                .resume
                .iter()
                .any(|e| matches!(e, Effect::CallMcpTool { tool_call_id: id, .. } if id == "tc-1")),
            "should recover pending tool call"
        );
    }

    #[test]
    fn recover_all_tools_done_last_msg_is_tool() {
        let mut state = created_state();

        let call_id = "call-1".to_string();
        let tool_call_id = "tc-1".to_string();

        apply_events(
            &mut state,
            vec![
                EventPayload::MessageUser(MessageUser {
                    message: Message {
                        role: Role::User,
                        content: Some("hi".into()),
                        tool_calls: vec![],
                        tool_call_id: None,
                        call_id: None,
                        token_count: None,
                    },
                    stream: false,
                }),
                EventPayload::LlmCallRequested(LlmCallRequested {
                    call_id: call_id.clone(),
                    request: mock_llm_request(),
                    stream: false,
                }),
                EventPayload::LlmCallCompleted(LlmCallCompleted {
                    call_id: call_id.clone(),
                    response: mock_llm_response(),
                }),
                EventPayload::MessageAssistant(MessageAssistant {
                    call_id: call_id.clone(),
                    message: Message {
                        role: Role::Assistant,
                        content: None,
                        tool_calls: vec![ToolCall {
                            id: tool_call_id.clone(),
                            name: "test_tool".into(),
                            arguments: "{}".into(),
                        }],
                        tool_call_id: None,
                        call_id: Some(call_id.clone()),
                        token_count: None,
                    },
                }),
                EventPayload::ToolCallRequested(ToolCallRequested {
                    tool_call_id: tool_call_id.clone(),
                    name: "test_tool".into(),
                    arguments: "{}".into(),
                }),
                EventPayload::ToolCallCompleted(ToolCallCompleted {
                    tool_call_id: tool_call_id.clone(),
                    name: "test_tool".into(),
                    result: "ok".into(),
                }),
                EventPayload::MessageTool(MessageTool {
                    message: Message {
                        role: Role::Tool,
                        content: Some("ok".into()),
                        tool_calls: vec![],
                        tool_call_id: Some(tool_call_id.clone()),
                        call_id: None,
                        token_count: None,
                    },
                }),
            ],
        );

        assert_eq!(state.agent_state.messages.last().unwrap().role, Role::Tool);

        let recovery = state.recover(None);
        assert!(
            recovery
                .resume
                .iter()
                .any(|e| matches!(e, Effect::Command(CommandPayload::RequestLlmCall { .. }))),
            "should trigger next LLM call when all tools done and last msg is tool"
        );
    }

    #[test]
    fn recover_no_stuck_conditions_returns_empty() {
        let state = created_state();
        let recovery = state.recover(None);
        assert!(recovery.setup.is_empty());
        assert!(recovery.resume.is_empty());
    }

    #[test]
    fn recover_during_interrupt_skips_agent_loop() {
        let mut state = created_state();

        // User message, LLM call, LLM completed, assistant message with tool call,
        // then session interrupted
        let call_id = "call-1".to_string();
        apply_events(
            &mut state,
            vec![
                EventPayload::MessageUser(MessageUser {
                    message: Message {
                        role: Role::User,
                        content: Some("hi".into()),
                        tool_calls: vec![],
                        tool_call_id: None,
                        call_id: None,
                        token_count: None,
                    },
                    stream: false,
                }),
                EventPayload::LlmCallRequested(LlmCallRequested {
                    call_id: call_id.clone(),
                    request: mock_llm_request(),
                    stream: false,
                }),
                EventPayload::LlmCallCompleted(LlmCallCompleted {
                    call_id: call_id.clone(),
                    response: mock_llm_response(),
                }),
                EventPayload::MessageAssistant(MessageAssistant {
                    call_id: call_id.clone(),
                    message: Message {
                        role: Role::Assistant,
                        content: Some("hello".into()),
                        tool_calls: vec![],
                        tool_call_id: None,
                        call_id: Some(call_id.clone()),
                        token_count: None,
                    },
                }),
                EventPayload::SessionInterrupted(SessionInterrupted {
                    interrupt_id: "int-1".into(),
                    reason: "approval_needed".into(),
                    payload: serde_json::json!({}),
                }),
            ],
        );

        let recovery = state.recover(None);
        // Setup should be empty (no MCP servers), resume should be empty (interrupted)
        assert!(recovery.setup.is_empty());
        assert!(
            recovery.resume.is_empty(),
            "interrupted session should not resume agent loop"
        );
    }

    #[test]
    fn recover_completed_llm_with_response_processed_is_not_stuck() {
        let mut state = created_state();

        let call_id = "call-1".to_string();
        apply_events(
            &mut state,
            vec![
                EventPayload::MessageUser(MessageUser {
                    message: Message {
                        role: Role::User,
                        content: Some("hi".into()),
                        tool_calls: vec![],
                        tool_call_id: None,
                        call_id: None,
                        token_count: None,
                    },
                    stream: false,
                }),
                EventPayload::LlmCallRequested(LlmCallRequested {
                    call_id: call_id.clone(),
                    request: mock_llm_request(),
                    stream: false,
                }),
                EventPayload::LlmCallCompleted(LlmCallCompleted {
                    call_id: call_id.clone(),
                    response: mock_llm_response(),
                }),
                EventPayload::MessageAssistant(MessageAssistant {
                    call_id: call_id.clone(),
                    message: Message {
                        role: Role::Assistant,
                        content: Some("hello".into()),
                        tool_calls: vec![],
                        tool_call_id: None,
                        call_id: Some(call_id.clone()),
                        token_count: None,
                    },
                }),
            ],
        );

        let recovery = state.recover(None);
        // No stuck conditions — completed LLM with message extracted, last msg is assistant (not tool)
        assert!(
            recovery.resume.is_empty(),
            "fully completed flow should have no recovery effects"
        );
    }
}
