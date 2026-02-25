use std::sync::Arc;

use chrono::Utc;
use ractor::{Actor, ActorProcessingErr, ActorRef, RpcReplyPort};
use uuid::Uuid;

use crate::domain::agent::AgentConfig;
use crate::domain::event::*;
use crate::domain::session::{
    AgentSession, AgentState, CommandPayload, Effect, SessionCommand, SessionError,
};

use super::event_store::{EventStore, StoreError};
use super::llm::{LlmClientProvider, StreamDelta};
use super::mcp::{Content, McpClient, McpClientProvider};
use super::strategy::StrategyProvider;

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
    /// Timer-triggered or scheduler-triggered wake.
    Wake,
}

// ---------------------------------------------------------------------------
// SessionInit — what the actor needs to start up
// ---------------------------------------------------------------------------

pub struct SessionInit {
    pub agent: AgentConfig,
    pub auth: SessionAuth,
}

pub struct SessionActorArgs {
    pub session_id: Uuid,
    pub init: SessionInit,
    pub store: Arc<dyn EventStore>,
    pub llm_provider: Arc<dyn LlmClientProvider>,
    pub mcp_provider: Arc<dyn McpClientProvider>,
    pub strategy_provider: Arc<dyn StrategyProvider>,
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
    let (events, snapshot) = state
        .session
        .process_command(cmd, &state.auth.tenant_id)?;

    if events.is_empty() {
        return Ok(vec![]);
    }

    state
        .store
        .append(
            state.session_id,
            &state.auth,
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
                                retryable: true,
                                source: None,
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
                Err(llm_error) => CommandPayload::FailLlmCall {
                    call_id,
                    error: llm_error.message,
                    retryable: llm_error.retryable,
                    source: serde_json::to_value(&llm_error.source).ok(),
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
    }
}

// ---------------------------------------------------------------------------
// Actor implementation
// ---------------------------------------------------------------------------

impl Actor for SessionActor {
    type Msg = SessionMessage;
    type State = SessionActorState;
    type Arguments = SessionActorArgs;

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        let session_id = args.session_id;
        let auth = args.init.auth;
        let agent = args.init.agent;

        // Try to resume from store; if not found, create a new session.
        let mut session = match args.store.load(session_id, &auth) {
            Ok(loaded) => {
                let snapshot = loaded.snapshot;
                let snap_agent = snapshot
                    .agent
                    .as_ref()
                    .ok_or("session has no agent config")?;
                let strategy = args
                    .strategy_provider
                    .resolve(snap_agent, &auth)
                    .await
                    .map_err(|e| format!("strategy resolution: {e}"))?;
                AgentSession::from_snapshot(snapshot, strategy)
            }
            Err(StoreError::SessionNotFound) => {
                let strategy = args
                    .strategy_provider
                    .resolve(&agent, &auth)
                    .await
                    .map_err(|e| format!("strategy resolution: {e}"))?;
                let mut session = AgentSession::new(session_id, strategy);
                let cmd = SessionCommand {
                    span: SpanContext::root(),
                    occurred_at: Utc::now(),
                    payload: CommandPayload::CreateSession {
                        agent,
                        auth: auth.clone(),
                    },
                };
                let (events, snapshot) = session
                    .process_command(cmd, &auth.tenant_id)
                    .map_err(|e| format!("init: {e}"))?;
                args.store
                    .append(session_id, &auth, events, snapshot)
                    .await
                    .map_err(|e| format!("store: {e}"))?;
                session
            }
            Err(e) => return Err(format!("load: {e}").into()),
        };

        // Start MCP servers
        let mut mcp_clients = Vec::new();
        if let Some(agent) = &session.agent_state.agent {
            for config in &agent.mcp_servers {
                match args.mcp_provider.start_server(config, &auth).await {
                    Ok(client) => mcp_clients.push(client),
                    Err(e) => eprintln!("MCP '{}' failed: {}", config.name, e),
                }
            }
        }

        // Prevent double-react on events already in the store
        session.agent_state.last_reacted = session.agent_state.last_applied;

        Ok(SessionActorState {
            session_id,
            session,
            store: args.store,
            auth,
            llm_provider: args.llm_provider,
            mcp_provider: args.mcp_provider,
            strategy_provider: args.strategy_provider,
            mcp_clients,
        })
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
                        let effects = state.session.react(tools, event).await;
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
            SessionMessage::Wake => {
                let cmd = SessionCommand {
                    span: SpanContext::root(),
                    occurred_at: Utc::now(),
                    payload: CommandPayload::Wake,
                };
                if let Err(e) = execute(state, cmd).await {
                    eprintln!("session wake error: {e}");
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    use crate::domain::agent::{AgentConfig, LlmConfig};
    use crate::domain::openai;
    use crate::domain::session::DefaultStrategy;

    fn far_future() -> chrono::DateTime<Utc> {
        Utc::now() + Duration::hours(1)
    }

    fn past() -> chrono::DateTime<Utc> {
        Utc::now() - Duration::seconds(10)
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
        let mut state = AgentSession::new(
            Uuid::new_v4(),
            Arc::new(DefaultStrategy::default()),
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

    // -----------------------------------------------------------------------
    // wake_at() tests
    // -----------------------------------------------------------------------

    #[test]
    fn wake_at_none_when_done() {
        let state = created_state();
        assert!(state.agent_state.wake_at().is_none());
    }

    #[test]
    fn wake_at_returns_pending_llm_deadline() {
        let mut state = created_state();
        let deadline = far_future();

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
                    deadline,
                }),
            ],
        );

        assert_eq!(state.agent_state.wake_at(), Some(deadline));
    }

    #[test]
    fn wake_at_returns_failed_retry_next_at() {
        let mut state = created_state();

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
                    deadline: far_future(),
                }),
                EventPayload::LlmCallErrored(LlmCallErrored {
                    call_id: "call-1".into(),
                    error: "timeout".into(),
                    retryable: true,
                    source: None,
                }),
            ],
        );

        let wake = state.agent_state.wake_at();
        assert!(wake.is_some(), "should have a retry wake_at");
    }

    #[test]
    fn wake_at_none_when_interrupted() {
        let mut state = created_state();

        apply_events(
            &mut state,
            vec![EventPayload::SessionInterrupted(SessionInterrupted {
                interrupt_id: "int-1".into(),
                reason: "approval".into(),
                payload: serde_json::json!({}),
            })],
        );

        assert!(state.agent_state.wake_at().is_none());
    }

    // -----------------------------------------------------------------------
    // wake() tests
    // -----------------------------------------------------------------------

    #[test]
    fn wake_pending_llm_call_re_issues() {
        let mut state = created_state();

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
                    deadline: far_future(),
                }),
            ],
        );

        let events = state.handle(CommandPayload::Wake).unwrap();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::LlmCallRequested(p) if p.call_id == "call-1")),
            "should re-issue pending LLM call"
        );
    }

    #[test]
    fn wake_timed_out_llm_call_emits_fail() {
        let mut state = created_state();

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
                    deadline: past(),
                }),
            ],
        );

        let events = state.handle(CommandPayload::Wake).unwrap();
        assert_eq!(events.len(), 1, "should produce exactly one event");
        assert!(
            matches!(&events[0], EventPayload::LlmCallErrored(p) if p.call_id == "call-1" && p.retryable),
            "should fail timed-out LLM call as retryable"
        );
    }

    #[test]
    fn wake_completed_llm_no_assistant_message() {
        let mut state = created_state();

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
                    deadline: far_future(),
                }),
                EventPayload::LlmCallCompleted(LlmCallCompleted {
                    call_id: "call-1".into(),
                    response: mock_llm_response(),
                }),
            ],
        );

        let events = state.handle(CommandPayload::Wake).unwrap();
        assert!(
            events.iter().any(|e| matches!(e, EventPayload::MessageAssistant(p) if p.call_id == "call-1")),
            "should emit MessageAssistant for unprocessed completed call"
        );
    }

    #[test]
    fn wake_pending_tool_call_re_issues() {
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
                    deadline: far_future(),
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
                    deadline: far_future(),
                }),
            ],
        );

        let events = state.handle(CommandPayload::Wake).unwrap();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::ToolCallRequested(p) if p.tool_call_id == "tc-1")),
            "should re-issue pending tool call"
        );
    }

    #[test]
    fn wake_all_tools_done_triggers_next_llm_call() {
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
                    deadline: far_future(),
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
                    deadline: far_future(),
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

        let events = state.handle(CommandPayload::Wake).unwrap();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::LlmCallRequested(_))),
            "should trigger next LLM call when all tools done and last msg is tool"
        );
    }

    #[test]
    fn wake_no_stuck_conditions_returns_empty() {
        let state = created_state();
        let events = state.handle(CommandPayload::Wake).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn wake_completed_llm_with_response_processed_returns_empty() {
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
                    deadline: far_future(),
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

        let events = state.handle(CommandPayload::Wake).unwrap();
        assert!(
            events.is_empty(),
            "fully completed flow should have no wake events"
        );
    }

    #[test]
    fn failed_llm_call_has_retry_wake_at() {
        let mut state = created_state();

        // Create a failed LLM call → apply_core computes retry.next_at
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
                    deadline: far_future(),
                }),
                EventPayload::LlmCallErrored(LlmCallErrored {
                    call_id: "call-1".into(),
                    error: "timeout".into(),
                    retryable: true,
                    source: None,
                }),
            ],
        );

        // apply_core computed retry.next_at for the failed call
        let wake_at = state.agent_state.wake_at();
        assert!(wake_at.is_some(), "should have a wake_at for retry");
    }

    #[test]
    fn wake_failed_llm_retry_reuses_call_id() {
        let mut state = created_state();

        // Set up: user message → LLM call requested → LLM call errored (retryable)
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
                    deadline: far_future(),
                }),
                EventPayload::LlmCallErrored(LlmCallErrored {
                    call_id: "call-1".into(),
                    error: "server error".into(),
                    retryable: true,
                    source: None,
                }),
            ],
        );

        // Manually set retry.next_at to past so wake triggers retry
        state
            .agent_state
            .llm_calls
            .get_mut("call-1")
            .unwrap()
            .retry
            .next_at = Some(past());

        let events = state.handle(CommandPayload::Wake).unwrap();

        // Should reuse "call-1" instead of generating a new call_id
        let retry_call_id = events.iter().find_map(|e| match e {
            EventPayload::LlmCallRequested(p) => Some(p.call_id.clone()),
            _ => None,
        });
        assert_eq!(
            retry_call_id.as_deref(),
            Some("call-1"),
            "wake should reuse the same call_id for retries"
        );

        // Apply the re-request events
        apply_events(&mut state, events);

        let call = state.agent_state.llm_calls.get("call-1").unwrap();
        assert_eq!(call.retry.attempts, 1, "retry count should be preserved from the previous failure");
        assert_eq!(call.status, crate::domain::session::LlmCallStatus::Pending);
    }

    #[tokio::test]
    async fn wake_failed_llm_retries_exhaust() {
        let mut state = created_state();

        // Set up: user message → LLM call requested → LLM call errored
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
                    deadline: far_future(),
                }),
                EventPayload::LlmCallErrored(LlmCallErrored {
                    call_id: "call-1".into(),
                    error: "server error".into(),
                    retryable: true,
                    source: None,
                }),
            ],
        );

        // First error: attempts=1, max_retries=3 → RetryScheduled
        let call = state.agent_state.llm_calls.get("call-1").unwrap();
        assert_eq!(
            call.status,
            crate::domain::session::LlmCallStatus::RetryScheduled,
            "first retryable error should schedule retry"
        );

        // Apply two more retryable errors to exhaust retries
        apply_events(
            &mut state,
            vec![
                EventPayload::LlmCallRequested(LlmCallRequested {
                    call_id: "call-1".into(),
                    request: mock_llm_request(),
                    stream: false,
                    deadline: far_future(),
                }),
                EventPayload::LlmCallErrored(LlmCallErrored {
                    call_id: "call-1".into(),
                    error: "server error".into(),
                    retryable: true,
                    source: None,
                }),
                EventPayload::LlmCallRequested(LlmCallRequested {
                    call_id: "call-1".into(),
                    request: mock_llm_request(),
                    stream: false,
                    deadline: far_future(),
                }),
                EventPayload::LlmCallErrored(LlmCallErrored {
                    call_id: "call-1".into(),
                    error: "server error".into(),
                    retryable: true,
                    source: None,
                }),
            ],
        );

        // After 3 errors (attempts=3, max_retries=3): Failed
        let call = state.agent_state.llm_calls.get("call-1").unwrap();
        assert_eq!(
            call.status,
            crate::domain::session::LlmCallStatus::Failed,
            "call should be Failed when retries exhausted"
        );

        // react() for the last LlmCallErrored should produce MarkDone
        let last_error_event = Event {
            id: Uuid::new_v4(),
            tenant_id: "t".into(),
            session_id: state.agent_state.session_id,
            sequence: 999, // doesn't matter for react
            span: SpanContext::root(),
            occurred_at: chrono::Utc::now(),
            payload: EventPayload::LlmCallErrored(LlmCallErrored {
                call_id: "call-1".into(),
                error: "server error".into(),
                retryable: true,
                source: None,
            }),
            derived: None,
        };
        let effects = state.react(None, &last_error_event).await;
        assert!(
            effects
                .iter()
                .any(|e| matches!(e, Effect::Command(CommandPayload::MarkDone))),
            "react should emit MarkDone when retries exhausted"
        );
    }

    #[tokio::test]
    async fn non_retryable_error_stops_immediately() {
        let mut state = created_state();

        // Set up: user message → LLM call requested → LLM call errored (non-retryable)
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
                    deadline: far_future(),
                }),
                EventPayload::LlmCallErrored(LlmCallErrored {
                    call_id: "call-1".into(),
                    error: "400 Bad Request".into(),
                    retryable: false,
                    source: Some(serde_json::json!({ "kind": "openai", "detail": { "status": 400 } })),
                }),
            ],
        );

        // Verify: call is Failed with no next_at
        let call = state.agent_state.llm_calls.get("call-1").unwrap();
        assert_eq!(
            call.status,
            crate::domain::session::LlmCallStatus::Failed,
            "non-retryable error should set status to Failed"
        );
        assert!(
            call.retry.next_at.is_none(),
            "non-retryable error should not set next_at"
        );

        // react() should emit MarkDone for non-retryable errors
        let error_event = Event {
            id: Uuid::new_v4(),
            tenant_id: "t".into(),
            session_id: state.agent_state.session_id,
            sequence: 999,
            span: SpanContext::root(),
            occurred_at: chrono::Utc::now(),
            payload: EventPayload::LlmCallErrored(LlmCallErrored {
                call_id: "call-1".into(),
                error: "400 Bad Request".into(),
                retryable: false,
                source: Some(serde_json::json!({ "kind": "openai", "detail": { "status": 400 } })),
            }),
            derived: None,
        };
        let effects = state.react(None, &error_event).await;
        assert!(
            effects
                .iter()
                .any(|e| matches!(e, Effect::Command(CommandPayload::MarkDone))),
            "react should emit MarkDone for non-retryable errors"
        );
    }
}
