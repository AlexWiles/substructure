use std::sync::Arc;

use ractor::{Actor, ActorProcessingErr, ActorRef, RpcReplyPort};
use uuid::Uuid;

use chrono::Utc;

use crate::domain::event::*;
use crate::domain::session::{
    CommandPayload, Effect, SessionCommand, SessionError, SessionState,
    extract_assistant_message, react, LlmCallStatus, ToolCallStatus,
};
use super::llm::StreamDelta;
use super::mcp::{McpClient, Content};
use super::event_store::{EventStore, StoreError, Version};
use super::llm::LlmClientProvider;
use super::mcp::McpClientProvider;

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
    GetState(RpcReplyPort<SessionState>),
    Events(Vec<Arc<Event>>),
    /// Resume effects to execute after MCP servers are ready.
    ResumeRecovery(Vec<Effect>),
}

// ---------------------------------------------------------------------------
// SessionActor — merged session + reactor
// ---------------------------------------------------------------------------

pub struct SessionActor;

pub struct SessionActorState {
    pub session_id: Uuid,
    pub session: SessionState,
    pub store: Arc<dyn EventStore>,
    pub auth: SessionAuth,
    pub llm_provider: Arc<dyn LlmClientProvider>,
    pub mcp_provider: Arc<dyn McpClientProvider>,
    pub mcp_clients: Vec<Arc<dyn McpClient>>,
}

impl SessionActorState {
    pub fn new(
        session_id: Uuid,
        store: Arc<dyn EventStore>,
        auth: SessionAuth,
        llm_provider: Arc<dyn LlmClientProvider>,
        mcp_provider: Arc<dyn McpClientProvider>,
    ) -> Self {
        SessionActorState {
            session_id,
            session: SessionState::new(session_id),
            store,
            auth,
            llm_provider,
            mcp_provider,
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
    actor_state: &mut SessionActorState,
    cmd: SessionCommand,
) -> Result<Vec<Event>, RuntimeError> {
    let payloads = actor_state.session.handle(cmd.payload)?;
    if payloads.is_empty() {
        return Ok(vec![]);
    }

    // Use local stream_version — not store.version() — so the version check
    // catches writes from other nodes that this actor hasn't seen yet.
    let version = Version(actor_state.session.stream_version);
    let events = actor_state
        .store
        .append(actor_state.session_id, &actor_state.auth, version, cmd.span, cmd.occurred_at, payloads)
        .await?;

    // Apply events to in-memory state immediately (so GetState is fresh).
    // React happens later when the dispatcher delivers these events back.
    for event in &events {
        actor_state.session.apply(event);
    }

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
        Effect::CallLlm { call_id, request, stream } => {
            let client_id = state.session.agent.as_ref()
                .map(|a| a.llm.client.clone())
                .unwrap_or_default();
            let client = match state.llm_provider.resolve(&client_id, &state.auth).await {
                Ok(c) => c,
                Err(e) => {
                    let _ = execute(state, SessionCommand {
                        span: span.child(),
                        occurred_at: Utc::now(),
                        payload: CommandPayload::FailLlmCall { call_id, error: e.to_string() },
                    }).await;
                    return;
                }
            };

            let result = if stream {
                let (chunk_tx, mut chunk_rx) =
                    tokio::sync::mpsc::unbounded_channel::<StreamDelta>();

                let call_id_fwd = call_id.clone();
                let span_fwd = span.clone();

                let (result, _) = tokio::join!(
                    client.call_streaming(&request, chunk_tx),
                    async {
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
                    }
                );
                result
            } else {
                client.call(&request).await
            };

            let payload = match result {
                Ok(response) => CommandPayload::CompleteLlmCall { call_id, response },
                Err(error) => CommandPayload::FailLlmCall { call_id, error },
            };
            let _ = execute(state, SessionCommand {
                span: span.child(),
                occurred_at: Utc::now(),
                payload,
            }).await;
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
                let _ = execute(state, SessionCommand {
                    span: span.child(),
                    occurred_at: Utc::now(),
                    payload,
                }).await;
            }
            // else: no MCP client — client-side tool, handled via event fan-out
        }
    }
}

// ---------------------------------------------------------------------------
// Recovery — rebuild state from events, detect stuck operations
// ---------------------------------------------------------------------------

/// Two-phase recovery effects: setup (MCP servers) must complete before
/// resume effects (LLM/tool calls) can be dispatched.
pub struct RecoveryEffects {
    pub setup: Vec<Effect>,
    pub resume: Vec<Effect>,
}

/// Scan rebuilt state for stuck conditions and return recovery effects.
pub fn recover(
    state: &SessionState,
    tools: Option<Vec<crate::domain::openai::Tool>>,
) -> RecoveryEffects {
    let mut setup = Vec::new();
    let mut resume = Vec::new();

    // MCP servers not running
    if let Some(agent) = &state.agent {
        if !agent.mcp_servers.is_empty() {
            setup.push(Effect::StartMcpServers(agent.mcp_servers.clone()));
        }
    }

    // Pending LLM call — re-issue the call
    for call in state.llm_calls.values() {
        if call.status == LlmCallStatus::Pending {
            resume.push(Effect::CallLlm {
                call_id: call.call_id.clone(),
                request: call.request.clone(),
                stream: state.stream,
            });
        }
    }

    // Completed LLM call, message not extracted — re-extract from stored response
    for call in state.llm_calls.values() {
        if call.status == LlmCallStatus::Completed && !call.response_processed {
            if let Some(ref response) = call.response {
                let (content, tool_calls, token_count) = extract_assistant_message(response);
                let mut effects = vec![Effect::Command(CommandPayload::SendAssistantMessage {
                    call_id: call.call_id.clone(),
                    content,
                    tool_calls: tool_calls.clone(),
                    token_count,
                })];
                for tc in &tool_calls {
                    effects.push(Effect::Command(CommandPayload::RequestToolCall {
                        tool_call_id: tc.id.clone(),
                        name: tc.name.clone(),
                        arguments: tc.arguments.clone(),
                    }));
                }
                resume.extend(effects);
            }
        }
    }

    // Pending tool calls — re-issue
    for tc in state.tool_calls.values() {
        if tc.status == ToolCallStatus::Pending {
            let args: serde_json::Value = state
                .messages
                .iter()
                .flat_map(|m| m.tool_calls.iter())
                .find(|t| t.id == tc.tool_call_id)
                .and_then(|t| serde_json::from_str(&t.arguments).ok())
                .unwrap_or_default();
            resume.push(Effect::CallMcpTool {
                tool_call_id: tc.tool_call_id.clone(),
                name: tc.name.clone(),
                arguments: args,
            });
        }
    }

    // All tools done, last message is Tool, no pending LLM call → trigger next LLM call
    let has_pending_llm = state.llm_calls.values().any(|c| c.status == LlmCallStatus::Pending);
    let last_is_tool = state.messages.last().is_some_and(|m| m.role == Role::Tool);
    if state.pending_tool_results == 0 && last_is_tool && !has_pending_llm {
        if let Some(request) = state.build_llm_request(tools) {
            resume.push(Effect::Command(CommandPayload::RequestLlmCall {
                call_id: Uuid::new_v4().to_string(),
                request,
                stream: state.stream,
            }));
        }
    }

    RecoveryEffects { setup, resume }
}

impl SessionActorState {
    /// Load events from the store and rebuild state, setting last_reacted = last_applied
    /// to prevent the dispatcher from re-reacting to old events.
    pub fn recover_from_store(&mut self) -> Result<(), StoreError> {
        let events = self.store.load(self.session_id, &self.auth)?;
        for event in &events {
            self.session.apply(event);
        }
        self.session.last_reacted = self.session.last_applied;
        Ok(())
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
                        .last_reacted
                        .is_some_and(|seq| event.sequence <= seq);
                    if !reacted {
                        let tools = state.all_tools();
                        let effects =
                            react(state.session_id, &mut state.session, tools, event);
                        for effect in effects {
                            handle_effect(state, &myself, &event.span, effect).await;
                        }
                        state.session.last_reacted = Some(event.sequence);
                    }
                }
            }
            SessionMessage::GetState(reply) => {
                let _ = reply.send(state.session.clone());
            }
            SessionMessage::ResumeRecovery(effects) => {
                let span = SpanContext::root();
                for effect in effects {
                    handle_effect(state, &myself, &span, effect).await;
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
    fn recover_pending_llm_call() {
        let mut state = created_state();

        // Add a user message and a pending LLM call
        apply_events(&mut state, vec![
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
        ]);

        let recovery = recover(&state, None);
        assert!(recovery.setup.is_empty());
        assert!(recovery.resume.iter().any(|e| matches!(e, Effect::CallLlm { call_id, .. } if call_id == "call-1")),
            "should recover pending LLM call");
    }

    #[test]
    fn recover_completed_llm_no_assistant_message() {
        let mut state = created_state();

        // User message, LLM call requested, LLM call completed — but no assistant message
        apply_events(&mut state, vec![
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
        ]);

        assert!(!state.llm_calls["call-1"].response_processed);

        let recovery = recover(&state, None);
        assert!(recovery.resume.iter().any(|e| matches!(e, Effect::Command(CommandPayload::SendAssistantMessage { call_id, .. }) if call_id == "call-1")),
            "should recover completed LLM call with no assistant message");
    }

    #[test]
    fn recover_pending_tool_call() {
        let mut state = created_state();

        // User msg, LLM call, LLM completed, assistant message with tool call, tool call requested
        let call_id = "call-1".to_string();
        let tool_call_id = "tc-1".to_string();

        apply_events(&mut state, vec![
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
        ]);

        let recovery = recover(&state, None);
        assert!(recovery.resume.iter().any(|e| matches!(e, Effect::CallMcpTool { tool_call_id: id, .. } if id == "tc-1")),
            "should recover pending tool call");
    }

    #[test]
    fn recover_all_tools_done_last_msg_is_tool() {
        let mut state = created_state();

        let call_id = "call-1".to_string();
        let tool_call_id = "tc-1".to_string();

        apply_events(&mut state, vec![
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
        ]);

        assert_eq!(state.pending_tool_results, 0);
        assert_eq!(state.messages.last().unwrap().role, Role::Tool);

        let recovery = recover(&state, None);
        assert!(recovery.resume.iter().any(|e| matches!(e, Effect::Command(CommandPayload::RequestLlmCall { .. }))),
            "should trigger next LLM call when all tools done and last msg is tool");
    }

    #[test]
    fn recover_no_stuck_conditions_returns_empty() {
        let state = created_state();
        let recovery = recover(&state, None);
        assert!(recovery.setup.is_empty());
        assert!(recovery.resume.is_empty());
    }

    #[test]
    fn recover_completed_llm_with_response_processed_is_not_stuck() {
        let mut state = created_state();

        let call_id = "call-1".to_string();
        apply_events(&mut state, vec![
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
        ]);

        assert!(state.llm_calls["call-1"].response_processed);

        let recovery = recover(&state, None);
        // No stuck conditions — completed LLM with message extracted, last msg is assistant (not tool)
        assert!(recovery.resume.is_empty(), "fully completed flow should have no recovery effects");
    }
}
