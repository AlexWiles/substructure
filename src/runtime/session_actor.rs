use std::sync::Arc;

use ractor::{Actor, ActorProcessingErr, ActorRef, RpcReplyPort};
use rand::RngExt;
use uuid::Uuid;

use crate::domain::event::*;
use crate::domain::session::{Effect, SessionCommand, SessionError, SessionState, react};
use super::llm::LlmClient;
use super::mcp::{McpClient, StdioMcpClient, Content};
use super::event_store::{EventStore, InMemoryEventStore, StoreError, Version};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error(transparent)]
    Session(#[from] SessionError),
    #[error(transparent)]
    Store(#[from] StoreError),
}

// ---------------------------------------------------------------------------
// SessionMessage — commands from API + events from dispatcher
// ---------------------------------------------------------------------------

pub enum SessionMessage {
    Execute(
        SpanContext,
        SessionCommand,
        RpcReplyPort<Result<Vec<Event>, RuntimeError>>,
    ),
    Cast(SpanContext, SessionCommand),
    GetState(RpcReplyPort<SessionState>),
    Events(Vec<Arc<Event>>),
}

// ---------------------------------------------------------------------------
// SessionActor — merged session + reactor
// ---------------------------------------------------------------------------

pub struct SessionActor;

pub struct SessionActorState {
    pub session_id: Uuid,
    pub session: SessionState,
    pub store: Arc<InMemoryEventStore>,
    pub client: Arc<dyn LlmClient>,
    pub mcp_clients: Vec<Arc<dyn McpClient>>,
}

impl SessionActorState {
    pub fn new(
        session_id: Uuid,
        store: Arc<InMemoryEventStore>,
        client: Arc<dyn LlmClient>,
    ) -> Self {
        SessionActorState {
            session_id,
            session: SessionState::new(session_id),
            store,
            client,
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
    span: SpanContext,
    cmd: SessionCommand,
) -> Result<Vec<Event>, RuntimeError> {
    let payloads = actor_state.session.handle(cmd)?;
    if payloads.is_empty() {
        return Ok(vec![]);
    }

    // Use local stream_version — not store.version() — so the version check
    // catches writes from other nodes that this actor hasn't seen yet.
    let version = Version(actor_state.session.stream_version);
    let events = actor_state
        .store
        .append(actor_state.session_id, version, span, payloads)
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

fn child_span(parent: &SpanContext) -> SpanContext {
    SpanContext {
        trace_id: parent.trace_id,
        span_id: rand::rng().random(),
        parent_span_id: Some(parent.span_id),
        trace_flags: parent.trace_flags,
        trace_state: parent.trace_state.clone(),
    }
}

async fn handle_effect(
    state: &mut SessionActorState,
    myself: &ActorRef<SessionMessage>,
    span: &SpanContext,
    effect: Effect,
) {
    match effect {
        Effect::Command(cmd) => {
            let _ = myself.send_message(SessionMessage::Cast(child_span(span), cmd));
        }
        Effect::StartMcpServers(configs) => {
            start_mcp_servers(state, &configs).await;
        }
        Effect::CallLlm { call_id, request } => {
            let client = Arc::clone(&state.client);
            let myself = myself.clone();
            let span = child_span(span);
            tokio::spawn(async move {
                let cmd = match client.call(&request).await {
                    Ok(response) => SessionCommand::CompleteLlmCall { call_id, response },
                    Err(error) => SessionCommand::FailLlmCall { call_id, error },
                };
                let _ = myself.send_message(SessionMessage::Cast(span, cmd));
            });
        }
        Effect::CallMcpTool {
            tool_call_id,
            name,
            arguments,
        } => {
            let mcp = state.find_mcp_client(&name).cloned();
            if let Some(mcp) = mcp {
                let myself = myself.clone();
                let span = child_span(span);
                tokio::spawn(async move {
                    let cmd = match mcp.call_tool(&name, arguments).await {
                        Ok(result) => {
                            let text = result
                                .content
                                .iter()
                                .map(|c| match c {
                                    Content::Text { text } => text.as_str(),
                                })
                                .collect::<Vec<_>>()
                                .join("\n");
                            if result.is_error {
                                SessionCommand::FailToolCall {
                                    tool_call_id,
                                    name,
                                    error: text,
                                }
                            } else {
                                SessionCommand::CompleteToolCall {
                                    tool_call_id,
                                    name,
                                    result: text,
                                }
                            }
                        }
                        Err(e) => SessionCommand::FailToolCall {
                            tool_call_id,
                            name,
                            error: e.to_string(),
                        },
                    };
                    let _ = myself.send_message(SessionMessage::Cast(span, cmd));
                });
            }
            // else: no MCP client — client-side tool, handled via event fan-out
        }
    }
}

async fn start_mcp_servers(state: &mut SessionActorState, configs: &[McpServerConfig]) {
    for config in configs {
        match &config.transport {
            McpTransportConfig::Stdio { command, args } => {
                match StdioMcpClient::new(command, args).await {
                    Ok(client) => state.mcp_clients.push(Arc::new(client)),
                    Err(e) => eprintln!("MCP '{}' failed: {}", config.name, e),
                }
            }
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
            SessionMessage::Execute(span, cmd, reply) => {
                let result = execute(state, span, cmd).await;
                let _ = reply.send(result);
            }
            SessionMessage::Cast(span, cmd) => {
                if let Err(e) = execute(state, span, cmd).await {
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
        }
        Ok(())
    }
}
