use std::collections::HashMap;
use std::sync::Arc;

use ractor::{call_t, Actor, ActorProcessingErr, ActorRef, RpcReplyPort, SupervisionEvent};
use uuid::Uuid;

use chrono::Utc;

use crate::domain::agent::AgentConfig;
use crate::domain::aggregate::DomainEvent;
use crate::domain::config::{BudgetPolicyConfig, EventStoreConfig, SystemConfig};
use crate::domain::event::{ClientIdentity, CompletionDelivery, SpanContext};
use crate::domain::session::{
    AgentState, BudgetActorRef, CommandPayload, IncomingMessage, McpToolEntry, SessionCommand,
    SessionContext, SessionStatus,
};
use dispatcher::spawn_aggregate_dispatcher;
use event_store::Event;
use wake_scheduler::spawn_wake_scheduler;

pub mod budget;
pub mod dispatcher;
pub mod event_store;
pub mod aggregate_actor;
pub mod jsonrpc;
pub mod llm;
pub mod mcp;
pub mod session_client;
pub mod wake_scheduler;

#[cfg(feature = "sqlite")]
pub use event_store::SqliteEventStore;
pub use event_store::{
    AggregateFilter, AggregateSort, AggregateSummary, EventStore, StoreError, StreamLoad, Version,
};
pub use llm::{LlmClient, MockLlmClient, OpenAiClient, StreamDelta};
pub use llm::{LlmClientFactory, LlmClientProvider, ProviderError, StaticLlmClientProvider};
pub use mcp::{mcp_actor_name, spawn_mcp_actor, McpActorClient, McpMessage};
pub use mcp::{
    CallToolResult, Content, McpClient, McpError, ServerCapabilities, ServerInfo, StdioMcpClient,
    ToolAnnotations, ToolDefinition,
};
pub use session_client::{
    Notification, OnSessionUpdate, SessionClientActor, SessionClientArgs, SessionUpdate,
};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error(transparent)]
    Session(#[from] crate::domain::session::SessionError),
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
}

// ---------------------------------------------------------------------------
// SessionMessage — used by session clients and process group routing
// ---------------------------------------------------------------------------

pub enum SessionMessage {
    Execute(
        SessionCommand,
        RpcReplyPort<Result<Vec<Arc<Event>>, RuntimeError>>,
    ),
    Cast(SessionCommand),
    GetState(RpcReplyPort<AgentState>),
    Events(Vec<Arc<DomainEvent<AgentState>>>),
    /// Timer-triggered or scheduler-triggered wake.
    Wake,
    /// Cancel this session (used by parent to cancel sub-agent).
    Cancel,
    /// Set client-provided tools (from AG-UI RunAgentInput).
    SetClientTools(Vec<crate::domain::openai::Tool>),
    /// Transient notification — broadcast to observers, never persisted.
    Notify(Arc<Notification>),
}

// ---------------------------------------------------------------------------
// SessionInit — what the runtime needs to start a session
// ---------------------------------------------------------------------------

pub struct SessionInit {
    pub agent: AgentConfig,
    pub auth: ClientIdentity,
    pub on_done: Option<CompletionDelivery>,
    pub span: SpanContext,
}

// ---------------------------------------------------------------------------
// Naming conventions for ractor registry / process groups
// ---------------------------------------------------------------------------

pub fn aggregate_actor_name(session_id: Uuid) -> String {
    format!("session-{session_id}")
}

pub fn session_group(session_id: Uuid) -> String {
    format!("session-group-{session_id}")
}

pub fn session_observer_group(session_id: Uuid) -> String {
    format!("session-observers-{session_id}")
}

/// Routing closure for the session aggregate dispatcher.
/// Broadcasts typed events to the session process group and aggregate actor.
fn session_route(aggregate_id: Uuid, events: Vec<Arc<DomainEvent<AgentState>>>) {
    let group = session_group(aggregate_id);
    for cell in ractor::pg::get_members(&group) {
        let actor: ActorRef<SessionMessage> = cell.into();
        let _ = actor.send_message(SessionMessage::Events(events.clone()));
    }

    if let Some(cell) = ractor::registry::where_is(aggregate_actor_name(aggregate_id)) {
        let actor: ActorRef<aggregate_actor::AggregateMessage<AgentState>> = cell.into();
        let _ = actor.send_message(aggregate_actor::AggregateMessage::Events(events));
    }
}

/// Broadcast a transient notification to session observers only.
pub fn notify_observers(session_id: Uuid, notification: Arc<Notification>) {
    let group = session_observer_group(session_id);
    for cell in ractor::pg::get_members(&group) {
        let actor: ActorRef<SessionMessage> = cell.into();
        let _ = actor.send_message(SessionMessage::Notify(Arc::clone(&notification)));
    }
}

// ---------------------------------------------------------------------------
// RuntimeMessage — commands handled by the RuntimeActor
// ---------------------------------------------------------------------------

pub enum RuntimeMessage {
    StartSession(
        Uuid,
        SessionInit,
        RpcReplyPort<Result<SessionHandle, RuntimeError>>,
    ),
    RunSubAgent(SubAgentRequest),
    WakeAggregate {
        aggregate_id: Uuid,
        aggregate_type: String,
        tenant_id: String,
    },
    /// Find-or-start the aggregate actor, then deliver a command.
    DeliverToSession {
        session_id: Uuid,
        payload: CommandPayload,
        span: SpanContext,
    },
}

pub struct SubAgentRequest {
    pub session_id: Uuid,
    pub agent_name: String,
    pub message: String,
    pub auth: ClientIdentity,
    pub delivery: CompletionDelivery,
    pub span: SpanContext,
    pub token_budget: Option<u64>,
    pub stream: bool,
}

// ---------------------------------------------------------------------------
// RuntimeActor — owns providers, spawns aggregate actors directly
// ---------------------------------------------------------------------------

struct RuntimeActor;

struct RuntimeState {
    myself: ActorRef<RuntimeMessage>,
    store: Arc<dyn EventStore>,
    agents: HashMap<String, AgentConfig>,
    llm_provider: Arc<dyn LlmClientProvider>,
    budget_policies: Vec<BudgetPolicyConfig>,
}

struct RuntimeArgs {
    store: Arc<dyn EventStore>,
    agents: HashMap<String, AgentConfig>,
    llm_provider: Arc<dyn LlmClientProvider>,
    budget_policies: Vec<BudgetPolicyConfig>,
}

// ---------------------------------------------------------------------------
// Build SessionContext — wires runtime resources into the domain context
// ---------------------------------------------------------------------------

fn build_session_context(
    session_id: Uuid,
    auth: &ClientIdentity,
    mcp_clients: &[Arc<dyn McpClient>],
    llm_provider: &Arc<dyn LlmClientProvider>,
    agents: &HashMap<String, AgentConfig>,
    agent: Option<&AgentConfig>,
    budget_actor: Option<ActorRef<budget::BudgetMessage>>,
    stream: bool,
) -> SessionContext {
    let mcp_tools: HashMap<String, McpToolEntry> = mcp_clients
        .iter()
        .flat_map(|c| {
            let info = c.server_info();
            let server_name = info.name.clone();
            let server_version = info.version.clone();
            c.tools().iter().map(move |t| {
                (
                    t.name.clone(),
                    McpToolEntry {
                        server_name: server_name.clone(),
                        server_version: server_version.clone(),
                    },
                )
            })
        })
        .collect();

    // Build all_tools from MCP + sub-agents + client tools (client tools added later via UpdateContext)
    let mut tools: Vec<crate::domain::openai::Tool> = mcp_clients
        .iter()
        .flat_map(|c| c.tools().iter().map(|t| t.to_openai_tool()))
        .collect();

    // Add sub-agent tools
    if let Some(agent) = agent {
        for name in &agent.sub_agents {
            if let Some(sub) = agents.get(name) {
                let tool_name = mcp::ToolDefinition::sanitized_name(name);
                tools.push(crate::domain::openai::Tool {
                    tool_type: "function".to_string(),
                    function: crate::domain::openai::ToolFunction {
                        name: tool_name,
                        description: sub.description.clone().unwrap_or_else(|| sub.name.clone()),
                        parameters: serde_json::json!({
                            "type": "object",
                            "properties": {
                                "message": {
                                    "type": "string",
                                    "description": "The message to send to the sub-agent"
                                }
                            },
                            "required": ["message"]
                        }),
                    },
                });
            }
        }
    }

    let all_tools = if tools.is_empty() { None } else { Some(tools) };

    // Wrap the LlmClientProvider in our domain-level trait
    let llm_adapter: Arc<dyn crate::domain::session::LlmClientTrait> =
        Arc::new(LlmProviderAdapter(llm_provider.clone()));

    // Wrap MCP clients
    let mcp_adapters: Vec<Arc<dyn crate::domain::session::McpClientTrait>> = mcp_clients
        .iter()
        .map(|c| Arc::new(McpClientAdapter(c.clone())) as Arc<dyn crate::domain::session::McpClientTrait>)
        .collect();

    // Wrap budget actor
    let budget_ref = budget_actor.map(|a| BudgetActorRef {
        inner: Box::new(a),
    });

    // Build callbacks
    let notify_chunk: crate::domain::session::NotifyChunkFn = Arc::new(
        |session_id, call_id, chunk_index, text| {
            notify_observers(
                session_id,
                Arc::new(Notification::LlmStreamChunk {
                    call_id,
                    chunk_index,
                    text,
                }),
            );
        },
    );

    // send_to_session is set below after we have the runtime ref

    SessionContext {
        mcp_tools,
        all_tools,
        session_id,
        auth: auth.clone(),
        stream,
        llm_provider: Some(llm_adapter),
        mcp_clients: mcp_adapters,
        agents: agents.clone(),
        client_tools: Vec::new(),
        budget_actor: budget_ref,
        notify_chunk: Some(notify_chunk),
        send_to_session: None, // set below after we have runtime ref
        spawn_sub_agent: None, // set below after we have runtime ref
    }
}

// ---------------------------------------------------------------------------
// Adapter: LlmClientProvider → domain LlmClientTrait
// ---------------------------------------------------------------------------

struct LlmProviderAdapter(Arc<dyn LlmClientProvider>);

impl crate::domain::session::LlmClientTrait for LlmProviderAdapter {
    fn resolve<'a>(
        &'a self,
        client_id: &'a str,
        auth: &'a ClientIdentity,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<Arc<dyn crate::domain::session::LlmCallable>, String>,
                > + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            let client = self
                .0
                .resolve(client_id, auth)
                .await
                .map_err(|e| e.to_string())?;
            Ok(Arc::new(LlmClientAdapter(client)) as Arc<dyn crate::domain::session::LlmCallable>)
        })
    }
}

struct LlmClientAdapter(Arc<dyn LlmClient>);

impl crate::domain::session::LlmCallable for LlmClientAdapter {
    fn call<'a>(
        &'a self,
        request: &'a crate::domain::event::LlmRequest,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<
                        crate::domain::event::LlmResponse,
                        crate::domain::session::LlmCallError,
                    >,
                > + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            self.0.call(request).await.map_err(|e| {
                crate::domain::session::LlmCallError {
                    message: e.message,
                    retryable: e.retryable,
                    source: serde_json::to_value(&e.source).ok(),
                }
            })
        })
    }

    fn call_streaming<'a>(
        &'a self,
        request: &'a crate::domain::event::LlmRequest,
        tx: tokio::sync::mpsc::UnboundedSender<crate::domain::session::StreamDelta>,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<
                        crate::domain::event::LlmResponse,
                        crate::domain::session::LlmCallError,
                    >,
                > + Send
                + 'a,
        >,
    > {
        // Bridge the runtime StreamDelta to domain StreamDelta
        let (bridge_tx, mut bridge_rx) =
            tokio::sync::mpsc::unbounded_channel::<llm::StreamDelta>();

        Box::pin(async move {
            let forward = tokio::spawn(async move {
                while let Some(delta) = bridge_rx.recv().await {
                    let _ = tx.send(crate::domain::session::StreamDelta {
                        text: delta.text,
                    });
                }
            });

            let result = self.0.call_streaming(request, bridge_tx).await.map_err(|e| {
                crate::domain::session::LlmCallError {
                    message: e.message,
                    retryable: e.retryable,
                    source: serde_json::to_value(&e.source).ok(),
                }
            });

            forward.abort();
            result
        })
    }
}

// ---------------------------------------------------------------------------
// Adapter: McpClient → domain McpClientTrait
// ---------------------------------------------------------------------------

struct McpClientAdapter(Arc<dyn McpClient>);

impl crate::domain::session::McpClientTrait for McpClientAdapter {
    fn server_info(&self) -> crate::domain::session::McpServerInfo {
        let info = self.0.server_info();
        crate::domain::session::McpServerInfo {
            name: info.name.clone(),
            version: info.version.clone(),
        }
    }

    fn tools(&self) -> Vec<crate::domain::session::McpToolDefinition> {
        self.0
            .tools()
            .iter()
            .map(|t| crate::domain::session::McpToolDefinition {
                name: t.name.clone(),
            })
            .collect()
    }

    fn call_tool<'a>(
        &'a self,
        name: &'a str,
        arguments: serde_json::Value,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<crate::domain::session::McpToolResult, String>,
                > + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            let result = self.0.call_tool(name, arguments).await.map_err(|e| e.to_string())?;
            Ok(crate::domain::session::McpToolResult {
                content: result
                    .content
                    .iter()
                    .map(|c| match c {
                        Content::Text { text } => {
                            crate::domain::session::McpToolContent::Text { text: text.clone() }
                        }
                        _ => crate::domain::session::McpToolContent::Other,
                    })
                    .collect(),
                is_error: result.is_error,
            })
        })
    }
}

// ---------------------------------------------------------------------------
// RuntimeState methods
// ---------------------------------------------------------------------------

impl RuntimeState {
    /// Look up a running aggregate actor by session ID and send a command.
    /// Returns `true` if the actor was found and the message was sent.
    fn try_send_to_aggregate(&self, session_id: Uuid, payload: CommandPayload, span: SpanContext) -> bool {
        if let Some(cell) = ractor::registry::where_is(aggregate_actor_name(session_id)) {
            let actor: ActorRef<aggregate_actor::AggregateMessage<AgentState>> = cell.into();
            let _ = actor.send_message(aggregate_actor::AggregateMessage::Cast {
                cmd: payload,
                span,
                occurred_at: Utc::now(),
            });
            true
        } else {
            false
        }
    }

    /// Get or spawn a budget actor for the given tenant.
    async fn get_or_spawn_budget_actor(
        &self,
        tenant_id: &str,
    ) -> Result<Option<ActorRef<budget::BudgetMessage>>, RuntimeError> {
        if self.budget_policies.is_empty() {
            return Ok(None);
        }
        let actor_name = budget::budget_actor_name(tenant_id);
        if let Some(cell) = ractor::registry::where_is(actor_name) {
            return Ok(Some(cell.into()));
        }
        let actor = budget::spawn_budget_actor(
            tenant_id.to_string(),
            self.budget_policies.clone(),
            self.store.clone(),
            self.myself.get_cell(),
        )
        .await
        .map_err(|e| RuntimeError::ActorCall(format!("budget: {e}")))?;
        Ok(Some(actor))
    }

    #[tracing::instrument(skip(self, agent), fields(agent = %agent.name))]
    async fn get_or_spawn_mcp_actors(
        &self,
        agent: &AgentConfig,
    ) -> Result<Vec<Arc<dyn McpClient>>, RuntimeError> {
        if agent.mcp_servers.is_empty() {
            return Ok(Vec::new());
        }
        let mut clients: Vec<Arc<dyn McpClient>> = Vec::with_capacity(agent.mcp_servers.len());
        for config in &agent.mcp_servers {
            let name = mcp::mcp_actor_name(&agent.name, &config.name);
            let actor_ref: ActorRef<mcp::McpMessage> =
                if let Some(cell) = ractor::registry::where_is(name) {
                    cell.into()
                } else {
                    mcp::spawn_mcp_actor(&agent.name, config.clone(), self.myself.get_cell())
                        .await
                        .map_err(|e| RuntimeError::ActorCall(format!("mcp {}: {e}", config.name)))?
                };
            let client = mcp::McpActorClient::from_actor(actor_ref)
                .await
                .map_err(|e| RuntimeError::ActorCall(format!("mcp {}: {e}", config.name)))?;
            clients.push(Arc::new(client));
        }
        Ok(clients)
    }

    #[tracing::instrument(skip(self, init), fields(%session_id))]
    async fn start_session(
        &self,
        session_id: Uuid,
        init: SessionInit,
    ) -> Result<SessionHandle, RuntimeError> {
        let auth = init.auth.clone();
        let agent = init.agent.clone();
        let actor_name = aggregate_actor_name(session_id);
        let already_running = ractor::registry::where_is(actor_name).is_some();

        if !already_running {
            let budget_actor = self.get_or_spawn_budget_actor(&auth.tenant_id).await?;
            let mcp_clients = self.get_or_spawn_mcp_actors(&init.agent).await?;

            let llm_provider = self.llm_provider.clone();
            let agents = self.agents.clone();
            let runtime_ref = self.myself.clone();
            let mcp_for_ctx = mcp_clients.clone();
            let auth_for_ctx = auth.clone();
            let agent_for_ctx = agent.clone();

            let aggregate_handle = aggregate_actor::spawn_aggregate_actor(
                aggregate_actor::AggregateActorArgs {
                    aggregate_id: session_id,
                    store: self.store.clone(),
                    tenant_id: auth.tenant_id.clone(),
                    init: Box::new(AgentState::new),
                    context_init: Box::new(move |state| {
                        let resolved_agent = state.agent.clone().unwrap_or(agent_for_ctx);
                        Box::pin(async move {
                            let mut ctx = build_session_context(
                                session_id,
                                &auth_for_ctx,
                                &mcp_for_ctx,
                                &llm_provider,
                                &agents,
                                Some(&resolved_agent),
                                budget_actor,
                                false,
                            );
                            // Wire up send_to_session (find-or-start via runtime)
                            let runtime_for_send = runtime_ref.clone();
                            ctx.send_to_session = Some(Arc::new(move |session_id, payload, span| {
                                let _ = runtime_for_send.send_message(
                                    RuntimeMessage::DeliverToSession { session_id, payload, span },
                                );
                            }));
                            // Wire up sub-agent spawning
                            let runtime = runtime_ref.clone();
                            ctx.spawn_sub_agent = Some(Arc::new(move |params| {
                                let _ = runtime.send_message(RuntimeMessage::RunSubAgent(
                                    SubAgentRequest {
                                        session_id: params.session_id,
                                        agent_name: params.agent_name,
                                        message: params.message,
                                        auth: params.auth,
                                        delivery: params.delivery,
                                        span: params.span,
                                        token_budget: params.token_budget,
                                        stream: params.stream,
                                    },
                                ));
                            }));
                            ctx
                        })
                    }),
                },
                self.myself.get_cell(),
            )
            .await
            .map_err(|e| RuntimeError::ActorCall(format!("aggregate actor spawn: {e}")))?;

            // Check if session needs creation
            let session = aggregate_handle.get_aggregate().await;
            let is_new = session.state.agent.is_none();

            if is_new {
                aggregate_handle
                    .send_command(
                        CommandPayload::CreateSession {
                            agent: init.agent,
                            auth: auth.clone(),
                            on_done: init.on_done,
                        },
                        init.span.child(),
                        Utc::now(),
                    )
                    .await
                    .map_err(|e| RuntimeError::ActorCall(format!("create session: {e}")))?;
            }

            // If resuming a completed sub-agent, deliver result to parent.
            // Deferred via message queue to avoid start_session ↔ wake_aggregate recursion.
            if session.state.status == SessionStatus::Done {
                if let Some(ref delivery) = session.state.on_done {
                    let result =
                        serde_json::to_string(&session.state.artifacts).unwrap_or_default();
                    let _ = self.myself.send_message(RuntimeMessage::DeliverToSession {
                        session_id: delivery.parent_session_id,
                        payload: CommandPayload::CompleteToolCall {
                            tool_call_id: delivery.tool_call_id.clone(),
                            name: delivery.tool_name.clone(),
                            result,
                        },
                        span: init.span.child(),
                    });
                }
            }
        }

        // Spawn a SessionClientActor for the caller
        let (client, _client_handle) = Actor::spawn(
            None,
            SessionClientActor,
            SessionClientArgs {
                session_id,
                auth,
                aggregate_actor_id: session_id,
                store: self.store.clone(),
                on_event: None,
            },
        )
        .await
        .map_err(|e| RuntimeError::ActorCall(format!("session client startup failed: {e}")))?;

        Ok(SessionHandle {
            session_id,
            session_client: client,
        })
    }

    async fn wake_aggregate(&self, aggregate_id: Uuid, aggregate_type: &str, tenant_id: &str) {
        match aggregate_type {
            "session" => {
                // If the aggregate actor is already running, send Wake command
                if let Some(cell) = ractor::registry::where_is(aggregate_actor_name(aggregate_id)) {
                    let actor: ActorRef<aggregate_actor::AggregateMessage<AgentState>> =
                        cell.into();
                    let _ = actor.send_message(aggregate_actor::AggregateMessage::Cast {
                        cmd: CommandPayload::Wake,
                        span: SpanContext::root(),
                        occurred_at: Utc::now(),
                    });
                    return;
                }
                // Not running — look up the agent name via list_aggregates and start the session.
                let filter = AggregateFilter {
                    aggregate_ids: Some(vec![aggregate_id]),
                    ..Default::default()
                };
                let results = self.store.list_aggregates(&filter).await;
                let summary = match results.into_iter().next() {
                    Some(s) => s,
                    None => {
                        tracing::warn!(%aggregate_id, "wake: session not found in store");
                        return;
                    }
                };
                let agent_name = match summary.label {
                    Some(name) => name,
                    None => {
                        tracing::warn!(%aggregate_id, "wake: session has no agent label");
                        return;
                    }
                };
                let agent = match self.agents.get(&agent_name) {
                    Some(a) => a.clone(),
                    None => {
                        tracing::warn!(agent = %agent_name, session = %aggregate_id, "wake: unknown agent");
                        return;
                    }
                };
                let auth = ClientIdentity {
                    tenant_id: tenant_id.to_string(),
                    sub: None,
                    attrs: Default::default(),
                };
                let init = SessionInit {
                    agent,
                    auth,
                    on_done: None,
                    span: SpanContext::root(),
                };
                if let Err(e) = self.start_session(aggregate_id, init).await {
                    tracing::warn!(session = %aggregate_id, error = %e, "wake: failed to start session");
                }
            }
            _ => {
                tracing::debug!(aggregate_type = %aggregate_type, %aggregate_id, "wake: no handler for aggregate type");
            }
        }
    }

    async fn run_sub_agent(&self, req: SubAgentRequest) -> Result<(), RuntimeError> {
        let mut agent = self
            .agents
            .get(&req.agent_name)
            .cloned()
            .ok_or_else(|| RuntimeError::UnknownAgent(req.agent_name.clone()))?;

        if let Some(budget) = req.token_budget {
            agent.token_budget = Some(budget);
        }

        let msg_span = req.span.child();

        let init = SessionInit {
            agent,
            auth: req.auth,
            on_done: Some(req.delivery),
            span: req.span,
        };

        // start_session spawns the aggregate actor and creates the session
        let handle = self.start_session(req.session_id, init).await?;

        // Send user message to the sub-agent
        let _ = handle
            .send_command(SessionCommand {
                span: msg_span,
                occurred_at: Utc::now(),
                payload: CommandPayload::SendMessage {
                    message: IncomingMessage::User {
                        content: req.message,
                    },
                    stream: req.stream,
                },
            })
            .await;

        Ok(())
    }
}

impl Actor for RuntimeActor {
    type Msg = RuntimeMessage;
    type State = RuntimeState;
    type Arguments = RuntimeArgs;

    async fn pre_start(
        &self,
        myself: ActorRef<Self::Msg>,
        args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        let state = RuntimeState {
            myself: myself.clone(),
            store: args.store.clone(),
            agents: args.agents,
            llm_provider: args.llm_provider,
            budget_policies: args.budget_policies,
        };

        // Spawn infrastructure actors (linked to RuntimeActor)
        tracing::debug!("spawning event dispatcher");
        spawn_aggregate_dispatcher::<AgentState>(
            &args.store,
            Arc::new(session_route),
            myself.get_cell(),
        )
        .await
        .map_err(|e| format!("failed to spawn dispatcher: {e}"))?;
        tracing::debug!("spawning wake scheduler");
        spawn_wake_scheduler(args.store, myself.clone(), myself.get_cell())
            .await
            .map_err(|e| format!("failed to spawn wake scheduler: {e}"))?;

        Ok(state)
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match message {
            RuntimeMessage::StartSession(session_id, init, reply) => {
                let result = state.start_session(session_id, init).await;
                let _ = reply.send(result);
            }
            RuntimeMessage::RunSubAgent(req) => {
                if let Err(e) = state.run_sub_agent(req).await {
                    tracing::error!(error = %e, "sub-agent error");
                }
            }
            RuntimeMessage::WakeAggregate {
                aggregate_id,
                aggregate_type,
                tenant_id,
            } => {
                state
                    .wake_aggregate(aggregate_id, &aggregate_type, &tenant_id)
                    .await;
            }
            RuntimeMessage::DeliverToSession {
                session_id,
                payload,
                span,
            } => {
                // Find-or-start: wake the aggregate if needed, then deliver
                state.wake_aggregate(session_id, "session", "").await;
                state.try_send_to_aggregate(session_id, payload, span);
            }
        }
        Ok(())
    }

    async fn handle_supervisor_evt(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: SupervisionEvent,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match &message {
            SupervisionEvent::ActorFailed(who, err) => {
                let name = who.get_name();
                tracing::error!(actor = ?name, error = %err, "child actor failed");
                restart_infrastructure(name, state).await;
            }
            SupervisionEvent::ActorTerminated(who, _, reason) => {
                let name = who.get_name();
                tracing::error!(actor = ?name, reason = ?reason, "child actor terminated");
                restart_infrastructure(name, state).await;
            }
            _ => {}
        }
        Ok(())
    }
}

/// Restart dispatcher, wake-scheduler, or budget actors if they died.
async fn restart_infrastructure(name: Option<String>, state: &RuntimeState) {
    match name.as_deref() {
        Some("session-dispatcher") => {
            tracing::info!("restarting session-dispatcher");
            if let Err(e) = spawn_aggregate_dispatcher::<AgentState>(
                &state.store,
                Arc::new(session_route),
                state.myself.get_cell(),
            )
            .await
            {
                tracing::error!(error = %e, "failed to restart session-dispatcher");
            }
        }
        Some("wake-scheduler") => {
            tracing::info!("restarting wake-scheduler");
            if let Err(e) = spawn_wake_scheduler(
                state.store.clone(),
                state.myself.clone(),
                state.myself.get_cell(),
            )
            .await
            {
                tracing::error!(error = %e, "failed to restart wake-scheduler");
            }
        }
        Some(name) if name.starts_with("mcp-") => {
            tracing::info!(server = %name, "MCP actor died, will re-spawn on next use");
        }
        Some(name) if name.starts_with("budget-") => {
            let tenant_id = &name["budget-".len()..];
            tracing::info!(tenant = %tenant_id, "restarting budget actor");
            if let Err(e) = budget::spawn_budget_actor(
                tenant_id.to_string(),
                state.budget_policies.clone(),
                state.store.clone(),
                state.myself.get_cell(),
            )
            .await
            {
                tracing::error!(error = %e, "failed to restart budget actor");
            }
        }
        _ => {
            // Aggregate actor death — just log
        }
    }
}

// ---------------------------------------------------------------------------
// Runtime — thin wrapper for the RuntimeActor
// ---------------------------------------------------------------------------

pub struct Runtime {
    actor: ActorRef<RuntimeMessage>,
    store: Arc<dyn EventStore>,
    agents: HashMap<String, AgentConfig>,
}

impl Runtime {
    pub async fn start(config: &SystemConfig) -> Result<Self, RuntimeError> {
        let store = create_event_store(&config.event_store).await;

        let llm_provider =
            StaticLlmClientProvider::from_config(&config.llm_clients, &default_llm_factories())
                .map_err(RuntimeError::ActorCall)?;

        let (actor, _handle) = Actor::spawn(
            Some("runtime".to_string()),
            RuntimeActor,
            RuntimeArgs {
                store: store.clone(),
                agents: config.agents.clone(),
                llm_provider: Arc::new(llm_provider),
                budget_policies: config.budgets.clone(),
            },
        )
        .await
        .map_err(|e| RuntimeError::ActorCall(e.to_string()))?;

        tracing::info!(
            agents = config.agents.len(),
            budgets = config.budgets.len(),
            "runtime started",
        );

        Ok(Runtime {
            actor,
            store,
            agents: config.agents.clone(),
        })
    }

    /// Look up an agent definition by name.
    pub fn agent(&self, name: &str) -> Option<&AgentConfig> {
        self.agents.get(name)
    }

    /// Create a new session for a named agent from the config.
    pub async fn create_session_for(
        &self,
        agent_name: &str,
        auth: ClientIdentity,
    ) -> Result<SessionHandle, RuntimeError> {
        self.start_session(Uuid::new_v4(), agent_name, auth).await
    }

    /// Start a session — resumes from store if it exists, otherwise creates new.
    pub async fn start_session(
        &self,
        session_id: Uuid,
        agent_name: &str,
        auth: ClientIdentity,
    ) -> Result<SessionHandle, RuntimeError> {
        let agent = self
            .agents
            .get(agent_name)
            .cloned()
            .ok_or_else(|| RuntimeError::UnknownAgent(agent_name.to_string()))?;

        let init = SessionInit {
            agent,
            auth,
            on_done: None,
            span: SpanContext::root(),
        };
        call_t!(
            self.actor,
            RuntimeMessage::StartSession,
            30_000,
            session_id,
            init
        )
        .map_err(|e| RuntimeError::ActorCall(e.to_string()))?
    }

    /// Spawn a new session client for an existing session.
    pub async fn connect(
        &self,
        session_id: Uuid,
        auth: ClientIdentity,
        on_event: Option<OnSessionUpdate>,
    ) -> Result<SessionHandle, RuntimeError> {
        // Verify aggregate actor is running
        if ractor::registry::where_is(aggregate_actor_name(session_id)).is_none() {
            return Err(RuntimeError::SessionNotFound);
        }

        let (client, _handle) = Actor::spawn(
            None,
            SessionClientActor,
            SessionClientArgs {
                session_id,
                auth,
                aggregate_actor_id: session_id,
                store: self.store.clone(),
                on_event,
            },
        )
        .await
        .map_err(|e| RuntimeError::ActorCall(format!("session client spawn failed: {e}")))?;

        Ok(SessionHandle {
            session_id,
            session_client: client,
        })
    }

    /// Check whether a session is currently running.
    pub fn session_is_running(&self, session_id: Uuid) -> bool {
        ractor::registry::where_is(aggregate_actor_name(session_id)).is_some()
    }

    pub fn store(&self) -> &Arc<dyn EventStore> {
        &self.store
    }

    /// Return the names of all configured agents.
    pub fn agent_names(&self) -> Vec<&str> {
        self.agents.keys().map(|s| s.as_str()).collect()
    }

    pub fn shutdown(self) {
        self.actor.stop(None);
    }
}

// ---------------------------------------------------------------------------
// SessionHandle — per-session interface
// ---------------------------------------------------------------------------

pub struct SessionHandle {
    pub session_id: Uuid,
    session_client: ActorRef<SessionMessage>,
}

impl SessionHandle {
    pub async fn send_command(&self, cmd: SessionCommand) -> Result<Vec<Arc<Event>>, RuntimeError> {
        let result = call_t!(self.session_client, SessionMessage::Execute, 5000, cmd)
            .map_err(|e| RuntimeError::ActorCall(e.to_string()))?;
        result
    }

    pub async fn get_state(&self) -> AgentState {
        call_t!(self.session_client, SessionMessage::GetState, 5000)
            .expect("failed to query session state")
    }

    pub fn shutdown(self) {
        self.session_client.stop(None);
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

async fn create_event_store(config: &EventStoreConfig) -> Arc<dyn EventStore> {
    match config {
        #[cfg(feature = "sqlite")]
        EventStoreConfig::Sqlite { path } => Arc::new(
            SqliteEventStore::new(path)
                .await
                .expect("failed to open SQLite event store"),
        ),
        #[cfg(not(feature = "sqlite"))]
        EventStoreConfig::Sqlite { .. } => {
            panic!("SQLite event store requires the 'sqlite' feature flag")
        }
    }
}

fn default_llm_factories() -> HashMap<String, LlmClientFactory> {
    let mut m: HashMap<String, LlmClientFactory> = HashMap::new();
    m.insert(
        "openai_compatible".into(),
        Box::new(OpenAiClient::from_config),
    );
    m.insert("mock".into(), Box::new(MockLlmClient::from_config));
    m
}
