use std::collections::HashMap;
use std::sync::Arc;

use ractor::{call_t, Actor, ActorProcessingErr, ActorRef, RpcReplyPort, SupervisionEvent};
use uuid::Uuid;

use chrono::Utc;

use crate::domain::agent::AgentConfig;
use crate::domain::config::{BudgetPolicyConfig, EventStoreConfig, SystemConfig};
use crate::domain::event::{CompletionDelivery, ClientIdentity, SpanContext};
use crate::domain::session::{AgentState, CommandPayload, IncomingMessage, SessionCommand};
use crate::domain::aggregate::DomainEvent;
use dispatcher::spawn_aggregate_dispatcher;
use event_store::Event;
use wake_scheduler::spawn_wake_scheduler;

pub mod budget;
pub mod dispatcher;
pub mod event_store;
pub mod jsonrpc;
pub mod llm;
pub mod mcp;
mod session_actor;
pub mod session_client;
mod strategy;
pub mod wake_scheduler;

#[cfg(feature = "sqlite")]
pub use event_store::SqliteEventStore;
pub use event_store::{
    AggregateFilter, AggregateSort, AggregateSummary, EventStore, StoreError, StreamLoad, Version,
};
pub use llm::{LlmClient, MockLlmClient, OpenAiClient, StreamDelta};
pub use llm::{LlmClientFactory, LlmClientProvider, ProviderError, StaticLlmClientProvider};
pub use mcp::{
    CallToolResult, Content, McpClient, McpError, ServerCapabilities, ServerInfo, StdioMcpClient,
    ToolAnnotations, ToolDefinition,
};
pub use mcp::{McpActorClient, McpMessage, mcp_actor_name, spawn_mcp_actor};
pub use session_actor::SessionActorArgs;
pub use session_actor::{
    RuntimeError, SessionActor, SessionActorState, SessionInit, SessionMessage,
};
pub use session_client::{
    Notification, OnSessionUpdate, SessionClientActor, SessionClientArgs, SessionUpdate,
};
pub use strategy::{StaticStrategyProvider, StrategyProvider, StrategyProviderError};

// ---------------------------------------------------------------------------
// Naming conventions for ractor registry / process groups
// ---------------------------------------------------------------------------

pub fn session_actor_name(session_id: Uuid) -> String {
    format!("session-{session_id}")
}

pub fn session_group(session_id: Uuid) -> String {
    format!("session-group-{session_id}")
}

pub fn session_observer_group(session_id: Uuid) -> String {
    format!("session-observers-{session_id}")
}

/// Routing closure for the session aggregate dispatcher.
/// Broadcasts typed events to the session process group.
fn session_route(aggregate_id: Uuid, events: Vec<Arc<DomainEvent<AgentState>>>) {
    let group = session_group(aggregate_id);
    for cell in ractor::pg::get_members(&group) {
        let actor: ActorRef<SessionMessage> = cell.into();
        let _ = actor.send_message(SessionMessage::Events(events.clone()));
    }
}

/// Look up a running session actor by ID and send it a message.
/// Silently no-ops if the actor is not running.
pub fn send_to_session(session_id: Uuid, message: SessionMessage) {
    if let Some(cell) = ractor::registry::where_is(session_actor_name(session_id)) {
        let actor: ActorRef<SessionMessage> = cell.into();
        let _ = actor.send_message(message);
    }
}

/// Broadcast a transient notification to session observers only.
/// The session actor is not in this group, so it never receives its own notifications.
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
// RuntimeActor — owns providers, spawns session actors
// ---------------------------------------------------------------------------

struct RuntimeActor;

struct RuntimeState {
    myself: ActorRef<RuntimeMessage>,
    store: Arc<dyn EventStore>,
    agents: HashMap<String, AgentConfig>,
    llm_provider: Arc<dyn LlmClientProvider>,
    strategy_provider: Arc<dyn StrategyProvider>,
    budget_policies: Vec<BudgetPolicyConfig>,
}

struct RuntimeArgs {
    store: Arc<dyn EventStore>,
    agents: HashMap<String, AgentConfig>,
    llm_provider: Arc<dyn LlmClientProvider>,
    strategy_provider: Arc<dyn StrategyProvider>,
    budget_policies: Vec<BudgetPolicyConfig>,
}

impl RuntimeState {
    /// Get or spawn a budget actor for the given tenant.
    /// Returns `None` if no budget policies are configured.
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
                        .map_err(|e| {
                            RuntimeError::ActorCall(format!("mcp {}: {e}", config.name))
                        })?
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
        let actor_name = session_actor_name(session_id);

        // Reuse existing actor if already running, otherwise spawn a new one.
        let actor: ActorRef<SessionMessage> =
            if let Some(cell) = ractor::registry::where_is(actor_name.clone()) {
                cell.into()
            } else {
                let budget_actor = self.get_or_spawn_budget_actor(&auth.tenant_id).await?;
                let mcp_clients = self.get_or_spawn_mcp_actors(&init.agent).await?;
                let (actor, _actor_handle) = Actor::spawn_linked(
                    Some(actor_name),
                    SessionActor,
                    SessionActorArgs {
                        session_id,
                        init,
                        store: self.store.clone(),
                        llm_provider: self.llm_provider.clone(),
                        mcp_clients,
                        strategy_provider: self.strategy_provider.clone(),
                        agents: self.agents.clone(),
                        runtime: self.myself.clone(),
                        budget_actor,
                    },
                    self.myself.get_cell(),
                )
                .await
                .map_err(|e| RuntimeError::ActorCall(format!("session startup failed: {e}")))?;
                actor
            };

        // SessionClientActor spawned standalone, then linked to SessionActor
        // so it dies automatically when the session dies
        let (client, _client_handle) = Actor::spawn(
            None,
            SessionClientActor,
            SessionClientArgs {
                session_id,
                auth,
                session_actor: actor.clone(),
                store: self.store.clone(),
                on_event: None,
            },
        )
        .await
        .map_err(|e| RuntimeError::ActorCall(format!("session client startup failed: {e}")))?;

        client.get_cell().link(actor.get_cell());

        Ok(SessionHandle {
            session_id,
            session_client: client,
        })
    }

    async fn wake_aggregate(&self, aggregate_id: Uuid, aggregate_type: &str, tenant_id: &str) {
        match aggregate_type {
            "session" => {
                // If the session actor is already running, just send Wake.
                if let Some(cell) = ractor::registry::where_is(session_actor_name(aggregate_id)) {
                    let actor: ActorRef<SessionMessage> = cell.into();
                    let _ = actor.send_message(SessionMessage::Wake);
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

        let budget_actor = self.get_or_spawn_budget_actor(&req.auth.tenant_id).await?;
        let mcp_clients = self.get_or_spawn_mcp_actors(&agent).await?;
        let msg_span = req.span.child();

        let init = SessionInit {
            agent,
            auth: req.auth,
            on_done: Some(req.delivery),
            span: req.span,
        };

        let actor_name = session_actor_name(req.session_id);
        let session_actor: ActorRef<SessionMessage> =
            if let Some(cell) = ractor::registry::where_is(actor_name.clone()) {
                cell.into()
            } else {
                let (actor, _) = Actor::spawn_linked(
                    Some(actor_name),
                    SessionActor,
                    SessionActorArgs {
                        session_id: req.session_id,
                        init,
                        store: self.store.clone(),
                        llm_provider: self.llm_provider.clone(),
                        mcp_clients,
                        strategy_provider: self.strategy_provider.clone(),
                        agents: self.agents.clone(),
                        runtime: self.myself.clone(),
                        budget_actor,
                    },
                    self.myself.get_cell(),
                )
                .await
                .map_err(|e| RuntimeError::ActorCall(format!("sub-agent: {e}")))?;
                actor
            };

        // Send user message to the sub-agent
        let _ = call_t!(
            session_actor,
            SessionMessage::Execute,
            5000,
            SessionCommand {
                span: msg_span,
                occurred_at: Utc::now(),
                payload: CommandPayload::SendMessage {
                    message: IncomingMessage::User {
                        content: req.message,
                    },
                    stream: req.stream,
                },
            }
        );

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
            strategy_provider: args.strategy_provider,
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
        spawn_wake_scheduler(
            args.store,
            myself.clone(),
            myself.get_cell(),
        )
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
            RuntimeMessage::WakeAggregate { aggregate_id, aggregate_type, tenant_id } => {
                state.wake_aggregate(aggregate_id, &aggregate_type, &tenant_id).await;
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

/// Restart dispatcher, wake-scheduler, or budget actors if they died; sessions just get logged.
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
            // SessionActor death — children auto-terminated, just log
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
                strategy_provider: Arc::new(StaticStrategyProvider),
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
    /// Generates a fresh session ID.
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
    /// The client joins the dispatcher's process group so it receives events
    /// and keeps its projected state up to date.
    /// An optional callback is invoked for each update (event or notification).
    pub async fn connect(
        &self,
        session_id: Uuid,
        auth: ClientIdentity,
        on_event: Option<OnSessionUpdate>,
    ) -> Result<SessionHandle, RuntimeError> {
        let session_actor: ActorRef<SessionMessage> =
            ractor::registry::where_is(session_actor_name(session_id))
                .ok_or(RuntimeError::SessionNotFound)?
                .into();

        let (client, _handle) = Actor::spawn(
            None,
            SessionClientActor,
            SessionClientArgs {
                session_id,
                auth,
                session_actor: session_actor.clone(),
                store: self.store.clone(),
                on_event,
            },
        )
        .await
        .map_err(|e| RuntimeError::ActorCall(format!("session client spawn failed: {e}")))?;

        client.get_cell().link(session_actor.get_cell());

        Ok(SessionHandle {
            session_id,
            session_client: client,
        })
    }

    /// Check whether a session actor is currently running.
    pub fn session_is_running(&self, session_id: Uuid) -> bool {
        ractor::registry::where_is(session_actor_name(session_id)).is_some()
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
        EventStoreConfig::Sqlite { path } => {
            Arc::new(
                SqliteEventStore::new(path)
                    .await
                    .expect("failed to open SQLite event store"),
            )
        }
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
