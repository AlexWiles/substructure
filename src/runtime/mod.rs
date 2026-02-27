use std::collections::HashMap;
use std::sync::Arc;

use ractor::{call_t, Actor, ActorProcessingErr, ActorRef, RpcReplyPort, SupervisionEvent};
use uuid::Uuid;

use chrono::Utc;

use crate::domain::agent::AgentConfig;
use crate::domain::config::{EventStoreConfig, SystemConfig};
use crate::domain::event::{CompletionDelivery, Event, SessionAuth, SpanContext};
use crate::domain::session::{AgentState, CommandPayload, IncomingMessage, SessionCommand};
use wake_scheduler::spawn_wake_scheduler;

pub mod dispatcher;
pub mod event_store;
pub mod jsonrpc;
pub mod llm;
pub mod mcp;
mod session_actor;
pub mod session_client;
mod strategy;
pub mod wake_scheduler;

pub use dispatcher::spawn_dispatcher;
#[cfg(feature = "sqlite")]
pub use event_store::SqliteEventStore;
pub use event_store::{
    EventStore, InMemoryEventStore, SessionFilter, SessionLoad, SessionSummary, StoreError, Version,
};
pub use llm::{LlmClient, MockLlmClient, OpenAiClient, StreamDelta};
pub use llm::{LlmClientFactory, LlmClientProvider, ProviderError, StaticLlmClientProvider};
pub use mcp::{
    CallToolResult, Content, McpClient, McpError, ServerCapabilities, ServerInfo, StdioMcpClient,
    ToolAnnotations, ToolDefinition,
};
pub use mcp::{McpClientProvider, StaticMcpClientProvider};
pub use session_actor::SessionActorArgs;
pub use session_actor::{
    RuntimeError, SessionActor, SessionActorState, SessionInit, SessionMessage,
};
pub use session_client::{ClientMessage, OnEvent, SessionClientActor, SessionClientArgs};
pub use strategy::{StaticStrategyProvider, StrategyProvider, StrategyProviderError};

// ---------------------------------------------------------------------------
// Naming conventions for ractor registry / process groups
// ---------------------------------------------------------------------------

pub fn session_actor_name(session_id: Uuid) -> String {
    format!("session-{session_id}")
}

pub fn session_clients_group(session_id: Uuid) -> String {
    format!("session-clients-{session_id}")
}

/// Look up a running session actor by ID and send it a message.
/// Silently no-ops if the actor is not running.
pub fn send_to_session(session_id: Uuid, message: SessionMessage) {
    if let Some(cell) = ractor::registry::where_is(session_actor_name(session_id)) {
        let actor: ActorRef<SessionMessage> = cell.into();
        let _ = actor.send_message(message);
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
}

pub struct SubAgentRequest {
    pub session_id: Uuid,
    pub agent_name: String,
    pub message: String,
    pub auth: SessionAuth,
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
    mcp_provider: Arc<dyn McpClientProvider>,
    strategy_provider: Arc<dyn StrategyProvider>,
}

struct RuntimeArgs {
    store: Arc<dyn EventStore>,
    agents: HashMap<String, AgentConfig>,
    llm_provider: Arc<dyn LlmClientProvider>,
    mcp_provider: Arc<dyn McpClientProvider>,
    strategy_provider: Arc<dyn StrategyProvider>,
}

impl RuntimeState {
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
                let (actor, _actor_handle) = Actor::spawn_linked(
                    Some(actor_name),
                    SessionActor,
                    SessionActorArgs {
                        session_id,
                        init,
                        store: self.store.clone(),
                        llm_provider: self.llm_provider.clone(),
                        mcp_provider: self.mcp_provider.clone(),
                        strategy_provider: self.strategy_provider.clone(),
                        agents: self.agents.clone(),
                        runtime: self.myself.clone(),
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

    async fn resume_all(&self) {
        let filter = SessionFilter {
            needs_wake: Some(true),
            ..Default::default()
        };
        let sessions = self.store.list_sessions(&filter);
        for summary in sessions {
            let agent = match self.agents.get(&summary.agent_name) {
                Some(a) => a.clone(),
                None => {
                    eprintln!(
                        "warn: unknown agent '{}' for session {}, skipping resume",
                        summary.agent_name, summary.session_id
                    );
                    continue;
                }
            };
            let auth = SessionAuth {
                tenant_id: summary.tenant_id,
                client_id: summary.client_id,
                sub: None,
            };
            let init = SessionInit {
                agent,
                auth,
                on_done: None,
                span: SpanContext::root(),
            };
            if let Err(e) = self.start_session(summary.session_id, init).await {
                eprintln!(
                    "warn: failed to resume session {}: {}",
                    summary.session_id, e
                );
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

        let actor_name = session_actor_name(req.session_id);
        let (session_actor, _) = Actor::spawn_linked(
            Some(actor_name),
            SessionActor,
            SessionActorArgs {
                session_id: req.session_id,
                init,
                store: self.store.clone(),
                llm_provider: self.llm_provider.clone(),
                mcp_provider: self.mcp_provider.clone(),
                strategy_provider: self.strategy_provider.clone(),
                agents: self.agents.clone(),
                runtime: self.myself.clone(),
            },
            self.myself.get_cell(),
        )
        .await
        .map_err(|e| RuntimeError::ActorCall(format!("sub-agent: {e}")))?;

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
            mcp_provider: args.mcp_provider,
            strategy_provider: args.strategy_provider,
        };

        // Spawn infrastructure actors (linked to RuntimeActor)
        spawn_dispatcher(args.store.clone(), myself.get_cell())
            .await
            .map_err(|e| format!("failed to spawn dispatcher: {e}"))?;
        spawn_wake_scheduler(args.store, myself.get_cell())
            .await
            .map_err(|e| format!("failed to spawn wake scheduler: {e}"))?;

        // Resume sessions that need waking after restart
        state.resume_all().await;

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
                    eprintln!("runtime: sub-agent error: {e}");
                }
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
                eprintln!("runtime: child {:?} failed: {err}", name);
                restart_infrastructure(name, state).await;
            }
            SupervisionEvent::ActorTerminated(who, _, reason) => {
                let name = who.get_name();
                eprintln!("runtime: child {:?} terminated: {reason:?}", name);
                restart_infrastructure(name, state).await;
            }
            _ => {}
        }
        Ok(())
    }
}

/// Restart dispatcher or wake-scheduler if they died; sessions just get logged.
async fn restart_infrastructure(name: Option<String>, state: &RuntimeState) {
    match name.as_deref() {
        Some("dispatcher") => {
            eprintln!("runtime: restarting dispatcher");
            if let Err(e) = spawn_dispatcher(state.store.clone(), state.myself.get_cell()).await {
                eprintln!("runtime: failed to restart dispatcher: {e}");
            }
        }
        Some("wake-scheduler") => {
            eprintln!("runtime: restarting wake-scheduler");
            if let Err(e) = spawn_wake_scheduler(state.store.clone(), state.myself.get_cell()).await
            {
                eprintln!("runtime: failed to restart wake-scheduler: {e}");
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
        let store = create_event_store(&config.event_store);

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
                mcp_provider: Arc::new(StaticMcpClientProvider),
                strategy_provider: Arc::new(StaticStrategyProvider),
            },
        )
        .await
        .map_err(|e| RuntimeError::ActorCall(e.to_string()))?;

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
        auth: SessionAuth,
    ) -> Result<SessionHandle, RuntimeError> {
        self.start_session(Uuid::new_v4(), agent_name, auth).await
    }

    /// Start a session — resumes from store if it exists, otherwise creates new.
    pub async fn start_session(
        &self,
        session_id: Uuid,
        agent_name: &str,
        auth: SessionAuth,
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
    /// An optional `on_event` callback is invoked for each event after it is applied.
    pub async fn connect(
        &self,
        session_id: Uuid,
        auth: SessionAuth,
        on_event: Option<OnEvent>,
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

    pub fn shutdown(self) {
        self.actor.stop(None);
    }
}

// ---------------------------------------------------------------------------
// SessionHandle — per-session interface
// ---------------------------------------------------------------------------

pub struct SessionHandle {
    pub session_id: Uuid,
    session_client: ActorRef<ClientMessage>,
}

impl SessionHandle {
    pub async fn send_command(&self, cmd: SessionCommand) -> Result<Vec<Event>, RuntimeError> {
        let result = call_t!(
            self.session_client,
            ClientMessage::SendCommand,
            5000,
            Box::new(cmd)
        )
        .map_err(|e| RuntimeError::ActorCall(e.to_string()))?;
        result
    }

    pub async fn get_state(&self) -> AgentState {
        call_t!(self.session_client, ClientMessage::GetState, 5000)
            .expect("failed to query session state")
    }

    pub fn shutdown(self) {
        self.session_client.stop(None);
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn create_event_store(config: &EventStoreConfig) -> Arc<dyn EventStore> {
    match config {
        EventStoreConfig::InMemory => Arc::new(InMemoryEventStore::new()),
        #[cfg(feature = "sqlite")]
        EventStoreConfig::Sqlite { path } => {
            Arc::new(SqliteEventStore::new(path).expect("failed to open SQLite event store"))
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
