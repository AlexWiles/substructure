use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;
use ractor::{call_t, Actor, ActorRef};
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::domain::agent::AgentConfig;
use crate::domain::config::{EventStoreConfig, SystemConfig};
use crate::domain::event::{Event, SessionAuth, SpanContext};
use crate::domain::session::{AgentSession, AgentState, CommandPayload, SessionCommand};

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
pub use event_store::{EventStore, InMemoryEventStore, SessionLoad, StoreError, Version};
pub use llm::{LlmClient, MockLlmClient, OpenAiClient, StreamDelta};
pub use llm::{LlmClientFactory, LlmClientProvider, ProviderError, StaticLlmClientProvider};
pub use mcp::{
    CallToolResult, Content, McpClient, McpError, ServerCapabilities, ServerInfo, StdioMcpClient,
    ToolAnnotations, ToolDefinition,
};
pub use mcp::{McpClientProvider, StaticMcpClientProvider};
pub use session_actor::{RuntimeError, SessionActor, SessionActorState, SessionMessage};
pub use session_client::{ClientMessage, SessionClientActor, SessionClientArgs};
pub use strategy::{StaticStrategyProvider, StrategyProvider, StrategyProviderError};

// ---------------------------------------------------------------------------
// Runtime — top-level handle for the system
// ---------------------------------------------------------------------------

pub struct Runtime {
    store: Arc<dyn EventStore>,
    dispatcher: ActorRef<dispatcher::DispatcherMessage>,
    dispatcher_handle: JoinHandle<()>,
    llm_provider: Arc<dyn LlmClientProvider>,
    mcp_provider: Arc<dyn McpClientProvider>,
    strategy_provider: Arc<dyn StrategyProvider>,
    agents: HashMap<String, AgentConfig>,
    wake_scheduler: ActorRef<wake_scheduler::WakeSchedulerMessage>,
    wake_scheduler_handle: JoinHandle<()>,
}

impl Runtime {
    pub async fn start(config: &SystemConfig) -> Self {
        let store = create_event_store(&config.event_store);
        let (dispatcher, dispatcher_handle) = spawn_dispatcher(store.clone()).await;

        let llm_provider =
            StaticLlmClientProvider::from_config(&config.llm_clients, &default_llm_factories())
                .expect("failed to create LLM clients from config");

        let (wake_scheduler, wake_scheduler_handle) = Actor::spawn(
            Some("wake-scheduler".to_string()),
            wake_scheduler::WakeScheduler,
            (),
        )
        .await
        .expect("failed to spawn wake scheduler");

        Runtime {
            store,
            dispatcher,
            dispatcher_handle,
            llm_provider: Arc::new(llm_provider),
            mcp_provider: Arc::new(StaticMcpClientProvider),
            strategy_provider: Arc::new(StaticStrategyProvider),
            agents: config.agents.clone(),
            wake_scheduler,
            wake_scheduler_handle,
        }
    }

    /// Replace the LLM client provider.
    pub fn with_llm_provider(mut self, provider: Arc<dyn LlmClientProvider>) -> Self {
        self.llm_provider = provider;
        self
    }

    /// Replace the MCP client provider.
    pub fn with_mcp_provider(mut self, provider: Arc<dyn McpClientProvider>) -> Self {
        self.mcp_provider = provider;
        self
    }

    /// Replace the strategy provider.
    pub fn with_strategy_provider(mut self, provider: Arc<dyn StrategyProvider>) -> Self {
        self.strategy_provider = provider;
        self
    }

    /// Look up an agent definition by name.
    pub fn agent(&self, name: &str) -> Option<&AgentConfig> {
        self.agents.get(name)
    }

    /// Create a new session for a named agent from the config.
    pub async fn create_session_for(
        &self,
        agent_name: &str,
        auth: SessionAuth,
    ) -> Result<SessionHandle, RuntimeError> {
        let agent = self
            .agents
            .get(agent_name)
            .cloned()
            .ok_or_else(|| RuntimeError::UnknownAgent(agent_name.to_string()))?;
        self.create_session(agent, auth).await
    }

    /// Create a new session, resolving LLM/MCP clients through providers.
    pub async fn create_session(
        &self,
        agent: AgentConfig,
        auth: SessionAuth,
    ) -> Result<SessionHandle, RuntimeError> {
        let session_id = Uuid::new_v4();

        let (strategy, strategy_state) = self
            .strategy_provider
            .resolve(&agent.strategy, &agent, &auth)
            .await
            .map_err(|e| RuntimeError::StrategyResolution(e.to_string()))?;

        let initial_state = SessionActorState::new(
            session_id,
            self.store.clone(),
            auth.clone(),
            self.llm_provider.clone(),
            self.mcp_provider.clone(),
            self.strategy_provider.clone(),
            strategy,
            strategy_state,
        );
        let (actor, session_handle) = Actor::spawn(
            Some(format!("session-{session_id}")),
            SessionActor,
            initial_state,
        )
        .await
        .expect("failed to spawn session actor");

        let (session_client, client_handle) = Actor::spawn(
            None,
            SessionClientActor,
            SessionClientArgs {
                session_id,
                auth: auth.clone(),
                session_actor: actor.clone(),
                store: self.store.clone(),
            },
        )
        .await
        .expect("failed to spawn session client");

        // Send CreateSession command to initialize domain state
        let result = call_t!(
            session_client,
            ClientMessage::SendCommand,
            5000,
            SessionCommand {
                span: SpanContext::root(),
                occurred_at: Utc::now(),
                payload: CommandPayload::CreateSession { agent, auth },
            }
        );
        if let Ok(inner) = result {
            inner.expect("CreateSession command failed");
        }

        Ok(SessionHandle {
            session_id,
            session_actor: actor,
            session_client,
            session_actor_handle: session_handle,
            session_client_handle: client_handle,
        })
    }

    /// Resume an existing session from stored snapshot.
    /// Recovery is handled by the wake scheduler — no special recovery path needed.
    pub async fn resume_session(
        &self,
        session_id: Uuid,
        auth: SessionAuth,
    ) -> Result<SessionHandle, RuntimeError> {
        let load = self.store.load(session_id, &auth)?;

        // Build session from snapshot (fast path) or cold replay (fallback)
        let mut session = if let Some(snapshot) = load.snapshot {
            let agent = snapshot
                .agent
                .as_ref()
                .ok_or(RuntimeError::SessionNotFound)?;
            let (strategy, _default_state) = self
                .strategy_provider
                .resolve(&agent.strategy, agent, &auth)
                .await
                .map_err(|e| RuntimeError::StrategyResolution(e.to_string()))?;

            AgentSession::from_snapshot(snapshot, strategy, &load.events)
        } else {
            // No snapshot — cold replay all events
            let mut core = AgentState::new(session_id);
            for event in &load.events {
                core.apply_core(event);
            }
            let agent = core.agent.as_ref().ok_or(RuntimeError::SessionNotFound)?;
            let (strategy, strategy_state) = self
                .strategy_provider
                .resolve(&agent.strategy, agent, &auth)
                .await
                .map_err(|e| RuntimeError::StrategyResolution(e.to_string()))?;

            AgentSession::from_core(core, strategy, strategy_state)
        };
        session.agent_state.last_reacted = session.agent_state.last_applied;

        // Compute setup effects (MCP servers — runtime state, never persisted)
        let setup = session.setup_effects();

        // Compute wake_at for scheduler registration
        let wake_at = session.agent_state.wake_at();

        let actor_state = SessionActorState::from_session(
            session,
            self.store.clone(),
            auth.clone(),
            self.llm_provider.clone(),
            self.mcp_provider.clone(),
            self.strategy_provider.clone(),
        );

        let (actor, session_handle) = Actor::spawn(
            Some(format!("session-{session_id}")),
            SessionActor,
            actor_state,
        )
        .await
        .expect("failed to spawn session actor");

        let (session_client, client_handle) = Actor::spawn(
            None,
            SessionClientActor,
            SessionClientArgs {
                session_id,
                auth,
                session_actor: actor.clone(),
                store: self.store.clone(),
            },
        )
        .await
        .expect("failed to spawn session client");

        // Start MCP servers (always needed — runtime state, never persisted)
        if !setup.is_empty() {
            let _ = actor.send_message(SessionMessage::SetupEffects(setup));
        }

        // Register with wake scheduler — it will immediately wake if wake_at is past due
        let _ = self
            .wake_scheduler
            .send_message(wake_scheduler::WakeSchedulerMessage::UpdateWakeAt {
                session_id,
                wake_at,
                actor_ref: actor.clone(),
            });

        Ok(SessionHandle {
            session_id,
            session_actor: actor,
            session_client,
            session_actor_handle: session_handle,
            session_client_handle: client_handle,
        })
    }

    pub fn store(&self) -> &Arc<dyn EventStore> {
        &self.store
    }

    pub fn wake_scheduler(&self) -> &ActorRef<wake_scheduler::WakeSchedulerMessage> {
        &self.wake_scheduler
    }

    pub async fn shutdown(self) {
        self.wake_scheduler.stop(None);
        let _ = self.wake_scheduler_handle.await;
        self.dispatcher.stop(None);
        let _ = self.dispatcher_handle.await;
    }
}

// ---------------------------------------------------------------------------
// SessionHandle — per-session interface
// ---------------------------------------------------------------------------

pub struct SessionHandle {
    pub session_id: Uuid,
    session_actor: ActorRef<SessionMessage>,
    session_client: ActorRef<ClientMessage>,
    session_actor_handle: JoinHandle<()>,
    session_client_handle: JoinHandle<()>,
}

impl SessionHandle {
    pub async fn send_command(&self, cmd: SessionCommand) -> Result<Vec<Event>, RuntimeError> {
        let result = call_t!(self.session_client, ClientMessage::SendCommand, 5000, cmd)
            .map_err(|e| RuntimeError::ActorCall(e.to_string()))?;
        result
    }

    pub async fn get_state(&self) -> AgentState {
        call_t!(self.session_client, ClientMessage::GetState, 5000)
            .expect("failed to query session state")
    }

    pub async fn shutdown(self) {
        self.session_client.stop(None);
        self.session_actor.stop(None);
        let _ = self.session_client_handle.await;
        let _ = self.session_actor_handle.await;
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn create_event_store(config: &EventStoreConfig) -> Arc<dyn EventStore> {
    match config {
        EventStoreConfig::InMemory => Arc::new(InMemoryEventStore::new()),
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
