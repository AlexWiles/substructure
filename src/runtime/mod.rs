use std::collections::HashMap;
use std::sync::Arc;

use ractor::{call_t, Actor, ActorRef};
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::domain::agent::AgentConfig;
use crate::domain::config::{EventStoreConfig, SystemConfig};
use crate::domain::event::{Event, SessionAuth};
use crate::domain::session::{AgentState, SessionCommand};
use session_actor::SessionActorArgs;
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
pub use session_actor::{RuntimeError, SessionActor, SessionActorState, SessionInit, SessionMessage};
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

        let (wake_scheduler, wake_scheduler_handle) =
            spawn_wake_scheduler(store.clone()).await;

        let runtime = Runtime {
            store,
            dispatcher,
            dispatcher_handle,
            llm_provider: Arc::new(llm_provider),
            mcp_provider: Arc::new(StaticMcpClientProvider),
            strategy_provider: Arc::new(StaticStrategyProvider),
            agents: config.agents.clone(),
            wake_scheduler,
            wake_scheduler_handle,
        };

        // Resume sessions that need waking after restart
        runtime.resume_all().await;

        runtime
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
        self.start_session(SessionInit::Create {
            session_id: Uuid::new_v4(),
            agent,
            auth,
        })
        .await
    }

    /// Start a session — either creating a new one or resuming from a snapshot.
    ///
    /// The actor's `pre_start` handles all startup logic: resolving the
    /// strategy, building or loading the session, persisting the initial
    /// `CreateSession` event (for new sessions), starting MCP servers,
    /// and setting `last_reacted` to prevent double-react.
    pub async fn start_session(&self, init: SessionInit) -> Result<SessionHandle, RuntimeError> {
        let session_id = init.session_id();
        let auth = init.auth().clone();

        let (actor, actor_handle) = Actor::spawn(
            Some(format!("session-{session_id}")),
            SessionActor,
            SessionActorArgs {
                init,
                store: self.store.clone(),
                llm_provider: self.llm_provider.clone(),
                mcp_provider: self.mcp_provider.clone(),
                strategy_provider: self.strategy_provider.clone(),
            },
        )
        .await
        .map_err(|e| RuntimeError::ActorCall(format!("session startup failed: {e}")))?;

        let (client, client_handle) = Actor::spawn(
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

        Ok(SessionHandle {
            session_id,
            session_actor: actor,
            session_client: client,
            session_actor_handle: actor_handle,
            session_client_handle: client_handle,
        })
    }

    /// Resume all sessions that need a wake, populating the wake scheduler.
    /// Called on startup to recover in-flight sessions after a restart.
    pub async fn resume_all(&self) -> Vec<SessionHandle> {
        let filter = SessionFilter {
            needs_wake: Some(true),
            ..Default::default()
        };
        let sessions = self.store.list_sessions(&filter);
        let mut handles = Vec::new();
        for summary in sessions {
            let auth = SessionAuth {
                tenant_id: summary.tenant_id,
                client_id: summary.client_id,
                sub: None,
            };
            let init = SessionInit::Resume {
                session_id: summary.session_id,
                auth,
            };
            match self.start_session(init).await {
                Ok(handle) => handles.push(handle),
                Err(e) => {
                    eprintln!(
                        "warn: failed to resume session {}: {}",
                        summary.session_id, e
                    );
                }
            }
        }
        handles
    }

    pub fn store(&self) -> &Arc<dyn EventStore> {
        &self.store
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
