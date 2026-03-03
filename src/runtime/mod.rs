use std::collections::HashMap;
use std::sync::Arc;

use ractor::{call_t, Actor, ActorRef};
use uuid::Uuid;

use crate::domain::agent::AgentConfig;
use crate::domain::config::{EventStoreConfig, SystemConfig};
use crate::domain::event::{ClientIdentity, SpanContext};

pub mod budget;
pub mod dispatcher;
pub mod event_store;
pub mod aggregate_actor;
pub mod jsonrpc;
pub mod llm;
pub mod mcp;
#[cfg(feature = "otel")]
pub mod otel;
pub mod session_client;
pub mod wake_scheduler;

mod adapters;
mod actor;
mod routing;
mod types;

// Re-export shared types
pub use types::{RuntimeError, RuntimeMessage, SessionHandle, SessionInit, SessionMessage, SubAgentRequest};

// Re-export routing utilities
pub use routing::{aggregate_actor_name, notify_observers, session_group, session_observer_group};

// Re-export event store types
#[cfg(feature = "sqlite")]
pub use event_store::SqliteEventStore;
pub use event_store::{
    AggregateFilter, AggregateSort, AggregateSummary, EventFilter, EventStore, StoreError,
    StreamLoad, Version,
};

// Re-export LLM types
pub use llm::{LlmClient, MockLlmClient, OpenAiClient, StreamDelta as LlmStreamDelta};
pub use llm::{LlmClientFactory, LlmClientProvider, ProviderError, StaticLlmClientProvider};

// Re-export MCP types
pub use mcp::{mcp_actor_name, spawn_mcp_actor, McpActorClient, McpMessage};
pub use mcp::{
    CallToolResult, Content, McpClient, McpError, ServerCapabilities, ServerInfo, StdioMcpClient,
    ToolAnnotations, ToolDefinition,
};

// Re-export session client types
pub use session_client::{
    Notification, OnSessionUpdate, SessionClientActor, SessionClientArgs, SessionUpdate,
};

use actor::{RuntimeActor, RuntimeArgs};

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
                #[cfg(feature = "otel")]
                otel: config.otel.clone(),
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
            Box::new(init)
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
            trace_id: None,
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
