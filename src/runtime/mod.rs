use std::collections::HashMap;
use std::sync::Arc;

use ractor::{call_t, Actor, ActorRef};
use uuid::Uuid;

// --- Modules (types, traits, config) ---
pub mod aggregate;
pub mod auth;
pub mod config;
pub mod defaults;
pub mod event;
pub mod message;
pub mod secret;
pub mod span;

// --- Runtime modules ---
pub mod budget;
pub mod event_store;
pub mod jsonrpc;
pub mod llm;
pub mod mcp;
#[cfg(feature = "otel")]
pub mod otel;
pub mod session;
pub mod wake_scheduler;

mod actor;
mod types;
pub(crate) mod decision_queue;
mod worker_client;
pub(crate) mod worker_convert;

use self::config::{AgentConfig, ClientIdentity, EventStoreConfig, SystemConfig};
use self::span::SpanContext;

// Re-export shared types
pub use types::{
    RuntimeError, RuntimeMessage, SessionHandle, SessionInit, SessionMessage, SubAgentRequest,
};

// Re-export routing utilities
pub use session::routing::{
    aggregate_actor_name, notify_observers, session_group, session_observer_group,
};

// Re-export event store types
pub use event_store::SqliteEventStore;
pub use event_store::{
    AggregateFilter, AggregateSort, AggregateSummary, EventFilter, EventStore, StoreError,
    StreamLoad, Version,
};

// Re-export LLM types
pub use llm::{
    LlmCallError, LlmCallable, LlmProviderTrait, MockLlmClient, OpenAiClient, StreamDelta,
};
pub use llm::{LlmClientFactory, StaticLlmClientProvider};

// Re-export MCP types
pub use mcp::{mcp_actor_name, spawn_mcp_actor, McpActorClient, McpMessage};
pub use mcp::{
    CallToolResult, Content, McpClient, McpError, ServerCapabilities, ServerInfo, StdioMcpClient,
    ToolAnnotations, ToolDefinition,
};

// Re-export budget types
pub use budget::{BudgetDenial, BudgetStatus};

// Re-export session types
pub use session::client::{
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
                tool_result_max_bytes: config.tool_result_max_bytes,
                gateway: config.gateway.clone(),
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
            stream: false,
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
                runtime: self.actor.clone(),
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

    /// Ensure an aggregate actor is running, waking it from the store if needed.
    /// Returns the actor cell from the registry.
    pub async fn ensure_aggregate(
        runtime: &ActorRef<RuntimeMessage>,
        aggregate_id: Uuid,
        aggregate_type: &str,
        tenant_id: &str,
    ) -> Result<ractor::ActorCell, RuntimeError> {
        let name = format!("{aggregate_type}-{aggregate_id}");

        if let Some(cell) = ractor::registry::where_is(name.clone()) {
            return Ok(cell);
        }

        let result = runtime
            .call(
                |reply| RuntimeMessage::EnsureAggregate {
                    aggregate_id,
                    aggregate_type: aggregate_type.into(),
                    tenant_id: tenant_id.into(),
                    reply,
                },
                Some(ractor::concurrency::Duration::from_millis(10_000)),
            )
            .await
            .map_err(|e| RuntimeError::ActorCall(e.to_string()))?;

        match result {
            ractor::rpc::CallResult::Success(inner) => inner?,
            ractor::rpc::CallResult::Timeout => {
                return Err(RuntimeError::ActorCall("ensure aggregate timed out".into()));
            }
            ractor::rpc::CallResult::SenderError => {
                return Err(RuntimeError::ActorCall(
                    "ensure aggregate sender error".into(),
                ));
            }
        }

        ractor::registry::where_is(name).ok_or(RuntimeError::SessionNotFound)
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
        EventStoreConfig::Sqlite { path } => Arc::new(
            SqliteEventStore::new(path)
                .await
                .expect("failed to open SQLite event store"),
        ),
    }
}

fn default_llm_factories() -> HashMap<String, LlmClientFactory> {
    let mut m: HashMap<String, LlmClientFactory> = HashMap::new();
    m.insert("openrouter".into(), Box::new(OpenAiClient::from_config));
    m.insert("mock".into(), Box::new(MockLlmClient::from_config));
    m
}
