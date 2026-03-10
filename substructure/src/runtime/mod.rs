use std::collections::HashSet;
use std::sync::Arc;

use ractor::{Actor, ActorRef};
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

mod actor;
pub(crate) mod grpc_gateway;
mod local_worker;
mod types;
pub(crate) mod work_dispatcher;
pub(crate) mod work_queue;
pub(crate) mod worker_convert;

use self::config::{EventStoreConfig, SystemConfig};
use self::session::CommandPayload;
use self::span::SpanContext;
use self::work_queue::{InMemoryWorkQueue, WorkQueue, WorkQueueError};

use actor::SupervisorActor;
use session::system::SessionSystem;

// Re-export shared types
pub use types::{
    RuntimeError, SessionHandle, SessionInit, SessionMessage, SubAgentRequest, SupervisorMessage,
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

use crate::worker::{self as proto, Worker};

// ---------------------------------------------------------------------------
// Runtime — thin infrastructure orchestrator
// ---------------------------------------------------------------------------

pub struct Runtime {
    supervisor: ActorRef<SupervisorMessage>,
    store: Arc<dyn EventStore>,
    queue: Arc<dyn WorkQueue>,
    system: SessionSystem,
}

impl Runtime {
    pub fn supervisor_cell(&self) -> ractor::ActorCell {
        self.supervisor.get_cell()
    }

    pub async fn start(
        config: &SystemConfig,
        worker: Option<Arc<dyn Worker>>,
    ) -> Result<Arc<Self>, RuntimeError> {
        let store = create_event_store(&config.event_store).await;

        let llm_client =
            StaticLlmClientProvider::from_config(&config.llm_clients, &default_llm_factories())
                .map_err(RuntimeError::ActorCall)?;
        let llm_provider: Arc<dyn LlmProviderTrait> = Arc::new(llm_client);

        // 1. Spawn the top-level runtime supervisor
        let (supervisor, _handle) = Actor::spawn(Some("runtime".to_string()), SupervisorActor, ())
            .await
            .map_err(|e| RuntimeError::ActorCall(e.to_string()))?;

        let supervisor_cell = supervisor.get_cell();

        // 2. Spawn the work queue actor (linked to runtime supervisor)
        let queue: Arc<dyn WorkQueue> = Arc::new(
            InMemoryWorkQueue::new(supervisor_cell.clone())
                .await
                .map_err(RuntimeError::ActorCall)?,
        );

        // 3. Spawn infrastructure children (linked to runtime supervisor)
        tracing::debug!("spawning event dispatcher");
        aggregate::dispatcher::spawn_aggregate_dispatcher::<session::SessionState>(
            &store,
            Arc::new(session::routing::session_route),
            supervisor_cell.clone(),
        )
        .await
        .map_err(|e| RuntimeError::ActorCall(format!("failed to spawn dispatcher: {e}")))?;

        #[cfg(feature = "otel")]
        if let Some(otel_config) = &config.otel {
            tracing::debug!("spawning otel exporter");
            if let Err(e) = otel::spawn_otel_exporter(
                &otel_config.endpoint,
                otel_config.service_name.clone(),
                supervisor_cell.clone(),
                &*store,
            )
            .await
            {
                tracing::warn!(error = %e, "failed to start otel exporter, continuing without it");
            }
        }

        // 4. Spawn the session supervisor
        let session_supervisor_ref =
            session::system::spawn_session_supervisor(supervisor_cell.clone())
                .await
                .map_err(|e| {
                    RuntimeError::ActorCall(format!("failed to spawn session supervisor: {e}"))
                })?;

        // 5. Build SessionSystem
        let worker_executor: Arc<dyn session::WorkerExecutor> =
            Arc::new(work_dispatcher::WorkDispatcher::new(queue.clone()));

        let system = SessionSystem::new(
            store.clone(),
            llm_provider,
            worker_executor,
            config.budgets.clone(),
        );

        // 6. Spawn wake scheduler (linked to session supervisor)
        tracing::debug!("spawning wake scheduler");
        session::wake_scheduler::spawn_wake_scheduler(
            store.clone(),
            system.clone(),
            session_supervisor_ref.get_cell(),
        )
        .await
        .map_err(|e| RuntimeError::ActorCall(format!("failed to spawn wake scheduler: {e}")))?;

        // 7. Build Runtime
        let runtime = Arc::new(Runtime {
            supervisor,
            store,
            queue,
            system,
        });

        // 8. Start gateway (if configured)
        if let Some(gw_config) = &config.gateway {
            let addr: std::net::SocketAddr = gw_config.addr.parse().map_err(|e| {
                RuntimeError::ActorCall(format!("invalid gateway addr '{}': {e}", gw_config.addr))
            })?;
            let gw_auth = auth::build_auth_resolver(&gw_config.auth).map_err(|e| {
                RuntimeError::ActorCall(format!("failed to build gateway auth: {e}"))
            })?;
            let gw = Arc::new(grpc_gateway::GrpcGateway::new(gw_auth, runtime.clone()));
            tokio::spawn(async move {
                if let Err(e) = gw.serve(addr).await {
                    tracing::error!(error = %e, "worker gateway server failed");
                }
            });
        }

        // 9. Start local worker (if provided)
        if let Some(worker) = worker {
            let rt = runtime.clone();
            tokio::spawn(async move {
                local_worker::run_local_worker(rt, worker).await;
            });
        }

        tracing::info!(budgets = config.budgets.len(), "runtime started",);

        Ok(runtime)
    }

    // -- public API (thin delegation) -----------------------------------------

    /// Create a new session for a named agent.
    pub async fn create_session_for(
        self: &Arc<Self>,
        agent_name: &str,
        auth: config::ClientIdentity,
    ) -> Result<SessionHandle, RuntimeError> {
        self.system.create_session_for(agent_name, auth).await
    }

    /// Start a session — resumes from store if it exists, otherwise creates new.
    pub async fn start_session(
        self: &Arc<Self>,
        session_id: Uuid,
        agent_name: &str,
        auth: config::ClientIdentity,
    ) -> Result<SessionHandle, RuntimeError> {
        self.system
            .start_session(session_id, agent_name, auth)
            .await
    }

    /// Spawn a new session client for an existing session.
    pub async fn connect(
        self: &Arc<Self>,
        session_id: Uuid,
        auth: config::ClientIdentity,
        on_event: Option<OnSessionUpdate>,
        stop_on_done: bool,
    ) -> Result<SessionHandle, RuntimeError> {
        self.system
            .connect(session_id, auth, on_event, stop_on_done)
            .await
    }

    /// Check whether a session is currently running.
    pub fn session_is_running(&self, session_id: Uuid) -> bool {
        self.system.session_is_running(session_id)
    }

    pub fn store(&self) -> &Arc<dyn EventStore> {
        &self.store
    }

    /// Block until a work item matching the given filters is available.
    pub async fn get_work(
        &self,
        agent_names: &HashSet<String>,
        tenant_id: Option<&str>,
    ) -> proto::WorkItem {
        self.queue.get_work(agent_names, tenant_id).await
    }

    /// Deliver a work result back to the originating session.
    pub async fn submit_result(&self, result: proto::WorkResult) -> Result<(), WorkQueueError> {
        let (session_id, payload, span) = match result.result {
            Some(proto::work_result::Result::Decision(r)) => parse_decision_result(r)?,
            Some(proto::work_result::Result::ToolCall(s)) => parse_tool_call_submission(s)?,
            None => return Err(WorkQueueError::InvalidArgument("result is required".into())),
        };
        self.system.deliver(session_id, payload, span).await;
        Ok(())
    }

    /// Deliver a command to a session, waking the aggregate if needed.
    pub async fn deliver(&self, session_id: Uuid, payload: CommandPayload, span: SpanContext) {
        self.system.deliver(session_id, payload, span).await;
    }

    pub fn shutdown(&self) {
        self.supervisor.stop(None);
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

fn parse_decision_result(
    r: proto::DecisionResult,
) -> Result<(Uuid, CommandPayload, SpanContext), WorkQueueError> {
    let session_id = r.session_id.parse::<Uuid>().map_err(|_| {
        WorkQueueError::InvalidArgument(format!("invalid session_id: {}", r.session_id))
    })?;
    let span = r
        .span
        .as_ref()
        .map(SpanContext::from)
        .unwrap_or_else(|| SpanContext::root().with_name("gateway.submit"));
    let decision = r
        .decision
        .ok_or_else(|| WorkQueueError::InvalidArgument("decision is required".into()))?;
    let payload = CommandPayload::SubmitWorkerDecision {
        decision_id: r.decision_id,
        actions: decision.actions,
        state: decision.state,
    };
    Ok((session_id, payload, span))
}

fn parse_tool_call_submission(
    s: proto::ToolCallSubmission,
) -> Result<(Uuid, CommandPayload, SpanContext), WorkQueueError> {
    let session_id = s.session_id.parse::<Uuid>().map_err(|_| {
        WorkQueueError::InvalidArgument(format!("invalid session_id: {}", s.session_id))
    })?;
    let span = s
        .span
        .as_ref()
        .map(SpanContext::from)
        .unwrap_or_else(|| SpanContext::root().with_name("gateway.tool_result"));
    let result = s
        .result
        .ok_or_else(|| WorkQueueError::InvalidArgument("result is required".into()))?;
    let payload = match result.outcome {
        Some(proto::tool_call_result::Outcome::Result(text)) => CommandPayload::CompleteToolCall {
            tool_call_id: result.tool_call_id,
            name: result.name,
            result: text,
            worker_state: result.worker_state,
        },
        Some(proto::tool_call_result::Outcome::Error(text)) => CommandPayload::FailToolCall {
            tool_call_id: result.tool_call_id,
            name: result.name,
            error: text,
            worker_state: result.worker_state,
        },
        None => CommandPayload::FailToolCall {
            tool_call_id: result.tool_call_id,
            name: result.name,
            error: "tool call returned no outcome".to_string(),
            worker_state: None,
        },
    };
    Ok((session_id, payload, span))
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

fn default_llm_factories() -> std::collections::HashMap<String, LlmClientFactory> {
    let mut m: std::collections::HashMap<String, LlmClientFactory> =
        std::collections::HashMap::new();
    m.insert("openrouter".into(), Box::new(OpenAiClient::from_config));
    m.insert("mock".into(), Box::new(MockLlmClient::from_config));
    m
}
