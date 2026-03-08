use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use chrono::Utc;
use ractor::{Actor, ActorCell, ActorRef};
use uuid::Uuid;

use crate::runtime::{
    aggregate::actor::{self as aggregate_actor, AggregateActorHandle, AggregateMessage},
    actor::SupervisorActor,
    budget,
    config::{BudgetPolicyConfig, ClientIdentity},
    event_store::{AggregateFilter, EventStore},
    llm::LlmProviderTrait,
    span::SpanContext,
    types::{RuntimeError, SessionHandle, SessionInit, SubAgentRequest, SupervisorMessage},
};

use super::client::{SessionClientActor, SessionClientArgs};
use super::routing::aggregate_actor_name;
use super::{
    CommandPayload, IncomingMessage, SessionCommand, SessionContext, SessionState,
    SessionStatus, WorkerExecutor,
};

/// Well-known registry name for the session supervisor actor.
pub const SESSION_SUPERVISOR_NAME: &str = "session-supervisor";

// ---------------------------------------------------------------------------
// SessionSystem — single cloneable infrastructure handle
// ---------------------------------------------------------------------------

/// All fields are cheap to clone (Arc, Vec<Clone>).
/// Created once at startup, cloned into each session context.
#[derive(Clone)]
pub struct SessionSystem {
    store: Arc<dyn EventStore>,
    llm_provider: Arc<dyn LlmProviderTrait>,
    worker_executor: Arc<dyn WorkerExecutor>,
    budget_policies: Vec<BudgetPolicyConfig>,
}

impl SessionSystem {
    pub fn new(
        store: Arc<dyn EventStore>,
        llm_provider: Arc<dyn LlmProviderTrait>,
        worker_executor: Arc<dyn WorkerExecutor>,
        budget_policies: Vec<BudgetPolicyConfig>,
    ) -> Self {
        Self {
            store,
            llm_provider,
            worker_executor,
            budget_policies,
        }
    }

    // -- accessors ------------------------------------------------------------

    pub fn store(&self) -> &Arc<dyn EventStore> {
        &self.store
    }

    pub fn llm_provider(&self) -> &Arc<dyn LlmProviderTrait> {
        &self.llm_provider
    }

    pub fn worker_executor(&self) -> &Arc<dyn WorkerExecutor> {
        &self.worker_executor
    }

    // -- budget ---------------------------------------------------------------

    /// Reserve budget for an LLM call. Returns `Ok(())` if granted, no policies
    /// configured, or on infrastructure errors (fail-open).
    pub async fn reserve_budget(
        &self,
        tenant_id: &str,
        session_id: Uuid,
        call_id: &str,
        context: budget::BudgetContext,
        breakdown: budget::UsageBreakdown,
        span: &SpanContext,
    ) -> Result<(), budget::BudgetError> {
        let Some(handle) = self.get_or_spawn_budget_actor(tenant_id).await.map_err(|e| {
            tracing::warn!(error = %e, "budget actor spawn failed — proceeding without reservation");
        }).ok().flatten() else {
            return Ok(());
        };

        match handle
            .send_command(
                budget::BudgetCommand::Reserve {
                    session_id,
                    call_id: call_id.to_string(),
                    context,
                    breakdown,
                    reserved_at: Utc::now(),
                },
                span.clone(),
                Utc::now(),
            )
            .await
        {
            Ok(_) => Ok(()),
            Err(crate::runtime::aggregate::actor::AggregateError::Command(e)) => Err(e),
            Err(other) => {
                tracing::warn!(error = %other, "budget reserve failed — proceeding without reservation");
                Ok(())
            }
        }
    }

    // -- inter-session methods ------------------------------------------------

    /// Deliver a command to a session. Direct registry send; if not running,
    /// wake from store then deliver.
    pub async fn deliver(
        &self,
        session_id: Uuid,
        payload: CommandPayload,
        span: SpanContext,
    ) {
        if try_send_to_aggregate(session_id, payload.clone(), span.clone()) {
            return;
        }
        // Aggregate not running — wake it and retry.
        self.wake_aggregate(session_id, "session", "").await;
        if !try_send_to_aggregate(session_id, payload, span) {
            tracing::warn!(%session_id, "deliver: aggregate not reachable after wake");
        }
    }

    /// Spawn a sub-agent session and send the initial message.
    pub async fn spawn_sub_agent(&self, req: SubAgentRequest) {
        self.run_sub_agent(req).await;
    }

    /// Ensure a session's aggregate actor is running. Returns its ActorCell.
    pub async fn ensure_session(
        &self,
        session_id: Uuid,
        tenant_id: &str,
    ) -> Result<ActorCell, RuntimeError> {
        if let Some(cell) = ractor::registry::where_is(aggregate_actor_name(session_id)) {
            return Ok(cell);
        }
        self.wake_aggregate(session_id, "session", tenant_id).await;
        ractor::registry::where_is(aggregate_actor_name(session_id))
            .ok_or(RuntimeError::SessionNotFound)
    }

    // -- public lifecycle API -------------------------------------------------

    /// Create a new session for a named agent.
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
        let init = SessionInit {
            agent_name: agent_name.to_string(),
            auth,
            on_done: None,
            span: SpanContext::root(),
            stream: false,
        };
        self.do_start_session(session_id, init).await
    }

    /// Spawn a new session client for an existing session.
    pub async fn connect(
        &self,
        session_id: Uuid,
        auth: ClientIdentity,
        on_event: Option<super::client::OnSessionUpdate>,
        stop_on_done: bool,
    ) -> Result<SessionHandle, RuntimeError> {
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
                system: self.clone(),
                on_event,
                stop_on_done,
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

    // -- internal methods -----------------------------------------------------

    /// Look up the session supervisor's ActorCell from the registry.
    fn supervisor_cell() -> ActorCell {
        ractor::registry::where_is(SESSION_SUPERVISOR_NAME.to_string())
            .expect("session supervisor must be running")
    }

    async fn get_or_spawn_budget_actor(
        &self,
        tenant_id: &str,
    ) -> Result<Option<AggregateActorHandle<budget::BudgetLedger>>, RuntimeError>
    {
        if self.budget_policies.is_empty() {
            return Ok(None);
        }
        let actor_name = budget::budget_actor_name(tenant_id);
        if let Some(cell) = ractor::registry::where_is(actor_name) {
            return Ok(Some(AggregateActorHandle {
                actor: cell.into(),
            }));
        }
        let handle = budget::spawn_budget_actor(
            tenant_id.to_string(),
            self.budget_policies.clone(),
            self.store.clone(),
            Self::supervisor_cell(),
        )
        .await
        .map_err(|e| RuntimeError::ActorCall(format!("budget: {e}")))?;
        Ok(Some(handle))
    }

    /// Core session lifecycle — Box::pin breaks recursive async
    /// (do_start_session → deliver → wake_aggregate → do_start_session).
    pub(crate) fn do_start_session(
        &self,
        session_id: Uuid,
        init: SessionInit,
    ) -> Pin<Box<dyn Future<Output = Result<SessionHandle, RuntimeError>> + Send + '_>> {
        Box::pin(async move {
            let trace_id = init.span.trace_id;
            let stream = init.stream;
            let auth = init.auth.clone();
            let actor_name = aggregate_actor_name(session_id);
            let already_running = ractor::registry::where_is(actor_name).is_some();

            if !already_running {
                let system = self.clone();
                let auth_for_ctx = auth.clone();

                let aggregate_handle = aggregate_actor::spawn_aggregate_actor(
                    aggregate_actor::AggregateActorArgs {
                        aggregate_id: session_id,
                        store: self.store.clone(),
                        tenant_id: auth.tenant_id.clone(),
                        init: Box::new(SessionState::new),
                        idle_timeout: None,
                        context_init: Box::new(move |_state| {
                            let system = system.clone();
                            let auth_for_ctx = auth_for_ctx.clone();
                            Box::pin(async move {
                                SessionContext {
                                    session_id,
                                    auth: auth_for_ctx,
                                    stream,
                                    system,
                                }
                            })
                        }),
                    },
                    Self::supervisor_cell(),
                )
                .await
                .map_err(|e| RuntimeError::ActorCall(format!("aggregate actor spawn: {e}")))?;

                // Check if session needs creation
                let session = aggregate_handle.get_aggregate().await;
                let is_new = session.state.agent_name.is_none();

                if is_new {
                    aggregate_handle
                        .send_command(
                            CommandPayload::CreateSession {
                                agent_name: init.agent_name,
                                auth: auth.clone(),
                                on_done: init.on_done,
                            },
                            init.span.child("session.create"),
                            Utc::now(),
                        )
                        .await
                        .map_err(|e| RuntimeError::ActorCall(format!("create session: {e}")))?;
                }

                // If resuming a completed sub-agent, deliver result to parent.
                if session.state.status == SessionStatus::Done {
                    if let Some(ref delivery) = session.state.on_done {
                        let result =
                            serde_json::to_string(&session.state.artifacts).unwrap_or_default();
                        self.deliver(
                            delivery.parent_session_id,
                            CommandPayload::CompleteToolCall {
                                tool_call_id: delivery.tool_call_id.clone(),
                                name: delivery.tool_name.clone(),
                                result,
                                worker_state: None,
                            },
                            init.span.child("sub_agent.deliver"),
                        )
                        .await;
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
                    system: self.clone(),
                    on_event: None,
                    stop_on_done: false,
                },
            )
            .await
            .map_err(|e| RuntimeError::ActorCall(format!("session client startup failed: {e}")))?;

            Ok(SessionHandle {
                session_id,
                trace_id: Some(trace_id),
                session_client: client,
            })
        })
    }

    pub(crate) async fn wake_aggregate(
        &self,
        aggregate_id: Uuid,
        aggregate_type: &str,
        tenant_id: &str,
    ) {
        match aggregate_type {
            "session" => {
                if let Some(cell) =
                    ractor::registry::where_is(aggregate_actor_name(aggregate_id))
                {
                    let actor: ActorRef<AggregateMessage<SessionState>> = cell.into();
                    let _ = actor.send_message(AggregateMessage::Cast {
                        cmd: CommandPayload::Wake,
                        span: SpanContext::root().with_name("wake"),
                        occurred_at: Utc::now(),
                    });
                    return;
                }
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
                let auth = ClientIdentity {
                    tenant_id: tenant_id.to_string(),
                    sub: None,
                    attrs: Default::default(),
                };
                let init = SessionInit {
                    agent_name,
                    auth,
                    on_done: None,
                    span: SpanContext::root().with_name("wake"),
                    stream: false,
                };
                if let Err(e) = self.do_start_session(aggregate_id, init).await {
                    tracing::warn!(session = %aggregate_id, error = %e, "wake: failed to start session");
                }
            }
            _ => {
                tracing::debug!(aggregate_type = %aggregate_type, %aggregate_id, "wake: no handler for aggregate type");
            }
        }
    }

    async fn run_sub_agent(&self, req: SubAgentRequest) {
        let msg_span = req.span.child("sub_agent.message");

        let init = SessionInit {
            agent_name: req.agent_name,
            auth: req.auth,
            on_done: Some(req.delivery),
            span: req.span,
            stream: req.stream,
        };

        let handle = match self.do_start_session(req.session_id, init).await {
            Ok(h) => h,
            Err(e) => {
                tracing::error!(error = %e, "sub-agent start failed");
                return;
            }
        };

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
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn try_send_to_aggregate(
    session_id: Uuid,
    payload: CommandPayload,
    span: SpanContext,
) -> bool {
    if let Some(cell) = ractor::registry::where_is(aggregate_actor_name(session_id)) {
        let actor: ActorRef<AggregateMessage<SessionState>> = cell.into();
        let _ = actor.send_message(AggregateMessage::Cast {
            cmd: payload,
            span,
            occurred_at: Utc::now(),
        });
        true
    } else {
        false
    }
}

/// Spawn the session supervisor actor (pure supervision — no messages).
/// Reuses SupervisorActor from actor.rs.
pub(crate) async fn spawn_session_supervisor(
    runtime_supervisor: ActorCell,
) -> Result<ActorRef<SupervisorMessage>, ractor::SpawnErr> {
    let (actor_ref, _handle) = Actor::spawn_linked(
        Some(SESSION_SUPERVISOR_NAME.to_string()),
        SupervisorActor,
        (),
        runtime_supervisor,
    )
    .await?;
    Ok(actor_ref)
}

// ---------------------------------------------------------------------------
// Test support
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use crate::runtime::event_store::{
        AggregateSummary, Event, EventFilter, EventBatch, StoreError, StreamLoad,
    };
    use crate::runtime::llm::LlmCallable;
    use crate::runtime::session::{ToolCallDispatch, WorkerDispatch};
    use ractor::port::output::OutputPort;

    struct NoopWorkerExecutor;

    impl WorkerExecutor for NoopWorkerExecutor {
        fn dispatch_decision(&self, _request: WorkerDispatch) {}
        fn dispatch_tool_call(&self, _request: ToolCallDispatch) {}
    }

    struct PanicStore(OutputPort<EventBatch>);

    #[async_trait::async_trait]
    impl EventStore for PanicStore {
        async fn append(
            &self, _: Uuid, _: &str, _: &str,
            _: Vec<Event>,
            _: serde_json::Value, _: u64, _: u64,
        ) -> Result<(), StoreError> {
            unimplemented!("test stub")
        }
        async fn load(&self, _: Uuid, _: &str) -> Result<StreamLoad, StoreError> {
            unimplemented!("test stub")
        }
        fn events(&self) -> &OutputPort<EventBatch> { &self.0 }
        async fn list_aggregates(&self, _: &AggregateFilter) -> Vec<AggregateSummary> {
            unimplemented!("test stub")
        }
        async fn query_events(&self, _: &EventFilter) -> Result<Vec<Event>, StoreError> {
            unimplemented!("test stub")
        }
    }

    struct PanicLlmProvider;

    #[async_trait::async_trait]
    impl LlmProviderTrait for PanicLlmProvider {
        async fn resolve(
            &self, _: &str, _: &ClientIdentity,
        ) -> Result<Arc<dyn LlmCallable>, String> {
            unimplemented!("test stub")
        }
    }

    impl SessionSystem {
        /// Create a SessionSystem for unit tests. All methods panic if called.
        pub fn for_test() -> Self {
            Self {
                store: Arc::new(PanicStore(OutputPort::default())),
                llm_provider: Arc::new(PanicLlmProvider),
                worker_executor: Arc::new(NoopWorkerExecutor),
                budget_policies: vec![],
            }
        }
    }
}
