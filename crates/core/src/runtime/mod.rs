use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::providers::memory_queue::TaskQueue;
use aggregate::{execute, Caller, ConflictRetry, ExecuteError, ExecuteInput};
use event_store::{EventStore, StoreError};
use llm::{
    spawn_llm_dispatch_processor, spawn_llm_task_executor, LlmProviderTrait, LlmTask, TokenDelta,
    TokenDeltaTransport,
};
use owner::SessionOwner;
use processor::ProcessorCheckpointStore;
use retry::{NoRetryResolver, WorkerRetryResolver};
use session::command::{CommandPayload, SessionError};
use session::decision::ClientPayload;
use session::index::{
    spawn_session_index_processor, SessionFilter, SessionIndexStore, SessionPage,
};
use session::state::SessionState;
use session::subscriptions::SessionSubscriptionSpec;
use span::SpanContext;
use sub_agent::{spawn_sub_agent_dispatch_processor, spawn_sub_agent_task_executor, SubAgentTask};
use wake::{spawn_wake_dispatcher, spawn_wake_processor, WakeScheduleStore};
use worker::spawn_worker_processor;
use worker::{DequeueFilter, FailDecision, SubmitDecision, WorkerDecisionRequest, WorkerQueue};

pub mod aggregate;
pub mod event_store;
pub mod llm;
pub mod owner;
pub mod processor;
pub mod retry;
pub mod serde_helpers;
pub mod session;
pub mod span;
pub mod sub_agent;
pub mod wake;
pub mod worker;

pub struct RuntimeConfig {
    pub llm_executor_workers: usize,
    pub sub_agent_executor_workers: usize,
    pub wake_poll_interval: std::time::Duration,
    pub shutdown_timeout: std::time::Duration,
    pub worker_retry_resolver: Arc<dyn WorkerRetryResolver>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            llm_executor_workers: 4,
            sub_agent_executor_workers: 2,
            wake_poll_interval: std::time::Duration::from_secs(30),
            shutdown_timeout: std::time::Duration::from_secs(5),
            worker_retry_resolver: Arc::new(NoRetryResolver),
        }
    }
}

pub struct Runtime {
    store: Arc<dyn EventStore>,
    queue: Arc<dyn WorkerQueue>,
    session_index: Arc<dyn SessionIndexStore>,
    session_subscriptions: session::subscriptions::SessionSubscriptions,
    token_delta_transport: Arc<dyn TokenDeltaTransport>,
    cancel: CancellationToken,
    handles: tokio::sync::Mutex<Vec<JoinHandle<()>>>,
    shutdown_timeout: Duration,
    worker_retry_resolver: Arc<dyn WorkerRetryResolver>,
}

pub struct SubmitClientPayload {
    pub session_id: String,
    pub caller: Caller,
    pub owner: SessionOwner,
    pub agent_id: String,
    pub payload: ClientPayload,
    /// Caller-provided turn ID for idempotency. Auto-generated if None.
    pub turn_id: Option<String>,
}

pub struct SubmitClientPayloadOutput {
    pub session_id: String,
    pub turn_id: String,
}

pub enum SubmitToolCallResult {
    Result { result: String },
    Error { error: String, retryable: bool },
}

pub struct SubmitToolCallResultInput {
    pub session_id: String,
    pub tool_call_id: String,
    pub attempt: u32,
    pub result: SubmitToolCallResult,
    pub caller: Caller,
    pub span: SpanContext,
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error(transparent)]
    Session(#[from] SessionError),
    #[error("{0}")]
    Internal(String),
}

impl From<ExecuteError<SessionError>> for RuntimeError {
    fn from(e: ExecuteError<SessionError>) -> Self {
        match e {
            ExecuteError::Command(c) => RuntimeError::Session(c),
            other => RuntimeError::Internal(other.to_string()),
        }
    }
}

impl Runtime {
    pub async fn shutdown(&self) {
        self.cancel.cancel();

        let mut guard = self.handles.lock().await;
        let handles: Vec<_> = guard.drain(..).collect();
        drop(guard);

        let abort_handles: Vec<_> = handles.iter().map(|h| h.abort_handle()).collect();

        let join_all = async {
            for handle in handles {
                let _ = handle.await;
            }
        };

        if tokio::time::timeout(self.shutdown_timeout, join_all)
            .await
            .is_err()
        {
            tracing::warn!("shutdown timed out, aborting remaining tasks");
            for handle in &abort_handles {
                handle.abort();
            }
        }
    }

    pub async fn dequeue_decision(&self, filter: &DequeueFilter) -> Option<WorkerDecisionRequest> {
        self.queue.dequeue(filter).await
    }

    /// Submit a client payload to a session. Returns immediately with the
    /// resolved session and turn ids; an idempotent re-submission resolves to
    /// the existing turn id. Use `stream` to observe events, including
    /// completion of an already-finished turn.
    pub async fn submit_client_payload(
        &self,
        input: SubmitClientPayload,
    ) -> Result<SubmitClientPayloadOutput, RuntimeError> {
        let session_id = input.session_id;
        let turn_id = input.turn_id.unwrap_or_else(|| Uuid::now_v7().to_string());

        let span = SpanContext::root();

        let worker_retry = self
            .worker_retry_resolver
            .resolve(input.caller.tenant_id())
            .await;

        // Create session (ignore if already exists)
        let create_result = execute::<SessionState>(
            &*self.store,
            ExecuteInput {
                aggregate_id: session_id.clone(),
                caller: input.caller.clone(),
                command: CommandPayload::CreateSession {
                    agent_id: input.agent_id,
                    owner: input.owner,
                    ancestry: vec![],
                    worker_retry,
                },
                span: span.child("create_session"),
            },
            &ConflictRetry::default(),
        )
        .await;

        match create_result {
            Ok(_) => {}
            Err(ExecuteError::Command(SessionError::SessionAlreadyCreated)) => {}
            Err(e) => return Err(e.into()),
        }

        // Try to submit the payload (idempotency guard may reject)
        let send_result = execute::<SessionState>(
            &*self.store,
            ExecuteInput {
                aggregate_id: session_id.clone(),
                caller: input.caller,
                command: CommandPayload::SubmitClientPayload {
                    payload: input.payload,
                    turn_id: Some(turn_id.clone()),
                },
                span: span.child("submit_client_payload"),
            },
            &ConflictRetry::default(),
        )
        .await;

        let effective_turn_id = match send_result {
            Ok(_) => turn_id,
            Err(ExecuteError::Command(SessionError::TurnAlreadyActive { turn_id })) => turn_id,
            Err(ExecuteError::Command(SessionError::TurnAlreadyCompleted { turn_id })) => turn_id,
            Err(e) => return Err(e.into()),
        };

        Ok(SubmitClientPayloadOutput {
            session_id,
            turn_id: effective_turn_id,
        })
    }

    /// Unified event stream. `spec` selects which events to observe (a single
    /// turn or the whole session); `sequence_after` optionally replays
    /// historical events with `sequence > N` before streaming live.
    pub async fn stream(
        &self,
        spec: SessionSubscriptionSpec,
        sequence_after: Option<u64>,
    ) -> Result<mpsc::Receiver<event_store::Event>, RuntimeError> {
        self.authorize_session_read(&spec.root_session_id, &spec.caller)
            .await?;
        Ok(self
            .session_subscriptions
            .stream(spec, sequence_after)
            .await)
    }

    /// Authorize a read of a session's events — the shared business-logic
    /// ownership check for every read path (live streams and history snapshots).
    /// An end-user owner may only read a session they own (its creating
    /// `owner.id` matches); privileged callers are unrestricted within the
    /// tenant. An uncreated session has no owner yet, so the read is allowed (it
    /// is simply empty, and the first turn binds the session to its caller).
    pub async fn authorize_session_read(
        &self,
        session_id: &str,
        caller: &Caller,
    ) -> Result<(), RuntimeError> {
        // Only a frontend caller is restricted — to sessions it owns, within its
        // own tenant. System and machine callers are unrestricted.
        let Caller::Frontend {
            tenant_id, user_id, ..
        } = caller
        else {
            return Ok(());
        };

        match self.store.load(tenant_id, session_id).await {
            Ok(snapshot) => {
                let agg: aggregate::Aggregate<SessionState> = serde_json::from_value(snapshot.data)
                    .map_err(|e| RuntimeError::Internal(e.to_string()))?;

                let owner = agg.state.owner.as_ref().and_then(|i| i.id.as_deref());

                if owner == Some(user_id) {
                    Ok(())
                } else {
                    Err(RuntimeError::Session(SessionError::SessionAccessDenied))
                }
            }
            // An uncreated session has no owner yet — nothing to leak.
            Err(StoreError::StreamNotFound) => Ok(()),
            Err(e) => Err(RuntimeError::Internal(e.to_string())),
        }
    }

    pub async fn subscribe_token_deltas(
        &self,
        tenant_id: &str,
        root_session_id: &str,
    ) -> mpsc::Receiver<TokenDelta> {
        self.token_delta_transport
            .subscribe(tenant_id, root_session_id)
            .await
    }

    // ---- Admin / inspection methods ----

    pub async fn list_sessions(&self, filter: &SessionFilter) -> Result<SessionPage, RuntimeError> {
        self.session_index
            .list_sessions(filter)
            .await
            .map_err(|e| RuntimeError::Internal(e.to_string()))
    }

    pub async fn count_sessions(&self, filter: &SessionFilter) -> Result<u64, RuntimeError> {
        self.session_index
            .count_sessions(filter)
            .await
            .map_err(|e| RuntimeError::Internal(e.to_string()))
    }

    pub async fn get_session(
        &self,
        tenant_id: &str,
        session_id: &str,
    ) -> Result<(event_store::Snapshot, SessionState), RuntimeError> {
        let snapshot = self
            .store
            .load(tenant_id, session_id)
            .await
            .map_err(|e| RuntimeError::Internal(e.to_string()))?;
        let agg: aggregate::Aggregate<SessionState> = serde_json::from_value(snapshot.data.clone())
            .map_err(|e| RuntimeError::Internal(e.to_string()))?;
        let state = agg.state;
        Ok((snapshot, state))
    }

    pub async fn get_session_events(
        &self,
        filter: &event_store::EventFilter,
    ) -> Result<Vec<event_store::Event>, RuntimeError> {
        self.store
            .query_events(filter)
            .await
            .map_err(|e| RuntimeError::Internal(e.to_string()))
    }

    /// Authorized read of a session's full event history, tenant-scoped.
    ///
    /// The snapshot twin of [`Runtime::stream`]: it runs the same
    /// `caller`-driven ownership gate before reading, so the authorization and
    /// the tenant scoping each live in exactly one place and there is no un-gated
    /// path to a session's history. Transport hands in `(tenant, session, caller)`
    /// and gets back authorized events to project; it never assembles an
    /// `EventFilter` or pairs a separate authorize call by hand.
    pub async fn read_session_events(
        &self,
        tenant_id: &str,
        session_id: &str,
        caller: &Caller,
    ) -> Result<Vec<event_store::Event>, RuntimeError> {
        self.authorize_session_read(session_id, caller).await?;
        let filter = event_store::EventFilter {
            aggregate_id: Some(session_id.to_string()),
            tenant_id: Some(tenant_id.to_string()),
            ..Default::default()
        };
        self.get_session_events(&filter).await
    }

    pub async fn submit_decision(&self, input: SubmitDecision) -> Result<(), RuntimeError> {
        execute::<SessionState>(
            &*self.store,
            ExecuteInput {
                aggregate_id: input.session_id.clone(),
                caller: input.caller,
                command: CommandPayload::SubmitWorkerDecision {
                    decision_id: input.decision_id,
                    actions: input.actions,
                    state: input.state,
                },
                span: input.span,
            },
            &ConflictRetry::default(),
        )
        .await
        .map(|_| ())
        .map_err(RuntimeError::from)
    }

    pub async fn submit_tool_call_result(
        &self,
        input: SubmitToolCallResultInput,
    ) -> Result<(), RuntimeError> {
        let command = match input.result {
            SubmitToolCallResult::Result { result } => CommandPayload::CompleteToolCall {
                tool_call_id: input.tool_call_id,
                attempt: input.attempt,
                result,
                worker_state: None,
            },
            SubmitToolCallResult::Error { error, retryable } => CommandPayload::FailToolCall {
                tool_call_id: input.tool_call_id,
                attempt: input.attempt,
                error,
                retryable,
                worker_state: None,
            },
        };

        execute::<SessionState>(
            &*self.store,
            ExecuteInput {
                aggregate_id: input.session_id,
                caller: input.caller,
                command,
                span: input.span,
            },
            &ConflictRetry::default(),
        )
        .await
        .map(|_| ())
        .map_err(RuntimeError::from)
    }

    pub async fn fail_decision(&self, input: FailDecision) -> Result<(), RuntimeError> {
        execute::<SessionState>(
            &*self.store,
            ExecuteInput {
                aggregate_id: input.session_id.clone(),
                caller: input.caller,
                command: CommandPayload::FailWorkerDecision {
                    decision_id: input.decision_id,
                    error: input.error,
                    retryable: input.retryable,
                },
                span: input.span,
            },
            &ConflictRetry::default(),
        )
        .await
        .map(|_| ())
        .map_err(RuntimeError::from)
    }
}

pub fn start(
    store: Arc<dyn EventStore>,
    llm_provider: Arc<dyn LlmProviderTrait>,
    llm_task_queue: Arc<dyn TaskQueue<LlmTask>>,
    sub_agent_task_queue: Arc<dyn TaskQueue<SubAgentTask>>,
    worker_queue: Arc<dyn WorkerQueue>,
    session_index_store: Arc<dyn SessionIndexStore>,
    checkpoint_store: Arc<dyn ProcessorCheckpointStore>,
    wake_store: Arc<dyn WakeScheduleStore>,
    token_delta_transport: Arc<dyn TokenDeltaTransport>,
    config: RuntimeConfig,
) -> Arc<Runtime> {
    let cancel = CancellationToken::new();

    let llm_processor_handle = spawn_llm_dispatch_processor(
        store.clone(),
        checkpoint_store.clone(),
        llm_task_queue.clone(),
        cancel.clone(),
    );
    let llm_executor_handles = spawn_llm_task_executor(
        store.clone(),
        llm_provider,
        llm_task_queue,
        token_delta_transport.clone(),
        config.llm_executor_workers,
        cancel.clone(),
    );

    let sub_agent_processor_handle = spawn_sub_agent_dispatch_processor(
        store.clone(),
        checkpoint_store.clone(),
        sub_agent_task_queue.clone(),
        cancel.clone(),
    );
    let sub_agent_executor_handles = spawn_sub_agent_task_executor(
        store.clone(),
        sub_agent_task_queue,
        config.sub_agent_executor_workers,
        cancel.clone(),
    );

    let worker_handle = spawn_worker_processor(
        store.clone(),
        checkpoint_store.clone(),
        worker_queue.clone(),
        cancel.clone(),
    );
    let session_index_processor_handle = spawn_session_index_processor(
        store.clone(),
        checkpoint_store.clone(),
        session_index_store.clone(),
        cancel.clone(),
    );
    let wake_processor_handle = spawn_wake_processor(
        store.clone(),
        checkpoint_store,
        wake_store.clone(),
        cancel.clone(),
    );

    let wake_dispatcher_handle = spawn_wake_dispatcher(
        store.clone(),
        wake_store,
        config.wake_poll_interval,
        cancel.clone(),
    );

    let session_subscriptions = session::subscriptions::SessionSubscriptions::new(store.clone());

    let mut handles = vec![
        llm_processor_handle,
        sub_agent_processor_handle,
        worker_handle,
        session_index_processor_handle,
        wake_processor_handle,
        wake_dispatcher_handle,
    ];
    handles.extend(llm_executor_handles);
    handles.extend(sub_agent_executor_handles);

    Arc::new(Runtime {
        store,
        queue: worker_queue,
        session_index: session_index_store,
        session_subscriptions,
        token_delta_transport,
        cancel,
        handles: tokio::sync::Mutex::new(handles),
        shutdown_timeout: config.shutdown_timeout,
        worker_retry_resolver: config.worker_retry_resolver,
    })
}
