use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::providers::memory_queue::TaskQueue;
use aggregate::{execute, ConflictRetry, ExecuteError, ExecuteInput};
use event_store::EventStore;
use identity::ClientIdentity;
use llm::{spawn_llm_dispatch_processor, spawn_llm_task_executor, LlmProviderTrait, LlmTask};
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
use worker::{DequeueFilter, SubmitDecision, WorkerDecisionRequest, WorkerQueue};

pub mod aggregate;
pub mod event_store;
pub mod identity;
pub mod llm;
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
    cancel: CancellationToken,
    handles: tokio::sync::Mutex<Vec<JoinHandle<()>>>,
    shutdown_timeout: Duration,
    worker_retry_resolver: Arc<dyn WorkerRetryResolver>,
}

pub struct SubmitClientPayload {
    pub session_id: String,
    pub tenant_id: String,
    pub identity: ClientIdentity,
    pub agent_id: String,
    pub payload: ClientPayload,
    /// Caller-provided turn ID for idempotency. Auto-generated if None.
    pub turn_id: Option<String>,
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct RuntimeError(String);

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

    /// Submit a client payload to a session and return a stream of events for that session.
    ///
    /// Subscribes to the event store broadcast *before* sending so no events are missed.
    /// The returned receiver yields events until the broadcast closes or the receiver is dropped.
    pub async fn submit_client_payload(
        &self,
        input: SubmitClientPayload,
    ) -> Result<(String, mpsc::Receiver<event_store::Event>), RuntimeError> {
        let session_id = input.session_id;
        let turn_id = input.turn_id.unwrap_or_else(|| Uuid::now_v7().to_string());

        let span = SpanContext::root();

        let worker_retry = self.worker_retry_resolver.resolve(&input.tenant_id).await;

        // Create session (ignore if already exists)
        let create_result = execute::<SessionState>(
            &*self.store,
            ExecuteInput {
                aggregate_id: session_id.clone(),
                tenant_id: input.tenant_id.clone(),
                command: CommandPayload::CreateSession {
                    agent_id: input.agent_id,
                    identity: input.identity.clone(),
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
            Err(e) => return Err(RuntimeError(e.to_string())),
        }

        // Try to submit the payload (idempotency guard may reject)
        let send_result = execute::<SessionState>(
            &*self.store,
            ExecuteInput {
                aggregate_id: session_id.clone(),
                tenant_id: input.tenant_id,
                command: CommandPayload::SubmitClientPayload {
                    payload: input.payload,
                    identity: input.identity,
                    turn_id: Some(turn_id.clone()),
                },
                span: span.child("submit_client_payload"),
            },
            &ConflictRetry::default(),
        )
        .await;

        // Determine effective turn_id and whether the turn is already completed
        let (effective_turn_id, is_completed) = match send_result {
            Ok(_) => (turn_id, false),
            Err(ExecuteError::Command(SessionError::TurnAlreadyActive { turn_id })) => {
                (turn_id, false)
            }
            Err(ExecuteError::Command(SessionError::TurnAlreadyCompleted { turn_id })) => {
                (turn_id, true)
            }
            Err(e) => return Err(RuntimeError(e.to_string())),
        };

        // Build event stream: replay historical + live (if active)
        if is_completed {
            self.session_subscriptions
                .replay(SessionSubscriptionSpec::Turn {
                    root_session_id: session_id,
                    turn_id: effective_turn_id,
                })
                .await
        } else {
            self.session_subscriptions
                .catchup(SessionSubscriptionSpec::Turn {
                    root_session_id: session_id,
                    turn_id: effective_turn_id,
                })
                .await
        }
    }

    /// Subscribe to events for a session, filtered by session ID.
    /// Closes when the given turn completes.
    pub fn subscribe_session(
        &self,
        session_id: String,
        turn_id: String,
    ) -> mpsc::Receiver<event_store::Event> {
        self.session_subscriptions
            .subscribe(SessionSubscriptionSpec::Turn {
                root_session_id: session_id,
                turn_id,
            })
    }

    /// Subscribe to all events for a session (and descendants).
    /// Never auto-closes — stays open until the receiver is dropped.
    pub fn subscribe_session_admin(
        &self,
        session_id: String,
    ) -> mpsc::Receiver<event_store::Event> {
        self.session_subscriptions
            .subscribe(SessionSubscriptionSpec::All {
                root_session_id: session_id,
            })
    }

    // ---- Admin / inspection methods ----

    pub async fn list_sessions(&self, filter: &SessionFilter) -> Result<SessionPage, RuntimeError> {
        self.session_index
            .list_sessions(filter)
            .await
            .map_err(|e| RuntimeError(e.to_string()))
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
            .map_err(|e| RuntimeError(e.to_string()))?;
        let agg: aggregate::Aggregate<SessionState> = serde_json::from_value(snapshot.data.clone())
            .map_err(|e| RuntimeError(e.to_string()))?;
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
            .map_err(|e| RuntimeError(e.to_string()))
    }

    pub async fn submit_decision(&self, input: SubmitDecision) -> Result<(), RuntimeError> {
        execute::<SessionState>(
            &*self.store,
            ExecuteInput {
                aggregate_id: input.session_id.clone(),
                tenant_id: input.tenant_id,
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
        .map_err(|e| RuntimeError(e.to_string()))
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
        cancel,
        handles: tokio::sync::Mutex::new(handles),
        shutdown_timeout: config.shutdown_timeout,
        worker_retry_resolver: config.worker_retry_resolver,
    })
}
