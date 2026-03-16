use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{broadcast, mpsc};
use tokio::task::JoinHandle;
use uuid::Uuid;

use aggregate::{execute, ConflictRetry, ExecuteError, ExecuteInput};
use event_store::{spawn_handler_pool, EventStore};
use identity::ClientIdentity;
use llm::handler::LlmEventHandler;
use llm::LlmProviderTrait;
use sub_agent::SubAgentHandler;
use retry::RetryPolicy;
use session::command::{CommandPayload, SessionError};
use session::state::SessionState;
use session_index::{SessionFilter, SessionIndex, SessionPage};
use span::SpanContext;
use wake::spawn_wake_scheduler;
use worker::spawn_worker_enqueue;
use worker::{DequeueFilter, WorkerDecisionRequest, SubmitDecision, WorkerQueue};

pub mod aggregate;
pub mod event_store;
pub mod identity;
pub mod llm;
pub mod retry;
pub mod serde_helpers;
pub mod session;
pub mod span;
pub mod sub_agent;
pub mod wake;
pub mod session_index;
pub mod worker;

pub struct RuntimeConfig {
    pub llm_pool_size: usize,
    pub wake_poll_interval: std::time::Duration,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            llm_pool_size: 4,
            wake_poll_interval: std::time::Duration::from_secs(30),
        }
    }
}

pub struct Runtime {
    store: Arc<dyn EventStore>,
    queue: Arc<dyn WorkerQueue>,
    session_index: Arc<dyn SessionIndex>,
    handles: Vec<JoinHandle<()>>,
}

pub struct SendMessage {
    pub session_id: Uuid,
    pub tenant_id: String,
    pub agent_id: String,
    pub content: String,
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct RuntimeError(String);

impl Runtime {
    pub fn shutdown(self) {
        for handle in self.handles {
            handle.abort();
        }
    }

    pub(crate) async fn dequeue_decision(&self, filter: &DequeueFilter) -> Option<WorkerDecisionRequest> {
        self.queue.dequeue(filter).await
    }

    /// Send a message to a session and return a stream of events for that session.
    ///
    /// Subscribes to the event store broadcast *before* sending so no events are missed.
    /// The returned receiver yields events until the broadcast closes or the receiver is dropped.
    pub async fn send_message(
        &self,
        input: SendMessage,
    ) -> Result<(Uuid, mpsc::Receiver<event_store::Event>), RuntimeError> {
        let session_id = input.session_id;
        let turn_id = Uuid::now_v7().to_string();

        let event_rx = self.subscribe_session(session_id, turn_id.clone());

        let span = SpanContext::root();

        // Create session (ignore if already exists)
        let create_result = execute::<SessionState>(
            &*self.store,
            ExecuteInput {
                aggregate_id: session_id,
                tenant_id: input.tenant_id.clone(),
                command: CommandPayload::CreateSession {
                    agent_id: input.agent_id,
                    auth: ClientIdentity {
                        tenant_id: input.tenant_id.clone(),
                        sub: None,
                        attrs: HashMap::new(),
                    },
                    ancestry: vec![],
                    worker_retry: RetryPolicy::no_retry(),
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

        // Send the message
        execute::<SessionState>(
            &*self.store,
            ExecuteInput {
                aggregate_id: session_id,
                tenant_id: input.tenant_id,
                command: CommandPayload::SendMessage {
                    message: session::message::Message {
                        role: session::message::Role::User,
                        content: Some(input.content),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    },
                    stream: false,
                    turn_id: Some(turn_id.clone()),
                },
                span: span.child("send_message"),
            },
            &ConflictRetry::default(),
        )
        .await
        .map_err(|e| RuntimeError(e.to_string()))?;

        Ok((session_id, event_rx))
    }

    /// Subscribe to events for a session, filtered by session ID.
    /// Closes when the given turn completes.
    /// Must be called *before* the command that triggers events so nothing is missed.
    pub fn subscribe_session(
        &self,
        session_id: Uuid,
        turn_id: String,
    ) -> mpsc::Receiver<event_store::Event> {
        let mut rx = self.store.subscribe();
        let (tx, event_rx) = mpsc::channel::<event_store::Event>(64);

        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(batch) => {
                        for event in batch.iter() {
                            let is_self = event.aggregate_id == session_id;
                            let is_descendant = !is_self
                                && event.derived.as_ref().and_then(|d| {
                                    serde_json::from_value::<session::state::DerivedState>(d.clone()).ok()
                                }).map_or(false, |d| d.ancestry.contains(&session_id));

                            if !is_self && !is_descendant {
                                continue;
                            }
                            if tx.send(event.clone()).await.is_err() {
                                return;
                            }
                            if is_self {
                                if let Ok(de) = aggregate::DomainEvent::<SessionState>::from_raw(event) {
                                    if matches!(&de.payload, session::events::EventPayload::TurnCompleted(tc) if tc.turn_id == turn_id) {
                                        return;
                                    }
                                }
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(_) => return,
                }
            }
        });

        event_rx
    }

    /// Subscribe to all events for a session (and descendants).
    /// Never auto-closes — stays open until the receiver is dropped.
    pub fn subscribe_session_admin(
        &self,
        session_id: Uuid,
    ) -> mpsc::Receiver<event_store::Event> {
        let mut rx = self.store.subscribe();
        let (tx, event_rx) = mpsc::channel::<event_store::Event>(64);

        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(batch) => {
                        for event in batch.iter() {
                            let is_self = event.aggregate_id == session_id;
                            let is_descendant = !is_self
                                && event.derived.as_ref().and_then(|d| {
                                    serde_json::from_value::<session::state::DerivedState>(
                                        d.clone(),
                                    )
                                    .ok()
                                })
                                .map_or(false, |d| d.ancestry.contains(&session_id));

                            if !is_self && !is_descendant {
                                continue;
                            }
                            if tx.send(event.clone()).await.is_err() {
                                return;
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(_) => return,
                }
            }
        });

        event_rx
    }

    // ---- Admin / inspection methods ----

    pub async fn list_sessions(
        &self,
        filter: &SessionFilter,
    ) -> Result<SessionPage, RuntimeError> {
        self.session_index
            .list_sessions(filter)
            .await
            .map_err(|e| RuntimeError(e.to_string()))
    }

    pub async fn get_session(
        &self,
        session_id: Uuid,
    ) -> Result<(event_store::Snapshot, SessionState), RuntimeError> {
        let snapshot = self
            .store
            .load(session_id)
            .await
            .map_err(|e| RuntimeError(e.to_string()))?;
        let agg: aggregate::Aggregate<SessionState> =
            serde_json::from_value(snapshot.data.clone())
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
                aggregate_id: input.session_id,
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
    worker_queue: Arc<dyn WorkerQueue>,
    session_index: Arc<dyn SessionIndex>,
    config: RuntimeConfig,
) -> Arc<Runtime> {
    let llm_handler = Arc::new(LlmEventHandler::new(store.clone(), llm_provider));
    let llm_handle =
        spawn_handler_pool::<SessionState>(store.clone(), llm_handler, config.llm_pool_size);

    let sub_agent_handler = Arc::new(SubAgentHandler::new(store.clone()));
    let sub_agent_handle =
        spawn_handler_pool::<SessionState>(store.clone(), sub_agent_handler, 2);

    let worker_handle = spawn_worker_enqueue(store.clone(), worker_queue.clone());
    let wake_handle = spawn_wake_scheduler(store.clone(), config.wake_poll_interval);

    Arc::new(Runtime {
        store,
        queue: worker_queue,
        session_index,
        handles: vec![llm_handle, sub_agent_handle, worker_handle, wake_handle],
    })
}
