use std::collections::HashMap;
use std::sync::Arc;

use tokio::task::JoinHandle;
use uuid::Uuid;

use aggregate::{execute, ExecuteError, ExecuteInput};
use event_store::{spawn_handler_pool, EventStore};
use identity::ClientIdentity;
use llm::handler::LlmEventHandler;
use llm::LlmProviderTrait;
use retry::RetryPolicy;
use session::command::{CommandPayload, SessionError};
use session::state::SessionState;
use span::SpanContext;
use wake::spawn_wake_scheduler;
use worker::spawn_worker_enqueue;
use worker::{DequeueFilter, PendingDecision, SubmitDecision, WorkerQueue};

pub mod aggregate;
pub mod event_store;
pub mod identity;
pub mod llm;
pub mod retry;
pub mod serde_helpers;
pub mod session;
pub mod span;
pub mod wake;
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

    pub async fn dequeue_decision(&self, filter: &DequeueFilter) -> Option<PendingDecision> {
        self.queue.dequeue(filter).await
    }

    pub async fn send_message(&self, input: SendMessage) -> Result<(), RuntimeError> {
        let span = SpanContext::root();

        // Create session (ignore if already exists)
        let create_result = execute::<SessionState>(
            &*self.store,
            ExecuteInput {
                aggregate_id: input.session_id,
                tenant_id: input.tenant_id.clone(),
                command: CommandPayload::CreateSession {
                    agent_id: input.agent_id,
                    auth: ClientIdentity {
                        tenant_id: input.tenant_id.clone(),
                        sub: None,
                        attrs: HashMap::new(),
                    },
                    on_done: None,
                    worker_retry: RetryPolicy::no_retry(),
                },
                span: span.child("create_session"),
            },
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
                aggregate_id: input.session_id,
                tenant_id: input.tenant_id,
                command: CommandPayload::SendUserMessage {
                    content: input.content,
                    stream: false,
                },
                span: span.child("send_message"),
            },
        )
        .await
        .map(|_| ())
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
    config: RuntimeConfig,
) -> Runtime {
    let llm_handler = Arc::new(LlmEventHandler::new(store.clone(), llm_provider));
    let llm_handle = spawn_handler_pool::<SessionState>(store.clone(), llm_handler, config.llm_pool_size);

    let worker_handle = spawn_worker_enqueue(store.clone(), worker_queue.clone());
    let wake_handle = spawn_wake_scheduler(store.clone(), config.wake_poll_interval);

    Runtime {
        store,
        queue: worker_queue,
        handles: vec![llm_handle, worker_handle, wake_handle],
    }
}
