use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::connectors::registry::Connections;
use crate::protocol::{
    AgentConfig, ClientAction, ClientAppend, ClientInput, ClientMessage, ClientMessages,
    ClientPayload, ErrorCode, ErrorInfo, InterruptResumption, SessionOwner, TokenDelta, WorkerRef,
};
use crate::providers::memory_queue::TaskQueue;
use crate::runtime::blob::BlobStore;
use connector::{spawn_connector_dispatch_processor, spawn_connector_task_executor, ConnectorTask};
use event_store::{EventStore, Seq};
use llm::{
    spawn_llm_dispatch_processor, spawn_llm_task_executor, LlmBlocks, LlmResolver, LlmTask,
    TokenDeltaTransport,
};
use processor::{
    EventProcessor, EventProcessorRunner, EventProcessorRunnerConfig, ProcessorCursorStore,
};
use retry::{DefaultWorkerRetryResolver, WorkerRetryResolver};
use session::command::{CommandPayload, Outcome, SessionError, SettleError, TurnTarget};
use session::decision::{EffectResultPayload, WorkKind};
use session::index::{
    spawn_session_index_processor, SessionFilter, SessionIndexStore, SessionPage,
};
use session::read::SessionReader;
use session::state::EffectKind;
use session::subscriptions::SessionSubscriptionSpec;
use session::{execute, ConflictRetry, ExecuteError, ExecuteInput, SessionAggregate, SessionEvent};
use span::SpanContext;
use subagent::{spawn_subagent_dispatch_processor, spawn_subagent_task_executor, SubagentTask};
use wake::{spawn_boot_reconciler, spawn_wake_dispatcher, spawn_wake_processor, WakeScheduleStore};
use worker::spawn_worker_processor;
use worker::{
    AgentDirectory, DequeueFilter, FailDecision, SubmitDecision, WorkerDecisionRequest, WorkerQueue,
};

pub mod blob;
mod caller;
pub mod connector;
pub mod event_store;
pub mod executor;
pub mod llm;
pub mod processor;
pub mod retry;
pub mod secret;
pub mod session;
pub mod span;
pub mod subagent;
pub mod wake;
pub mod worker;

pub use caller::Caller;

pub struct RuntimeConfig {
    pub llm_executor_workers: usize,
    pub subagent_executor_workers: usize,
    pub connector_executor_workers: usize,
    pub executor_concurrency: usize,
    pub wake_poll_interval: std::time::Duration,
    pub shutdown_timeout: std::time::Duration,
    pub worker_retry_resolver: Arc<dyn WorkerRetryResolver>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            llm_executor_workers: 32,
            subagent_executor_workers: 32,
            connector_executor_workers: 64,
            executor_concurrency: 32,
            wake_poll_interval: std::time::Duration::from_secs(30),
            shutdown_timeout: std::time::Duration::from_secs(5),
            worker_retry_resolver: Arc::new(DefaultWorkerRetryResolver),
        }
    }
}

impl RuntimeConfig {
    fn pool(&self, workers: usize) -> executor::ExecutorPool {
        executor::ExecutorPool {
            workers,
            concurrency: self.executor_concurrency,
        }
    }
}

pub struct Runtime {
    store: Arc<dyn EventStore>,
    queue: Arc<dyn WorkerQueue>,
    agents: Arc<dyn AgentDirectory>,
    session_index: Arc<dyn SessionIndexStore>,
    session_subscriptions: session::subscriptions::SessionSubscriptions,
    token_delta_transport: Arc<dyn TokenDeltaTransport>,
    cursor_store: Arc<dyn ProcessorCursorStore>,
    cancel: CancellationToken,
    handles: tokio::sync::Mutex<Vec<JoinHandle<()>>>,
    shutdown_timeout: Duration,
    worker_retry_resolver: Arc<dyn WorkerRetryResolver>,
    blobs: Arc<dyn BlobStore>,
}

pub struct SubmitClientPayload {
    pub session_id: String,
    pub caller: Caller,
    pub owner: SessionOwner,
    pub agent_id: String,
    pub payload: ClientPayload,

    pub agent: Option<AgentConfig>,

    pub worker: Option<WorkerRef>,

    pub turn_id: Option<String>,

    pub continue_turn: bool,

    pub queue: bool,
}

pub struct SubmitClientPayloadOutput {
    pub session_id: String,
    pub turn_id: String,

    pub queued: bool,
}

pub struct HandleClientInput {
    pub session_id: String,
    pub caller: Caller,
    pub owner: SessionOwner,
    pub input: ClientInput,
    pub agent: Option<AgentConfig>,
    pub worker: Option<WorkerRef>,
    pub span: SpanContext,
}

pub struct ClientInputOutput {
    pub session_id: String,
    pub turn_id: String,

    pub queued: bool,
}

#[derive(Debug)]
pub enum EffectSettlement {
    Result(EffectResultPayload),
    Error {
        error: String,
        retryable: bool,
        code: Option<ErrorCode>,
        detail: Option<serde_json::Value>,
    },
}

pub struct SettleEffectInput {
    pub session_id: String,
    pub kind: WorkKind,
    pub id: String,

    pub attempt: Option<u32>,
    pub settlement: EffectSettlement,
    pub caller: Caller,
    pub span: SpanContext,
}

pub struct InterruptSessionInput {
    pub session_id: String,
    pub interrupt_id: String,
    pub reason: String,
    pub payload: serde_json::Value,
    pub caller: Caller,
    pub span: SpanContext,
}

pub struct ResumeInterruptInput {
    pub session_id: String,
    pub interrupt_id: String,
    pub payload: serde_json::Value,
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

impl From<ExecuteError> for RuntimeError {
    fn from(e: ExecuteError) -> Self {
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

    pub async fn create_session(
        &self,
        session_id: &str,
        caller: &Caller,
        agent_id: String,
        owner: SessionOwner,
        agent: Option<AgentConfig>,
        worker: Option<WorkerRef>,
    ) -> Result<bool, RuntimeError> {
        let worker_retry = self.worker_retry_resolver.resolve(caller.tenant_id()).await;
        let result = execute(
            &*self.store,
            ExecuteInput {
                session_id: session_id.to_string(),
                caller: caller.clone(),
                command: worker::directory::create_session_command(
                    &*self.agents,
                    agent_id,
                    owner,
                    vec![],
                    worker_retry,
                    agent,
                    worker,
                ),
                span: SpanContext::root().child("create_session"),
            },
            &ConflictRetry::default(),
        )
        .await;

        match result {
            Ok(_) => Ok(true),
            Err(ExecuteError::Command(SessionError::SessionAlreadyCreated)) => Ok(false),
            Err(e) => Err(e.into()),
        }
    }

    pub async fn submit_client_payload(
        &self,
        input: SubmitClientPayload,
    ) -> Result<SubmitClientPayloadOutput, RuntimeError> {
        let session_id = input.session_id;
        let turn_id = input.turn_id.unwrap_or_else(|| Uuid::now_v7().to_string());
        let caller = input.caller.clone();

        let span = SpanContext::root();

        self.create_session(
            &session_id,
            &input.caller,
            input.agent_id,
            input.owner,
            input.agent,
            input.worker,
        )
        .await?;

        let send_result = execute(
            &*self.store,
            ExecuteInput {
                session_id: session_id.clone(),
                caller: input.caller,
                command: CommandPayload::SubmitClientPayload {
                    turn: match (&input.payload, input.continue_turn) {
                        (ClientPayload::Action(_), _) => TurnTarget::Detached,
                        (_, true) => TurnTarget::Continue(turn_id.clone()),
                        (_, false) => TurnTarget::Open(turn_id.clone()),
                    },
                    payload: input.payload,
                    queue: input.queue,
                },
                span: span.child("submit_client_payload"),
            },
            &ConflictRetry::default(),
        )
        .await;

        let (effective_turn_id, queued) = match send_result {
            Ok(out) if input.continue_turn => (out.turn_id.unwrap_or(turn_id), false),
            Ok(out) => {
                let queued = input.queue && out.turn_id.as_deref() != Some(turn_id.as_str());
                (turn_id, queued)
            }

            Err(ExecuteError::Command(SessionError::TurnAlreadyActive { turn_id: held })) => {
                let queued = input.queue
                    && held == turn_id
                    && !matches!(
                        self.active_turn_id(&caller, &session_id).await,
                        Ok(active) if active == held
                    );
                (held, queued)
            }
            Err(ExecuteError::Command(SessionError::TurnAlreadyCompleted { turn_id })) => {
                (turn_id, false)
            }
            Err(e) => return Err(e.into()),
        };

        Ok(SubmitClientPayloadOutput {
            session_id,
            turn_id: effective_turn_id,
            queued,
        })
    }

    pub async fn handle_client_input(
        &self,
        input: HandleClientInput,
    ) -> Result<ClientInputOutput, RuntimeError> {
        let HandleClientInput {
            session_id,
            caller,
            owner,
            input,
            agent,
            worker,
            span,
        } = input;

        let (agent_id, turn_id, payload, queue) = match input {
            ClientInput::Message {
                agent_id,
                turn_id,
                message,
                stream,
                queue,
            } => (
                agent_id,
                turn_id,
                ClientPayload::Message(ClientMessage { message, stream }),
                queue,
            ),
            ClientInput::Messages {
                agent_id,
                turn_id,
                messages,
                stream,
                client,
            } => (
                agent_id,
                turn_id,
                ClientPayload::Messages(ClientMessages {
                    messages,
                    stream,
                    client,
                }),
                false,
            ),
            ClientInput::Append {
                agent_id,
                turn_id,
                messages,
                stream,
                client,
                queue,
            } => (
                agent_id,
                turn_id,
                ClientPayload::Append(ClientAppend {
                    messages,
                    stream,
                    client,
                }),
                queue,
            ),
            ClientInput::Action {
                agent_id,
                turn_id,
                name,
                args,
            } => (
                agent_id,
                turn_id,
                ClientPayload::Action(ClientAction { name, args }),
                false,
            ),
            ClientInput::InterruptResume {
                resumption:
                    InterruptResumption {
                        interrupt_id,
                        payload,
                    },
            } => {
                let turn_id = self.active_turn_id(&caller, &session_id).await?;
                self.resume_interrupt(ResumeInterruptInput {
                    session_id: session_id.clone(),
                    interrupt_id,
                    payload,
                    caller,
                    span,
                })
                .await?;
                return Ok(ClientInputOutput {
                    session_id,
                    turn_id,
                    queued: false,
                });
            }
            ClientInput::ToolResult {
                id,
                attempt,
                result,
                content,
                structured_content,
                is_error,
            } => {
                let answered = crate::protocol::ToolResult::from_action(
                    result,
                    content,
                    structured_content,
                    is_error,
                )
                .map_err(|e| RuntimeError::Internal(e.to_string()))?;
                return self
                    .settle_client_tool(
                        session_id,
                        caller,
                        span,
                        id,
                        attempt,
                        EffectSettlement::Result(EffectResultPayload::ToolCall {
                            result: answered,
                        }),
                    )
                    .await;
            }
            ClientInput::ToolError {
                id,
                error,
                retryable,
                attempt,
            } => {
                return self
                    .settle_client_tool(
                        session_id,
                        caller,
                        span,
                        id,
                        attempt,
                        EffectSettlement::Error {
                            error,
                            retryable,
                            code: None,
                            detail: None,
                        },
                    )
                    .await;
            }
        };

        let out = self
            .submit_client_payload(SubmitClientPayload {
                session_id,
                caller,
                owner,
                agent_id,
                payload,
                agent,
                worker,
                turn_id,
                continue_turn: false,
                queue,
            })
            .await?;
        Ok(ClientInputOutput {
            session_id: out.session_id,
            turn_id: out.turn_id,
            queued: out.queued,
        })
    }

    async fn settle_client_tool(
        &self,
        session_id: String,
        caller: Caller,
        span: SpanContext,
        id: String,
        attempt: Option<u32>,
        settlement: EffectSettlement,
    ) -> Result<ClientInputOutput, RuntimeError> {
        let turn_id = self.active_turn_id(&caller, &session_id).await?;
        self.settle_effect(SettleEffectInput {
            session_id: session_id.clone(),
            kind: WorkKind::ToolCall,
            id,
            attempt,
            settlement,
            caller,
            span,
        })
        .await?;
        Ok(ClientInputOutput {
            session_id,
            turn_id,
            queued: false,
        })
    }

    async fn active_turn_id(
        &self,
        caller: &Caller,
        session_id: &str,
    ) -> Result<String, RuntimeError> {
        match self.get_session(caller.tenant_id(), session_id).await {
            Ok(session) => session
                .state
                .turn_id()
                .map(str::to_string)
                .ok_or(RuntimeError::Session(SessionError::NoActiveTurn)),
            Err(_) => Err(RuntimeError::Session(SessionError::NoActiveTurn)),
        }
    }

    pub async fn stream(
        &self,
        spec: SessionSubscriptionSpec,
        after: Option<Seq>,
    ) -> Result<mpsc::Receiver<SessionEvent>, RuntimeError> {
        self.authorize_session_read(&spec.session_id, &spec.caller)
            .await?;
        Ok(self.session_subscriptions.stream(spec, after).await)
    }

    pub async fn authorize_session_read(
        &self,
        session_id: &str,
        caller: &Caller,
    ) -> Result<(), RuntimeError> {
        self.reader().authorize(session_id, caller).await
    }

    pub async fn spawn_processor(
        &self,
        processor: Arc<dyn EventProcessor>,
        config: EventProcessorRunnerConfig,
        start_at_tail: bool,
    ) -> Result<(), RuntimeError> {
        if start_at_tail {
            self.cursor_store
                .seed_at_tail(processor.name())
                .await
                .map_err(|e| RuntimeError::Internal(e.to_string()))?;
        }
        let handle = EventProcessorRunner::new(
            self.store.clone(),
            self.cursor_store.clone(),
            processor,
            config,
            self.cancel.clone(),
        )
        .spawn();
        self.handles.lock().await.push(handle);
        Ok(())
    }

    pub async fn subscribe_token_deltas(
        &self,
        caller: &Caller,
        root_session_id: &str,
    ) -> mpsc::Receiver<TokenDelta> {
        self.token_delta_transport
            .subscribe(caller.tenant_id(), root_session_id)
            .await
    }

    pub fn token_delta_transport(&self) -> Arc<dyn TokenDeltaTransport> {
        self.token_delta_transport.clone()
    }

    pub fn agents(&self) -> Arc<dyn AgentDirectory> {
        self.agents.clone()
    }

    pub fn llm_blocks(&self, tenant_id: &str) -> LlmBlocks {
        self.agents.tenant(tenant_id).llm
    }

    pub fn reader(&self) -> SessionReader {
        SessionReader::new(self.store.clone(), self.session_index.clone())
    }

    pub async fn list_sessions(&self, filter: &SessionFilter) -> Result<SessionPage, RuntimeError> {
        self.reader().list(filter).await
    }

    pub async fn count_sessions(&self, filter: &SessionFilter) -> Result<u64, RuntimeError> {
        self.reader().count(filter).await
    }

    pub async fn get_session(
        &self,
        tenant_id: &str,
        session_id: &str,
    ) -> Result<SessionAggregate, RuntimeError> {
        self.reader().session(tenant_id, session_id).await
    }

    pub async fn read_session_events(
        &self,
        caller: &Caller,
        session_id: &str,
        after: Option<Seq>,
        limit: Option<usize>,
    ) -> Result<Vec<SessionEvent>, RuntimeError> {
        self.reader().events(caller, session_id, after, limit).await
    }

    pub async fn submit_decision(&self, input: SubmitDecision) -> Result<(), RuntimeError> {
        execute(
            &*self.store,
            ExecuteInput {
                session_id: input.session_id.clone(),
                caller: input.caller,
                command: CommandPayload::SubmitWorkerDecision {
                    decision_id: input.decision_id,
                    transcript: input.transcript,
                    actions: input.actions,
                    state: input.state,
                    agent: input.agent,
                    channels: input.channels,
                },
                span: input.span,
            },
            &ConflictRetry::default(),
        )
        .await
        .map(|_| ())
        .map_err(RuntimeError::from)
    }

    pub(crate) fn blob_store(&self) -> &dyn BlobStore {
        self.blobs.as_ref()
    }

    pub(crate) async fn stored(
        &self,
        result: crate::protocol::ToolResult,
        tenant_id: &str,
    ) -> crate::protocol::StoredResult {
        blob::store(result, self.blob_store(), tenant_id).await
    }

    pub async fn settle_effect(&self, input: SettleEffectInput) -> Result<(), RuntimeError> {
        let outcome = match input.settlement {
            EffectSettlement::Result(EffectResultPayload::ToolCall { result }) => Outcome::Tool {
                result: self.stored(result, input.caller.tenant_id()).await,
            },
            EffectSettlement::Result(EffectResultPayload::LlmCall { response }) => {
                Outcome::Llm(response)
            }

            EffectSettlement::Error {
                error,
                retryable,
                code,
                detail,
            } => SettleError::new(
                ErrorInfo::handler(error).or_code(code).or_detail(detail),
                retryable,
            )
            .into(),
        };
        let kind = match input.kind {
            WorkKind::ToolCall => EffectKind::ToolCall,
            WorkKind::LlmCall => EffectKind::LlmCall,
        };
        let command = CommandPayload::settle(kind, input.id, input.attempt, outcome);

        execute(
            &*self.store,
            ExecuteInput {
                session_id: input.session_id,
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

    pub async fn interrupt_session(
        &self,
        input: InterruptSessionInput,
    ) -> Result<(), RuntimeError> {
        execute(
            &*self.store,
            ExecuteInput {
                session_id: input.session_id,
                caller: input.caller,
                command: CommandPayload::Interrupt {
                    interrupt_id: input.interrupt_id,
                    reason: input.reason,
                    payload: input.payload,
                },
                span: input.span,
            },
            &ConflictRetry::default(),
        )
        .await
        .map(|_| ())
        .map_err(RuntimeError::from)
    }

    pub async fn resume_interrupt(&self, input: ResumeInterruptInput) -> Result<(), RuntimeError> {
        execute(
            &*self.store,
            ExecuteInput {
                session_id: input.session_id,
                caller: input.caller,
                command: CommandPayload::ResumeInterrupt {
                    interrupt_id: input.interrupt_id,
                    payload: input.payload,
                },
                span: input.span,
            },
            &ConflictRetry::default(),
        )
        .await
        .map(|_| ())
        .map_err(RuntimeError::from)
    }

    pub async fn fail_decision(&self, input: FailDecision) -> Result<(), RuntimeError> {
        execute(
            &*self.store,
            ExecuteInput {
                session_id: input.session_id.clone(),
                caller: input.caller,
                command: CommandPayload::settle(
                    EffectKind::Decision,
                    input.decision_id,
                    None,
                    SettleError::new(input.error, input.retryable),
                ),
                span: input.span,
            },
            &ConflictRetry::default(),
        )
        .await
        .map(|_| ())
        .map_err(RuntimeError::from)
    }
}

pub struct RuntimeDeps {
    pub store: Arc<dyn EventStore>,

    pub agents: Arc<dyn AgentDirectory>,

    pub llm: Arc<dyn LlmResolver>,
    pub llm_task_queue: Arc<dyn TaskQueue<LlmTask>>,
    pub subagent_task_queue: Arc<dyn TaskQueue<SubagentTask>>,
    pub connections: Option<Arc<Connections>>,

    pub plugins: Arc<dyn crate::plugins::PluginResolver>,
    pub connector_task_queue: Arc<dyn TaskQueue<ConnectorTask>>,
    pub worker_queue: Arc<dyn WorkerQueue>,
    pub channel_proposers: Vec<Arc<dyn worker::ChannelProposer>>,
    pub session_index_store: Arc<dyn SessionIndexStore>,
    pub cursor_store: Arc<dyn ProcessorCursorStore>,
    pub wake_store: Arc<dyn WakeScheduleStore>,
    pub token_delta_transport: Arc<dyn TokenDeltaTransport>,
    pub blobs: Arc<dyn BlobStore>,
}

pub fn start(deps: RuntimeDeps, config: RuntimeConfig) -> Arc<Runtime> {
    let RuntimeDeps {
        store,
        agents,
        llm,
        llm_task_queue,
        subagent_task_queue,
        connections,
        plugins,
        connector_task_queue,
        worker_queue,
        channel_proposers,
        session_index_store,
        cursor_store,
        wake_store,
        token_delta_transport,
        blobs,
    } = deps;
    let cancel = CancellationToken::new();

    let mut llm_handles: Vec<tokio::task::JoinHandle<()>> = Vec::new();
    if !llm.is_empty() {
        llm_handles.push(spawn_llm_dispatch_processor(
            store.clone(),
            cursor_store.clone(),
            llm_task_queue.clone(),
            cancel.clone(),
        ));
        llm_handles.extend(spawn_llm_task_executor(
            store.clone(),
            llm,
            llm_task_queue,
            token_delta_transport.clone(),
            blobs.clone(),
            config.pool(config.llm_executor_workers),
            cancel.clone(),
        ));
    }

    let mut connector_handles: Vec<tokio::task::JoinHandle<()>> = Vec::new();
    connector_handles.push(spawn_connector_dispatch_processor(
        store.clone(),
        cursor_store.clone(),
        connector_task_queue.clone(),
        cancel.clone(),
    ));
    connector_handles.extend(spawn_connector_task_executor(
        store.clone(),
        connections,
        plugins,
        connector_task_queue,
        config.pool(config.connector_executor_workers),
        cancel.clone(),
    ));

    let subagent_processor_handle = spawn_subagent_dispatch_processor(
        store.clone(),
        cursor_store.clone(),
        subagent_task_queue.clone(),
        cancel.clone(),
    );
    let subagent_executor_handles = spawn_subagent_task_executor(
        store.clone(),
        agents.clone(),
        subagent_task_queue,
        config.pool(config.subagent_executor_workers),
        cancel.clone(),
    );

    let worker_handle = spawn_worker_processor(
        store.clone(),
        cursor_store.clone(),
        worker_queue.clone(),
        agents.clone(),
        channel_proposers,
        blobs.clone(),
        cancel.clone(),
    );
    let session_index_processor_handle = spawn_session_index_processor(
        store.clone(),
        cursor_store.clone(),
        session_index_store.clone(),
        cancel.clone(),
    );
    let wake_processor_handle = spawn_wake_processor(
        store.clone(),
        cursor_store.clone(),
        wake_store.clone(),
        cancel.clone(),
    );

    let wake_dispatcher_handle = spawn_wake_dispatcher(
        store.clone(),
        wake_store.clone(),
        config.wake_poll_interval,
        cancel.clone(),
    );

    let boot_reconciler_handle = spawn_boot_reconciler(store.clone(), wake_store);

    let session_subscriptions = session::subscriptions::SessionSubscriptions::new(store.clone());

    let mut handles = vec![
        subagent_processor_handle,
        worker_handle,
        session_index_processor_handle,
        wake_processor_handle,
        boot_reconciler_handle,
        wake_dispatcher_handle,
    ];
    handles.extend(llm_handles);
    handles.extend(connector_handles);
    handles.extend(subagent_executor_handles);

    Arc::new(Runtime {
        store,
        queue: worker_queue,
        agents,
        session_index: session_index_store,
        session_subscriptions,
        token_delta_transport,
        cursor_store,
        cancel,
        handles: tokio::sync::Mutex::new(handles),
        shutdown_timeout: config.shutdown_timeout,
        worker_retry_resolver: config.worker_retry_resolver,
        blobs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use session::ConflictRetry;

    #[test]
    fn concurrency_fits_the_conflict_budget() {
        let config = RuntimeConfig::default();
        let budget = ConflictRetry::default().max_retries as usize;
        assert!(
            config.executor_concurrency <= budget + 1,
            "{} tasks at once can conflict {} times, and the budget is {budget}",
            config.executor_concurrency,
            config.executor_concurrency - 1,
        );
    }

    #[test]
    fn every_worker_count_is_at_least_one() {
        let config = RuntimeConfig::default();
        for workers in [
            config.llm_executor_workers,
            config.subagent_executor_workers,
            config.connector_executor_workers,
        ] {
            assert!(workers >= 1, "a subsystem with no worker drains nothing");
        }
    }
}
