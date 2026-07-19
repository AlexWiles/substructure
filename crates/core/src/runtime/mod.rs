use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::protocol::{
    ClientAction, ClientInput, ClientMessage, ClientMessages, ClientPayload, ErrorCode,
    InterruptResumption, SessionOwner, TokenDelta,
};
use crate::providers::memory_queue::TaskQueue;
use event_store::{EventFilter, EventStore, Seq, StoreError};
use llm::{
    spawn_llm_dispatch_processor, spawn_llm_task_executor, LlmProviderTrait, LlmTask,
    TokenDeltaTransport,
};
use processor::ProcessorCheckpointStore;
use retry::{NoRetryResolver, WorkerRetryResolver};
use session::command::{CommandPayload, SessionError};
use session::decision::{EffectResultPayload, WorkKind};
use session::index::{
    spawn_session_index_processor, SessionFilter, SessionIndexStore, SessionPage,
};
use session::subscriptions::SessionSubscriptionSpec;
use session::wire::result_to_string;
use session::{execute, ConflictRetry, ExecuteError, ExecuteInput, SessionAggregate, SessionEvent};
use span::SpanContext;
use sub_agent::{spawn_sub_agent_dispatch_processor, spawn_sub_agent_task_executor, SubAgentTask};
use wake::{spawn_wake_dispatcher, spawn_wake_processor, WakeScheduleStore};
use worker::spawn_worker_processor;
use worker::{DequeueFilter, FailDecision, SubmitDecision, WorkerDecisionRequest, WorkerQueue};

mod caller;
pub mod event_store;
pub mod llm;
pub mod processor;
pub mod retry;
pub mod session;
pub mod span;
pub mod sub_agent;
pub mod wake;
pub mod worker;

pub use caller::Caller;

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

/// A client input plus the ambient context an engine call needs. The single entry point
/// for the whole client input surface — HTTP, CLI, and AG-UI all build one of these and
/// hand it to [`Runtime::handle_client_input`]. `session_id` is the universal address
/// (minted by the caller when absent); the submit-only `agent_id`/`turn_id` ride inside
/// `input`.
pub struct HandleClientInput {
    pub session_id: String,
    pub caller: Caller,
    pub owner: SessionOwner,
    pub input: ClientInput,
    pub span: SpanContext,
}

pub struct ClientInputOutput {
    pub session_id: String,
    pub turn_id: String,
}

/// How an effect settled out-of-band. `Result` carries a tool result or llm
/// response; `Error` is uniform across kinds.
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
    /// `None` settles the current attempt; `Some` fences a stale executor.
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
        let create_result = execute(
            &*self.store,
            ExecuteInput {
                session_id: session_id.clone(),
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
        let send_result = execute(
            &*self.store,
            ExecuteInput {
                session_id: session_id.clone(),
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

    /// The single seam for the whole client input surface. A submit delegates to
    /// [`Self::submit_client_payload`]; a resume/settle continues the session's active turn
    /// — looked up here, `NoActiveTurn` if there is none — via [`Self::resume_interrupt`] /
    /// [`Self::settle_effect`]. Those three methods stay public for the machine and embedded
    /// surfaces, which address effects directly rather than the active turn.
    pub async fn handle_client_input(
        &self,
        input: HandleClientInput,
    ) -> Result<ClientInputOutput, RuntimeError> {
        let HandleClientInput {
            session_id,
            caller,
            owner,
            input,
            span,
        } = input;

        // Submit variants carry their addressing inline; resume/settle continue the active
        // turn. The submit arms yield the payload and fall through to one submit call; the
        // rest handle their own dispatch and return.
        let (agent_id, turn_id, payload) = match input {
            ClientInput::Message {
                agent_id,
                turn_id,
                message,
                stream,
            } => (
                agent_id,
                turn_id,
                ClientPayload::Message(ClientMessage { message, stream }),
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
                });
            }
            ClientInput::ToolResult {
                id,
                attempt,
                result,
            } => {
                return self
                    .settle_client_tool(
                        session_id,
                        caller,
                        span,
                        id,
                        attempt,
                        EffectSettlement::Result(EffectResultPayload::ToolCall {
                            result: result_to_string(result),
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
                turn_id,
            })
            .await?;
        Ok(ClientInputOutput {
            session_id: out.session_id,
            turn_id: out.turn_id,
        })
    }

    /// Settle a client-handled effect against the session's active turn (returned for
    /// stream scoping); `NoActiveTurn` if there is none.
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
        })
    }

    /// The active turn a resume/settle continues. `NoActiveTurn` when the session has
    /// none (or does not exist yet); the subsequent resume/settle enforces ownership.
    async fn active_turn_id(
        &self,
        caller: &Caller,
        session_id: &str,
    ) -> Result<String, RuntimeError> {
        match self.get_session(caller.tenant_id(), session_id).await {
            Ok(session) => session
                .state
                .turn_id
                .ok_or(RuntimeError::Session(SessionError::NoActiveTurn)),
            Err(_) => Err(RuntimeError::Session(SessionError::NoActiveTurn)),
        }
    }

    /// Unified event stream. `spec` selects which events to observe (a single
    /// turn or the whole session); `after` (a per-stream cursor) optionally
    /// replays historical events with `seq > N` before streaming live.
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
        let Caller::Frontend {
            tenant_id, user_id, ..
        } = caller
        else {
            return Ok(());
        };

        let session = match self.store.load(tenant_id, session_id).await {
            Ok(session) => session,
            // An uncreated session has no owner yet — nothing to leak; the read
            // is simply empty, and the first turn binds the session to its owner.
            Err(StoreError::StreamNotFound) => return Ok(()),
            Err(e) => return Err(RuntimeError::Internal(e.to_string())),
        };

        let owner_id = session.state.owner.as_ref().and_then(|o| o.id.as_deref());

        if owner_id == Some(user_id.as_str()) {
            Ok(())
        } else {
            Err(RuntimeError::Session(SessionError::SessionAccessDenied))
        }
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
    ) -> Result<SessionAggregate, RuntimeError> {
        self.store
            .load(tenant_id, session_id)
            .await
            .map_err(|e| RuntimeError::Internal(e.to_string()))
    }

    pub async fn read_session_events(
        &self,
        caller: &Caller,
        session_id: &str,
        after: Option<Seq>,
        limit: Option<usize>,
    ) -> Result<Vec<SessionEvent>, RuntimeError> {
        self.authorize_session_read(session_id, caller).await?;
        let filter = EventFilter {
            session_id: Some(session_id.to_string()),
            tenant_id: Some(caller.tenant_id().to_string()),
            after_seq: after,
            limit,
            ..Default::default()
        };
        self.store
            .query_events(&filter)
            .await
            .map_err(|e| RuntimeError::Internal(e.to_string()))
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
                },
                span: input.span,
            },
            &ConflictRetry::default(),
        )
        .await
        .map(|_| ())
        .map_err(RuntimeError::from)
    }

    pub async fn settle_effect(&self, input: SettleEffectInput) -> Result<(), RuntimeError> {
        let command = match input.settlement {
            EffectSettlement::Result(EffectResultPayload::ToolCall { result }) => {
                CommandPayload::CompleteToolCall {
                    tool_call_id: input.id,
                    attempt: input.attempt,
                    result,
                }
            }
            EffectSettlement::Result(EffectResultPayload::LlmCall { response }) => {
                CommandPayload::CompleteLlmCall {
                    call_id: input.id,
                    attempt: input.attempt,
                    response,
                }
            }
            EffectSettlement::Error {
                error,
                retryable,
                code,
                detail,
            } => match input.kind {
                WorkKind::ToolCall => CommandPayload::FailToolCall {
                    tool_call_id: input.id,
                    attempt: input.attempt,
                    error,
                    retryable,
                },
                WorkKind::LlmCall => CommandPayload::FailLlmCall {
                    call_id: input.id,
                    attempt: input.attempt,
                    error,
                    retryable,
                    code,
                    detail,
                },
            },
        };

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
    llm_provider: Option<Arc<dyn LlmProviderTrait>>,
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

    // The LLM dispatch processor + executor only run server-side calls. With no
    // provider configured the server handles LLM calls worker-side only, so we
    // skip that subsystem entirely.
    let mut llm_handles: Vec<tokio::task::JoinHandle<()>> = Vec::new();
    if let Some(llm_provider) = llm_provider {
        llm_handles.push(spawn_llm_dispatch_processor(
            store.clone(),
            checkpoint_store.clone(),
            llm_task_queue.clone(),
            cancel.clone(),
        ));
        llm_handles.extend(spawn_llm_task_executor(
            store.clone(),
            llm_provider,
            llm_task_queue,
            token_delta_transport.clone(),
            config.llm_executor_workers,
            cancel.clone(),
        ));
    }

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
        sub_agent_processor_handle,
        worker_handle,
        session_index_processor_handle,
        wake_processor_handle,
        wake_dispatcher_handle,
    ];
    handles.extend(llm_handles);
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
