use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::connectors::registry::Connections;
use crate::protocol::{
    ClientAction, ClientAppend, ClientInput, ClientMessage, ClientMessages, ClientPayload,
    ErrorCode, ErrorInfo, InterruptResumption, SessionOwner, TokenDelta,
};
use crate::providers::memory_queue::TaskQueue;
use connector::{spawn_connector_dispatch_processor, spawn_connector_task_executor, ConnectorTask};
use event_store::{EventFilter, EventStore, Seq, StoreError};
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
use session::state::EffectKind;
use session::subscriptions::SessionSubscriptionSpec;
use session::wire::result_to_string;
use session::{execute, ConflictRetry, ExecuteError, ExecuteInput, SessionAggregate, SessionEvent};
use span::SpanContext;
use sub_agent::{spawn_sub_agent_dispatch_processor, spawn_sub_agent_task_executor, SubAgentTask};
use wake::{spawn_boot_reconciler, spawn_wake_dispatcher, spawn_wake_processor, WakeScheduleStore};
use worker::spawn_worker_processor;
use worker::{
    AgentDirectory, DequeueFilter, FailDecision, SubmitDecision, WorkerDecisionRequest, WorkerQueue,
};

mod caller;
pub mod connector;
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
    pub connector_executor_workers: usize,
    pub wake_poll_interval: std::time::Duration,
    pub shutdown_timeout: std::time::Duration,
    pub worker_retry_resolver: Arc<dyn WorkerRetryResolver>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            llm_executor_workers: 4,
            sub_agent_executor_workers: 2,
            connector_executor_workers: 2,
            wake_poll_interval: std::time::Duration::from_secs(30),
            shutdown_timeout: std::time::Duration::from_secs(5),
            worker_retry_resolver: Arc::new(DefaultWorkerRetryResolver),
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
}

pub struct SubmitClientPayload {
    pub session_id: String,
    pub caller: Caller,
    pub owner: SessionOwner,
    pub agent_id: String,
    pub payload: ClientPayload,
    /// Caller-provided turn ID for idempotency. Auto-generated if None.
    pub turn_id: Option<String>,
    /// Record into the turn already running instead of opening one, leaving
    /// `turn_id` as the fallback for when none is. For input that continues a
    /// turn rather than requesting one — an interrupt resume that also revises
    /// the view, which AG-UI requires to arrive as a single input.
    pub continue_turn: bool,
    /// Hold the payload for the next turn instead of refusing it when one is
    /// already running. Only the message shapes queue; the rest are refused as
    /// before. The submit answers as soon as the payload is durable — the turn
    /// starts when the running one ends.
    pub queue: bool,
}

pub struct SubmitClientPayloadOutput {
    pub session_id: String,
    pub turn_id: String,
    /// The turn was accepted but has not started: another turn holds the
    /// session. A snapshot taken as the submit answers, so a turn that starts
    /// immediately after reads as `false`.
    pub queued: bool,
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
    /// The turn was accepted but has not started — see
    /// [`SubmitClientPayloadOutput::queued`]. Always false for the inputs that
    /// cannot queue.
    pub queued: bool,
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
        let caller = input.caller.clone();

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
                    turn: match input.continue_turn {
                        true => TurnTarget::Continue(turn_id.clone()),
                        false => TurnTarget::Open(turn_id.clone()),
                    },
                    queue: input.queue,
                },
                span: span.child("submit_client_payload"),
            },
            &ConflictRetry::default(),
        )
        .await;

        // The turn the caller must watch, and whether it is waiting rather than
        // running. The command reports the turn it left the session on, so an
        // accepted submit answers both from what it just committed.
        let (effective_turn_id, queued) = match send_result {
            // A continued submit landed in whatever turn was running, which is
            // the one the caller must watch — not the fallback it offered.
            Ok(out) if input.continue_turn => (out.turn_id.unwrap_or(turn_id), false),
            Ok(out) => {
                let queued = input.queue && out.turn_id.as_deref() != Some(turn_id.as_str());
                (turn_id, queued)
            }
            // A turn id the session already holds. Naming our own id means it
            // was taken and still owes a run — running or waiting, a difference
            // only the session knows, so a retrying submitter pays one read for
            // it. Naming another turn means this one was refused, not taken.
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
            queued: false,
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
                .turn_id()
                .map(str::to_string)
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

    /// Attach a durable event processor, joined into runtime shutdown. With
    /// `start_at_tail`, every stream's cursor is parked at its head so a new
    /// processor renders the future, not the whole history.
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

    /// What this deployment declares — the routing truth, and the blocks a
    /// decision seam resolves an `llm.call` against.
    pub fn agents(&self) -> Arc<dyn AgentDirectory> {
        self.agents.clone()
    }

    pub fn llm_blocks(&self, tenant_id: &str) -> LlmBlocks {
        self.agents.llm(tenant_id)
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
        let outcome = match input.settlement {
            EffectSettlement::Result(EffectResultPayload::ToolCall { result }) => {
                Outcome::Tool { result }
            }
            EffectSettlement::Result(EffectResultPayload::LlmCall { response }) => {
                Outcome::Llm(Box::new(response))
            }
            // Worker- and client-authored: flat on the wire, and it may
            // classify itself or not. Unclassified is a handler failure — the
            // thing asked to do the work reported that it could not.
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

/// The stores, queues and providers a runtime drives.
pub struct RuntimeDeps {
    pub store: Arc<dyn EventStore>,
    /// What this deployment declares: its agents, their hosting, and the
    /// `[llm.*]` blocks they name.
    pub agents: Arc<dyn AgentDirectory>,
    /// The client for an engine-run llm block, resolved per call.
    pub llm: Arc<dyn LlmResolver>,
    pub llm_task_queue: Arc<dyn TaskQueue<LlmTask>>,
    pub sub_agent_task_queue: Arc<dyn TaskQueue<SubAgentTask>>,
    pub connections: Option<Arc<Connections>>,
    pub connector_task_queue: Arc<dyn TaskQueue<ConnectorTask>>,
    pub worker_queue: Arc<dyn WorkerQueue>,
    pub session_index_store: Arc<dyn SessionIndexStore>,
    pub cursor_store: Arc<dyn ProcessorCursorStore>,
    pub wake_store: Arc<dyn WakeScheduleStore>,
    pub token_delta_transport: Arc<dyn TokenDeltaTransport>,
}

pub fn start(deps: RuntimeDeps, config: RuntimeConfig) -> Arc<Runtime> {
    let RuntimeDeps {
        store,
        agents,
        llm,
        llm_task_queue,
        sub_agent_task_queue,
        connections,
        connector_task_queue,
        worker_queue,
        session_index_store,
        cursor_store,
        wake_store,
        token_delta_transport,
    } = deps;
    let cancel = CancellationToken::new();

    // The LLM dispatch processor + executor only run engine-side calls. With no
    // engine-run block declared, every model call belongs to a worker, so we
    // skip that subsystem entirely.
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
            config.llm_executor_workers,
            cancel.clone(),
        ));
    }

    // McpServer work only exists where connections are configured; with none,
    // nothing can name one, so the subsystem is skipped entirely.
    let mut connector_handles: Vec<tokio::task::JoinHandle<()>> = Vec::new();
    if let Some(connections) = connections {
        connector_handles.push(spawn_connector_dispatch_processor(
            store.clone(),
            cursor_store.clone(),
            connector_task_queue.clone(),
            cancel.clone(),
        ));
        connector_handles.extend(spawn_connector_task_executor(
            store.clone(),
            connections,
            connector_task_queue,
            config.connector_executor_workers,
            cancel.clone(),
        ));
    }

    let sub_agent_processor_handle = spawn_sub_agent_dispatch_processor(
        store.clone(),
        cursor_store.clone(),
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
        cursor_store.clone(),
        worker_queue.clone(),
        agents.clone(),
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

    // After the dispatcher: its event subscription must exist before the
    // reconciler commits failures, or their retry timers oversleep to the
    // fallback poll.
    let boot_reconciler_handle = spawn_boot_reconciler(store.clone(), wake_store);

    let session_subscriptions = session::subscriptions::SessionSubscriptions::new(store.clone());

    let mut handles = vec![
        sub_agent_processor_handle,
        worker_handle,
        session_index_processor_handle,
        wake_processor_handle,
        boot_reconciler_handle,
        wake_dispatcher_handle,
    ];
    handles.extend(llm_handles);
    handles.extend(connector_handles);
    handles.extend(sub_agent_executor_handles);

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
    })
}
