use std::collections::HashMap;
use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction};
use napi_derive::napi;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tracing_subscriber::EnvFilter;

use base64::Engine;
use substructure_core::identity::ClientIdentity;
use substructure_core::providers::memory_queue::{ShardedInMemoryQueue, TaskQueue};
use substructure_core::providers::openrouter::{OpenRouterConfig, OpenRouterProvider};
use substructure_core::providers::sqlite::{
    SqliteCheckpointStore, SqliteDb, SqliteEventStore, SqliteSessionIndexStore, SqliteWakeStore,
};
use substructure_core::providers::worker_queue::InMemoryWorkerQueue;
use substructure_core::session::decision::ClientPayload;
use substructure_core::span::SpanContext as CoreSpanContext;
use substructure_core::worker::{DequeueFilter, FailDecision, SubmitDecision};
use substructure_core::{
    Caller, Runtime, RuntimeConfig, SubmitClientPayload, SubmitToolCallResult,
    SubmitToolCallResultInput,
};

/// Result returned by `submitPayload`.
#[napi(object)]
pub struct SubmitPayloadResult {
    pub session_id: String,
    pub turn_id: String,
}

/// Configuration for creating a new Runtime.
#[napi(object)]
pub struct RuntimeOptions {
    /// SQLite database path
    pub db: String,
    /// OpenRouter API base URL (default: "https://openrouter.ai/api")
    pub openrouter_base_url: Option<String>,
    /// OpenRouter API key
    pub openrouter_api_key: Option<String>,
    /// Number of concurrent LLM handler tasks (default: 4)
    pub llm_pool_size: Option<u32>,
}

/// The Substructure runtime, running in-process.
#[napi]
pub struct EmbeddedRuntime {
    inner: Arc<Runtime>,
    worker_handles: Mutex<HashMap<String, JoinHandle<()>>>,
}

#[napi]
impl EmbeddedRuntime {
    #[napi(constructor)]
    pub fn new(options: RuntimeOptions) -> Result<Self> {
        // Initialize tracing (ignore if already set)
        let _ = tracing_subscriber::fmt()
            .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
            .try_init();

        let db = SqliteDb::open(&options.db, std::time::Duration::from_secs(5))
            .map_err(|e| Error::from_reason(format!("failed to open database: {e}")))?;

        let event_store = Arc::new(
            SqliteEventStore::new(db.clone())
                .map_err(|e| Error::from_reason(format!("failed to init event store: {e}")))?,
        );
        let checkpoint_store =
            Arc::new(SqliteCheckpointStore::new(db.clone()).map_err(|e| {
                Error::from_reason(format!("failed to init checkpoint store: {e}"))
            })?);
        let wake_store = Arc::new(
            SqliteWakeStore::new(db.clone())
                .map_err(|e| Error::from_reason(format!("failed to init wake store: {e}")))?,
        );
        let session_index_store = Arc::new(
            SqliteSessionIndexStore::new(db)
                .map_err(|e| Error::from_reason(format!("failed to init session index: {e}")))?,
        );

        let config = RuntimeConfig {
            llm_executor_workers: options.llm_pool_size.unwrap_or(4) as usize,
            ..Default::default()
        };

        let queue = Arc::new(InMemoryWorkerQueue::new());
        let llm_task_queue: Arc<dyn TaskQueue<substructure_core::llm::LlmTask>> = Arc::new(
            ShardedInMemoryQueue::new(config.llm_executor_workers as u32),
        );
        let sub_agent_task_queue: Arc<dyn TaskQueue<substructure_core::sub_agent::SubAgentTask>> =
            Arc::new(ShardedInMemoryQueue::new(
                config.sub_agent_executor_workers as u32,
            ));
        let llm_provider = Arc::new(OpenRouterProvider::new(OpenRouterConfig {
            base_url: options
                .openrouter_base_url
                .unwrap_or_else(|| "https://openrouter.ai/api".to_string()),
            api_key: options.openrouter_api_key.unwrap_or_default(),
        }));
        let token_delta_transport =
            Arc::new(substructure_core::llm::InMemoryTokenDeltaTransport::new());

        // Enter the NAPI tokio runtime so background tasks spawned by start() work
        let rt = tokio::runtime::Handle::current();
        let inner = rt.block_on(async {
            substructure_core::start(
                event_store,
                llm_provider,
                llm_task_queue,
                sub_agent_task_queue,
                queue,
                session_index_store,
                checkpoint_store,
                wake_store,
                token_delta_transport,
                config,
            )
        });

        Ok(Self {
            inner,
            worker_handles: Mutex::new(HashMap::new()),
        })
    }

    /// Register a JavaScript function as a worker for the given tenant and agent IDs.
    #[napi(
        ts_args_type = "tenantId: string, agentIds: string[], callback: (decision: string) => Promise<string>"
    )]
    pub async fn register_worker(
        &self,
        tenant_id: String,
        #[allow(unused)] agent_ids: Vec<String>,
        callback: ThreadsafeFunction<String, ErrorStrategy::Fatal>,
    ) -> Result<()> {
        let runtime = self.inner.clone();
        let filter_tenant = tenant_id.clone();

        let handle = tokio::spawn(async move {
            let filter = DequeueFilter {
                tenant_id: filter_tenant,
            };

            loop {
                let decision = match runtime.dequeue_decision(&filter).await {
                    Some(d) => d,
                    None => continue,
                };

                let decision_json = match serde_json::to_string(&decision) {
                    Ok(j) => j,
                    Err(e) => {
                        tracing::warn!(error = %e, "failed to serialize decision");
                        continue;
                    }
                };

                let result: Result<String> =
                    match callback.call_async::<Promise<String>>(decision_json).await {
                        Ok(promise) => promise.await,
                        Err(e) => Err(e),
                    };

                match result {
                    Ok(response_json) => {
                        match serde_json::from_str::<WorkerResponse>(&response_json) {
                            Ok(submit) => {
                                let submit_decision = SubmitDecision {
                                    session_id: decision.session_id,
                                    tenant_id: decision.tenant_id.clone(),
                                    caller: Caller::System,
                                    decision_id: decision.decision_id.clone(),
                                    actions: submit.actions,
                                    state: submit
                                        .state
                                        .map(|s| {
                                            base64::engine::general_purpose::STANDARD
                                                .decode(s)
                                                .unwrap_or_default()
                                        })
                                        .unwrap_or_default(),
                                    span: decision.span.child("js_worker"),
                                };

                                if let Err(e) = runtime.submit_decision(submit_decision).await {
                                    tracing::warn!(
                                        decision_id = %decision.decision_id,
                                        error = %e,
                                        "failed to submit worker decision"
                                    );
                                }
                            }
                            Err(e) => {
                                tracing::warn!(
                                    decision_id = %decision.decision_id,
                                    error = %e,
                                    "failed to parse worker response"
                                );
                                let fail = FailDecision {
                                    session_id: decision.session_id,
                                    tenant_id: decision.tenant_id.clone(),
                                    caller: Caller::System,
                                    decision_id: decision.decision_id.clone(),
                                    error: format!("failed to parse worker response: {e}"),
                                    retryable: false,
                                    span: decision.span.child("js_worker"),
                                };
                                if let Err(e) = runtime.fail_decision(fail).await {
                                    tracing::warn!(
                                        decision_id = %decision.decision_id,
                                        error = %e,
                                        "fail_decision failed"
                                    );
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            decision_id = %decision.decision_id,
                            error = %e,
                            "js worker callback failed"
                        );
                        let fail = FailDecision {
                            session_id: decision.session_id,
                            tenant_id: decision.tenant_id.clone(),
                            caller: Caller::System,
                            decision_id: decision.decision_id.clone(),
                            error: format!("js worker callback failed: {e}"),
                            retryable: true,
                            span: decision.span.child("js_worker"),
                        };
                        if let Err(e) = runtime.fail_decision(fail).await {
                            tracing::warn!(
                                decision_id = %decision.decision_id,
                                error = %e,
                                "fail_decision failed"
                            );
                        }
                    }
                }
            }
        });

        let key = tenant_id;
        let mut handles = self.worker_handles.lock().await;
        if let Some(old) = handles.insert(key, handle) {
            old.abort();
        }

        Ok(())
    }

    /// Submit a client payload to an agent session. Returns immediately;
    /// use `streamSession` to observe events.
    #[napi(
        js_name = "submitPayload",
        ts_args_type = "sessionId: string, agentId: string, payloadJson: string, identityJson: string, turnId: string | undefined"
    )]
    pub async fn submit_client_payload(
        &self,
        session_id: String,
        agent_id: String,
        payload_json: String,
        identity_json: String,
        turn_id: Option<String>,
    ) -> Result<SubmitPayloadResult> {
        let payload: ClientPayload = serde_json::from_str(&payload_json)
            .map_err(|e| Error::from_reason(format!("invalid payloadJson: {e}")))?;
        let identity: ClientIdentity = serde_json::from_str(&identity_json)
            .map_err(|e| Error::from_reason(format!("invalid identityJson: {e}")))?;
        if identity.id.as_deref().is_none_or(str::is_empty) {
            return Err(Error::from_reason("identity.id is required"));
        }

        let output = self
            .inner
            .submit_client_payload(SubmitClientPayload {
                session_id,
                tenant_id: identity.tenant_id.clone(),
                caller: Caller::System,
                identity,
                agent_id,
                payload,
                turn_id,
            })
            .await
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(SubmitPayloadResult {
            session_id: output.session_id,
            turn_id: output.turn_id,
        })
    }

    /// Complete (or fail) a tool call out-of-band. Mirrors the HTTP
    /// `POST /api/machine/sessions/{id}/tool-call-results` endpoint and
    /// is the embedded counterpart to `BackendClient.submitToolCallResult`.
    #[napi(
        js_name = "submitToolCallResult",
        ts_args_type = "sessionId: string, tenantId: string, toolCallId: string, attempt: number, resultJson: string | undefined, errorMessage: string | undefined, retryable: boolean | undefined"
    )]
    pub async fn submit_tool_call_result(
        &self,
        session_id: String,
        tenant_id: String,
        tool_call_id: String,
        attempt: u32,
        result_json: Option<String>,
        error_message: Option<String>,
        retryable: Option<bool>,
    ) -> Result<()> {
        let result = match (result_json, error_message) {
            (Some(result), None) => SubmitToolCallResult::Result { result },
            (None, Some(error)) => SubmitToolCallResult::Error {
                error,
                retryable: retryable.unwrap_or(false),
            },
            (Some(_), Some(_)) => {
                return Err(Error::from_reason(
                    "submitToolCallResult: provide either resultJson or errorMessage, not both",
                ));
            }
            (None, None) => {
                return Err(Error::from_reason(
                    "submitToolCallResult: provide resultJson or errorMessage",
                ));
            }
        };

        self.inner
            .submit_tool_call_result(SubmitToolCallResultInput {
                session_id,
                tenant_id,
                tool_call_id,
                attempt,
                result,
                caller: Caller::System,
                span: CoreSpanContext::root().child("napi_tool_call_result"),
            })
            .await
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(())
    }

    /// Stream events for a session. If `turnId` is provided, events are
    /// filtered to that turn and the stream auto-closes on completion;
    /// otherwise all events for the session are streamed. `sequenceAfter`
    /// optionally replays historical events with `sequence > N` first.
    #[napi(
        js_name = "streamSession",
        ts_args_type = "tenantId: string, sessionId: string, turnId: string | undefined, sequenceAfter: number | undefined, onEvent: (event: string) => void"
    )]
    pub async fn stream_session(
        &self,
        tenant_id: String,
        session_id: String,
        turn_id: Option<String>,
        sequence_after: Option<i64>,
        on_event: ThreadsafeFunction<String, ErrorStrategy::Fatal>,
    ) -> Result<()> {
        use substructure_core::session::subscriptions::{
            SessionRequester, SessionSubscriptionSpec,
        };

        // The native binding is a privileged machine caller.
        let spec = match turn_id {
            Some(tid) => SessionSubscriptionSpec::Turn {
                tenant_id: tenant_id.clone(),
                root_session_id: session_id,
                turn_id: tid,
                requester: SessionRequester::Privileged,
            },
            None => SessionSubscriptionSpec::All {
                tenant_id: tenant_id.clone(),
                root_session_id: session_id,
                requester: SessionRequester::Privileged,
            },
        };
        let cursor = sequence_after.map(|n| n.max(0) as u64);
        let mut rx = self
            .inner
            .stream(spec, cursor)
            .await
            .map_err(|e| Error::from_reason(e.to_string()))?;

        while let Some(event) = rx.recv().await {
            let json =
                serde_json::to_string(&event).map_err(|e| Error::from_reason(e.to_string()))?;
            on_event.call(
                json,
                napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
            );
        }

        Ok(())
    }

    /// Shut down the runtime and all worker loops.
    #[napi]
    pub async fn shutdown(&self) -> Result<()> {
        let mut handles = self.worker_handles.lock().await;
        for (_, handle) in handles.drain() {
            handle.abort();
        }
        self.inner.shutdown().await;
        Ok(())
    }
}

/// Response format expected from JS worker callbacks.
#[derive(serde::Deserialize)]
struct WorkerResponse {
    actions: Vec<substructure_core::session::decision::WorkerAction>,
    #[serde(default)]
    state: Option<String>,
}
