use std::collections::HashMap;
use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction};
use napi_derive::napi;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tracing_subscriber::EnvFilter;

use base64::Engine;
use substructure::providers::llm::openrouter::{OpenRouterConfig, OpenRouterProvider};
use substructure::providers::sqlite::SqliteStore;
use substructure::providers::worker::memory_queue::InMemoryWorkerQueue;
use substructure_core::llm::InMemoryLlmTaskQueue;
use substructure_core::worker::{DequeueFilter, SubmitDecision};
use substructure_core::{Runtime, RuntimeConfig, SendMessage};

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
pub struct JsRuntime {
    inner: Arc<Runtime>,
    worker_handles: Mutex<HashMap<String, JoinHandle<()>>>,
}

#[napi]
impl JsRuntime {
    #[napi(constructor)]
    pub fn new(options: RuntimeOptions) -> Result<Self> {
        // Initialize tracing (ignore if already set)
        let _ = tracing_subscriber::fmt()
            .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
            .try_init();

        let store = Arc::new(
            SqliteStore::new(&options.db)
                .map_err(|e| Error::from_reason(format!("failed to open database: {e}")))?,
        );
        let queue = Arc::new(InMemoryWorkerQueue::new());
        let llm_task_queue = Arc::new(InMemoryLlmTaskQueue::new());
        let llm_provider = Arc::new(OpenRouterProvider::new(OpenRouterConfig {
            base_url: options
                .openrouter_base_url
                .unwrap_or_else(|| "https://openrouter.ai/api".to_string()),
            api_key: options.openrouter_api_key.unwrap_or_default(),
        }));

        let config = RuntimeConfig {
            llm_executor_workers: options.llm_pool_size.unwrap_or(4) as usize,
            ..Default::default()
        };

        // Enter the NAPI tokio runtime so background tasks spawned by start() work
        let rt = tokio::runtime::Handle::current();
        let inner = rt.block_on(async {
            substructure_core::start(
                store.clone(),
                llm_provider,
                llm_task_queue,
                queue,
                store.clone(),
                store.clone(),
                store.clone(),
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
        agent_ids: Vec<String>,
        callback: ThreadsafeFunction<String, ErrorStrategy::Fatal>,
    ) -> Result<()> {
        let runtime = self.inner.clone();
        let filter_tenant = tenant_id.clone();
        let filter_agents = agent_ids.clone();

        let handle = tokio::spawn(async move {
            let filter = DequeueFilter {
                tenant_id: filter_tenant,
                agent_ids: filter_agents,
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
                        let submit: WorkerResponse = match serde_json::from_str(&response_json) {
                            Ok(r) => r,
                            Err(e) => {
                                tracing::warn!(error = %e, "failed to parse worker response");
                                continue;
                            }
                        };

                        let submit_decision = SubmitDecision {
                            session_id: decision.session_id,
                            tenant_id: decision.tenant_id.clone(),
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
                            "js worker callback failed"
                        );
                    }
                }
            }
        });

        let key = format!("{tenant_id}:{}", agent_ids.join(","));
        let mut handles = self.worker_handles.lock().await;
        if let Some(old) = handles.insert(key, handle) {
            old.abort();
        }

        Ok(())
    }

    /// Send a message to an agent session, calling `onEvent` for each event as it arrives.
    #[napi(
        ts_args_type = "sessionId: string, tenantId: string, agentId: string, content: string, turnId: string | undefined, onEvent: (event: string) => void"
    )]
    pub async fn send_message(
        &self,
        session_id: String,
        tenant_id: String,
        agent_id: String,
        content: String,
        turn_id: Option<String>,
        on_event: ThreadsafeFunction<String, ErrorStrategy::Fatal>,
    ) -> Result<()> {
        let (_, mut rx) = self
            .inner
            .send_message(SendMessage {
                session_id,
                tenant_id,
                agent_id,
                content,
                turn_id,
            })
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
