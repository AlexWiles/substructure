use std::collections::HashMap;
use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

use substructure_core::worker::{DequeueFilter, SubmitDecision, WorkerDecisionRequest};
use substructure_core::Runtime;

/// Configuration for creating a new Runtime.
#[napi(object)]
pub struct RuntimeOptions {
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
    /// Register a JavaScript function as a worker for the given tenant and agent IDs.
    ///
    /// The callback receives a decision as a JSON string and must return a JSON string
    /// with `{ actions: [...], state: "base64..." }`.
    ///
    /// ```js
    /// rt.registerWorker('default', ['weather-agent'], async (decisionJson) => {
    ///   const decision = JSON.parse(decisionJson)
    ///   return JSON.stringify({ actions: [...], state: '' })
    /// })
    /// ```
    #[napi(ts_args_type = "tenantId: string, agentIds: string[], callback: (decision: string) => Promise<string>")]
    pub async fn register_worker(
        &self,
        tenant_id: String,
        agent_ids: Vec<String>,
        callback: ThreadsafeFunction<String, ErrorStrategy::CalleeHandled>,
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

                let result: Result<String> = match callback
                    .call_async::<Promise<String>>(Ok(decision_json))
                    .await
                {
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
                            state: submit.state.unwrap_or_default(),
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

    /// Send a message to an agent session and return the stream of events as JSON strings.
    #[napi]
    pub async fn send_message(
        &self,
        session_id: String,
        tenant_id: String,
        agent_id: String,
        content: String,
    ) -> Result<Vec<String>> {
        let sid = uuid::Uuid::parse_str(&session_id)
            .map_err(|e| Error::from_reason(format!("invalid session_id: {e}")))?;

        let (_, mut rx) = self
            .inner
            .send_message(substructure_core::SendMessage {
                session_id: sid,
                tenant_id,
                agent_id,
                content,
            })
            .await
            .map_err(|e| Error::from_reason(e.to_string()))?;

        let mut events = Vec::new();
        while let Some(event) = rx.recv().await {
            let json = serde_json::to_string(&event)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            events.push(json);
        }

        Ok(events)
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
    state: Option<Vec<u8>>,
}
