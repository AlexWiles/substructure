use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tokio::sync::Semaphore;
use tokio::task::JoinHandle;

use crate::protocol::{DecisionResponse, ErrorCode};
use crate::runtime::secret::{get_or_mint, SecretPath, SecretStore};
use crate::session::wire::resolve_response;
use crate::transport::http_push::HttpDecider;
use crate::worker::push::{Decider, PushError};
use crate::worker::{
    AgentDirectory, DequeueFilter, FailDecision, Hosting, SubmitDecision, WorkerDecisionRequest,
};
use crate::{Caller, Runtime};

enum Route {
    Worker { id: String, url: String },
    Engine,
}

pub struct PushAdapter {
    pub runtime: Arc<Runtime>,
    router: Arc<Router>,
    handles: Mutex<HashMap<String, JoinHandle<()>>>,
    semaphore: Arc<Semaphore>,
}

impl PushAdapter {
    pub fn new(
        runtime: Arc<Runtime>,
        agents: Arc<dyn AgentDirectory>,
        secrets: Arc<dyn SecretStore>,
        concurrency: usize,
    ) -> Self {
        Self {
            runtime,
            router: Arc::new(Router {
                agents,
                secrets,
                http: HttpDecider::client(),
            }),
            handles: Mutex::new(HashMap::new()),
            semaphore: Arc::new(Semaphore::new(concurrency)),
        }
    }

    pub fn start(&self) {
        for tenant_id in self.router.agents.tenants() {
            self.spawn_loop(tenant_id);
        }
    }

    fn spawn_loop(&self, tenant_id: String) {
        let mut handles = self.handles.lock().unwrap();
        if let Some(existing) = handles.get(&tenant_id) {
            existing.abort();
        }

        let runtime = self.runtime.clone();
        let router = self.router.clone();
        let semaphore = self.semaphore.clone();
        let tid = tenant_id.clone();

        let handle = tokio::spawn(async move {
            let filter = DequeueFilter { tenant_id };
            loop {
                let decision = match runtime.dequeue_decision(&filter).await {
                    Some(d) => {
                        tracing::debug!(
                            decision_id = %d.decision_id,
                            agent_id = %d.agent_id,
                            "decision loop dequeued decision"
                        );
                        d
                    }
                    None => continue,
                };

                let permit = match semaphore.clone().acquire_owned().await {
                    Ok(p) => p,
                    Err(_) => return,
                };

                let router = router.clone();
                let runtime = runtime.clone();
                tokio::spawn(async move {
                    let _permit = permit;
                    decide(&runtime, &router, decision).await;
                });
            }
        });

        handles.insert(tid, handle);
    }
}

struct Router {
    agents: Arc<dyn AgentDirectory>,
    secrets: Arc<dyn SecretStore>,
    http: reqwest::Client,
}

impl Router {
    fn route(&self, tenant_id: &str, decision: &WorkerDecisionRequest) -> Result<Route, String> {
        let agent_id = &decision.agent_id;
        let hosting = match &decision.worker {
            Some(w) => Some(Hosting::Worker(w.id.clone())),
            None => self.agents.agent(tenant_id, agent_id).map(|e| e.hosting),
        };
        let worker_id = match hosting {
            Some(Hosting::Engine) => return Ok(Route::Engine),
            Some(Hosting::Worker(id)) => id,
            None if decision.agent.is_some() => return Ok(Route::Engine),
            None => {
                return Err(format!(
                    "no [agent.{agent_id}], no worker on the session, and no config on the \
                     session. Declared agents: {}",
                    crate::copy::declared(self.agents.agent_ids(tenant_id))
                ))
            }
        };
        let Some(block) = self.agents.worker(tenant_id, &worker_id) else {
            return Err(format!("no [worker.{worker_id}] in subs.toml"));
        };
        let Some(url) = decision
            .worker
            .as_ref()
            .and_then(|w| w.url.clone())
            .or(block.url)
        else {
            return Err(format!(
                "[worker.{worker_id}] has no `url` and the session brought none"
            ));
        };
        Ok(Route::Worker { id: worker_id, url })
    }

    async fn transport(
        &self,
        tenant_id: &str,
        id: &str,
        url: String,
    ) -> Result<HttpDecider, PushError> {
        let path = SecretPath::Worker(id.to_string());
        let secret = get_or_mint(&*self.secrets, tenant_id, &path.secret_ref())
            .await
            .map_err(|e| {
                PushError::retryable(
                    ErrorCode::Internal,
                    format!("reading the signing secret for worker.{id} failed: {e}"),
                )
            })?;
        Ok(HttpDecider::new(self.http.clone(), url, secret))
    }
}

async fn decide(runtime: &Runtime, router: &Router, decision: WorkerDecisionRequest) {
    let tenant_id = decision.tenant_id().to_string();

    match router.route(&tenant_id, &decision) {
        Err(e) => {
            tracing::warn!(agent_id = %decision.agent_id, error = %e, "decision is unroutable");
            let failure = PushError::fatal(ErrorCode::Unroutable, e);
            record_failure(runtime, &decision, &tenant_id, failure).await;
        }
        Ok(Route::Engine) => decide_in_engine(runtime, decision, &tenant_id).await,
        Ok(Route::Worker { id, url }) => {
            let transport = match router.transport(&tenant_id, &id, url).await {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(decision_id = %decision.decision_id, error = %e, "no transport");
                    record_failure(runtime, &decision, &tenant_id, e).await;
                    return;
                }
            };
            let token_delta_transport = runtime.token_delta_transport();
            match transport.push(&decision, token_delta_transport).await {
                Ok(resp) => submit(runtime, decision, &tenant_id, resp, "push_worker").await,
                Err(e) => {
                    tracing::warn!(
                        decision_id = %decision.decision_id,
                        error = %e,
                        code = ?e.error.code,
                        param = ?e.error.param,
                        "push dispatch failed"
                    );
                    record_failure(runtime, &decision, &tenant_id, e).await;
                }
            }
        }
    }
}

async fn decide_in_engine(runtime: &Runtime, decision: WorkerDecisionRequest, tenant_id: &str) {
    let span = decision.span.child("engine_decide");
    if decision.proposed.authors_nothing() {
        let error = format!(
            "the engine has no proposal for `{}`, and no worker is attached to agent `{}`",
            decision.trigger.kind(),
            decision.agent_id
        );
        tracing::warn!(decision_id = %decision.decision_id, %error, "engine cannot decide");
        let failure = PushError::fatal(ErrorCode::Unroutable, error);
        record_failure(runtime, &decision, tenant_id, failure).await;
        return;
    }
    tracing::debug!(
        parent: None,
        decision_id = %decision.decision_id,
        agent_id = %decision.agent_id,
        trigger = decision.trigger.kind(),
        traceparent = %span.traceparent(),
        "deciding in engine"
    );
    let proposed = decision.proposed.clone();
    submit(runtime, decision, tenant_id, proposed, "engine_decide").await;
}

async fn record_failure(
    runtime: &Runtime,
    decision: &WorkerDecisionRequest,
    tenant_id: &str,
    error: PushError,
) {
    let failed = runtime
        .fail_decision(FailDecision {
            session_id: decision.session_id.clone(),
            caller: Caller::System {
                tenant_id: tenant_id.to_string(),
            },
            decision_id: decision.decision_id.clone(),
            error: error.error,
            retryable: error.retryable,
            span: decision.span.child("push_fail_decision"),
        })
        .await;
    if let Err(e) = failed {
        tracing::warn!(
            decision_id = %decision.decision_id,
            error = %e,
            "recording worker decision failure failed"
        );
    }
}

async fn submit(
    runtime: &Runtime,
    decision: WorkerDecisionRequest,
    tenant_id: &str,
    response: DecisionResponse,
    span_name: &str,
) {
    let resolved = match resolve_response(
        response,
        decision.agent.as_ref(),
        Some(&decision.trigger),
        &runtime.llm_blocks(tenant_id),
        runtime.blob_store(),
        tenant_id,
    )
    .await
    {
        Ok(resolved) => resolved,
        Err(e) => {
            tracing::warn!(
                decision_id = %decision.decision_id,
                error = %e,
                "unresolvable decision"
            );
            let mut failure =
                PushError::fatal(ErrorCode::InvalidResponse, e.to_string()).with_detail(e.detail());
            if let Some(param) = e.param() {
                failure = failure.with_param(param);
            }
            record_failure(runtime, &decision, tenant_id, failure).await;
            return;
        }
    };
    let submit = SubmitDecision {
        session_id: decision.session_id,
        caller: Caller::System {
            tenant_id: tenant_id.to_string(),
        },
        decision_id: decision.decision_id.clone(),
        transcript: resolved.messages,
        actions: resolved.actions,
        state: resolved.state,
        agent: resolved.agent,
        channels: resolved.channels,
        span: decision.span.child(span_name),
    };
    if let Err(e) = runtime.submit_decision(submit).await {
        tracing::warn!(
            decision_id = %decision.decision_id,
            error = %e,
            "decision submit failed"
        );
    }
}
