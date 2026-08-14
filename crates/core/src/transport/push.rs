//! Where a decision goes.
//!
//! One rule, read from the agent directory: an agent with a `worker` URL has
//! its decisions POSTed there; an agent without one is decided here, by
//! accepting the engine's own proposal; an agent nobody declared fails
//! immediately, saying so. There is no registration step and no tenant-wide
//! default, so a routing question always has exactly one answer and the file is
//! the whole of it.
//!
//! The engine-hosted path is deliberately the worker path minus the HTTP: the
//! same proposal, the same `resolve_response`, the same `submit_decision`. It
//! therefore inherits the queue's durability, restart recovery, the retry
//! ladder, and single-live-decision ordering rather than reimplementing them.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tokio::sync::Semaphore;
use tokio::task::JoinHandle;

use crate::protocol::{DecisionResponse, ErrorCode};
use crate::session::wire::resolve_response;
use crate::worker::directory::declared;
use crate::worker::push::{PushError, PushTransport, TransportRegistry};
use crate::worker::{
    AgentDirectory, DequeueFilter, FailDecision, SubmitDecision, WorkerDecisionRequest,
};
use crate::{Caller, Runtime};

/// Who decides for one agent.
enum Route {
    Push(Arc<dyn PushTransport>),
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
        transports: TransportRegistry,
        concurrency: usize,
    ) -> Self {
        Self {
            runtime,
            router: Arc::new(Router {
                agents,
                transports,
                built: Mutex::new(HashMap::new()),
            }),
            handles: Mutex::new(HashMap::new()),
            semaphore: Arc::new(Semaphore::new(concurrency)),
        }
    }

    /// A decision loop per tenant the directory declares anything for. A tenant
    /// with nothing declared has no agent that could be routed, so it has
    /// nothing to dequeue.
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

/// The routing half, shared by every in-flight decision.
struct Router {
    agents: Arc<dyn AgentDirectory>,
    transports: TransportRegistry,
    /// One transport per `(tenant, agent)` with a worker, built on first use: a
    /// transport holds a connection pool, so rebuilding it per decision would
    /// throw away every keep-alive.
    built: Mutex<HashMap<(String, String), Arc<dyn PushTransport>>>,
}

impl Router {
    fn route(&self, tenant_id: &str, agent_id: &str) -> Result<Route, String> {
        let Some(entry) = self.agents.agent(tenant_id, agent_id) else {
            return Err(format!(
                "no [agent.{agent_id}] in substructure.toml. Declared: {}",
                declared(&self.agents.agent_ids(tenant_id))
            ));
        };
        let Some(worker) = entry.worker else {
            return Ok(Route::Engine);
        };

        let key = (tenant_id.to_string(), agent_id.to_string());
        if let Some(t) = self.built.lock().unwrap().get(&key) {
            return Ok(Route::Push(t.clone()));
        }
        // A worker that names no secret is called unsigned: signing with a
        // secret only this process knows proves nothing to anybody.
        let transport = self.transports.create(
            "http",
            serde_json::json!({
                "endpoint_url": worker.url,
                "signing_secret": worker.signing_secret,
            }),
        )?;
        self.built.lock().unwrap().insert(key, transport.clone());
        Ok(Route::Push(transport))
    }
}

async fn decide(runtime: &Runtime, router: &Router, decision: WorkerDecisionRequest) {
    // Bind the tenant before moving fields out of `decision`: `tenant_id()`
    // borrows the whole struct.
    let tenant_id = decision.tenant_id().to_string();

    match router.route(&tenant_id, &decision.agent_id) {
        // An undeclared agent will not become declared by waiting, so it fails
        // now rather than climbing a ten-minute retry ladder.
        Err(e) => {
            tracing::warn!(agent_id = %decision.agent_id, error = %e, "decision is unroutable");
            let failure = PushError::fatal(ErrorCode::Unroutable, e);
            record_failure(runtime, &decision, &tenant_id, failure).await;
        }
        Ok(Route::Engine) => decide_in_engine(runtime, decision, &tenant_id).await,
        Ok(Route::Push(transport)) => {
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

/// Accept the engine's own proposal as the decision. An empty proposal is a
/// trigger the engine cannot answer, and no amount of retrying gives it an
/// answer, so it fails loudly instead of quietly climbing the ladder.
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

/// Record the failure now: the queue never redelivers a dequeued decision, so an
/// unrecorded failure hangs its turn until the decision's deadline. `retryable`
/// comes from the transport — a 4xx or an unparseable reply won't get better on a
/// second attempt, so retrying it only delays the terminal.
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

/// Lower a decision against the trigger it answers and the config resolved for
/// it — so omitted settle ids, flat `llm.call` fields, and the block a call runs
/// on are all resolved before the core sees it — then submit it.
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
