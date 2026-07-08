use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::Semaphore;
use tokio::task::JoinHandle;

use crate::session::wire::resolve_actions;
use crate::worker::push::{PushRegistrationRecord, PushRegistry};
use crate::worker::{DequeueFilter, SubmitDecision};
use crate::{Caller, Runtime};

pub struct PushAdapter {
    pub runtime: Arc<Runtime>,
    registry: Arc<PushRegistry>,
    handles: std::sync::Mutex<HashMap<String, JoinHandle<()>>>,
    semaphore: Arc<Semaphore>,
}

impl PushAdapter {
    pub fn new(runtime: Arc<Runtime>, registry: PushRegistry, concurrency: usize) -> Self {
        Self {
            runtime,
            registry: Arc::new(registry),
            handles: std::sync::Mutex::new(HashMap::new()),
            semaphore: Arc::new(Semaphore::new(concurrency)),
        }
    }

    pub async fn start(&self) {
        let tenants = self.registry.tenants().await;
        for tenant_id in tenants {
            self.spawn_loop(tenant_id);
        }
    }

    pub async fn register(&self, record: PushRegistrationRecord) -> Result<(), String> {
        let tenant_id = record.tenant_id.clone();
        self.registry.register(record).await?;
        self.spawn_loop(tenant_id);
        Ok(())
    }

    pub async fn registration(&self, tenant_id: &str) -> Option<PushRegistrationRecord> {
        self.registry.registration(tenant_id).await
    }

    pub async fn unregister(&self, tenant_id: &str) -> Result<(), String> {
        let handle = self.handles.lock().unwrap().remove(tenant_id);
        if let Some(h) = handle {
            h.abort();
        }
        self.registry.unregister(tenant_id).await
    }

    fn spawn_loop(&self, tenant_id: String) {
        let mut handles = self.handles.lock().unwrap();
        if let Some(existing) = handles.get(&tenant_id) {
            existing.abort();
        }

        let runtime = self.runtime.clone();
        let registry = self.registry.clone();
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
                            "push loop dequeued decision"
                        );
                        d
                    }
                    None => continue,
                };

                let permit = match semaphore.clone().acquire_owned().await {
                    Ok(p) => p,
                    Err(_) => return,
                };

                let registry = registry.clone();
                let runtime = runtime.clone();

                tokio::spawn(async move {
                    let _permit = permit;

                    // Bind the tenant before moving fields out of `decision`
                    // below: `tenant_id()` borrows the whole struct.
                    let tenant_id = decision.tenant_id().to_string();
                    let transport = match registry.lookup(&tenant_id).await {
                        Some(t) => t,
                        None => return,
                    };
                    let token_delta_transport = runtime.token_delta_transport();

                    match transport.push(&decision, token_delta_transport).await {
                        Ok(resp) => {
                            // Resolve wire actions against the trigger this response
                            // answers, so any omitted settle id is filled in before the
                            // core ever sees it.
                            let actions =
                                match resolve_actions(resp.actions, Some(&decision.trigger)) {
                                    Ok(actions) => actions,
                                    Err(e) => {
                                        tracing::warn!(
                                            decision_id = %decision.decision_id,
                                            error = %e,
                                            "unresolvable worker action"
                                        );
                                        return;
                                    }
                                };
                            let submit = SubmitDecision {
                                session_id: decision.session_id,
                                caller: Caller::System {
                                    tenant_id: tenant_id.clone(),
                                },
                                decision_id: decision.decision_id.clone(),
                                transcript: resp.messages,
                                actions,
                                state: resp.state,
                                span: decision.span.child("push_worker"),
                            };
                            if let Err(e) = runtime.submit_decision(submit).await {
                                tracing::warn!(
                                    decision_id = %decision.decision_id,
                                    error = %e,
                                    "push worker submit failed"
                                );
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                decision_id = %decision.decision_id,
                                error = %e,
                                "push dispatch failed"
                            );
                        }
                    }
                });
            }
        });

        handles.insert(tid, handle);
    }
}
