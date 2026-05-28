use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::Semaphore;
use tokio::task::JoinHandle;

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

                    let transport = match registry.lookup(&decision.tenant_id).await {
                        Some(t) => t,
                        None => return,
                    };

                    match transport.push(&decision).await {
                        Ok(resp) => {
                            let submit = SubmitDecision {
                                session_id: decision.session_id,
                                tenant_id: decision.tenant_id.clone(),
                                caller: Caller::System,
                                decision_id: decision.decision_id.clone(),
                                actions: resp.actions,
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
