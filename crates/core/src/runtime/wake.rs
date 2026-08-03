use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::runtime::event_store::EventStore;
use crate::runtime::processor::{
    EventProcessor, EventProcessorRunner, EventProcessorRunnerConfig, ProcessorCursorStore,
    ProcessorError,
};
use crate::runtime::session::command::CommandPayload;
use crate::runtime::session::{execute, ConflictRetry, ExecuteInput, SessionEvent};
use crate::runtime::span::SpanContext;
use crate::runtime::Caller;

#[derive(Debug, Clone)]
pub struct WakeScheduleItem {
    pub tenant_id: String,
    pub session_id: String,
    pub wake_at: DateTime<Utc>,
}

#[async_trait::async_trait]
pub trait WakeScheduleStore: Send + Sync {
    async fn upsert_wake(
        &self,
        tenant_id: &str,
        session_id: &str,
        wake_at: DateTime<Utc>,
    ) -> Result<(), String>;
    async fn remove_wake(&self, tenant_id: &str, session_id: &str) -> Result<(), String>;
    async fn list_due_wakes(
        &self,
        now: DateTime<Utc>,
        limit: usize,
    ) -> Result<Vec<WakeScheduleItem>, String>;
    async fn next_wake_at(&self) -> Result<Option<DateTime<Utc>>, String>;
}

struct WakeScheduleProjection {
    wake_store: Arc<dyn WakeScheduleStore>,
}

impl WakeScheduleProjection {
    fn new(wake_store: Arc<dyn WakeScheduleStore>) -> Self {
        Self { wake_store }
    }
}

#[async_trait::async_trait]
impl EventProcessor for WakeScheduleProjection {
    fn name(&self) -> &'static str {
        "wake_schedule"
    }

    async fn apply(&self, event: SessionEvent) -> Result<(), ProcessorError> {
        match event.meta.wake_at {
            Some(wake_at) => self
                .wake_store
                .upsert_wake(&event.tenant_id, &event.session_id, wake_at)
                .await
                .map_err(ProcessorError::Apply),
            None => self
                .wake_store
                .remove_wake(&event.tenant_id, &event.session_id)
                .await
                .map_err(ProcessorError::Apply),
        }
    }
}

pub fn spawn_wake_processor(
    store: Arc<dyn EventStore>,
    cursor_store: Arc<dyn ProcessorCursorStore>,
    wake_store: Arc<dyn WakeScheduleStore>,
    cancel: CancellationToken,
) -> JoinHandle<()> {
    let projection = Arc::new(WakeScheduleProjection::new(wake_store));
    EventProcessorRunner::new(
        store,
        cursor_store,
        projection,
        EventProcessorRunnerConfig::default(),
        cancel,
    )
    .spawn()
}

/// One-shot boot recovery: `ReconcileDispatch` every session holding a timer, so
/// work orphaned by the previous process fails and re-issues now instead of at
/// its deadline. Sessions with nothing orphaned no-op.
pub fn spawn_boot_reconciler(
    store: Arc<dyn EventStore>,
    wake_store: Arc<dyn WakeScheduleStore>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let horizon = Utc::now() + chrono::Duration::days(36500);
        let items = match wake_store.list_due_wakes(horizon, 100_000).await {
            Ok(items) => items,
            Err(e) => {
                tracing::warn!(error = %e, "boot reconcile: listing wakes failed");
                return;
            }
        };
        for item in items {
            let _ = execute(
                store.as_ref(),
                ExecuteInput {
                    session_id: item.session_id,
                    caller: Caller::System {
                        tenant_id: item.tenant_id,
                    },
                    command: CommandPayload::ReconcileDispatch,
                    span: SpanContext::root(),
                },
                &ConflictRetry::default(),
            )
            .await;
        }
    })
}

pub fn spawn_wake_dispatcher(
    store: Arc<dyn EventStore>,
    wake_store: Arc<dyn WakeScheduleStore>,
    poll_interval: Duration,
    cancel: CancellationToken,
) -> JoinHandle<()> {
    let mut rx = store.subscribe();
    tokio::spawn(async move {
        loop {
            if cancel.is_cancelled() {
                break;
            }
            let now = Utc::now();
            fire_due(&store, &wake_store, now).await;
            let next_wake = wake_store.next_wake_at().await.unwrap_or(None);
            let mut deadline = match next_wake {
                Some(at) if at <= now => continue,
                Some(at) => at.min(now + chrono::Duration::from_std(poll_interval).unwrap()),
                None => now + chrono::Duration::from_std(poll_interval).unwrap(),
            };

            // An event's own `wake_at` is authoritative for the sleep: the
            // projection it lands in may not have applied yet, so re-reading the
            // store here can miss it and oversleep to the fallback poll.
            loop {
                tokio::select! {
                    _ = tokio::time::sleep(chrono_to_std(deadline - Utc::now())) => break,
                    batch = rx.recv() => {
                        if let Some(events) = batch {
                            let earliest = events
                                .iter()
                                .filter_map(extract_wake_at)
                                .min();
                            if let Some(at) = earliest {
                                deadline = deadline.min(at);
                            }
                        }
                    }
                    _ = cancel.cancelled() => return,
                }
            }
        }
    })
}

async fn fire_due(
    store: &Arc<dyn EventStore>,
    wake_store: &Arc<dyn WakeScheduleStore>,
    now: DateTime<Utc>,
) {
    let due = match wake_store.list_due_wakes(now, 256).await {
        Ok(items) => items,
        Err(_) => return,
    };
    for item in due {
        let _ = execute(
            store.as_ref(),
            ExecuteInput {
                session_id: item.session_id,
                caller: Caller::System {
                    tenant_id: item.tenant_id,
                },
                command: CommandPayload::Wake { now },
                span: SpanContext::root(),
            },
            &ConflictRetry::default(),
        )
        .await;
    }
}

fn extract_wake_at(event: &SessionEvent) -> Option<DateTime<Utc>> {
    event.meta.wake_at
}

fn chrono_to_std(d: chrono::Duration) -> Duration {
    d.to_std().unwrap_or(Duration::ZERO)
}
