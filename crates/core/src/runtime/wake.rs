use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::runtime::event_store::EventStore;
use crate::runtime::processor::{
    EventProcessor, EventProcessorRunner, EventProcessorRunnerConfig, ProcessorCheckpointStore,
    ProcessorError,
};
use crate::runtime::session::command::CommandPayload;
use crate::runtime::session::{execute, ConflictRetry, ExecuteInput, SessionEvent};
use crate::runtime::span::SpanContext;
use crate::runtime::Caller;

#[derive(Debug, Clone)]
pub struct WakeScheduleItem {
    pub tenant_id: String,
    pub aggregate_id: String,
    pub wake_at: DateTime<Utc>,
}

#[async_trait::async_trait]
pub trait WakeScheduleStore: Send + Sync {
    async fn upsert_wake(
        &self,
        tenant_id: &str,
        aggregate_id: &str,
        wake_at: DateTime<Utc>,
    ) -> Result<(), String>;
    async fn remove_wake(&self, tenant_id: &str, aggregate_id: &str) -> Result<(), String>;
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
        match event.derived.and_then(|d| d.wake_at) {
            Some(wake_at) => self
                .wake_store
                .upsert_wake(&event.tenant_id, &event.aggregate_id, wake_at)
                .await
                .map_err(ProcessorError::Apply),
            None => self
                .wake_store
                .remove_wake(&event.tenant_id, &event.aggregate_id)
                .await
                .map_err(ProcessorError::Apply),
        }
    }
}

pub fn spawn_wake_processor(
    store: Arc<dyn EventStore>,
    checkpoint_store: Arc<dyn ProcessorCheckpointStore>,
    wake_store: Arc<dyn WakeScheduleStore>,
    cancel: CancellationToken,
) -> JoinHandle<()> {
    let projection = Arc::new(WakeScheduleProjection::new(wake_store));
    EventProcessorRunner::new(
        store,
        checkpoint_store,
        projection,
        EventProcessorRunnerConfig::default(),
        cancel,
    )
    .spawn()
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
            let deadline = match next_wake {
                Some(at) if at <= now => continue,
                Some(at) => at.min(now + chrono::Duration::from_std(poll_interval).unwrap()),
                None => now + chrono::Duration::from_std(poll_interval).unwrap(),
            };

            tokio::select! {
                _ = tokio::time::sleep(chrono_to_std(deadline - now)) => {}
                batch = rx.recv() => {
                    if let Ok(events) = batch {
                        let earliest = events
                            .iter()
                            .filter_map(extract_wake_at)
                            .min();
                        if earliest.is_some_and(|at| at < deadline) {
                            continue;
                        }
                    }
                }
                _ = cancel.cancelled() => break,
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
                aggregate_id: item.aggregate_id,
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
    event.derived.as_ref()?.wake_at
}

fn chrono_to_std(d: chrono::Duration) -> Duration {
    d.to_std().unwrap_or(Duration::ZERO)
}
