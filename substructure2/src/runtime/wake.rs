use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use tokio::task::JoinHandle;

use crate::runtime::aggregate::{execute, AggregateState, ExecuteInput};
use crate::runtime::event_store::{AggregateFilter, AggregateSort, Event, EventStore};
use crate::runtime::session::command::CommandPayload;
use crate::runtime::session::state::SessionState;
use crate::runtime::span::SpanContext;

pub fn spawn_wake_scheduler(
    store: Arc<dyn EventStore>,
    poll_interval: Duration,
) -> JoinHandle<()> {
    let mut rx = store.subscribe();
    tokio::spawn(async move {
        loop {
            let now = Utc::now();
            fire_due(&store, now).await;
            let next_wake = query_next_wake(&store).await;
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
            }
        }
    })
}

async fn fire_due(store: &Arc<dyn EventStore>, now: DateTime<Utc>) {
    let filter = AggregateFilter {
        aggregate_type: Some(SessionState::AGGREGATE_TYPE.to_string()),
        wake_at_before: Some(now),
        sort: AggregateSort::WakeAtAsc,
        ..Default::default()
    };
    let summaries = match store.list_aggregates(&filter).await {
        Ok(s) => s,
        Err(_) => return,
    };
    for summary in summaries {
        let _ = execute::<SessionState>(
            store.as_ref(),
            ExecuteInput {
                aggregate_id: summary.aggregate_id,
                tenant_id: summary.tenant_id,
                command: CommandPayload::Wake { now },
                span: SpanContext::root(),
            },
        )
        .await;
    }
}

async fn query_next_wake(store: &Arc<dyn EventStore>) -> Option<DateTime<Utc>> {
    let filter = AggregateFilter {
        aggregate_type: Some(SessionState::AGGREGATE_TYPE.to_string()),
        wake_at_not_null: true,
        sort: AggregateSort::WakeAtAsc,
        limit: Some(1),
        ..Default::default()
    };
    store
        .list_aggregates(&filter)
        .await
        .ok()?
        .into_iter()
        .next()?
        .wake_at
}

fn extract_wake_at(event: &Event) -> Option<DateTime<Utc>> {
    event
        .derived
        .as_ref()?
        .get("wake_at")?
        .as_str()?
        .parse()
        .ok()
}

fn chrono_to_std(d: chrono::Duration) -> Duration {
    d.to_std().unwrap_or(Duration::ZERO)
}
