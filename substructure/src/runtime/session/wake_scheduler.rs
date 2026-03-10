use std::sync::Arc;

use chrono::{DateTime, Utc};
use ractor::{Actor, ActorCell, ActorProcessingErr, ActorRef, SpawnErr};
use tokio::task::AbortHandle;

use crate::runtime::event_store::{AggregateFilter, AggregateSort, EventBatch, EventStore};
use super::system::SessionSystem;

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

pub enum WakeSchedulerMessage {
    /// New events from the store — may require rescheduling.
    Events(EventBatch),
    /// Timer tick — fire due aggregates and reschedule.
    Tick,
}

// ---------------------------------------------------------------------------
// Actor
// ---------------------------------------------------------------------------

pub struct WakeScheduler;

pub struct WakeSchedulerState {
    store: Arc<dyn EventStore>,
    sessions: SessionSystem,
    myself: ActorRef<WakeSchedulerMessage>,
    next_tick_at: Option<DateTime<Utc>>,
    timer_handle: Option<AbortHandle>,
}

pub struct WakeSchedulerArgs {
    pub store: Arc<dyn EventStore>,
    pub sessions: SessionSystem,
}

fn schedule(state: &mut WakeSchedulerState, at: DateTime<Utc>) {
    if let Some(handle) = state.timer_handle.take() {
        handle.abort();
    }

    let delay = (at - Utc::now())
        .to_std()
        .unwrap_or(std::time::Duration::ZERO);

    let myself = state.myself.clone();
    let handle = tokio::spawn(async move {
        tokio::time::sleep(delay).await;
        if let Err(e) = myself.send_message(WakeSchedulerMessage::Tick) {
            tracing::warn!(error = %e, "failed to send scheduled tick");
        }
    });

    state.next_tick_at = Some(at);
    state.timer_handle = Some(handle.abort_handle());
}

/// Query the store for the next *future* wake time and schedule a timer.
/// Past-due aggregates are handled by the current Tick; scheduling them
/// again would produce a zero-delay timer and spin in an infinite loop.
async fn reschedule(state: &mut WakeSchedulerState) {
    if let Some(handle) = state.timer_handle.take() {
        handle.abort();
    }
    state.next_tick_at = None;

    let now = Utc::now();
    let filter = AggregateFilter {
        wake_at_not_null: true,
        sort: AggregateSort::WakeAtAsc,
        limit: Some(1),
        ..Default::default()
    };
    let results = state.store.list_aggregates(&filter).await;
    if let Some(next) = results.first().and_then(|s| s.wake_at) {
        if next > now {
            schedule(state, next);
        }
    }
}

impl Actor for WakeScheduler {
    type Msg = WakeSchedulerMessage;
    type State = WakeSchedulerState;
    type Arguments = WakeSchedulerArgs;

    async fn pre_start(
        &self,
        myself: ActorRef<Self::Msg>,
        args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        // Send an initial Tick to catch up on any due aggregates at startup.
        if let Err(e) = myself.send_message(WakeSchedulerMessage::Tick) {
            tracing::warn!(error = %e, "failed to send initial tick");
        }

        Ok(WakeSchedulerState {
            store: args.store,
            sessions: args.sessions,
            myself,
            next_tick_at: None,
            timer_handle: None,
        })
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match message {
            WakeSchedulerMessage::Events(events) => {
                // Check if any event's wake_at is sooner than our current timer.
                // Ignore past wake_at values — those are handled by the current
                // or next Tick, not by scheduling a zero-delay timer.
                let now = Utc::now();
                for event in &events {
                    if let Some(wake_at) = event.wake_at {
                        if wake_at > now {
                            let sooner = state.next_tick_at.is_none_or(|next| wake_at < next);
                            if sooner {
                                schedule(state, wake_at);
                            }
                        }
                    }
                }
            }
            WakeSchedulerMessage::Tick => {
                // Fire all due aggregates.
                let filter = AggregateFilter {
                    wake_at_before: Some(Utc::now()),
                    wake_at_not_null: true,
                    ..Default::default()
                };
                let due = state.store.list_aggregates(&filter).await;
                for agg in due {
                    let ss = state.sessions.clone();
                    tokio::spawn(async move {
                        ss.wake_aggregate(agg.aggregate_id, &agg.aggregate_type, &agg.tenant_id)
                            .await;
                    });
                }

                // Schedule next tick from the store.
                reschedule(state).await;
            }
        }
        Ok(())
    }
}

pub async fn spawn_wake_scheduler(
    store: Arc<dyn EventStore>,
    sessions: SessionSystem,
    supervisor: ActorCell,
) -> Result<ActorRef<WakeSchedulerMessage>, SpawnErr> {
    let (actor_ref, _handle) = Actor::spawn_linked(
        Some("wake-scheduler".to_string()),
        WakeScheduler,
        WakeSchedulerArgs {
            store: store.clone(),
            sessions,
        },
        supervisor,
    )
    .await?;

    store.events().subscribe(actor_ref.clone(), |batch| {
        Some(WakeSchedulerMessage::Events(batch))
    });

    Ok(actor_ref)
}
