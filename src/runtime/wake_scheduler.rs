use std::sync::Arc;

use chrono::{DateTime, Utc};
use ractor::{Actor, ActorCell, ActorProcessingErr, ActorRef, SpawnErr};
use tokio::task::AbortHandle;

use super::event_store::{AggregateFilter, AggregateSort, EventBatch, EventStore};
use super::RuntimeMessage;

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
    runtime: ActorRef<RuntimeMessage>,
    myself: ActorRef<WakeSchedulerMessage>,
    next_tick_at: Option<DateTime<Utc>>,
    timer_handle: Option<AbortHandle>,
}

pub struct WakeSchedulerArgs {
    pub store: Arc<dyn EventStore>,
    pub runtime: ActorRef<RuntimeMessage>,
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
        let _ = myself.send_message(WakeSchedulerMessage::Tick);
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
        let _ = myself.send_message(WakeSchedulerMessage::Tick);

        Ok(WakeSchedulerState {
            store: args.store,
            runtime: args.runtime,
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
                            let sooner =
                                state.next_tick_at.is_none_or(|next| wake_at < next);
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
                    let _ = state.runtime.send_message(RuntimeMessage::WakeAggregate {
                        aggregate_id: agg.aggregate_id,
                        aggregate_type: agg.aggregate_type,
                        tenant_id: agg.tenant_id,
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
    runtime: ActorRef<RuntimeMessage>,
    supervisor: ActorCell,
) -> Result<ActorRef<WakeSchedulerMessage>, SpawnErr> {
    let (actor_ref, _handle) = Actor::spawn_linked(
        Some("wake-scheduler".to_string()),
        WakeScheduler,
        WakeSchedulerArgs {
            store: store.clone(),
            runtime,
        },
        supervisor,
    )
    .await?;

    store.events().subscribe(actor_ref.clone(), |batch| {
        Some(WakeSchedulerMessage::Events(batch))
    });

    Ok(actor_ref)
}
