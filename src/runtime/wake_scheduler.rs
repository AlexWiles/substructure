use std::sync::Arc;

use chrono::{DateTime, Utc};
use ractor::{Actor, ActorCell, ActorProcessingErr, ActorRef, SpawnErr};
use tokio::task::AbortHandle;

use crate::domain::session::DerivedState;

use super::event_store::session_index::{SessionFilter, SessionIndex};
use super::event_store::{EventBatch, EventStore};
use super::session_actor::SessionMessage;

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

pub enum WakeSchedulerMessage {
    /// New events from the store — may require rescheduling.
    Events(EventBatch),
    /// Timer tick — fire due sessions and reschedule.
    Tick,
}

// ---------------------------------------------------------------------------
// Actor
// ---------------------------------------------------------------------------

pub struct WakeScheduler;

pub struct WakeSchedulerState {
    session_index: Arc<dyn SessionIndex>,
    myself: ActorRef<WakeSchedulerMessage>,
    next_tick_at: Option<DateTime<Utc>>,
    timer_handle: Option<AbortHandle>,
}

pub struct WakeSchedulerArgs {
    pub session_index: Arc<dyn SessionIndex>,
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

/// Query the session index for the next wake time and schedule a timer.
async fn reschedule(state: &mut WakeSchedulerState) {
    if let Some(handle) = state.timer_handle.take() {
        handle.abort();
    }
    state.next_tick_at = None;

    if let Some(next) = state.session_index.next_wake_at().await {
        schedule(state, next);
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
        // Send an initial Tick to catch up on any due sessions.
        let _ = myself.send_message(WakeSchedulerMessage::Tick);

        Ok(WakeSchedulerState {
            session_index: args.session_index,
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
                // Check if any event sets a wake_at sooner than our current timer.
                for event in &events {
                    if let Some(derived_value) = &event.derived {
                        if let Ok(derived) =
                            serde_json::from_value::<DerivedState>(derived_value.clone())
                        {
                            if let Some(wake_at) = derived.wake_at {
                                let sooner = state
                                    .next_tick_at
                                    .is_none_or(|next| wake_at < next);
                                if sooner {
                                    schedule(state, wake_at);
                                }
                            }
                        }
                    }
                }
            }
            WakeSchedulerMessage::Tick => {
                // Fire all due sessions.
                let due = state.session_index.list_sessions(&SessionFilter {
                    needs_wake: Some(true),
                    ..Default::default()
                }).await;
                for session in due {
                    super::send_to_session(session.session_id, SessionMessage::Wake);
                }

                // Schedule next tick from the index.
                reschedule(state).await;
            }
        }
        Ok(())
    }
}

pub async fn spawn_wake_scheduler(
    store: Arc<dyn EventStore>,
    session_index: Arc<dyn SessionIndex>,
    supervisor: ActorCell,
) -> Result<ActorRef<WakeSchedulerMessage>, SpawnErr> {
    let (actor_ref, _handle) = Actor::spawn_linked(
        Some("wake-scheduler".to_string()),
        WakeScheduler,
        WakeSchedulerArgs { session_index },
        supervisor,
    )
    .await?;

    store.events().subscribe(actor_ref.clone(), |batch| {
        Some(WakeSchedulerMessage::Events(batch))
    });

    Ok(actor_ref)
}
