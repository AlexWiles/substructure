use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use ractor::{Actor, ActorCell, ActorProcessingErr, ActorRef, SpawnErr};
use tokio::task::AbortHandle;
use uuid::Uuid;

use crate::domain::session::DerivedState;

use super::event_store::session_index::{SessionFilter, SessionIndex};
use super::event_store::{EventBatch, EventStore};
use super::session_actor::SessionMessage;

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

pub enum WakeSchedulerMessage {
    /// New events from the store — update wake times.
    Events(EventBatch),
    /// Timer tick — fire due sessions.
    Tick,
}

// ---------------------------------------------------------------------------
// Actor
// ---------------------------------------------------------------------------

pub struct WakeScheduler;

pub struct WakeSchedulerState {
    sessions: HashMap<Uuid, Option<DateTime<Utc>>>,
    myself: Option<ActorRef<WakeSchedulerMessage>>,
    timer_handle: Option<AbortHandle>,
}

pub struct WakeSchedulerArgs {
    pub store: Arc<dyn EventStore>,
    pub session_index: Arc<dyn SessionIndex>,
}

fn reschedule(state: &mut WakeSchedulerState) {
    // Abort existing timer
    if let Some(handle) = state.timer_handle.take() {
        handle.abort();
    }

    // Find the earliest wake_at
    let next_wake = state.sessions.values().filter_map(|t| *t).min();

    let wake_at = match next_wake {
        Some(t) => t,
        None => return,
    };

    let myself = match &state.myself {
        Some(r) => r.clone(),
        None => return,
    };

    // Compute delay, clamped to zero if past-due
    let now = Utc::now();
    let delay = (wake_at - now)
        .to_std()
        .unwrap_or(std::time::Duration::ZERO);

    let handle = tokio::spawn(async move {
        tokio::time::sleep(delay).await;
        let _ = myself.send_message(WakeSchedulerMessage::Tick);
    });

    state.timer_handle = Some(handle.abort_handle());
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
        // Catch up on existing sessions that need waking
        let mut sessions: HashMap<Uuid, Option<DateTime<Utc>>> = HashMap::new();
        let summaries = args.session_index.list_sessions(&SessionFilter {
            needs_wake: Some(true),
            ..Default::default()
        });
        for s in summaries {
            sessions.insert(s.session_id, s.wake_at);
        }

        let mut state = WakeSchedulerState {
            sessions,
            myself: Some(myself),
            timer_handle: None,
        };
        reschedule(&mut state);
        Ok(state)
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match message {
            WakeSchedulerMessage::Events(events) => {
                for event in &events {
                    if let Some(derived_value) = &event.derived {
                        if let Ok(derived) =
                            serde_json::from_value::<DerivedState>(derived_value.clone())
                        {
                            match derived.wake_at {
                                Some(wake_at) => {
                                    state
                                        .sessions
                                        .insert(event.aggregate_id, Some(wake_at));
                                }
                                None => {
                                    state.sessions.remove(&event.aggregate_id);
                                }
                            };
                        }
                    }
                }
                reschedule(state);
            }
            WakeSchedulerMessage::Tick => {
                let now = Utc::now();
                let due: Vec<Uuid> = state
                    .sessions
                    .iter()
                    .filter_map(|(id, wake_at)| wake_at.filter(|t| *t <= now).map(|_| *id))
                    .collect();

                for session_id in due {
                    super::send_to_session(session_id, SessionMessage::Wake);
                    state.sessions.remove(&session_id);
                }
                reschedule(state);
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
        WakeSchedulerArgs {
            store: store.clone(),
            session_index,
        },
        supervisor,
    )
    .await?;

    store.events().subscribe(actor_ref.clone(), |batch| {
        Some(WakeSchedulerMessage::Events(batch))
    });

    Ok(actor_ref)
}
