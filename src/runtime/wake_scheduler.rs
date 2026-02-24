use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use ractor::{Actor, ActorCell, ActorProcessingErr, ActorRef, SpawnErr};
use tokio::task::AbortHandle;
use uuid::Uuid;

use super::event_store::EventStore;
use super::session_actor::SessionMessage;

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

pub enum WakeSchedulerMessage {
    /// New events in store — read and update wake times.
    Wake,
    /// Timer tick — fire due sessions.
    Tick,
}

// ---------------------------------------------------------------------------
// Actor
// ---------------------------------------------------------------------------

pub struct WakeScheduler;

pub struct WakeSchedulerState {
    store: Arc<dyn EventStore>,
    offset: u64,
    sessions: HashMap<Uuid, Option<DateTime<Utc>>>,
    myself: Option<ActorRef<WakeSchedulerMessage>>,
    timer_handle: Option<AbortHandle>,
}

pub struct WakeSchedulerArgs {
    pub store: Arc<dyn EventStore>,
}

fn reschedule(state: &mut WakeSchedulerState) {
    // Abort existing timer
    if let Some(handle) = state.timer_handle.take() {
        handle.abort();
    }

    // Find the earliest wake_at
    let next_wake = state
        .sessions
        .values()
        .filter_map(|t| *t)
        .min();

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
        Ok(WakeSchedulerState {
            store: args.store,
            offset: 0,
            sessions: HashMap::new(),
            myself: Some(myself),
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
            WakeSchedulerMessage::Wake => {
                loop {
                    let events = state.store.read_from(state.offset, 100);
                    if events.is_empty() {
                        break;
                    }

                    for event in &events {
                        if let Some(derived) = &event.derived {
                            match derived.wake_at {
                                Some(wake_at) => {
                                    state.sessions.insert(event.session_id, Some(wake_at));
                                }
                                None => {
                                    state.sessions.remove(&event.session_id);
                                }
                            };
                        }
                    }

                    state.offset += events.len() as u64;
                }
                reschedule(state);
            }
            WakeSchedulerMessage::Tick => {
                let now = Utc::now();
                let due: Vec<Uuid> = state
                    .sessions
                    .iter()
                    .filter_map(|(id, wake_at)| {
                        wake_at.filter(|t| *t <= now).map(|_| *id)
                    })
                    .collect();

                for session_id in due {
                    let name = format!("session-{session_id}");
                    if let Some(cell) = ractor::registry::where_is(name) {
                        let actor: ActorRef<SessionMessage> = cell.into();
                        let _ = actor.send_message(SessionMessage::Wake);
                    }
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
    supervisor: ActorCell,
) -> Result<ActorRef<WakeSchedulerMessage>, SpawnErr> {
    let (actor_ref, _handle) = Actor::spawn_linked(
        Some("wake-scheduler".to_string()),
        WakeScheduler,
        WakeSchedulerArgs {
            store: store.clone(),
        },
        supervisor,
    )
    .await?;

    store
        .notify()
        .subscribe(actor_ref.clone(), |_| Some(WakeSchedulerMessage::Wake));

    // Send initial Wake to catch up on existing log
    let _ = actor_ref.send_message(WakeSchedulerMessage::Wake);

    Ok(actor_ref)
}
