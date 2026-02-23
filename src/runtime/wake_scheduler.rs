use std::collections::HashMap;

use chrono::{DateTime, Utc};
use ractor::{Actor, ActorProcessingErr, ActorRef};
use uuid::Uuid;

use super::session_actor::SessionMessage;

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

pub enum WakeSchedulerMessage {
    /// Register or update a session's wake_at time.
    UpdateWakeAt {
        session_id: Uuid,
        wake_at: Option<DateTime<Utc>>,
        actor_ref: ActorRef<SessionMessage>,
    },
    /// Remove a session from the scheduler (e.g., on shutdown).
    Deregister { session_id: Uuid },
    /// Internal tick — check all sessions.
    Tick,
}

// ---------------------------------------------------------------------------
// Actor
// ---------------------------------------------------------------------------

pub struct WakeScheduler;

pub struct WakeSchedulerState {
    sessions: HashMap<Uuid, WakeEntry>,
}

struct WakeEntry {
    wake_at: Option<DateTime<Utc>>,
    actor_ref: ActorRef<SessionMessage>,
}

impl Actor for WakeScheduler {
    type Msg = WakeSchedulerMessage;
    type State = WakeSchedulerState;
    type Arguments = ();

    async fn pre_start(
        &self,
        myself: ActorRef<Self::Msg>,
        _args: (),
    ) -> Result<Self::State, ActorProcessingErr> {
        // Start the tick loop
        let myself_clone = myself.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                if myself_clone
                    .send_message(WakeSchedulerMessage::Tick)
                    .is_err()
                {
                    break;
                }
            }
        });

        Ok(WakeSchedulerState {
            sessions: HashMap::new(),
        })
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match message {
            WakeSchedulerMessage::UpdateWakeAt {
                session_id,
                wake_at,
                actor_ref,
            } => {
                state
                    .sessions
                    .insert(session_id, WakeEntry { wake_at, actor_ref });
            }
            WakeSchedulerMessage::Deregister { session_id } => {
                state.sessions.remove(&session_id);
            }
            WakeSchedulerMessage::Tick => {
                let now = Utc::now();
                for entry in state.sessions.values() {
                    if let Some(wake_at) = entry.wake_at {
                        if wake_at <= now {
                            let _ = entry.actor_ref.send_message(SessionMessage::Wake);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
