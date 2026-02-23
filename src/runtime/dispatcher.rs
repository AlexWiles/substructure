use std::collections::HashMap;
use std::sync::Arc;

use ractor::{Actor, ActorProcessingErr, ActorRef, SupervisionEvent};
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::domain::event::Event;
use super::session_actor::SessionMessage;
use super::session_client::ClientMessage;
use super::event_store::EventStore;

// ---------------------------------------------------------------------------
// Dispatcher actor — reads global log, delivers events to session actors
// ---------------------------------------------------------------------------

pub enum DispatcherMessage {
    Wake,
}

pub struct DispatcherActor;

pub struct DispatcherState {
    store: Arc<dyn EventStore>,
    offset: u64,
}

pub struct DispatcherArgs {
    store: Arc<dyn EventStore>,
}

impl Actor for DispatcherActor {
    type Msg = DispatcherMessage;
    type State = DispatcherState;
    type Arguments = DispatcherArgs;

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        Ok(DispatcherState {
            store: args.store,
            offset: 0,
        })
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match message {
            DispatcherMessage::Wake => {
                loop {
                    let events = state.store.read_from(state.offset, 100);
                    if events.is_empty() {
                        break;
                    }

                    // Group events by session_id
                    let mut by_session: HashMap<Uuid, Vec<Arc<Event>>> = HashMap::new();
                    for event in &events {
                        by_session
                            .entry(event.session_id)
                            .or_default()
                            .push(Arc::clone(event));
                    }

                    // Deliver to session actors via registry
                    for (session_id, session_events) in by_session {
                        let name = format!("session-{session_id}");
                        match ractor::registry::where_is(name) {
                            Some(cell) => {
                                let actor: ActorRef<SessionMessage> = cell.into();
                                let _ = actor
                                    .send_message(SessionMessage::Events(session_events.clone()));
                            }
                            None => {
                                eprintln!(
                                    "dispatcher: session-{session_id} not found in registry"
                                );
                            }
                        }

                        // Fan-out to session clients via process group
                        let client_group = format!("session-clients-{session_id}");
                        for cell in ractor::pg::get_members(&client_group) {
                            let client: ActorRef<ClientMessage> = cell.into();
                            let _ = client
                                .send_message(ClientMessage::Events(session_events.clone()));
                        }
                    }

                    // Advance offset (position in global log, decoupled from
                    // session-local sequences)
                    state.offset += events.len() as u64;
                }
            }
        }
        Ok(())
    }

    async fn handle_supervisor_evt(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: SupervisionEvent,
        _state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match &message {
            SupervisionEvent::ActorFailed(who, err) => {
                eprintln!("session {:?} failed: {err}", who.get_name());
            }
            SupervisionEvent::ActorTerminated(who, _, reason) => {
                eprintln!("session {:?} terminated: {reason:?}", who.get_name());
            }
            _ => {}
        }
        Ok(())
    }
}

pub async fn spawn_dispatcher(
    store: Arc<dyn EventStore>,
) -> (ActorRef<DispatcherMessage>, JoinHandle<()>) {
    let (actor_ref, handle) = Actor::spawn(
        Some("dispatcher".to_string()),
        DispatcherActor,
        DispatcherArgs {
            store: store.clone(),
        },
    )
    .await
    .expect("failed to spawn dispatcher");

    store
        .notify()
        .subscribe(actor_ref.clone(), |_| Some(DispatcherMessage::Wake));

    (actor_ref, handle)
}
