use std::collections::HashMap;
use std::sync::Arc;

use ractor::{Actor, ActorCell, ActorProcessingErr, ActorRef, SpawnErr};
use uuid::Uuid;

use super::event_store::{EventBatch, EventStore};
use super::session_actor::SessionMessage;
use super::session_client::ClientMessage;
use crate::domain::event::Event;

// ---------------------------------------------------------------------------
// Dispatcher actor — fans out new events to session actors and clients
// ---------------------------------------------------------------------------

pub enum DispatcherMessage {
    Events(EventBatch),
}

pub struct DispatcherActor;

impl Actor for DispatcherActor {
    type Msg = DispatcherMessage;
    type State = ();
    type Arguments = ();

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        _args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        Ok(())
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        _state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match message {
            DispatcherMessage::Events(events) => {
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
                    if let Some(cell) = ractor::registry::where_is(name) {
                        let actor: ActorRef<SessionMessage> = cell.into();
                        let _ = actor.send_message(SessionMessage::Events(session_events.clone()));
                    }

                    // Fan-out to session clients via process group
                    let client_group = format!("session-clients-{session_id}");
                    for cell in ractor::pg::get_members(&client_group) {
                        let client: ActorRef<ClientMessage> = cell.into();
                        let _ = client.send_message(ClientMessage::Events(session_events.clone()));
                    }
                }
            }
        }
        Ok(())
    }
}

pub async fn spawn_dispatcher(
    store: Arc<dyn EventStore>,
    supervisor: ActorCell,
) -> Result<ActorRef<DispatcherMessage>, SpawnErr> {
    let (actor_ref, _handle) = Actor::spawn_linked(
        Some("dispatcher".to_string()),
        DispatcherActor,
        (),
        supervisor,
    )
    .await?;

    store.events().subscribe(actor_ref.clone(), |batch| {
        Some(DispatcherMessage::Events(batch))
    });

    Ok(actor_ref)
}
