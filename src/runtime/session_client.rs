use std::sync::Arc;

use ractor::{call_t, Actor, ActorProcessingErr, ActorRef, RpcReplyPort};
use uuid::Uuid;

use crate::domain::session::{SessionCommand, SessionState};
use crate::domain::event::{Event, SessionAuth};
use super::session_actor::{RuntimeError, SessionMessage};
use super::event_store::EventStore;

// ---------------------------------------------------------------------------
// ClientMessage — commands from transport + events from dispatcher
// ---------------------------------------------------------------------------

pub enum ClientMessage {
    /// From transport: forward a command to the session
    SendCommand(
        SessionCommand,
        RpcReplyPort<Result<Vec<Event>, RuntimeError>>,
    ),
    /// From dispatcher: events for this session
    Events(Vec<Arc<Event>>),
    /// From transport: query current session state
    GetState(RpcReplyPort<SessionState>),
}

// ---------------------------------------------------------------------------
// SessionClientActor
// ---------------------------------------------------------------------------

pub struct SessionClientActor;

pub struct SessionClientState {
    session_id: Uuid,
    session: SessionState,
    session_actor: ActorRef<SessionMessage>,
}

pub struct SessionClientArgs {
    pub session_id: Uuid,
    pub auth: SessionAuth,
    pub session_actor: ActorRef<SessionMessage>,
    pub store: Arc<dyn EventStore>,
}

impl Actor for SessionClientActor {
    type Msg = ClientMessage;
    type State = SessionClientState;
    type Arguments = SessionClientArgs;

    async fn pre_start(
        &self,
        myself: ActorRef<Self::Msg>,
        args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        let mut session = SessionState::new(args.session_id);

        // Replay existing history so the client starts up-to-date
        if let Ok(events) = args.store.load(args.session_id, &args.auth) {
            for event in &events {
                session.apply(event);
            }
        }

        // Join the process group *after* replay so we don't double-apply
        // events that arrive from the dispatcher while we're catching up
        let group = format!("session-clients-{}", args.session_id);
        ractor::pg::join(group, vec![myself.get_cell()]);

        Ok(SessionClientState {
            session_id: args.session_id,
            session,
            session_actor: args.session_actor,
        })
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match message {
            ClientMessage::SendCommand(cmd, reply) => {
                let result = call_t!(
                    state.session_actor,
                    SessionMessage::Execute,
                    5000,
                    cmd
                );
                match result {
                    Ok(inner) => {
                        let _ = reply.send(inner);
                    }
                    Err(e) => {
                        eprintln!(
                            "session-client[{}]: call to session actor failed: {e}",
                            state.session_id
                        );
                    }
                }
            }
            ClientMessage::Events(events) => {
                for event in &events {
                    state.session.apply(event);
                }
            }
            ClientMessage::GetState(reply) => {
                let _ = reply.send(state.session.clone());
            }
        }
        Ok(())
    }
}
