use std::sync::Arc;

use ractor::{call_t, Actor, ActorProcessingErr, ActorRef, RpcReplyPort};
use uuid::Uuid;

use super::event_store::EventStore;
use super::session_actor::{RuntimeError, SessionMessage};
use crate::domain::event::{Event, SessionAuth};
use crate::domain::session::{AgentState, SessionCommand};

// ---------------------------------------------------------------------------
// ClientMessage — commands from transport + events from dispatcher
// ---------------------------------------------------------------------------

pub enum ClientMessage {
    /// From transport: forward a command to the session
    SendCommand(
        Box<SessionCommand>,
        RpcReplyPort<Result<Vec<Event>, RuntimeError>>,
    ),
    /// From dispatcher: events for this session
    Events(Vec<Arc<Event>>),
    /// From transport: query current session state
    GetState(RpcReplyPort<AgentState>),
}

// ---------------------------------------------------------------------------
// SessionClientActor
// ---------------------------------------------------------------------------

pub struct SessionClientActor;

/// Callback invoked for each event after it is applied.
pub type OnEvent = Box<dyn Fn(&Event) + Send + Sync>;

pub struct SessionClientState {
    session_id: Uuid,
    core: AgentState,
    session_actor: ActorRef<SessionMessage>,
    on_event: Option<OnEvent>,
}

pub struct SessionClientArgs {
    pub session_id: Uuid,
    pub auth: SessionAuth,
    pub session_actor: ActorRef<SessionMessage>,
    pub store: Arc<dyn EventStore>,
    pub on_event: Option<OnEvent>,
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
        let core = match args.store.load(args.session_id, &args.auth) {
            Ok(load) => load.snapshot,
            Err(_) => AgentState::new(args.session_id),
        };

        let group = format!("session-clients-{}", args.session_id);

        ractor::pg::join(group, vec![myself.get_cell()]);

        Ok(SessionClientState {
            session_id: args.session_id,
            core,
            session_actor: args.session_actor,
            on_event: args.on_event,
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
                let result = call_t!(state.session_actor, SessionMessage::Execute, 5000, *cmd);
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
                    state.core.apply_core(event);
                    if let Some(f) = &state.on_event {
                        f(event);
                    }
                }
            }
            ClientMessage::GetState(reply) => {
                let _ = reply.send(state.core.clone());
            }
        }
        Ok(())
    }
}
