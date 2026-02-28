use std::sync::Arc;

use ractor::{call_t, Actor, ActorProcessingErr, ActorRef};
use uuid::Uuid;

use super::event_store::EventStore;
use super::session_actor::SessionMessage;
use crate::domain::aggregate::{Aggregate, DomainEvent};
use crate::domain::event::ClientIdentity;
use crate::domain::session::AgentState;

// ---------------------------------------------------------------------------
// SessionClientActor
// ---------------------------------------------------------------------------

pub struct SessionClientActor;

/// Callback invoked for each typed event after it is applied.
pub type OnEvent = Box<dyn Fn(&DomainEvent<AgentState>) + Send + Sync>;

pub struct SessionClientState {
    session_id: Uuid,
    core: Aggregate<AgentState>,
    session_actor: ActorRef<SessionMessage>,
    on_event: Option<OnEvent>,
}

pub struct SessionClientArgs {
    pub session_id: Uuid,
    pub auth: ClientIdentity,
    pub session_actor: ActorRef<SessionMessage>,
    pub store: Arc<dyn EventStore>,
    pub on_event: Option<OnEvent>,
}

impl Actor for SessionClientActor {
    type Msg = SessionMessage;
    type State = SessionClientState;
    type Arguments = SessionClientArgs;

    async fn pre_start(
        &self,
        myself: ActorRef<Self::Msg>,
        args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        let core = match args.store.load(args.session_id, &args.auth.tenant_id).await {
            Ok(load) => serde_json::from_value(load.snapshot)
                .unwrap_or_else(|_| Aggregate::new(AgentState::new(args.session_id))),
            Err(_) => Aggregate::new(AgentState::new(args.session_id)),
        };

        let group = super::session_group(args.session_id);
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
            SessionMessage::Execute(cmd, reply) => {
                let result = call_t!(state.session_actor, SessionMessage::Execute, 5000, cmd);
                match result {
                    Ok(inner) => {
                        let _ = reply.send(inner);
                    }
                    Err(e) => {
                        tracing::error!(session = %state.session_id, error = %e, "call to session actor failed");
                    }
                }
            }
            SessionMessage::Events(typed_events) => {
                for typed in &typed_events {
                    state.core.apply(&typed.payload, typed.sequence, typed.occurred_at);
                    if let Some(f) = &state.on_event {
                        f(typed);
                    }
                }
            }
            SessionMessage::GetState(reply) => {
                let _ = reply.send(state.core.state.clone());
            }
            _ => {} // Wake, Cancel, Cast, SetClientTools — not for clients
        }
        Ok(())
    }
}
