use ractor::{Actor, ActorProcessingErr, ActorRef, SupervisionEvent};

use super::types::SupervisorMessage;

// ---------------------------------------------------------------------------
// SupervisorActor — pure supervisor, no message processing
// ---------------------------------------------------------------------------

pub(crate) struct SupervisorActor;

impl Actor for SupervisorActor {
    type Msg = SupervisorMessage;
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
        match message {} // unreachable — SupervisorMessage is uninhabited
    }

    async fn handle_supervisor_evt(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: SupervisionEvent,
        _state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match &message {
            SupervisionEvent::ActorFailed(who, err) => {
                tracing::error!(actor = ?who.get_name(), error = %err, "child actor failed");
            }
            SupervisionEvent::ActorTerminated(who, _, reason) => {
                if reason.is_some() {
                    tracing::error!(actor = ?who.get_name(), reason = ?reason, "child actor terminated unexpectedly");
                } else {
                    tracing::debug!(actor = ?who.get_name(), "child actor stopped");
                }
            }
            _ => {}
        }
        Ok(())
    }
}
