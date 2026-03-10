use uuid::Uuid;

use crate::runtime::aggregate::{AggregateState, ApplyContext};

use super::command::{CommandPayload, SessionError};
use super::events::EventPayload;
use super::state::{DerivedState, SessionState};

impl AggregateState for SessionState {
    type Event = EventPayload;
    type Command = CommandPayload;
    type Error = SessionError;
    type Derived = DerivedState;

    const AGGREGATE_TYPE: &'static str = "session";

    fn initial(id: Uuid) -> Self {
        SessionState::new(id)
    }

    fn apply(&mut self, event: &EventPayload, ctx: &ApplyContext) {
        SessionState::apply(self, event, ctx);
    }

    fn handle_command(&self, cmd: CommandPayload) -> Result<Vec<EventPayload>, SessionError> {
        self.handle(cmd)
    }

    fn derived_state(&self) -> DerivedState {
        SessionState::derived_state(self)
    }

    fn wake_at(&self) -> Option<chrono::DateTime<chrono::Utc>> {
        SessionState::wake_at(self)
    }

    fn label(&self) -> Option<String> {
        self.agent_id.clone()
    }
}
