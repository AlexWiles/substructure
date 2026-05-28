use crate::runtime::aggregate::{AggregateState, ApplyContext, Caller};

use super::command::{CommandPayload, SessionError};
use super::events::EventPayload;
use super::state::{DerivedState, SessionState};

impl AggregateState for SessionState {
    type Event = EventPayload;
    type Command = CommandPayload;
    type Error = SessionError;
    type Derived = DerivedState;

    const AGGREGATE_TYPE: &'static str = "session";

    fn initial(id: String) -> Self {
        SessionState::new(id)
    }

    fn apply(&mut self, event: &EventPayload, ctx: &ApplyContext) {
        SessionState::apply(self, event, ctx);
    }

    fn handle_command(
        &self,
        cmd: CommandPayload,
        caller: &Caller,
    ) -> Result<Vec<EventPayload>, SessionError> {
        self.handle(cmd, caller)
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
