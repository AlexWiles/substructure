use std::sync::Arc;

use chrono::Utc;
use uuid::Uuid;

use super::agent_state::{AgentState, StrategySlot};
use super::command_handler::{SessionCommand, SessionError};
use super::strategy::Strategy;
use crate::domain::event::*;

pub(super) fn new_call_id() -> String {
    Uuid::new_v4().to_string()
}

pub struct AgentSession {
    pub agent_state: AgentState,
}

impl AgentSession {
    pub fn new(session_id: Uuid, strategy: Arc<dyn Strategy>) -> Self {
        let mut agent_state = AgentState::new(session_id);
        agent_state.strategy_state = strategy.default_state();
        agent_state.strategy = StrategySlot(Some(strategy));
        AgentSession { agent_state }
    }

    /// Build from a stored snapshot.
    pub fn from_snapshot(snapshot: AgentState, strategy: Arc<dyn Strategy>) -> Self {
        let mut agent_state = snapshot;
        agent_state.strategy = StrategySlot(Some(strategy));
        AgentSession { agent_state }
    }

    /// Take a snapshot of the current session state.
    pub fn snapshot(&self) -> AgentState {
        self.agent_state.clone()
    }

    pub fn apply(&mut self, event: &Event) {
        self.agent_state.apply_core(event);
    }

    /// Compute the derived state envelope for the current session state.
    pub fn derived_state(&self) -> DerivedState {
        DerivedState {
            status: self.agent_state.status.clone(),
            wake_at: self.agent_state.wake_at(),
        }
    }

    /// Process a command: validate, build events, apply, and stamp derived state.
    ///
    /// Returns the fully-stamped events and a post-apply snapshot, ready for
    /// persistence. The caller only needs to persist — no domain logic leaks out.
    pub fn process_command(
        &mut self,
        cmd: SessionCommand,
        tenant_id: &str,
    ) -> Result<(Vec<Event>, AgentState), SessionError> {
        let payloads = self.handle(cmd.payload)?;
        if payloads.is_empty() {
            return Ok((vec![], self.snapshot()));
        }

        let base_seq = self.agent_state.stream_version + 1;
        let mut events: Vec<Event> = payloads
            .into_iter()
            .enumerate()
            .map(|(i, payload)| Event {
                id: Uuid::new_v4(),
                tenant_id: tenant_id.to_string(),
                session_id: self.agent_state.session_id,
                sequence: base_seq + i as u64,
                span: cmd.span.clone(),
                occurred_at: cmd.occurred_at,
                payload,
                derived: None,
            })
            .collect();

        for event in &events {
            self.apply(event);
        }
        let derived = Some(self.derived_state());
        for event in &mut events {
            event.derived = derived.clone();
        }

        Ok((events, self.snapshot()))
    }

    /// Compute LLM call deadline from agent config.
    pub(super) fn llm_deadline(&self) -> chrono::DateTime<Utc> {
        let timeout = self
            .agent_state
            .agent
            .as_ref()
            .map(|a| a.retry.llm_timeout_secs)
            .unwrap_or(60);
        Utc::now() + chrono::Duration::seconds(timeout as i64)
    }

    /// Compute tool call deadline from agent config.
    pub(super) fn tool_deadline(&self) -> chrono::DateTime<Utc> {
        let timeout = self
            .agent_state
            .agent
            .as_ref()
            .map(|a| a.retry.tool_timeout_secs)
            .unwrap_or(120);
        Utc::now() + chrono::Duration::seconds(timeout as i64)
    }
}
