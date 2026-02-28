use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::agent_state::{AgentState, DerivedState, StrategySlot};
use super::command_handler::{SessionCommand, SessionError};
use super::strategy::Strategy;
use crate::domain::aggregate::{Aggregate, AggregateStatus, DomainEvent, Reducer};
use crate::domain::event::*;

pub(super) fn new_call_id() -> String {
    Uuid::new_v4().to_string()
}

/// MCP server info associated with a tool name (transient, populated by runtime).
#[derive(Debug, Clone)]
pub struct McpToolEntry {
    pub server_name: String,
    pub server_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSession {
    pub state: AgentState,
    /// MCP tool name → server info (transient, populated by runtime).
    #[serde(skip, default)]
    pub mcp_tools: HashMap<String, McpToolEntry>,
}

impl AgentSession {
    pub fn new(session_id: Uuid, strategy: Arc<dyn Strategy>) -> Self {
        let mut state = AgentState::new(session_id);
        state.strategy_state = strategy.default_state();
        state.strategy = StrategySlot(Some(strategy));
        AgentSession {
            state,
            mcp_tools: HashMap::new(),
        }
    }

    /// Build from a stored snapshot.
    pub fn from_snapshot(
        mut snapshot: Aggregate<AgentSession>,
        strategy: Arc<dyn Strategy>,
    ) -> Aggregate<AgentSession> {
        snapshot.state.attach_strategy(strategy);
        snapshot
    }

    pub fn attach_strategy(&mut self, strategy: Arc<dyn Strategy>) {
        self.state.strategy = StrategySlot(Some(strategy));
    }

    /// Compute LLM call deadline from agent config.
    pub(super) fn llm_deadline(&self) -> chrono::DateTime<Utc> {
        let timeout = self
            .state
            .agent
            .as_ref()
            .map(|a| a.retry.llm_timeout_secs)
            .unwrap_or(60);
        Utc::now() + chrono::Duration::seconds(timeout as i64)
    }

    /// Compute tool call deadline from agent config.
    pub(super) fn tool_deadline(&self) -> chrono::DateTime<Utc> {
        let timeout = self
            .state
            .agent
            .as_ref()
            .map(|a| a.retry.tool_timeout_secs)
            .unwrap_or(120);
        Utc::now() + chrono::Duration::seconds(timeout as i64)
    }

    /// Compute tool call metadata based on the tool name.
    pub(super) fn tool_call_meta(&self, name: &str, tool_call_id: &str) -> Option<ToolCallMeta> {
        // Check sub-agents
        if let Some(agent_name) = self
            .state
            .agent
            .as_ref()
            .and_then(|a| a.sub_agents.iter().find(|s| s.as_str() == name))
        {
            return Some(ToolCallMeta::SubAgent {
                child_session_id: Uuid::new_v5(&self.state.session_id, tool_call_id.as_bytes()),
                agent_name: agent_name.clone(),
            });
        }
        // Check MCP tools
        if let Some(entry) = self.mcp_tools.get(name) {
            return Some(ToolCallMeta::Mcp {
                server_name: entry.server_name.clone(),
                server_version: entry.server_version.clone(),
            });
        }
        None
    }

    pub fn derived_state(&self) -> DerivedState {
        self.state.derived_state()
    }

    pub fn wake_at(&self) -> Option<chrono::DateTime<Utc>> {
        self.state.wake_at()
    }

    pub fn label(&self) -> Option<String> {
        self.state.agent.as_ref().map(|a| a.name.clone())
    }

    pub fn agent(&self) -> Option<&AgentConfig> {
        self.state.agent.as_ref()
    }

    pub fn tool_calls(
        &self,
    ) -> &std::collections::HashMap<String, super::agent_state::ToolCallState> {
        &self.state.tool_calls
    }

    pub fn set_last_reacted(&mut self, seq: Option<u64>) {
        self.state.last_reacted = seq;
    }

    pub fn last_reacted(&self) -> Option<u64> {
        self.state.last_reacted
    }

    pub fn status(&self) -> super::agent_state::SessionStatus {
        self.state.status.clone()
    }

    pub fn on_done(&self) -> Option<&CompletionDelivery> {
        self.state.on_done.as_ref()
    }

    pub fn artifacts(&self) -> &[Artifact] {
        &self.state.artifacts
    }

    pub fn messages(&self) -> &Vec<Message> {
        &self.state.messages
    }

    pub fn llm_call(&self, call_id: &str) -> Option<&super::agent_state::LlmCallState> {
        self.state.llm_calls.get(call_id)
    }

    pub fn cloned_state(&self) -> AgentState {
        self.state.clone()
    }

    pub fn session_id(&self) -> Uuid {
        self.state.session_id
    }

    pub fn set_token_usage_total(&mut self, total: u64) {
        self.state.token_usage.total_tokens = total;
    }

    pub fn llm_calls(
        &self,
    ) -> &std::collections::HashMap<String, super::agent_state::LlmCallState> {
        &self.state.llm_calls
    }
}

impl Reducer for AgentSession {
    type Event = EventPayload;
    type Derived = DerivedState;

    fn aggregate_type() -> &'static str {
        "session"
    }

    fn apply(&mut self, event: &Self::Event) {
        self.state.apply_core(event);
    }

    fn wake_at(&self) -> Option<chrono::DateTime<Utc>> {
        self.state.wake_at()
    }

    fn status(&self) -> AggregateStatus {
        self.state.aggregate_status()
    }

    fn label(&self) -> Option<String> {
        self.label()
    }
}

impl Aggregate<AgentSession> {
    pub fn process_command(
        &mut self,
        cmd: SessionCommand,
        tenant_id: &str,
    ) -> Result<(Vec<DomainEvent<AgentSession>>, Aggregate<AgentSession>), SessionError> {
        let payloads = self.state.handle(cmd.payload)?;
        if payloads.is_empty() {
            return Ok((vec![], self.clone()));
        }

        let base_seq = self.stream_version + 1;

        // Apply each payload to the state
        for (i, payload) in payloads.iter().enumerate() {
            self.apply(payload, base_seq + i as u64, cmd.occurred_at);
        }

        // Compute derived state after all events applied
        let derived = self.state.derived_state();

        // Build domain events
        let events: Vec<DomainEvent<AgentSession>> = payloads
            .into_iter()
            .enumerate()
            .map(|(i, payload)| DomainEvent {
                id: Uuid::new_v4(),
                tenant_id: tenant_id.to_string(),
                aggregate_id: self.state.session_id(),
                sequence: base_seq + i as u64,
                span: cmd.span.clone(),
                occurred_at: cmd.occurred_at,
                payload,
                derived: Some(derived.clone()),
            })
            .collect();

        Ok((events, self.clone()))
    }
}
