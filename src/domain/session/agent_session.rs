use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;
use uuid::Uuid;

use super::agent_state::{AgentState, StrategySlot};
use super::command_handler::{SessionCommand, SessionError};
use super::strategy::Strategy;
use crate::domain::aggregate::DomainEvent;
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

pub struct AgentSession {
    pub agent_state: AgentState,
    /// MCP tool name → server info (transient, populated by runtime).
    pub mcp_tools: HashMap<String, McpToolEntry>,
}

impl AgentSession {
    pub fn new(session_id: Uuid, strategy: Arc<dyn Strategy>) -> Self {
        let mut agent_state = AgentState::new(session_id);
        agent_state.strategy_state = strategy.default_state();
        agent_state.strategy = StrategySlot(Some(strategy));
        AgentSession {
            agent_state,
            mcp_tools: HashMap::new(),
        }
    }

    /// Build from a stored snapshot.
    pub fn from_snapshot(snapshot: AgentState, strategy: Arc<dyn Strategy>) -> Self {
        let mut agent_state = snapshot;
        agent_state.strategy = StrategySlot(Some(strategy));
        AgentSession {
            agent_state,
            mcp_tools: HashMap::new(),
        }
    }

    /// Take a snapshot of the current session state.
    pub fn snapshot(&self) -> AgentState {
        self.agent_state.clone()
    }

    pub fn apply(&mut self, payload: &EventPayload, sequence: u64) {
        self.agent_state.apply_core(payload, sequence);
    }

    /// Process a command: validate, build events, apply, and stamp derived state.
    ///
    /// Returns the fully-stamped domain events and a post-apply snapshot, ready for
    /// persistence. The caller converts to raw events before storing.
    pub fn process_command(
        &mut self,
        cmd: SessionCommand,
        tenant_id: &str,
    ) -> Result<(Vec<DomainEvent<AgentState>>, AgentState), SessionError> {
        let payloads = self.handle(cmd.payload)?;
        if payloads.is_empty() {
            return Ok((vec![], self.snapshot()));
        }

        let base_seq = self.agent_state.stream_version + 1;

        // Apply each payload to the state
        for (i, payload) in payloads.iter().enumerate() {
            self.apply(payload, base_seq + i as u64);
        }

        // Compute derived state after all events applied
        let derived = self.agent_state.derived_state();

        // Build domain events
        let events: Vec<DomainEvent<AgentState>> = payloads
            .into_iter()
            .enumerate()
            .map(|(i, payload)| DomainEvent {
                id: Uuid::new_v4(),
                tenant_id: tenant_id.to_string(),
                aggregate_id: self.agent_state.session_id,
                sequence: base_seq + i as u64,
                span: cmd.span.clone(),
                occurred_at: cmd.occurred_at,
                payload,
                derived: Some(derived.clone()),
            })
            .collect();

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

    /// Compute tool call metadata based on the tool name.
    pub(super) fn tool_call_meta(&self, name: &str, tool_call_id: &str) -> Option<ToolCallMeta> {
        // Check sub-agents
        if let Some(agent_name) = self
            .agent_state
            .agent
            .as_ref()
            .and_then(|a| a.sub_agents.iter().find(|s| s.as_str() == name))
        {
            return Some(ToolCallMeta::SubAgent {
                child_session_id: Uuid::new_v5(
                    &self.agent_state.session_id,
                    tool_call_id.as_bytes(),
                ),
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
}
