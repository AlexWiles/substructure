use serde_json::Value;
use uuid::Uuid;

use crate::connectors::Principal;
use crate::runtime::span::SpanContext;

/// Work the engine owes a connection, dispatched off the events that request
/// it. [`ConnectorTask::Answer`] reaches no network, and runs here so that one
/// seam settles every tool call.
#[derive(Debug, Clone)]
pub enum ConnectorTask {
    /// Fetch one connection's tool list.
    Sync {
        source_event_id: Uuid,
        session_id: String,
        tenant_id: String,
        connection_id: String,
        /// Resolved from the session owner at dispatch, never persisted.
        principal: Principal,
        attempt: u32,
        span: SpanContext,
    },
    /// Execute one tool call against the connection it was routed to.
    CallTool {
        source_event_id: Uuid,
        session_id: String,
        tenant_id: String,
        tool_call_id: String,
        attempt: u32,
        connection_id: String,
        principal: Principal,
        /// The name the connection knows the tool by, not the model's alias.
        remote_name: String,
        arguments: Value,
        span: SpanContext,
    },
    /// Answer one of the engine's own tools, from the session.
    Answer {
        source_event_id: Uuid,
        session_id: String,
        tenant_id: String,
        tool_call_id: String,
        attempt: u32,
        span: SpanContext,
    },
}

impl ConnectorTask {
    pub fn dedupe_key(&self) -> String {
        match self {
            // Keyed by attempt, so a retry is new work rather than a duplicate
            // of the attempt it replaces.
            ConnectorTask::Sync {
                session_id,
                connection_id,
                attempt,
                ..
            } => format!("connector:sync:{session_id}:{connection_id}:{attempt}"),
            ConnectorTask::CallTool {
                session_id,
                tool_call_id,
                attempt,
                ..
            } => format!("connector:call:{session_id}:{tool_call_id}:{attempt}"),
            ConnectorTask::Answer {
                session_id,
                tool_call_id,
                attempt,
                ..
            } => format!("connector:answer:{session_id}:{tool_call_id}:{attempt}"),
        }
    }

    pub fn session_id(&self) -> &str {
        match self {
            ConnectorTask::Sync { session_id, .. }
            | ConnectorTask::CallTool { session_id, .. }
            | ConnectorTask::Answer { session_id, .. } => session_id,
        }
    }
}
