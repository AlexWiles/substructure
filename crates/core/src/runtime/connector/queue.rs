use chrono::{DateTime, Utc};
use serde_json::Value;
use uuid::Uuid;

use crate::connectors::registry::ConnectionPath;
use crate::connectors::Requester;
use crate::protocol::RetryPolicy;
use crate::runtime::executor::{BoundedTask, TaskBound};
use crate::runtime::session::state::EffectKind;
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
        connection_id: ConnectionPath,
        /// Resolved from the session owner at dispatch, never persisted.
        requester: Requester,
        attempt: u32,
        retry: RetryPolicy,
        enqueued_at: DateTime<Utc>,
        span: SpanContext,
    },
    /// Execute one tool call against the connection it was routed to.
    CallTool {
        source_event_id: Uuid,
        session_id: String,
        tenant_id: String,
        tool_call_id: String,
        attempt: u32,
        connection_id: ConnectionPath,
        requester: Requester,
        /// The name the connection knows the tool by, not the model's alias.
        remote_name: String,
        arguments: Value,
        retry: RetryPolicy,
        enqueued_at: DateTime<Utc>,
        span: SpanContext,
    },
    /// Answer one of the engine's own tools, from the session.
    Answer {
        source_event_id: Uuid,
        session_id: String,
        tenant_id: String,
        tool_call_id: String,
        attempt: u32,
        retry: RetryPolicy,
        enqueued_at: DateTime<Utc>,
        span: SpanContext,
    },
}

impl BoundedTask for ConnectorTask {
    fn bound(&self) -> Option<TaskBound> {
        let (kind, id) = match self {
            ConnectorTask::Sync { connection_id, .. } => {
                (EffectKind::ConnectorSync, connection_id.to_string())
            }
            ConnectorTask::CallTool { tool_call_id, .. }
            | ConnectorTask::Answer { tool_call_id, .. } => {
                (EffectKind::ToolCall, tool_call_id.clone())
            }
        };
        let (tenant_id, session_id, attempt, retry, enqueued_at, span) = match self {
            ConnectorTask::Sync {
                tenant_id,
                session_id,
                attempt,
                retry,
                enqueued_at,
                span,
                ..
            }
            | ConnectorTask::CallTool {
                tenant_id,
                session_id,
                attempt,
                retry,
                enqueued_at,
                span,
                ..
            }
            | ConnectorTask::Answer {
                tenant_id,
                session_id,
                attempt,
                retry,
                enqueued_at,
                span,
                ..
            } => (tenant_id, session_id, attempt, retry, enqueued_at, span),
        };
        Some(TaskBound {
            tenant_id: tenant_id.clone(),
            session_id: session_id.clone(),
            kind,
            id,
            attempt: Some(*attempt),
            enqueued_at: *enqueued_at,
            queue_timeout: retry.queue_timeout(),
            run_timeout: retry.run_timeout(),
            span: span.clone(),
        })
    }
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
