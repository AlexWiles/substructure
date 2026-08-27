use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use uuid::Uuid;

use crate::protocol::ErrorInfo;
use crate::protocol::{DraftMessage, RetryPolicy, SessionOwner, Usage};
use crate::runtime::executor::{BoundedTask, TaskBound};
use crate::runtime::session::state::EffectKind;
use crate::runtime::span::SpanContext;

#[derive(Debug, Clone)]
pub enum SubagentTask {
    SpawnSubagent {
        source_event_id: Uuid,
        parent_session_id: String,
        tenant_id: String,
        tool_call_id: String,
        child_session_id: String,
        agent_id: String,
        owner: SessionOwner,
        ancestry: Vec<String>,
        message: Option<DraftMessage>,
        retry: RetryPolicy,
        enqueued_at: DateTime<Utc>,
        span: SpanContext,
    },
    SendSessionMessage {
        source_event_id: Uuid,
        tenant_id: String,
        target_session_id: String,
        message: DraftMessage,
        span: SpanContext,
    },
    CompleteSubagentTurn {
        source_event_id: Uuid,
        parent_session_id: String,
        tenant_id: String,
        child_session_id: String,
        agent_id: String,
        turn_id: String,
        data: serde_json::Value,
        cost: Decimal,
        token_usage: Usage,
        error: Option<ErrorInfo>,
        span: SpanContext,
    },
    CancelSubagent {
        source_event_id: Uuid,
        tenant_id: String,
        child_session_id: String,
        span: SpanContext,
    },
}

impl BoundedTask for SubagentTask {
    fn bound(&self) -> Option<TaskBound> {
        let SubagentTask::SpawnSubagent {
            parent_session_id,
            tenant_id,
            tool_call_id,
            retry,
            enqueued_at,
            span,
            ..
        } = self
        else {
            return None;
        };
        Some(TaskBound {
            tenant_id: tenant_id.clone(),
            session_id: parent_session_id.clone(),
            kind: EffectKind::Subagent,
            id: tool_call_id.clone(),
            attempt: None,
            enqueued_at: *enqueued_at,
            queue_timeout: retry.queue_timeout(),
            run_timeout: retry.run_timeout(),
            span: span.clone(),
        })
    }
}

impl SubagentTask {
    pub fn dedupe_key(&self) -> String {
        match self {
            SubagentTask::SpawnSubagent {
                source_event_id,
                parent_session_id,
                tool_call_id,
                ..
            } => format!("subagent:spawn:{parent_session_id}:{tool_call_id}:{source_event_id}"),
            SubagentTask::SendSessionMessage {
                source_event_id,
                target_session_id,
                ..
            } => format!("subagent:send_message:{target_session_id}:{source_event_id}"),
            SubagentTask::CompleteSubagentTurn {
                source_event_id,
                parent_session_id,
                child_session_id,
                turn_id,
                ..
            } => format!(
                "subagent:complete_turn:{parent_session_id}:{child_session_id}:{turn_id}:{source_event_id}"
            ),
            SubagentTask::CancelSubagent {
                source_event_id,
                child_session_id,
                ..
            } => format!("subagent:cancel:{child_session_id}:{source_event_id}"),
        }
    }
}
