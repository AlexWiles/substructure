use rust_decimal::Decimal;
use uuid::Uuid;

use crate::protocol::{DraftMessage, RetryPolicy, SessionOwner};
use crate::runtime::span::SpanContext;

#[derive(Debug, Clone)]
pub enum SubAgentTask {
    SpawnSubAgent {
        source_event_id: Uuid,
        parent_session_id: String,
        tenant_id: String,
        child_session_id: String,
        agent_id: String,
        owner: SessionOwner,
        ancestry: Vec<String>,
        retry: RetryPolicy,
        span: SpanContext,
    },
    SendSessionMessage {
        source_event_id: Uuid,
        tenant_id: String,
        target_session_id: String,
        message: DraftMessage,
        span: SpanContext,
    },
    CompleteSubAgentTurn {
        source_event_id: Uuid,
        parent_session_id: String,
        tenant_id: String,
        child_session_id: String,
        agent_id: String,
        turn_id: String,
        data: serde_json::Value,
        cost: Decimal,
        token_usage: std::collections::BTreeMap<String, u64>,
        span: SpanContext,
    },
    CancelSubAgent {
        source_event_id: Uuid,
        tenant_id: String,
        child_session_id: String,
        span: SpanContext,
    },
}

impl SubAgentTask {
    pub fn dedupe_key(&self) -> String {
        match self {
            SubAgentTask::SpawnSubAgent {
                source_event_id,
                parent_session_id,
                child_session_id,
                ..
            } => format!(
                "subagent:spawn:{parent_session_id}:{child_session_id}:{source_event_id}"
            ),
            SubAgentTask::SendSessionMessage {
                source_event_id,
                target_session_id,
                ..
            } => format!("subagent:send_message:{target_session_id}:{source_event_id}"),
            SubAgentTask::CompleteSubAgentTurn {
                source_event_id,
                parent_session_id,
                child_session_id,
                turn_id,
                ..
            } => format!(
                "subagent:complete_turn:{parent_session_id}:{child_session_id}:{turn_id}:{source_event_id}"
            ),
            SubAgentTask::CancelSubAgent {
                source_event_id,
                child_session_id,
                ..
            } => format!("subagent:cancel:{child_session_id}:{source_event_id}"),
        }
    }
}
