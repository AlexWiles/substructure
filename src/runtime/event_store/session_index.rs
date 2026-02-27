use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::domain::session::SessionStatus;

// ---------------------------------------------------------------------------
// Session-specific query layer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct SessionFilter {
    pub tenant_id: Option<String>,
    pub statuses: Option<Vec<SessionStatus>>,
    pub needs_wake: Option<bool>,
    pub agent_name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SessionSummary {
    pub session_id: Uuid,
    pub tenant_id: String,
    pub client_id: String,
    pub status: SessionStatus,
    pub wake_at: Option<DateTime<Utc>>,
    pub agent_name: String,
    pub message_count: usize,
    pub token_usage: u64,
    pub stream_version: u64,
}

/// Session-specific query interface layered on top of the generic event store.
///
/// Implemented by store backends that can filter/list session aggregates.
pub trait SessionIndex: Send + Sync {
    fn list_sessions(&self, filter: &SessionFilter) -> Vec<SessionSummary>;
}
