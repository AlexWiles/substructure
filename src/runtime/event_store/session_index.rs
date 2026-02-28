use async_trait::async_trait;
use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::domain::session::SessionStatus;

// ---------------------------------------------------------------------------
// Session-specific query layer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionSort {
    #[default]
    LastEventDesc,
    FirstEventDesc,
    FirstEventAsc,
}

#[derive(Debug, Clone, Default)]
pub struct SessionFilter {
    pub tenant_id: Option<String>,
    pub statuses: Option<Vec<SessionStatus>>,
    pub needs_wake: Option<bool>,
    pub agent_name: Option<String>,
    pub sort: SessionSort,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionSummary {
    pub session_id: Uuid,
    pub tenant_id: String,
    pub status: SessionStatus,
    pub wake_at: Option<DateTime<Utc>>,
    pub agent_name: String,
    pub message_count: usize,
    pub token_usage: u64,
    pub stream_version: u64,
    pub first_event_at: Option<DateTime<Utc>>,
    pub last_event_at: Option<DateTime<Utc>>,
}

/// Session-specific query interface layered on top of the generic event store.
///
/// Implemented by store backends that can filter/list session aggregates.
#[async_trait]
pub trait SessionIndex: Send + Sync {
    async fn list_sessions(&self, filter: &SessionFilter) -> Vec<SessionSummary>;

    /// Return the earliest future `wake_at` across all sessions, if any.
    async fn next_wake_at(&self) -> Option<DateTime<Utc>>;
}
