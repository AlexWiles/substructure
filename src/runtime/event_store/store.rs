use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use ractor::OutputPort;
use uuid::Uuid;

use crate::domain::event::{Event, SessionAuth};
use crate::domain::session::{AgentState, SessionStatus};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Version(pub u64);

impl Version {
    pub fn initial() -> Self {
        Version(0)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("version conflict: expected {}, actual {}", expected.0, actual.0)]
    VersionConflict { expected: Version, actual: Version },
    #[error("session not found")]
    SessionNotFound,
    #[error("tenant mismatch")]
    TenantMismatch,
}

pub struct SessionLoad {
    pub snapshot: AgentState,
}

// ---------------------------------------------------------------------------
// Session listing
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

#[async_trait]
pub trait EventStore: Send + Sync {
    /// Persist pre-built events + snapshot atomically. Version check for
    /// concurrency control.
    async fn append(
        &self,
        session_id: Uuid,
        auth: &SessionAuth,
        expected_version: Version,
        events: Vec<Event>,
        snapshot: AgentState,
    ) -> Result<(), StoreError>;

    /// Load latest snapshot + any events after it.
    fn load(&self, session_id: Uuid, auth: &SessionAuth) -> Result<SessionLoad, StoreError>;

    /// Read events from the global log starting at `offset`, up to `limit` events.
    fn read_from(&self, offset: u64, limit: usize) -> Vec<Arc<Event>>;

    /// List sessions matching the given filter. Empty filter returns all sessions.
    fn list_sessions(&self, filter: &SessionFilter) -> Vec<SessionSummary>;

    /// Returns the notification port that fires when new events are appended.
    fn notify(&self) -> &OutputPort<()>;
}
