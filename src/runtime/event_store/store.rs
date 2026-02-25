use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use ractor::OutputPort;
use uuid::Uuid;

use crate::domain::event::{Event, SessionAuth};
use crate::domain::session::{AgentState, SessionStatus};

/// Events broadcast by the store after a successful append.
pub type EventBatch = Vec<Arc<Event>>;

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
    #[error("internal store error: {0}")]
    Internal(String),
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
    /// Persist pre-built events + snapshot atomically. The expected
    /// (pre-append) version is derived from `snapshot.stream_version - events.len()`.
    async fn append(
        &self,
        session_id: Uuid,
        auth: &SessionAuth,
        events: Vec<Event>,
        snapshot: AgentState,
    ) -> Result<(), StoreError>;

    /// Load latest snapshot + any events after it.
    fn load(&self, session_id: Uuid, auth: &SessionAuth) -> Result<SessionLoad, StoreError>;

    /// List sessions matching the given filter. Empty filter returns all sessions.
    fn list_sessions(&self, filter: &SessionFilter) -> Vec<SessionSummary>;

    /// Returns the port that broadcasts newly appended events.
    fn events(&self) -> &OutputPort<EventBatch>;
}
