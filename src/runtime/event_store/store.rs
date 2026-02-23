use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use ractor::OutputPort;
use uuid::Uuid;

use crate::domain::event::{Event, EventPayload, SessionAuth, SpanContext};

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

#[async_trait]
pub trait EventStore: Send + Sync {
    async fn append(
        &self,
        session_id: Uuid,
        auth: &SessionAuth,
        expected_version: Version,
        span: SpanContext,
        occurred_at: DateTime<Utc>,
        payloads: Vec<EventPayload>,
    ) -> Result<Vec<Event>, StoreError>;

    fn load(&self, session_id: Uuid, auth: &SessionAuth) -> Result<Vec<Event>, StoreError>;

    fn version(&self, session_id: Uuid) -> Version;

    /// Read events from the global log starting at `offset`, up to `limit` events.
    fn read_from(&self, offset: u64, limit: usize) -> Vec<Arc<Event>>;

    /// Returns the notification port that fires when new events are appended.
    fn notify(&self) -> &OutputPort<()>;
}
