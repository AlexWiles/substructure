use async_trait::async_trait;
use uuid::Uuid;

use crate::domain::event::{Event, EventPayload, SpanContext};

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
}

#[async_trait]
pub trait EventStore: Send + Sync {
    async fn append(
        &self,
        session_id: Uuid,
        expected_version: Version,
        span: SpanContext,
        payloads: Vec<EventPayload>,
    ) -> Result<Vec<Event>, StoreError>;

    fn load(&self, session_id: Uuid) -> Result<Vec<Event>, StoreError>;

    fn version(&self, session_id: Uuid) -> Version;
}
