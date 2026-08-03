use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::runtime::event_store::EventTap;
use crate::runtime::session::{SessionAggregate, SessionEvent};

/// Monotonic version within a single session's stream. The only cursor there
/// is: meaningful only alongside a `session_id`, since each stream numbers
/// from 1. Streams are independent — the store defines no order between them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Seq(pub u64);

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("version conflict: expected {}, actual {}", expected.0, actual.0)]
    VersionConflict { expected: Seq, actual: Seq },
    #[error("stream not found")]
    StreamNotFound,
    #[error("store operation cancelled")]
    Cancelled,
    #[error("internal store error: {0}")]
    Internal(String),
}

pub struct AppendInput {
    pub events: Vec<SessionEvent>,
    pub snapshot: SessionAggregate,
    pub expected_version: Seq,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct EventFilter {
    pub session_id: Option<String>,
    pub tenant_id: Option<String>,
    /// Keep events whose `seq` is greater. Only unambiguous with `session_id`
    /// set, since versions restart per stream.
    pub after_seq: Option<Seq>,
    pub limit: Option<usize>,
}

#[async_trait]
pub trait EventStore: Send + Sync {
    /// Persist events and snapshot atomically, guarded by `expected_version`
    /// (`VersionConflict` on mismatch). Implementations must notify
    /// subscribers with the appended events after a successful append.
    async fn append(&self, input: AppendInput) -> Result<(), StoreError>;

    /// Load the latest session, fully hydrated (tenant-scoped). History logs
    /// must come back in seq order with their original seq tags — `rewind`
    /// depends on both.
    async fn load(&self, tenant_id: &str, session_id: &str)
        -> Result<SessionAggregate, StoreError>;

    /// Query events with filtering and pagination, decoded from storage.
    ///
    /// Implementations must return events grouped by stream and ascending by
    /// `seq` within one, and fail with `StoreError` on an undecodable stored
    /// event.
    async fn query_events(&self, filter: &EventFilter) -> Result<Vec<SessionEvent>, StoreError>;

    /// Tap the store's [`EventBus`](super::EventBus): a best-effort hint that
    /// events were appended. Batches may be missed; consumers needing every
    /// event must replay by cursor via [`query_events`](Self::query_events).
    fn subscribe(&self) -> EventTap;
}
