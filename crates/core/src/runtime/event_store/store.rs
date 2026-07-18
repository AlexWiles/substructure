use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use std::sync::Arc;

use tokio::sync::broadcast;

use crate::runtime::session::{NewSessionEvent, SessionAggregate, SessionEvent};

/// Monotonic position in the store-wide event log, across every session.
/// The global cursor: use it to read or resume the whole log in commit order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GlobalPosition(pub u64);

/// Monotonic version within a single session's stream. The per-stream cursor:
/// meaningful only alongside a `session_id`, since each stream numbers from 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Seq(pub u64);

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("version conflict: expected {}, actual {}", expected.0, actual.0)]
    VersionConflict { expected: Seq, actual: Seq },
    #[error("stream not found")]
    StreamNotFound,
    #[error("internal store error: {0}")]
    Internal(String),
}

pub struct AppendInput {
    pub events: Vec<NewSessionEvent>,
    pub snapshot: SessionAggregate,
    pub expected_version: Seq,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct EventFilter {
    /// Global cursor: keep events whose `global_position` is greater. Reads the
    /// whole log in commit order; needs no `session_id`.
    pub after_global_position: Option<GlobalPosition>,
    pub session_id: Option<String>,
    pub tenant_id: Option<String>,
    /// Per-stream cursor: keep events whose `seq` is greater. Only unambiguous
    /// with `session_id` set, since versions restart per stream.
    pub after_seq: Option<Seq>,
    pub limit: Option<usize>,
}

#[async_trait]
pub trait EventStore: Send + Sync {
    /// Persist events and snapshot atomically, guarded by `expected_version`
    /// (`VersionConflict` on mismatch). Implementations must notify
    /// subscribers with the position-assigned events after a successful
    /// append.
    async fn append(&self, input: AppendInput) -> Result<(), StoreError>;

    /// Load the latest session, fully hydrated (tenant-scoped). History logs
    /// must come back in seq order with their original seq tags — `rewind`
    /// depends on both.
    async fn load(&self, tenant_id: &str, session_id: &str)
        -> Result<SessionAggregate, StoreError>;

    /// Query events with filtering and pagination, decoded from storage.
    ///
    /// Implementations must return events in ascending `position` order and
    /// fail with `StoreError` on an undecodable stored event.
    async fn query_events(&self, filter: &EventFilter) -> Result<Vec<SessionEvent>, StoreError>;

    /// Subscribe to new events as they are appended.
    fn subscribe(&self) -> broadcast::Receiver<Arc<Vec<SessionEvent>>>;
}
