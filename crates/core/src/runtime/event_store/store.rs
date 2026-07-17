use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use std::sync::Arc;

use tokio::sync::broadcast;

use crate::runtime::session::{NewSessionEvent, SessionAggregate, SessionEvent};

/// Monotonic position in the store-wide event log, across every aggregate.
/// The global cursor: use it to read or resume the whole log in commit order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GlobalPosition(pub u64);

/// Monotonic version within a single aggregate's stream. The per-stream cursor:
/// meaningful only alongside an `aggregate_id`, since each stream numbers from 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StreamVersion(pub u64);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Version(pub u64);

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("version conflict: expected {}, actual {}", expected.0, actual.0)]
    VersionConflict { expected: Version, actual: Version },
    #[error("stream not found")]
    StreamNotFound,
    #[error("internal store error: {0}")]
    Internal(String),
}

pub struct AppendInput {
    pub events: Vec<NewSessionEvent>,
    pub snapshot: SessionAggregate,
    pub expected_version: u64,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AggregateSort {
    #[default]
    LastEventDesc,
    FirstEventAsc,
    FirstEventDesc,
    WakeAtAsc,
}

#[derive(Debug, Clone, Default)]
pub struct AggregateFilter {
    pub aggregate_type: Option<String>,
    pub aggregate_ids: Option<Vec<String>>,
    pub tenant_id: Option<String>,
    /// Only include aggregates with `wake_at <= t`.
    pub wake_at_before: Option<DateTime<Utc>>,
    /// Only include aggregates that have a non-null `wake_at`.
    pub wake_at_not_null: bool,
    pub sort: AggregateSort,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AggregateSummary {
    pub aggregate_id: String,
    pub aggregate_type: String,
    pub tenant_id: String,
    pub wake_at: Option<DateTime<Utc>>,
    pub stream_version: StreamVersion,
    pub first_event_at: Option<DateTime<Utc>>,
    pub last_event_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct EventFilter {
    /// Global cursor: keep events whose `global_position` is greater. Reads the
    /// whole log in commit order; needs no `aggregate_id`.
    pub after_global_position: Option<GlobalPosition>,
    pub aggregate_id: Option<String>,
    pub tenant_id: Option<String>,
    pub trace_id: Option<String>,
    /// Per-stream cursor: keep events whose `stream_version` is greater. Only
    /// unambiguous with `aggregate_id` set, since versions restart per stream.
    pub after_stream_version: Option<StreamVersion>,
    pub occurred_after: Option<DateTime<Utc>>,
    pub occurred_before: Option<DateTime<Utc>>,
    pub limit: Option<usize>,
}

#[async_trait]
pub trait EventStore: Send + Sync {
    /// Persist events and snapshot atomically.
    /// Implementations must notify subscribers after a successful append.
    async fn append(&self, input: AppendInput) -> Result<(), StoreError>;

    /// Load the latest snapshot for a stream (tenant-scoped).
    async fn load(
        &self,
        tenant_id: &str,
        aggregate_id: &str,
    ) -> Result<SessionAggregate, StoreError>;

    /// Query aggregates with filtering, sorting, and pagination.
    async fn list_aggregates(
        &self,
        filter: &AggregateFilter,
    ) -> Result<Vec<AggregateSummary>, StoreError>;

    /// Query events with filtering and pagination, decoded from storage.
    ///
    /// Implementations must return events in ascending `position` order and
    /// fail with `StoreError` on an undecodable stored event.
    async fn query_events(&self, filter: &EventFilter) -> Result<Vec<SessionEvent>, StoreError>;

    /// Subscribe to new events as they are appended.
    fn subscribe(&self) -> broadcast::Receiver<Arc<Vec<SessionEvent>>>;
}
