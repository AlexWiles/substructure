use std::collections::HashMap;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::runtime::span::SpanContext;

/// The raw event envelope persisted by the store.
///
/// Payload and derived are opaque `serde_json::Value`s. Domain code works
/// with the typed `DomainEvent<R>` and converts at the boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: Uuid,
    pub tenant_id: String,
    pub aggregate_type: String,
    pub aggregate_id: Uuid,
    pub sequence: u64,
    pub span: SpanContext,
    pub occurred_at: DateTime<Utc>,
    pub payload: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derived: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
    /// Wall-clock start of the execute() call that produced this event.
    pub start_time: DateTime<Utc>,
    /// Wall-clock end of the execute() call that produced this event.
    pub end_time: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Version(pub u64);

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("version conflict: expected {}, actual {}", expected.0, actual.0)]
    VersionConflict { expected: Version, actual: Version },
    #[error("stream not found")]
    StreamNotFound,
    #[error("tenant mismatch")]
    TenantMismatch,
    #[error("internal store error: {0}")]
    Internal(String),
}

pub struct Snapshot {
    pub aggregate_id: Uuid,
    pub tenant_id: String,
    pub aggregate_type: String,
    pub data: serde_json::Value,
    pub stream_version: u64,
    pub wake_at: Option<DateTime<Utc>>,
    pub first_event_at: Option<DateTime<Utc>>,
    pub last_event_at: Option<DateTime<Utc>>,
}

pub struct AppendInput {
    pub events: Vec<Event>,
    pub snapshot: Snapshot,
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
    pub aggregate_ids: Option<Vec<Uuid>>,
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
    pub aggregate_id: Uuid,
    pub aggregate_type: String,
    pub tenant_id: String,
    pub wake_at: Option<DateTime<Utc>>,
    pub stream_version: u64,
    pub first_event_at: Option<DateTime<Utc>>,
    pub last_event_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct EventFilter {
    pub aggregate_id: Option<Uuid>,
    pub aggregate_type: Option<String>,
    pub tenant_id: Option<String>,
    pub trace_id: Option<String>,
    pub sequence_after: Option<u64>,
    pub occurred_after: Option<DateTime<Utc>>,
    pub occurred_before: Option<DateTime<Utc>>,
    pub limit: Option<usize>,
}

#[async_trait]
pub trait EventStore: Send + Sync {
    /// Persist events and snapshot atomically.
    ///
    /// Returns `VersionConflict` if `expected_version` doesn't match the
    /// current stream version. Returns `TenantMismatch` if the stream
    /// belongs to a different tenant.
    async fn append(&self, input: AppendInput) -> Result<(), StoreError>;

    /// Load the latest snapshot for a stream.
    async fn load(&self, aggregate_id: Uuid, tenant_id: &str) -> Result<Snapshot, StoreError>;

    /// Query aggregates with filtering, sorting, and pagination.
    async fn list_aggregates(
        &self,
        filter: &AggregateFilter,
    ) -> Result<Vec<AggregateSummary>, StoreError>;

    /// Query events with filtering and pagination.
    async fn query_events(&self, filter: &EventFilter) -> Result<Vec<Event>, StoreError>;
}
