use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use ractor::OutputPort;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::domain::aggregate::AggregateStatus;
use crate::domain::span::SpanContext;

// ---------------------------------------------------------------------------
// Store-level Event — aggregate-agnostic raw envelope
// ---------------------------------------------------------------------------

/// The raw event envelope persisted and broadcast by the store.
///
/// Payload and derived are opaque `serde_json::Value`s. Domain code works
/// with the typed `DomainEvent<A>` and converts at the boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: Uuid,
    pub tenant_id: String,
    pub aggregate_type: String,
    pub aggregate_id: Uuid,
    pub event_type: String,
    pub sequence: u64,
    pub span: SpanContext,
    pub occurred_at: DateTime<Utc>,
    pub payload: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derived: Option<serde_json::Value>,
    /// Snapshot-level wake_at, stamped by the store during broadcast.
    /// Not persisted in the events table.
    #[serde(skip)]
    pub wake_at: Option<DateTime<Utc>>,
}

/// Events broadcast by the store after a successful append.
pub type EventBatch = Vec<Arc<Event>>;

// ---------------------------------------------------------------------------
// Version & errors
// ---------------------------------------------------------------------------

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
    #[error("stream not found")]
    StreamNotFound,
    #[error("tenant mismatch")]
    TenantMismatch,
    #[error("internal store error: {0}")]
    Internal(String),
}

// ---------------------------------------------------------------------------
// Load result
// ---------------------------------------------------------------------------

pub struct StreamLoad {
    pub snapshot: serde_json::Value,
    pub stream_version: u64,
}

// ---------------------------------------------------------------------------
// Aggregate query types
// ---------------------------------------------------------------------------

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
    pub status: Option<Vec<AggregateStatus>>,
    pub label: Option<String>,
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
    pub status: AggregateStatus,
    pub label: Option<String>,
    pub wake_at: Option<DateTime<Utc>>,
    pub stream_version: u64,
    pub first_event_at: Option<DateTime<Utc>>,
    pub last_event_at: Option<DateTime<Utc>>,
}

// ---------------------------------------------------------------------------
// EventStore trait — generic, aggregate-agnostic
// ---------------------------------------------------------------------------

#[async_trait]
pub trait EventStore: Send + Sync {
    /// Persist events + snapshot atomically.
    ///
    /// - `aggregate_id`: the stream/aggregate identity
    /// - `tenant_id`: for tenant isolation
    /// - `aggregate_type`: discriminator string (e.g. "session")
    /// - `events`: raw event envelopes
    /// - `snapshot`: serialized aggregate state
    /// - `expected_version`: the stream version before this append
    /// - `new_version`: the stream version after this append
    async fn append(
        &self,
        aggregate_id: Uuid,
        tenant_id: &str,
        aggregate_type: &str,
        events: Vec<Event>,
        snapshot: serde_json::Value,
        expected_version: u64,
        new_version: u64,
    ) -> Result<(), StoreError>;

    /// Load latest snapshot for a stream.
    async fn load(&self, aggregate_id: Uuid, tenant_id: &str) -> Result<StreamLoad, StoreError>;

    /// Returns the port that broadcasts newly appended events.
    fn events(&self) -> &OutputPort<EventBatch>;

    /// Query aggregates with filtering, sorting, and pagination.
    async fn list_aggregates(&self, filter: &AggregateFilter) -> Vec<AggregateSummary>;
}
