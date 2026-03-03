use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use ractor::OutputPort;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::runtime::aggregate::AggregateStatus;
use crate::runtime::span::SpanContext;

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
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
    /// Wall-clock start of the execute() call that produced this event.
    pub start_time: DateTime<Utc>,
    /// Wall-clock end of the execute() call that produced this event.
    pub end_time: DateTime<Utc>,
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
// Event query types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Deserialize)]
pub struct EventFilter {
    pub aggregate_id: Option<Uuid>,
    pub aggregate_type: Option<String>,
    pub tenant_id: Option<String>,
    pub trace_id: Option<String>,
    pub event_type: Option<String>,
    pub sequence_after: Option<u64>,
    pub occurred_after: Option<DateTime<Utc>>,
    pub occurred_before: Option<DateTime<Utc>>,
    pub limit: Option<usize>,
}

// ---------------------------------------------------------------------------
// Span reconstruction from persisted events
// ---------------------------------------------------------------------------

/// A reconstructed span summary built from persisted events.
///
/// Groups events that share a `span_id` (all events from one `execute()` call)
/// and extracts timing, naming, attributes, and error status.
#[derive(Debug, Clone, Serialize)]
pub struct SpanSummary {
    pub span: SpanContext,
    pub name: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub aggregate_type: String,
    pub aggregate_id: Uuid,
    pub error: Option<String>,
    pub attributes: Vec<(String, String)>,
    pub event_types: Vec<String>,
}

/// Reconstruct span summaries from persisted events.
///
/// Groups events by `span_id`, then builds one `SpanSummary` per group using
/// persisted `start_time`/`end_time` and metadata.
pub fn reconstruct_span_summaries(events: &[&Event]) -> Vec<SpanSummary> {
    use crate::runtime::span::SpanId;
    use std::collections::BTreeMap;

    let mut groups: BTreeMap<SpanId, Vec<&Event>> = BTreeMap::new();
    for event in events {
        groups.entry(event.span.span_id).or_default().push(event);
    }

    groups
        .into_values()
        .map(|group| {
            let first = group[0];

            let base_name = first
                .span
                .name
                .clone()
                .or_else(|| group.first().map(|e| e.event_type.clone()))
                .unwrap_or_else(|| "execute".to_string());

            let mut span_label: Option<String> = None;
            let mut span_error: Option<String> = None;
            let mut attributes = vec![
                ("aggregate.type".into(), first.aggregate_type.clone()),
                ("aggregate.id".into(), first.aggregate_id.to_string()),
            ];
            let mut event_types = Vec::new();

            for (i, event) in group.iter().enumerate() {
                event_types.push(event.event_type.clone());
                attributes.push((format!("event.{i}.type"), event.event_type.clone()));
                for (k, v) in &event.metadata {
                    match k.as_str() {
                        "span.label" if span_label.is_none() => span_label = Some(v.clone()),
                        "span.error" => span_error = Some(v.clone()),
                        _ => attributes.push((k.clone(), v.clone())),
                    }
                }
            }

            let name = match span_label {
                Some(label) => format!("{base_name}: {label}"),
                None => base_name,
            };

            SpanSummary {
                span: first.span.clone(),
                name,
                start_time: first.start_time,
                end_time: first.end_time,
                aggregate_type: first.aggregate_type.clone(),
                aggregate_id: first.aggregate_id,
                error: span_error,
                attributes,
                event_types,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// EventStore trait — generic, aggregate-agnostic
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
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

    /// Query events with filtering and pagination.
    async fn query_events(&self, filter: &EventFilter) -> Result<Vec<Event>, StoreError>;
}
