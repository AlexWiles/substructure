use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use ractor::OutputPort;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

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
    fn load(&self, aggregate_id: Uuid, tenant_id: &str) -> Result<StreamLoad, StoreError>;

    /// Returns the port that broadcasts newly appended events.
    fn events(&self) -> &OutputPort<EventBatch>;
}
