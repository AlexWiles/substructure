use chrono::{DateTime, Utc};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::span::SpanContext;

// ---------------------------------------------------------------------------
// Reducer — pure event-application trait
// ---------------------------------------------------------------------------

/// Defines how events are applied to produce aggregate state.
///
/// Each reducer has its own typed event and derived-state enums,
/// ensuring type safety while keeping the store generic.
/// Metadata (version, timestamps) is managed by [`Aggregate<R>`].
pub trait Reducer: Sized + Serialize + DeserializeOwned + Clone + Send + Sync + 'static {
    /// Typed event payload enum for this reducer.
    type Event: Serialize + DeserializeOwned + Clone + Send + Sync + 'static;
    /// Typed derived state for query optimization (stamped on events).
    type Derived: Serialize + DeserializeOwned + Clone + Send + Sync + 'static;

    /// Discriminator string stored alongside events and snapshots (e.g. "session").
    fn aggregate_type() -> &'static str;
    /// Apply a single event to this state.
    fn apply(&mut self, event: &Self::Event);
}

// ---------------------------------------------------------------------------
// Aggregate<R> — reducer state + shared metadata
// ---------------------------------------------------------------------------

/// The full aggregate: reducer state plus shared metadata.
///
/// Manages stream versioning, dedup, and timestamps generically
/// for any [`Reducer`] implementation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Aggregate<R: Reducer> {
    pub state: R,
    pub stream_version: u64,
    pub last_applied: Option<u64>,
    pub first_event_at: Option<DateTime<Utc>>,
    pub last_event_at: Option<DateTime<Utc>>,
}

impl<R: Reducer> Aggregate<R> {
    pub fn new(state: R) -> Self {
        Aggregate {
            state,
            stream_version: 0,
            last_applied: None,
            first_event_at: None,
            last_event_at: None,
        }
    }

    /// Apply an event with dedup, version tracking, and timestamp updates.
    /// Returns `true` if the event was applied (not a duplicate).
    pub fn apply(&mut self, event: &R::Event, sequence: u64, occurred_at: DateTime<Utc>) -> bool {
        if self.last_applied.is_some_and(|seq| sequence <= seq) {
            return false;
        }
        self.state.apply(event);
        self.last_applied = Some(sequence);
        self.stream_version += 1;
        if self.first_event_at.is_none() {
            self.first_event_at = Some(occurred_at);
        }
        self.last_event_at = Some(occurred_at);
        true
    }
}

// ---------------------------------------------------------------------------
// DomainEvent<R> — typed event envelope
// ---------------------------------------------------------------------------

/// A typed domain event, parameterized by reducer.
///
/// Domain code works with `DomainEvent<R>` for type safety.
/// Converted to/from the store-level `event_store::Event` at the boundary.
#[derive(Debug, Clone)]
pub struct DomainEvent<R: Reducer> {
    pub id: Uuid,
    pub tenant_id: String,
    pub aggregate_id: Uuid,
    pub sequence: u64,
    pub span: SpanContext,
    pub occurred_at: DateTime<Utc>,
    pub payload: R::Event,
    pub derived: Option<R::Derived>,
}

impl<R: Reducer> DomainEvent<R> {
    /// Convert to a store-level raw event by serializing payload and derived to Values.
    pub fn into_raw(self) -> crate::runtime::event_store::Event {
        let payload_value =
            serde_json::to_value(&self.payload).expect("event payload serialization");

        // Extract event_type from the serde tag (e.g. {"type": "session.created", ...})
        let event_type = payload_value
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let derived_value = self
            .derived
            .as_ref()
            .map(|d| serde_json::to_value(d).expect("derived state serialization"));

        crate::runtime::event_store::Event {
            id: self.id,
            tenant_id: self.tenant_id,
            aggregate_type: R::aggregate_type().to_string(),
            aggregate_id: self.aggregate_id,
            event_type,
            sequence: self.sequence,
            span: self.span,
            occurred_at: self.occurred_at,
            payload: payload_value,
            derived: derived_value,
        }
    }

    /// Deserialize from a store-level raw event.
    pub fn from_raw(raw: &crate::runtime::event_store::Event) -> Result<Self, serde_json::Error> {
        let payload: R::Event = serde_json::from_value(raw.payload.clone())?;
        let derived: Option<R::Derived> = raw
            .derived
            .as_ref()
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()?;

        Ok(DomainEvent {
            id: raw.id,
            tenant_id: raw.tenant_id.clone(),
            aggregate_id: raw.aggregate_id,
            sequence: raw.sequence,
            span: raw.span.clone(),
            occurred_at: raw.occurred_at,
            payload,
            derived,
        })
    }
}
