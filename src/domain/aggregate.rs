use chrono::{DateTime, Utc};
use serde::de::DeserializeOwned;
use serde::Serialize;
use uuid::Uuid;

use super::span::SpanContext;

/// Defines an aggregate root for event sourcing.
///
/// Each aggregate has its own typed event and derived-state enums,
/// ensuring type safety while keeping the store generic.
pub trait Aggregate: Sized + Serialize + DeserializeOwned + Clone + Send + Sync + 'static {
    /// Typed event payload enum for this aggregate.
    type Event: Serialize + DeserializeOwned + Clone + Send + Sync + 'static;
    /// Typed derived state for query optimization (stamped on events).
    type Derived: Serialize + DeserializeOwned + Clone + Send + Sync + 'static;

    /// Discriminator string stored alongside events and snapshots (e.g. "session").
    fn aggregate_type() -> &'static str;
    /// Current stream version (for optimistic concurrency).
    fn stream_version(&self) -> u64;
    /// Apply a single event to this aggregate's state.
    fn apply(&mut self, event: &Self::Event, sequence: u64);
}

/// A typed domain event, parameterized by aggregate.
///
/// Domain code works with `DomainEvent<A>` for type safety.
/// Converted to/from the store-level `event_store::Event` at the boundary.
#[derive(Debug, Clone)]
pub struct DomainEvent<A: Aggregate> {
    pub id: Uuid,
    pub tenant_id: String,
    pub aggregate_id: Uuid,
    pub sequence: u64,
    pub span: SpanContext,
    pub occurred_at: DateTime<Utc>,
    pub payload: A::Event,
    pub derived: Option<A::Derived>,
}

impl<A: Aggregate> DomainEvent<A> {
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
            aggregate_type: A::aggregate_type().to_string(),
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
        let payload: A::Event = serde_json::from_value(raw.payload.clone())?;
        let derived: Option<A::Derived> = raw
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
