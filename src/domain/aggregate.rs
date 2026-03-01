use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::span::SpanContext;

// ---------------------------------------------------------------------------
// AggregateState — unified trait for event sourced aggregates
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AggregateStatus {
    /// Work in flight (LLM calls, tool calls, etc.).
    Active,
    /// Nothing in flight; may have a `wake_at` for scheduled retry.
    #[default]
    Idle,
    /// Terminal — aggregate has completed its lifecycle.
    Done,
}

#[async_trait]
pub trait AggregateState: Sized + Serialize + DeserializeOwned + Clone + Send + Sync + 'static {
    type Event: Serialize + DeserializeOwned + Clone + Send + Sync + 'static;
    type Command: Send + Sync + 'static;
    type Error: Send + Sync + 'static;
    type Context: Send + Sync + 'static;
    type Derived: Serialize + DeserializeOwned + Clone + Send + Sync + 'static;

    fn aggregate_type() -> &'static str;
    fn apply(&mut self, event: &Self::Event);
    fn handle_command(&self, cmd: Self::Command, ctx: &Self::Context) -> Result<Vec<Self::Event>, Self::Error>;
    async fn on_event(&self, event: &Self::Event, ctx: &Self::Context, span: &SpanContext) -> Option<Self::Command>;
    /// Compute a derived-state snapshot (stamped on events for query optimization).
    fn derived_state(&self) -> Self::Derived;

    fn wake_at(&self) -> Option<DateTime<Utc>> {
        None
    }
    fn status(&self) -> AggregateStatus {
        AggregateStatus::Idle
    }
    /// Human-readable label for this aggregate (e.g. agent name for sessions).
    fn label(&self) -> Option<String> {
        None
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Aggregate<R: AggregateState> {
    pub state: R,
    pub stream_version: u64,
    pub last_applied: Option<u64>,
    pub first_event_at: Option<DateTime<Utc>>,
    pub last_event_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub wake_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub status: AggregateStatus,
    #[serde(default)]
    pub label: Option<String>,
}

impl<R: AggregateState> Aggregate<R> {
    pub fn new(state: R) -> Self {
        Aggregate {
            state,
            stream_version: 0,
            last_applied: None,
            first_event_at: None,
            last_event_at: None,
            wake_at: None,
            status: AggregateStatus::default(),
            label: None,
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
        self.wake_at = self.state.wake_at();
        self.status = self.state.status();
        self.label = self.state.label();
        true
    }

    /// Apply event payloads to the aggregate and wrap them as domain events.
    ///
    /// This is the mutation step — call it after the pure decision
    /// (`handle_command`) has produced payloads.
    pub fn commit(
        &mut self,
        payloads: Vec<R::Event>,
        aggregate_id: Uuid,
        span: SpanContext,
        occurred_at: DateTime<Utc>,
        tenant_id: &str,
    ) -> Vec<DomainEvent<R>> {
        if payloads.is_empty() {
            return vec![];
        }

        let base_seq = self.stream_version + 1;

        // Apply each payload to the state
        for (i, payload) in payloads.iter().enumerate() {
            self.apply(payload, base_seq + i as u64, occurred_at);
        }

        // Compute derived state after all events applied
        let derived = self.state.derived_state();

        // Wrap payloads as domain events
        payloads
            .into_iter()
            .enumerate()
            .map(|(i, payload)| DomainEvent {
                id: Uuid::new_v4(),
                tenant_id: tenant_id.to_string(),
                aggregate_id,
                sequence: base_seq + i as u64,
                span: span.clone(),
                occurred_at,
                payload,
                derived: Some(derived.clone()),
            })
            .collect()
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
pub struct DomainEvent<R: AggregateState> {
    pub id: Uuid,
    pub tenant_id: String,
    pub aggregate_id: Uuid,
    pub sequence: u64,
    pub span: SpanContext,
    pub occurred_at: DateTime<Utc>,
    pub payload: R::Event,
    pub derived: Option<R::Derived>,
}

impl<R: AggregateState> DomainEvent<R> {
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
            wake_at: None,
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
