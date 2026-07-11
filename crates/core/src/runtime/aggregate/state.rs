use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::de::DeserializeOwned;
use serde::Serialize;
use uuid::Uuid;

use crate::runtime::event_store::Event as StoreEvent;
use crate::runtime::event_store::{GlobalPosition, StreamVersion};
use crate::runtime::span::SpanContext;

pub struct ApplyContext {
    pub occurred_at: DateTime<Utc>,
    pub sequence: u64,
}

/// A typed domain event, parameterized by aggregate state.
///
/// Carries the domain payload alongside envelope fields (span, sequence,
/// timestamps, metadata). Domain code works with `DomainEvent<R>` for type
/// safety; the store converts to/from an opaque representation at the boundary.
#[derive(Debug, Clone)]
pub struct DomainEvent<R: AggregateState> {
    pub id: Uuid,
    pub tenant_id: String,
    pub aggregate_id: String,
    pub sequence: u64,
    pub span: SpanContext,
    pub occurred_at: DateTime<Utc>,
    pub payload: R::Event,
    pub derived: Option<R::Derived>,
    pub metadata: HashMap<String, String>,
}

impl<R: AggregateState> DomainEvent<R> {
    pub fn from_raw(raw: &StoreEvent) -> Result<Self, serde_json::Error> {
        let payload: R::Event = serde_json::from_value(raw.payload.clone())?;
        let derived: Option<R::Derived> = raw
            .derived
            .as_ref()
            .map(|d| serde_json::from_value(d.clone()))
            .transpose()?;
        Ok(DomainEvent {
            id: raw.id,
            tenant_id: raw.tenant_id.clone(),
            aggregate_id: raw.aggregate_id.clone(),
            sequence: raw.stream_version.0,
            span: raw.span.clone(),
            occurred_at: raw.occurred_at,
            payload,
            derived,
            metadata: raw.metadata.clone(),
        })
    }

    /// Convert to a store-level raw event by serializing payload and derived.
    ///
    /// `start_time`/`end_time` are the wall-clock bounds of the `execute()` call
    /// that produced this event, used for span reconstruction.
    pub fn into_raw(
        self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<StoreEvent, serde_json::Error> {
        let payload = serde_json::to_value(&self.payload)?;
        let derived = self
            .derived
            .as_ref()
            .map(serde_json::to_value)
            .transpose()?;

        Ok(StoreEvent {
            global_position: GlobalPosition(0),
            id: self.id,
            tenant_id: self.tenant_id,
            aggregate_type: R::AGGREGATE_TYPE.to_string(),
            aggregate_id: self.aggregate_id,
            stream_version: StreamVersion(self.sequence),
            span: self.span,
            occurred_at: self.occurred_at,
            payload,
            derived,
            metadata: self.metadata,
            start_time,
            end_time,
        })
    }
}

pub trait AggregateState:
    Sized + Serialize + DeserializeOwned + Clone + Send + Sync + 'static
{
    type Event: Serialize + DeserializeOwned + Clone + Send + Sync + 'static;
    type Command: Clone + Send + Sync + 'static;
    type Error: std::fmt::Debug + Send + Sync + 'static;
    type Derived: Serialize + DeserializeOwned + Clone + Send + Sync + 'static;

    const AGGREGATE_TYPE: &'static str;

    fn initial(id: String) -> Self;

    fn apply(&mut self, event: &Self::Event, ctx: &ApplyContext);

    fn handle_command(
        &self,
        cmd: Self::Command,
        caller: &super::caller::Caller,
    ) -> Result<Vec<Self::Event>, Self::Error>;

    fn derived_state(&self) -> Self::Derived;

    fn wake_at(&self) -> Option<DateTime<Utc>> {
        None
    }

    fn label(&self) -> Option<String> {
        None
    }
}
