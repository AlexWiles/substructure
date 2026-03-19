use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::runtime::event_store::{EventStore, Snapshot, StoreError};
use crate::runtime::span::{SpanContext, TraceId};

use super::state::{AggregateState, ApplyContext, DomainEvent};

pub struct CommitContext {
    pub span: SpanContext,
    pub occurred_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Aggregate<R: AggregateState> {
    pub id: String,
    pub tenant_id: String,
    pub state: R,
    pub stream_version: u64,
    pub last_applied: Option<u64>,
    pub first_event_at: Option<DateTime<Utc>>,
    pub last_event_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub wake_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub label: Option<String>,
    /// The trace_id of the aggregate's origin span, set on first commit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<TraceId>,
}

impl<R: AggregateState> Aggregate<R> {
    pub fn new(id: String, tenant_id: String, state: R) -> Self {
        Aggregate {
            id,
            tenant_id,
            state,
            stream_version: 0,
            last_applied: None,
            first_event_at: None,
            last_event_at: None,
            wake_at: None,
            label: None,
            trace_id: None,
        }
    }

    pub async fn load_or_create(
        store: &dyn EventStore,
        id: String,
        tenant_id: String,
    ) -> Result<(Self, u64), StoreError> {
        match store.load(&tenant_id, &id).await {
            Ok(snapshot) => {
                if snapshot.tenant_id != tenant_id {
                    return Err(StoreError::Internal("tenant mismatch".into()));
                }
                let agg: Self = serde_json::from_value(snapshot.data)
                    .map_err(|e| StoreError::Internal(e.to_string()))?;
                Ok((agg, snapshot.stream_version))
            }
            Err(StoreError::StreamNotFound) => {
                Ok((Self::new(id.clone(), tenant_id, R::initial(id)), 0))
            }
            Err(e) => Err(e),
        }
    }

    /// Apply a domain event with dedup, version tracking, and timestamp updates.
    /// Returns `true` if the event was applied (not a duplicate).
    pub fn apply(&mut self, event: &DomainEvent<R>) -> bool {
        if self.last_applied.is_some_and(|seq| event.sequence <= seq) {
            return false;
        }
        let ctx = ApplyContext {
            occurred_at: event.occurred_at,
        };
        self.state.apply(&event.payload, &ctx);
        self.last_applied = Some(event.sequence);
        self.stream_version += 1;
        if self.first_event_at.is_none() {
            self.first_event_at = Some(event.occurred_at);
        }
        self.last_event_at = Some(event.occurred_at);
        self.wake_at = self.state.wake_at();
        self.label = self.state.label();
        true
    }

    /// Apply event payloads to the aggregate and wrap them as domain events.
    ///
    /// This is the mutation step — call it after the pure decision
    /// (`handle_command`) has produced event payloads.
    pub fn commit(&mut self, events: Vec<R::Event>, ctx: &CommitContext) -> Vec<DomainEvent<R>> {
        if events.is_empty() {
            return vec![];
        }

        if self.trace_id.is_none() {
            self.trace_id = Some(ctx.span.trace_id);
        }

        let apply_ctx = ApplyContext {
            occurred_at: ctx.occurred_at,
        };

        let mut domain_events = Vec::with_capacity(events.len());

        for payload in events {
            self.stream_version += 1;
            self.state.apply(&payload, &apply_ctx);
            self.last_applied = Some(self.stream_version);
            if self.first_event_at.is_none() {
                self.first_event_at = Some(ctx.occurred_at);
            }
            self.last_event_at = Some(ctx.occurred_at);
            self.wake_at = self.state.wake_at();
            self.label = self.state.label();

            domain_events.push(DomainEvent {
                id: Uuid::now_v7(),
                tenant_id: self.tenant_id.clone(),
                aggregate_id: self.id.clone(),
                sequence: self.stream_version,
                span: ctx.span.clone(),
                occurred_at: ctx.occurred_at,
                payload,
                derived: Some(self.state.derived_state()),
                metadata: HashMap::new(),
            });
        }

        domain_events
    }

    pub fn to_snapshot(&self) -> Result<Snapshot, serde_json::Error> {
        Ok(Snapshot {
            aggregate_id: self.id.clone(),
            tenant_id: self.tenant_id.clone(),
            aggregate_type: R::AGGREGATE_TYPE.to_string(),
            data: serde_json::to_value(self)?,
            stream_version: self.stream_version,
            wake_at: self.wake_at,
            first_event_at: self.first_event_at,
            last_event_at: self.last_event_at,
        })
    }
}
