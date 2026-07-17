use std::time::Duration;

use chrono::{DateTime, Utc};
use rand::Rng;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::runtime::caller::Caller;
use crate::runtime::event_store::{AppendInput, EventStore, GlobalPosition, StoreError};
use crate::runtime::span::{SpanContext, TraceId};

use super::command::{CommandPayload, SessionError};
use super::events::EventPayload;
use super::state::{ApplyContext, DerivedState, SessionState};

pub(crate) const SESSION_TYPE: &str = "session";

pub struct CommitContext {
    pub span: SpanContext,
    pub occurred_at: DateTime<Utc>,
}

/// A committed-but-unpersisted event: everything but the store-assigned
/// `global_position`. Also the persisted `data` shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewSessionEvent {
    pub id: Uuid,
    pub tenant_id: String,
    pub aggregate_id: String,
    #[serde(rename = "stream_version")]
    pub sequence: u64,
    pub span: SpanContext,
    pub occurred_at: DateTime<Utc>,
    pub payload: EventPayload,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derived: Option<DerivedState>,
    /// Wall-clock bounds of the execute() call that produced this event.
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
}

impl NewSessionEvent {
    pub fn into_event(self, global_position: GlobalPosition) -> SessionEvent {
        SessionEvent {
            global_position,
            id: self.id,
            tenant_id: self.tenant_id,
            aggregate_id: self.aggregate_id,
            sequence: self.sequence,
            span: self.span,
            occurred_at: self.occurred_at,
            payload: self.payload,
            derived: self.derived,
            start_time: self.start_time,
            end_time: self.end_time,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEvent {
    pub global_position: GlobalPosition,
    pub id: Uuid,
    pub tenant_id: String,
    pub aggregate_id: String,
    #[serde(rename = "stream_version")]
    pub sequence: u64,
    pub span: SpanContext,
    pub occurred_at: DateTime<Utc>,
    pub payload: EventPayload,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derived: Option<DerivedState>,
    /// Wall-clock bounds of the execute() call that produced this event.
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
}

impl SessionEvent {
    /// The payload's serde `"type"` tag.
    pub fn payload_type(&self) -> String {
        serde_json::to_value(&self.payload)
            .ok()
            .and_then(|v| v.get("type")?.as_str().map(str::to_owned))
            .unwrap_or_else(|| "unknown".to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionAggregate {
    pub id: String,
    pub tenant_id: String,
    pub state: SessionState,
    pub stream_version: u64,
    pub last_applied: Option<u64>,
    pub first_event_at: Option<DateTime<Utc>>,
    pub last_event_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub wake_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<TraceId>,
}

impl SessionAggregate {
    pub fn new(id: String, tenant_id: String, state: SessionState) -> Self {
        Self {
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
            Ok(session) => {
                if session.tenant_id != tenant_id {
                    return Err(StoreError::Internal("tenant mismatch".into()));
                }
                let expected_version = session.stream_version;
                Ok((session, expected_version))
            }
            Err(StoreError::StreamNotFound) => {
                Ok((Self::new(id.clone(), tenant_id, SessionState::new(id)), 0))
            }
            Err(error) => Err(error),
        }
    }

    pub fn commit(
        &mut self,
        events: Vec<EventPayload>,
        context: &CommitContext,
    ) -> Vec<NewSessionEvent> {
        if events.is_empty() {
            return vec![];
        }
        if self.trace_id.is_none() {
            self.trace_id = Some(context.span.trace_id);
        }

        let mut session_events = Vec::with_capacity(events.len());
        for payload in events {
            self.stream_version += 1;
            self.state.apply(
                &payload,
                &ApplyContext {
                    occurred_at: context.occurred_at,
                    sequence: self.stream_version,
                },
            );
            self.last_applied = Some(self.stream_version);
            if self.first_event_at.is_none() {
                self.first_event_at = Some(context.occurred_at);
            }
            self.last_event_at = Some(context.occurred_at);
            self.wake_at = self.state.wake_at();
            self.label = self.state.agent_id.clone();
            session_events.push(NewSessionEvent {
                id: Uuid::now_v7(),
                tenant_id: self.tenant_id.clone(),
                aggregate_id: self.id.clone(),
                sequence: self.stream_version,
                span: context.span.clone(),
                occurred_at: context.occurred_at,
                payload,
                derived: Some(self.state.derived_state()),
                start_time: context.occurred_at,
                end_time: context.occurred_at,
            });
        }
        session_events
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ExecuteError {
    #[error("command error: {0:?}")]
    Command(SessionError),
    #[error(transparent)]
    Store(#[from] StoreError),
}

pub struct ExecuteInput {
    pub aggregate_id: String,
    pub caller: Caller,
    pub command: CommandPayload,
    pub span: SpanContext,
}

#[derive(Debug, Clone)]
pub struct ConflictRetry {
    pub max_retries: u32,
    pub base_delay: Duration,
    pub max_delay: Duration,
    pub jitter: bool,
}

impl Default for ConflictRetry {
    fn default() -> Self {
        Self {
            max_retries: 5,
            base_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(500),
            jitter: true,
        }
    }
}

impl ConflictRetry {
    fn delay_for(&self, attempt: u32) -> Duration {
        let exponential = self.base_delay.saturating_mul(1 << attempt.min(16));
        let capped = exponential.min(self.max_delay);
        if !self.jitter {
            return capped;
        }
        let millis = capped.as_millis() as u64;
        if millis == 0 {
            return Duration::ZERO;
        }
        Duration::from_millis(rand::rng().random_range(0..=millis))
    }
}

pub async fn execute(
    store: &dyn EventStore,
    input: ExecuteInput,
    retry: &ConflictRetry,
) -> Result<(), ExecuteError> {
    let mut attempt = 0;
    loop {
        let start_time = Utc::now();
        let command = input.command.clone();
        let (mut session, expected_version) = SessionAggregate::load_or_create(
            store,
            input.aggregate_id.clone(),
            input.caller.tenant_id().to_string(),
        )
        .await?;
        let events = session
            .state
            .handle(command, &input.caller)
            .map_err(ExecuteError::Command)?;
        if events.is_empty() {
            return Ok(());
        }

        let mut events = session.commit(
            events,
            &CommitContext {
                span: input.span.clone(),
                occurred_at: Utc::now(),
            },
        );
        let end_time = Utc::now();
        for event in &mut events {
            event.start_time = start_time;
            event.end_time = end_time;
        }

        match store
            .append(AppendInput {
                events,
                snapshot: session,
                expected_version,
            })
            .await
        {
            Ok(()) => return Ok(()),
            Err(StoreError::VersionConflict { .. }) if attempt < retry.max_retries => {
                let delay = retry.delay_for(attempt);
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
                attempt += 1;
            }
            Err(error) => return Err(error.into()),
        }
    }
}
