use std::time::Duration;

use chrono::{DateTime, Utc};
use rand::Rng;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::runtime::caller::Caller;
use crate::runtime::event_store::{AppendInput, EventStore, Seq, StoreError};
use crate::runtime::span::SpanContext;

use super::command::{CommandPayload, SessionError, Working};
use super::events::EventPayload;
use super::state::{waiting_on_client, ApplyContext, EventMeta, SessionState};

pub struct CommitContext {
    pub span: SpanContext,
    pub occurred_at: DateTime<Utc>,
}

/// One committed event. `(session_id, seq)` locates it; nothing orders it
/// against another stream's events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEvent {
    pub id: Uuid,
    pub tenant_id: String,
    pub session_id: String,
    pub seq: u64,
    pub span: SpanContext,
    pub occurred_at: DateTime<Utc>,
    pub payload: EventPayload,
    pub meta: EventMeta,
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

    /// Whether the engine has nothing more to send until the caller acts. The
    /// turn is not always over: a client tool parks it, and the same turn
    /// resumes when the result arrives.
    pub fn ends_run(&self) -> bool {
        match &self.payload {
            EventPayload::TurnCompleted(_)
            | EventPayload::SessionInterrupted(_)
            | EventPayload::SessionCancelled => true,
            EventPayload::LlmCallErrored(e) => !e.retryable,
            _ => waiting_on_client(&self.meta.calls),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SessionAggregate {
    pub id: String,
    pub tenant_id: String,
    pub state: SessionState,
    pub seq: u64,
    pub first_event_at: Option<DateTime<Utc>>,
    pub last_event_at: Option<DateTime<Utc>>,
    pub wake_at: Option<DateTime<Utc>>,
}

impl SessionAggregate {
    pub fn new(id: String, tenant_id: String, state: SessionState) -> Self {
        Self {
            id,
            tenant_id,
            state,
            seq: 0,
            first_event_at: None,
            last_event_at: None,
            wake_at: None,
        }
    }

    pub async fn load_or_create(
        store: &dyn EventStore,
        id: String,
        tenant_id: String,
    ) -> Result<(Self, Seq), StoreError> {
        match store.load(&tenant_id, &id).await {
            Ok(session) => {
                if session.tenant_id != tenant_id {
                    return Err(StoreError::Internal("tenant mismatch".into()));
                }
                let expected_version = Seq(session.seq);
                Ok((session, expected_version))
            }
            Err(StoreError::StreamNotFound) => Ok((
                Self::new(id.clone(), tenant_id, SessionState::new(id)),
                Seq(0),
            )),
            Err(error) => Err(error),
        }
    }

    /// Handle a command against a working copy of the state: each emitted
    /// event is applied as it is emitted, so handler reads always see the
    /// events already produced. The scratch copy is discarded; `commit`
    /// applies the returned events to the canonical state.
    pub fn handle(
        &self,
        cmd: CommandPayload,
        caller: &Caller,
        now: DateTime<Utc>,
    ) -> Result<Vec<EventPayload>, SessionError> {
        let mut working = Working::new(self.state.clone(), self.seq, now);
        working.run(cmd, caller)?;
        Ok(working.into_events())
    }

    /// [`handle`](Self::handle) with the current clock; test convenience.
    #[cfg(test)]
    pub fn try_handle(
        &self,
        cmd: CommandPayload,
        caller: &Caller,
    ) -> Result<Vec<EventPayload>, SessionError> {
        self.handle(cmd, caller, Utc::now())
    }

    pub fn commit(
        &mut self,
        events: Vec<EventPayload>,
        context: &CommitContext,
    ) -> Vec<SessionEvent> {
        if events.is_empty() {
            return vec![];
        }

        let mut session_events = Vec::with_capacity(events.len());
        for payload in events {
            self.seq += 1;
            self.state.apply(
                &payload,
                &ApplyContext {
                    occurred_at: context.occurred_at,
                    sequence: self.seq,
                },
            );
            if self.first_event_at.is_none() {
                self.first_event_at = Some(context.occurred_at);
            }
            self.last_event_at = Some(context.occurred_at);
            let meta = self.state.event_meta(context.occurred_at);
            self.wake_at = meta.wake_at;
            session_events.push(SessionEvent {
                id: Uuid::now_v7(),
                tenant_id: self.tenant_id.clone(),
                session_id: self.id.clone(),
                seq: self.seq,
                span: context.span.clone(),
                occurred_at: context.occurred_at,
                payload,
                meta,
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
    pub session_id: String,
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
            max_retries: 32,
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

/// What a committed command left behind. Facts the caller would otherwise have
/// to re-read the session to learn — so the answer comes from the state the
/// command itself committed, not from a second look at the store.
#[derive(Debug, Clone, Default)]
pub struct ExecuteOutput {
    /// The turn the session is on, as of this command. `None` before the first
    /// one. Matches what a `turn_id` stamp on these events would say.
    pub turn_id: Option<String>,
}

pub async fn execute(
    store: &dyn EventStore,
    input: ExecuteInput,
    retry: &ConflictRetry,
) -> Result<ExecuteOutput, ExecuteError> {
    let mut attempt = 0;
    loop {
        let start_time = Utc::now();
        let command = input.command.clone();
        let (mut session, expected_version) = SessionAggregate::load_or_create(
            store,
            input.session_id.clone(),
            input.caller.tenant_id().to_string(),
        )
        .await?;

        // One clock per attempt: handling and commit see the same instant.
        let now = Utc::now();
        let events = session
            .handle(command, &input.caller, now)
            .map_err(ExecuteError::Command)?;

        if events.is_empty() {
            return Ok(ExecuteOutput {
                turn_id: session.state.turn_id().map(str::to_string),
            });
        }

        let mut events = session.commit(
            events,
            &CommitContext {
                span: input.span.clone(),
                occurred_at: now,
            },
        );
        let end_time = Utc::now();
        for event in &mut events {
            event.start_time = start_time;
            event.end_time = end_time;
        }

        // Read before the snapshot moves into the append: this is the state the
        // events being written leave the session in.
        let output = ExecuteOutput {
            turn_id: session.state.turn_id().map(str::to_string),
        };
        match store
            .append(AppendInput {
                events,
                snapshot: session,
                expected_version,
            })
            .await
        {
            Ok(()) => return Ok(output),
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

#[cfg(test)]
mod ends_run_tests {
    use super::*;
    use crate::protocol::{Effect, EffectKind, EffectStatus, Handler};
    use crate::runtime::session::events::DecisionQueued;
    use crate::runtime::session::state::{EventMeta, SessionStatus};

    fn call(id: &str, handler: Handler) -> Effect {
        Effect {
            id: id.into(),
            kind: EffectKind::ToolCall,
            status: EffectStatus::Pending,
            attempt: 0,
            deadline: None,
            anchor: None,
            name: None,
            arguments: None,
            handler: Some(handler),
            stream: None,
            agent_id: None,
            tool_call_id: None,
        }
    }

    fn event(payload: EventPayload, calls: Vec<Effect>) -> SessionEvent {
        let at = chrono::DateTime::UNIX_EPOCH;
        SessionEvent {
            id: Uuid::nil(),
            tenant_id: "t".into(),
            session_id: "s".into(),
            seq: 1,
            span: SpanContext::root(),
            occurred_at: at,
            payload,
            meta: EventMeta {
                status: SessionStatus::Idle,
                wake_at: None,
                owner: None,
                agent_id: None,
                ancestry: Vec::new(),
                turn_id: None,
                cost: Default::default(),
                sub_agent_cost: Default::default(),
                head_id: None,
                calls,
                decisions: Vec::new(),
            },
            start_time: at,
            end_time: at,
        }
    }

    fn queued(calls: Vec<Effect>) -> SessionEvent {
        event(
            EventPayload::DecisionQueued(DecisionQueued {
                id: "d1".into(),
                trigger: crate::runtime::session::decision::Trigger::SessionStart,
            }),
            calls,
        )
    }

    #[test]
    fn a_completed_turn_ends_the_run() {
        let done = event(
            EventPayload::TurnCompleted(crate::runtime::session::events::TurnCompleted {
                turn_id: "t1".into(),
                data: serde_json::Value::Null,
                turn_cost: Default::default(),
                turn_token_usage: Default::default(),
                error: None,
            }),
            vec![call("c", Handler::Client)],
        );
        assert!(done.ends_run());
    }

    #[test]
    fn an_outstanding_client_call_ends_the_run() {
        assert!(queued(vec![call("c", Handler::Client)]).ends_run());
    }

    #[test]
    fn a_worker_call_alongside_it_does_not() {
        assert!(!queued(vec![call("c", Handler::Client), call("w", Handler::Worker)]).ends_run());
    }

    #[test]
    fn a_sub_agent_names_no_handler_so_it_holds_the_run() {
        let mut sub = call("s", Handler::Client);
        sub.kind = EffectKind::SubAgent;
        sub.handler = None;
        assert!(!queued(vec![call("c", Handler::Client), sub]).ends_run());
    }

    #[test]
    fn nothing_outstanding_is_not_an_ending() {
        assert!(!queued(Vec::new()).ends_run());
    }

    #[test]
    fn a_retryable_llm_error_keeps_the_run() {
        let payload = |retryable: bool| {
            EventPayload::LlmCallErrored(crate::runtime::session::events::LlmCallErrored {
                id: "l".into(),
                attempt: 0,
                error: crate::protocol::ErrorInfo::new(
                    crate::protocol::ErrorCode::Internal,
                    "boom",
                ),
                retryable,
            })
        };
        assert!(!event(payload(true), Vec::new()).ends_run());
        assert!(event(payload(false), Vec::new()).ends_run());
    }
}
