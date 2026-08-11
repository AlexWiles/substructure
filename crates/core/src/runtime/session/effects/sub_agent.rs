//! A turn delegated to a child session.
//!
//! ```text
//! Queued ─dispatch→ Pending ─started─────────────→ Completed
//!    ↑                 │ ─error[retries left]────→ RetryScheduled ─due→ Queued
//!    │                 │ ─error[exhausted]───────→ Failed ⇒ queue sub_agent.finished(err)
//!    └──── requeue ────┘ ─void──────────────────→ Failed
//! ```
//!
//! The effect is named by the child session, so `Completed` means the child
//! started, not that it answered. Its turn result arrives later as its own
//! event and queues `sub_agent.finished(ok)`; a delegation that never starts
//! folds the error back as the delegation's result instead.

use rust_decimal::Decimal;

use super::{decision_queued, fail, mismatched, void_events, KindSpec, Outcome, SettleError};
use crate::protocol::{DraftMessage, RetryPolicy, Usage};
use crate::runtime::session::command::SessionError;
use crate::runtime::session::decision::Trigger;
use crate::runtime::session::events::*;
use crate::runtime::session::state::{json_to_string, EffectKind, SessionState};
use crate::runtime::Caller;

pub struct SubAgentSpec;

impl KindSpec for SubAgentSpec {
    fn kind(&self) -> EffectKind {
        EffectKind::SubAgent
    }

    fn settle(&self, state: &SessionState, id: &str, outcome: Outcome) -> Vec<EventPayload> {
        match outcome {
            Outcome::SubAgentStarted => {
                vec![EventPayload::SubAgentStarted(SubAgentStarted {
                    id: id.to_string(),
                })]
            }
            Outcome::Error(e) => fail(self, state, id, &e),
            other => mismatched(self.kind(), &other),
        }
    }

    fn errored(&self, _state: &SessionState, id: &str, e: &SettleError) -> Option<EventPayload> {
        Some(EventPayload::SubAgentErrored(SubAgentErrored {
            id: id.to_string(),
            error: e.error.clone(),
            retryable: e.retryable,
        }))
    }

    /// The error is the delegation's result: `sub_agent.finished` is keyed by
    /// the model tool call the delegation answers, not by the child session.
    fn terminal(&self, state: &SessionState, id: &str, e: &SettleError) -> Vec<EventPayload> {
        let Some(sa) = state.sub_agent(id) else {
            return Vec::new();
        };
        vec![decision_queued(Trigger::SubAgentFinished {
            id: sa.tool_call_id.clone(),
            ok: false,
            session_id: id.to_string(),
            agent_id: sa.agent_id.clone(),
            result: None,
            error: Some(e.error.clone()),
        })]
    }

    fn dispatch(&self, state: &SessionState, id: &str) -> Vec<EventPayload> {
        if state.has_effect(EffectKind::SubAgent, id) {
            vec![EventPayload::SubAgentDispatched(SubAgentDispatched {
                id: id.to_string(),
            })]
        } else {
            void_events(EffectKind::SubAgent, id.to_string())
        }
    }

    fn retry(&self, state: &SessionState, id: &str) -> Vec<EventPayload> {
        let Some(effect) = state.effect(EffectKind::SubAgent, id) else {
            return Vec::new();
        };
        let (t, Some(sa)) = (&effect.tracking, effect.sub_agent()) else {
            return Vec::new();
        };
        vec![EventPayload::SubAgentRequested(SubAgentRequested {
            id: id.to_string(),
            agent_id: sa.agent_id.clone(),
            tool_call_id: sa.tool_call_id.clone(),
            message: sa.message.clone(),
            retry: t.retry_policy.clone(),
        })]
    }
}

/// Delegate a turn to a child session. Idempotent by child session id.
pub(in crate::runtime::session) fn request(
    state: &SessionState,
    session_id: String,
    agent_id: String,
    tool_call_id: String,
    message: Option<DraftMessage>,
    retry: RetryPolicy,
    caller: &Caller,
) -> Result<Vec<EventPayload>, SessionError> {
    SessionState::ensure_internal(caller)?;
    if state.has_effect(EffectKind::SubAgent, &session_id) {
        return Ok(Vec::new());
    }
    Ok(vec![EventPayload::SubAgentRequested(SubAgentRequested {
        id: session_id,
        agent_id,
        tool_call_id,
        message,
        retry,
    })])
}

/// A child's turn result: record it and queue `sub_agent.finished` so the
/// delegation folds back into the parent's loop. Empty when the call is gone.
///
/// Not a settle: the delegation itself completed when the child started, and
/// this arrives against a `Completed` effect.
pub(in crate::runtime::session) fn complete_turn(
    state: &SessionState,
    session_id: String,
    data: serde_json::Value,
    cost: Decimal,
    token_usage: Usage,
    caller: &Caller,
) -> Result<Vec<EventPayload>, SessionError> {
    SessionState::ensure_internal(caller)?;
    let Some(sa) = state.sub_agent(&session_id) else {
        return Ok(Vec::new());
    };
    let tool_call_id = sa.tool_call_id.clone();
    let agent_id = sa.agent_id.clone();
    let result = json_to_string(&data);
    Ok(vec![
        EventPayload::SubAgentTurnCompleted(SubAgentTurnCompleted {
            id: session_id.clone(),
            cost,
            token_usage,
            data,
        }),
        decision_queued(Trigger::SubAgentFinished {
            id: tool_call_id,
            ok: true,
            session_id,
            agent_id,
            result: Some(result),
            error: None,
        }),
    ])
}
