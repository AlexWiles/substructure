use rust_decimal::Decimal;
use uuid::Uuid;

use super::{decision_queued, fail, mismatched, void_events, KindSpec, Outcome, SettleError};
use crate::protocol::{DraftMessage, ErrorCode, ErrorInfo, RetryPolicy, Usage};
use crate::runtime::session::command::SessionError;
use crate::runtime::session::decision::Trigger;
use crate::runtime::session::events::*;
use crate::runtime::session::state::{json_to_string, EffectKind, SessionState};
use crate::runtime::Caller;

pub struct SubagentSpec;

impl KindSpec for SubagentSpec {
    fn kind(&self) -> EffectKind {
        EffectKind::Subagent
    }

    fn voids_when_running(&self) -> bool {
        true
    }

    fn settle(&self, state: &SessionState, id: &str, outcome: Outcome) -> Vec<EventPayload> {
        match outcome {
            Outcome::SubagentStarted => {
                vec![EventPayload::SubagentStarted(SubagentStarted {
                    id: id.to_string(),
                })]
            }
            Outcome::Error(e) => fail(self, state, id, &e),
            other => mismatched(self.kind(), &other),
        }
    }

    fn errored(&self, _state: &SessionState, id: &str, e: &SettleError) -> Option<EventPayload> {
        Some(EventPayload::SubagentErrored(SubagentErrored {
            id: id.to_string(),
            error: e.error.clone(),
            retryable: e.retryable,
        }))
    }

    fn terminal(&self, state: &SessionState, id: &str, e: &SettleError) -> Vec<EventPayload> {
        let Some(sa) = state.subagent(id) else {
            return Vec::new();
        };
        vec![decision_queued(Trigger::SubagentFinished {
            id: id.to_string(),
            ok: false,
            session_id: sa.session_id.clone(),
            agent_id: sa.agent_id.clone(),
            result: None,
            error: Some(e.error.clone()),
        })]
    }

    fn dispatch(&self, state: &SessionState, id: &str) -> Vec<EventPayload> {
        if state.has_effect(EffectKind::Subagent, id) {
            vec![EventPayload::SubagentDispatched(SubagentDispatched {
                id: id.to_string(),
            })]
        } else {
            void_events(EffectKind::Subagent, id.to_string())
        }
    }

    fn retry(&self, state: &SessionState, id: &str) -> Vec<EventPayload> {
        let Some(effect) = state.effect(EffectKind::Subagent, id) else {
            return Vec::new();
        };
        let (t, Some(sa)) = (&effect.tracking, effect.subagent()) else {
            return Vec::new();
        };
        vec![EventPayload::SubagentRequested(SubagentRequested {
            id: id.to_string(),
            agent_id: sa.agent_id.clone(),
            session_id: sa.session_id.clone(),
            message: sa.message.clone(),
            retry: t.retry_policy.clone(),
        })]
    }
}

const SPAWN_NAMESPACE: Uuid = Uuid::from_u128(0x7b5e_1f2a_3c4d_5e6f_8091_a2b3_c4d5_e6f7);

fn child_session_id(decision_id: &str, tool_call_id: &str) -> String {
    Uuid::new_v5(
        &SPAWN_NAMESPACE,
        format!("{decision_id}:{tool_call_id}").as_bytes(),
    )
    .to_string()
}

pub(in crate::runtime::session) struct Spawn {
    pub tool_call_id: String,
    pub agent_id: String,
    pub session_id: Option<String>,
    pub message: Option<DraftMessage>,
    pub retry: RetryPolicy,
    pub decision_id: String,
}

pub(in crate::runtime::session) fn request(
    state: &SessionState,
    spawn: Spawn,
    caller: &Caller,
) -> Result<Vec<EventPayload>, SessionError> {
    SessionState::ensure_internal(caller)?;
    if state.has_effect(EffectKind::Subagent, &spawn.tool_call_id) {
        return Ok(Vec::new());
    }
    let session_id = spawn
        .session_id
        .clone()
        .unwrap_or_else(|| child_session_id(&spawn.decision_id, &spawn.tool_call_id));
    let answer = |error: ErrorInfo| {
        Ok(vec![decision_queued(Trigger::SubagentFinished {
            id: spawn.tool_call_id.clone(),
            ok: false,
            session_id: session_id.clone(),
            agent_id: spawn.agent_id.clone(),
            result: None,
            error: Some(error),
        })])
    };
    let config = state.at_head().resolve_agent_for().unwrap_or_default();
    if !config.may_spawn_subagent(state.depth()) {
        let limit = config.depth_limit();
        return answer(ErrorInfo::new(
            ErrorCode::BudgetExceeded,
            format!("subagent depth limit reached: max_subagent_depth is {limit}"),
        ));
    }
    if let Some(named) = spawn.session_id.as_deref() {
        match state.subagent_session(named) {
            None => {
                return answer(ErrorInfo::new(
                    ErrorCode::Unroutable,
                    format!("`{named}` names no session this agent delegated to"),
                ))
            }
            Some(sa) if sa.agent_id != spawn.agent_id => {
                return answer(ErrorInfo::new(
                    ErrorCode::Unroutable,
                    format!(
                        "`{named}` is a session of `{}`, not of `{}`",
                        sa.agent_id, spawn.agent_id
                    ),
                ))
            }
            Some(_) => {}
        }
    }
    if state.subagent_awaiting(&session_id).is_some() {
        return answer(ErrorInfo::new(
            ErrorCode::Unroutable,
            format!(
                "{} is already answering; wait for its result",
                spawn.agent_id
            ),
        ));
    }
    Ok(vec![EventPayload::SubagentRequested(SubagentRequested {
        id: spawn.tool_call_id,
        agent_id: spawn.agent_id,
        session_id,
        message: spawn.message,
        retry: spawn.retry,
    })])
}

pub(in crate::runtime::session) fn complete_turn(
    tool_call_id: String,
    session_id: String,
    agent_id: String,
    data: serde_json::Value,
    cost: Decimal,
    token_usage: Usage,
) -> Vec<EventPayload> {
    let result = json_to_string(&data);
    vec![
        EventPayload::SubagentTurnCompleted(SubagentTurnCompleted {
            id: tool_call_id.clone(),
            cost,
            token_usage,
            data,
        }),
        decision_queued(Trigger::SubagentFinished {
            id: tool_call_id,
            ok: true,
            session_id,
            agent_id,
            result: Some(result),
            error: None,
        }),
    ]
}
