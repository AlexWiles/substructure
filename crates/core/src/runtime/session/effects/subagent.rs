use rust_decimal::Decimal;
use uuid::Uuid;

use super::{decision_queued, fail, mismatched, void_events, KindSpec, Outcome, SettleError};
use crate::protocol::{
    Content, DraftMessage, ErrorCode, ErrorInfo, RetryPolicy, Role, SpawnMode, SubagentMode, Usage,
};
use crate::runtime::session::command::SessionError;
use crate::runtime::session::decision::Trigger;
use crate::runtime::session::events::*;
use crate::runtime::session::state::{json_to_string, EffectKind, SessionState};
use crate::runtime::Caller;

pub const DETACHED_STARTED: &str = "working detached; the result arrives as a later message";

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
                let mut events = vec![EventPayload::SubagentStarted(SubagentStarted {
                    id: id.to_string(),
                })];
                if let Some(sa) = state
                    .subagent(id)
                    .filter(|sa| sa.mode == SpawnMode::Detached)
                {
                    events.push(decision_queued(Trigger::SubagentFinished {
                        id: id.to_string(),
                        ok: true,
                        session_id: sa.session_id.clone(),
                        agent_id: sa.agent_id.clone(),
                        result: Some(DETACHED_STARTED.to_string()),
                        error: None,
                    }));
                }
                events
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
            mode: sa.mode,
        })]
    }
}

fn withdraw_notice(state: &SessionState, session_id: &str) -> Vec<EventPayload> {
    let Some(prior) = state.queued_subagent_notice() else {
        return Vec::new();
    };
    if !prior.sessions.iter().any(|s| s == session_id) {
        return Vec::new();
    }
    let mut events = vec![EventPayload::DecisionDropped(DecisionDropped {
        id: prior.decision_id.to_string(),
    })];
    let (messages, sessions): (Vec<DraftMessage>, Vec<String>) = prior
        .messages
        .iter()
        .zip(prior.sessions)
        .filter(|(_, s)| *s != session_id)
        .map(|(m, s)| (m.clone(), s.clone()))
        .unzip();
    if !messages.is_empty() {
        events.push(decision_queued(Trigger::SubagentNotice {
            messages,
            sessions,
            turn_id: prior.turn_id.to_string(),
        }));
    }
    events
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
    pub mode: Option<SpawnMode>,
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
            Some(sa) if !spawn.agent_id.is_empty() && sa.agent_id != spawn.agent_id => {
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
    let requested = spawn.mode.unwrap_or_default();
    let mode = match config.subagent_mode(&spawn.agent_id) {
        None | Some(SubagentMode::Any) => requested,
        Some(_) if requested == SpawnMode::Wait => SpawnMode::Wait,
        Some(SubagentMode::Blocking) => SpawnMode::Blocking,
        Some(SubagentMode::Detached) => SpawnMode::Detached,
    };
    if mode == SpawnMode::Wait {
        let Some((_, sa)) = spawn
            .session_id
            .as_deref()
            .and_then(|named| state.subagent_detached(named))
        else {
            return answer(ErrorInfo::new(
                ErrorCode::Unroutable,
                "`wait` needs a `session` from an earlier detached call".to_string(),
            ));
        };
        let agent_id = sa.agent_id.clone();
        let Some(text) = sa.result.clone() else {
            return Ok(vec![EventPayload::SubagentRequested(SubagentRequested {
                id: spawn.tool_call_id,
                agent_id,
                session_id,
                message: None,
                retry: spawn.retry,
                mode: SpawnMode::Wait,
            })]);
        };
        let mut events = withdraw_notice(state, &session_id);
        events.push(decision_queued(match sa.is_error {
            false => Trigger::SubagentFinished {
                id: spawn.tool_call_id,
                ok: true,
                session_id,
                agent_id,
                result: Some(text),
                error: None,
            },
            true => Trigger::SubagentFinished {
                id: spawn.tool_call_id,
                ok: false,
                session_id,
                agent_id,
                result: None,
                error: Some(ErrorInfo::internal(text)),
            },
        }));
        return Ok(events);
    }
    if state
        .subagent_detached(&session_id)
        .is_some_and(|(_, sa)| sa.result.is_none())
    {
        return answer(ErrorInfo::new(
            ErrorCode::Unroutable,
            format!(
                "{} is still working detached; wait for its result, or start a fresh session",
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
        mode,
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
            turn_id: None,
            error: None,
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

pub(in crate::runtime::session) fn notice(
    session_id: &str,
    agent_id: &str,
    completed: &SubagentTurnCompleted,
) -> DraftMessage {
    let body = match &completed.error {
        Some(e) => e.message.clone(),
        None => json_to_string(&completed.data),
    };
    let attr = match completed.error.is_some() {
        true => " error=\"true\"",
        false => "",
    };
    DraftMessage {
        id: None,
        role: Role::User,
        content: Some(Content::Text(format!(
            "<subagent_result agent=\"{agent_id}\" session=\"{session_id}\"{attr}>\n{body}\n\
             </subagent_result>"
        ))),
        tool_calls: None,
        tool_call_id: None,
        name: None,
        reasoning: None,
    }
}
