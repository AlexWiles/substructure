use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;

use super::decision::{Action, LlmHandler, Trigger};
use super::effects::{self, decision_queued, void_events, Settle};
use super::events::*;
use super::schedule::{self, ScheduleStep};
use super::state::{
    new_call_id, EffectKind, EffectTracking, SessionState, SessionStatus, TurnPhase,
};
use crate::protocol::ErrorInfo;
use crate::protocol::{
    AgentConfig, ClientContext, ClientPayload, DraftMessage, EffectStatus, InterruptOrigin,
    LlmFormat, LlmRequest, RetryOverride, RetryPolicy, Role, SessionOwner, SpawnMode, Usage,
    WorkerRef, WorkerState,
};
use crate::runtime::retry::RetryTarget;
use crate::runtime::Caller;

pub use super::effects::{Outcome, SettleError};

#[derive(Debug, Clone)]
pub enum CommandPayload {
    CreateSession {
        agent_id: String,
        owner: SessionOwner,
        ancestry: Vec<String>,
        worker_retry: RetryPolicy,
        agent: Option<AgentConfig>,
        worker: Option<WorkerRef>,
    },
    SubmitClientPayload {
        payload: ClientPayload,
        turn: TurnTarget,
        queue: bool,
    },
    SendMessage {
        message: DraftMessage,
        #[allow(dead_code)]
        stream: bool,
        turn_id: Option<String>,
        parent_id: Option<String>,
    },
    RequestLlmCall {
        call_id: String,
        llm: String,
        request: LlmRequest,
        stream: bool,
        retry: RetryPolicy,
        handler: LlmHandler,
        format: Option<LlmFormat>,
    },
    SettleEffect {
        kind: EffectKind,
        id: String,
        attempt: Option<u32>,
        outcome: Outcome,
    },
    RequestToolCall {
        tool_call_id: String,
        name: String,
        arguments: String,
        retry: Option<RetryOverride>,
    },
    RequestSubagent {
        session_id: Option<String>,
        agent_id: String,
        tool_call_id: String,
        message: Option<DraftMessage>,
        retry: RetryPolicy,
        decision_id: String,
        mode: Option<SpawnMode>,
    },
    CompleteSubagentTurn {
        session_id: String,
        agent_id: String,
        turn_id: String,
        data: serde_json::Value,
        cost: Decimal,
        token_usage: Usage,
        error: Option<ErrorInfo>,
    },
    Interrupt {
        interrupt_id: String,
        reason: String,
        payload: serde_json::Value,
    },
    ResumeInterrupt {
        interrupt_id: String,
        payload: serde_json::Value,
    },
    SubmitWorkerDecision {
        decision_id: String,
        transcript: Vec<DraftMessage>,
        actions: Vec<Action>,
        state: Option<WorkerState>,
        agent: Option<AgentConfig>,
        channels: BTreeMap<String, serde_json::Value>,
    },
    CancelSession,
    FinishTurn {
        data: serde_json::Value,
    },
    CompleteTurn,
    Wake {
        now: DateTime<Utc>,
    },
    ReconcileDispatch,
}

impl CommandPayload {
    pub fn settle(
        kind: EffectKind,
        id: impl Into<String>,
        attempt: Option<u32>,
        outcome: impl Into<Outcome>,
    ) -> Self {
        CommandPayload::SettleEffect {
            kind,
            id: id.into(),
            attempt,
            outcome: outcome.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TurnTarget {
    Open(String),
    Continue(String),
    Detached,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum SessionError {
    #[error("session has not been created")]
    SessionNotCreated,
    #[error("session already exists")]
    SessionAlreadyCreated,
    #[error("session is interrupted")]
    SessionInterrupted,
    #[error("session start failed")]
    SessionStartFailed,
    #[error("turn already active: {turn_id}")]
    TurnAlreadyActive { turn_id: String },
    #[error("turn already completed: {turn_id}")]
    TurnAlreadyCompleted { turn_id: String },
    #[error("session has no active turn")]
    NoActiveTurn,
    #[error("client subject is required")]
    MissingSubject,
    #[error("session access denied")]
    SessionAccessDenied,
    #[error("effect not found")]
    EffectNotFound,
    #[error("effect is not pending")]
    EffectNotPending,
    #[error("effect attempt mismatch")]
    EffectAttemptMismatch,
    #[error("caller may not settle this effect")]
    EffectWrongHandler,
}

impl Action {
    fn into_command(self, decision_id: &str) -> Option<CommandPayload> {
        match self {
            Action::CallLlm {
                id,
                llm,
                request,
                stream,
                retry,
                handler,
                format,
            } => Some(CommandPayload::RequestLlmCall {
                call_id: id,
                llm,
                request,
                stream,
                retry,
                handler,
                format,
            }),
            Action::CallTool {
                id,
                name,
                arguments,
                retry,
            } => Some(CommandPayload::RequestToolCall {
                tool_call_id: id,
                name,
                arguments,
                retry,
            }),
            Action::SpawnSubagent {
                session_id,
                agent_id,
                tool_call_id,
                message,
                retry,
                mode,
            } => Some(CommandPayload::RequestSubagent {
                session_id,
                agent_id,
                tool_call_id,
                message,
                retry,
                decision_id: decision_id.to_string(),
                mode,
            }),
            Action::ToolResult {
                id,
                attempt,
                result,
            } => Some(CommandPayload::settle(
                EffectKind::ToolCall,
                id,
                attempt,
                Outcome::Tool { result },
            )),
            Action::LlmResult {
                id,
                attempt,
                response,
            } => Some(CommandPayload::settle(
                EffectKind::LlmCall,
                id,
                attempt,
                Outcome::Llm(Box::new(response)),
            )),
            Action::ToolError {
                id,
                attempt,
                error,
                retryable,
                ..
            } => Some(CommandPayload::settle(
                EffectKind::ToolCall,
                id,
                attempt,
                Outcome::from(SettleError::new(error, retryable)),
            )),
            Action::LlmError {
                id,
                attempt,
                error,
                retryable,
            } => Some(CommandPayload::settle(
                EffectKind::LlmCall,
                id,
                attempt,
                Outcome::from(SettleError::new(error, retryable)),
            )),
            Action::SendMessage { .. }
            | Action::Interrupt { .. }
            | Action::ResolveInterrupt { .. }
            | Action::SyncConnector { .. }
            | Action::Done { .. } => None,
        }
    }
}

struct Completion {
    index: usize,
    tool_call_id: String,
    name: String,
    result: String,
}

impl SessionState {
    fn stale_decision(&self, node: Option<&str>, trigger: &Trigger) -> bool {
        !self.at(node).anchor_on_path(self.trigger_anchor(trigger))
    }

    pub(in crate::runtime::session) fn void_effects(
        &self,
        stranded: impl Fn(EffectKind, &EffectTracking, Option<&str>) -> bool,
    ) -> Vec<EventPayload> {
        self.effects
            .values()
            .filter(|e| {
                matches!(
                    e.kind(),
                    EffectKind::ToolCall | EffectKind::LlmCall | EffectKind::Subagent
                )
            })
            .filter(|e| stranded(e.kind(), &e.tracking, e.anchor.as_deref()))
            .map(|e| {
                EventPayload::CallVoided(CallVoided {
                    kind: e.kind(),
                    id: e.id.clone(),
                })
            })
            .collect()
    }

    fn void_stranded_work(&self, node: Option<&str>) -> Vec<EventPayload> {
        let at = self.at(node);
        let mut events = self.void_effects(|_, tracking, anchor| {
            matches!(
                tracking.status(),
                EffectStatus::Queued | EffectStatus::Pending | EffectStatus::RetryScheduled
            ) && !at.anchor_on_path(anchor)
        });
        for e in self.effects_of(EffectKind::Decision) {
            let Some(wd) = e.decision() else { continue };
            if matches!(
                e.tracking.status(),
                EffectStatus::Queued | EffectStatus::RetryScheduled
            ) && self.stale_decision(node, &wd.trigger)
            {
                events.push(EventPayload::DecisionDropped(DecisionDropped {
                    id: e.id.clone(),
                }));
            }
        }
        events
    }

    fn finish_turn_events(
        &self,
        data: serde_json::Value,
        caller: &Caller,
    ) -> Result<Vec<EventPayload>, SessionError> {
        SessionState::ensure_internal(caller)?;
        match &self.phase {
            TurnPhase::Active { turn_id } => Ok(vec![decision_queued(Trigger::turn_finished(
                turn_id.clone(),
                data,
                self.turn_cost,
                self.turn_token_usage.clone(),
            ))]),
            TurnPhase::Idle | TurnPhase::Finalizing(_) => {
                Ok(vec![EventPayload::SessionDone(SessionDone {})])
            }
        }
    }

    fn worker_state_events(&self, state: Option<WorkerState>) -> Vec<EventPayload> {
        let Some(state) = state else {
            return Vec::new();
        };
        if state == self.at_head().resolve_state_for() {
            return Vec::new();
        }
        vec![EventPayload::WorkerStateUpdated(WorkerStateUpdated {
            state,
            anchor: self.head_id.clone(),
        })]
    }

    fn agent_config_events(&self, config: Option<AgentConfig>) -> Vec<EventPayload> {
        let Some(config) = config else {
            return Vec::new();
        };
        if Some(&config) == self.at_head().resolve_agent_for().as_ref() {
            return Vec::new();
        }
        let mut events = effects::connector::sync(self, &config);
        events.push(EventPayload::AgentConfigUpdated(AgentConfigUpdated {
            config,
            anchor: self.head_id.clone(),
        }));
        events
    }
}

pub struct Working {
    state: SessionState,
    seq: u64,
    now: DateTime<Utc>,
    plan_now: DateTime<Utc>,
    events: Vec<EventPayload>,
}

impl std::ops::Deref for Working {
    type Target = SessionState;
    fn deref(&self) -> &SessionState {
        &self.state
    }
}

impl Working {
    pub fn new(state: SessionState, seq: u64, now: DateTime<Utc>) -> Self {
        Self {
            state,
            seq,
            now,
            plan_now: now,
            events: Vec::new(),
        }
    }

    fn emit(&mut self, payload: EventPayload) {
        self.seq += 1;
        self.state.apply(
            &payload,
            &super::state::ApplyContext {
                occurred_at: self.now,
                sequence: self.seq,
            },
        );
        self.events.push(payload);
        #[cfg(debug_assertions)]
        self.check_invariants();
    }

    #[cfg(debug_assertions)]
    fn check_invariants(&self) {
        if let Err(violation) = self.state.check_invariants() {
            panic!("{violation}; events: {:?}", self.events);
        }
    }

    fn emit_all(&mut self, events: Vec<EventPayload>) {
        for event in events {
            self.emit(event);
        }
    }

    fn then(&mut self, step: impl FnOnce(&SessionState) -> Vec<EventPayload>) -> &mut Self {
        let events = step(&self.state);
        self.emit_all(events);
        self
    }

    fn try_then(
        &mut self,
        step: impl FnOnce(&SessionState) -> Result<Vec<EventPayload>, SessionError>,
    ) -> Result<&mut Self, SessionError> {
        let events = step(&self.state)?;
        self.emit_all(events);
        Ok(self)
    }

    pub fn into_events(self) -> Vec<EventPayload> {
        self.events
    }

    fn schedule(&mut self) {
        let mut swept = false;
        loop {
            let Some(step) = schedule::plan(&self.state, self.plan_now)
                .into_iter()
                .find(|step| !(swept && step.is_sweep()))
            else {
                return;
            };
            swept |= step.is_sweep();
            self.execute(step);
        }
    }

    fn execute(&mut self, step: ScheduleStep) {
        match step {
            ScheduleStep::RequestFetch { connection_id } => {
                let config = self.state.retry_config();
                self.emit(EventPayload::ConnectorSyncRequested(
                    ConnectorSyncRequested {
                        path: connection_id,
                        attempt: 0,
                        retry: RetryPolicy::resolve(
                            None,
                            config.as_ref(),
                            RetryTarget::ConnectorSync,
                        ),
                    },
                ));
            }
            ScheduleStep::Dispatch { kind, id } => self.dispatch(kind, id),
            ScheduleStep::VoidPhantom { kind, id } => {
                self.then(|_| void_events(kind, id));
            }
            ScheduleStep::TimeOut { kind, id } => self.time_out(kind, id),
            ScheduleStep::Retry { kind, id } => {
                self.then(|s| kind.spec().retry(s, &id));
            }
            ScheduleStep::RedispatchDecision { decision_id } => {
                self.emit(EventPayload::DecisionDispatched(DecisionDispatched {
                    id: decision_id,
                }));
            }
        }
    }

    fn dispatch(&mut self, kind: EffectKind, id: String) {
        let spec = kind.spec();
        self.then(|s| spec.dispatch(s, &id));
        if let Some(trigger) = spec.execute_trigger(&self.state, &id) {
            self.emit(decision_queued(trigger));
        }
    }

    fn time_out(&mut self, kind: EffectKind, id: String) {
        let spec = kind.spec();
        let now = self.plan_now;
        self.then(|s| {
            let total = s.tracking(kind, &id).is_some_and(|t| t.total_expired(now));
            spec.settle(s, &id, Outcome::Error(spec.timeout_error(total)))
        });
    }

    fn settle_effect(
        &mut self,
        kind: EffectKind,
        id: String,
        attempt: Option<u32>,
        outcome: Outcome,
        caller: &Caller,
    ) -> Result<(), SessionError> {
        let spec = kind.spec();
        self.try_then(|s| {
            spec.authorize(s, &id, caller)?;
            Ok(match spec.resolve(s, &id, attempt, caller)? {
                Settle::Live => spec.settle(s, &id, outcome),
                Settle::Drop => Vec::new(),
            })
        })
        .map(|_| ())
    }

    fn queue_decision(&mut self, trigger: Trigger) -> String {
        let decision_id = new_call_id();
        self.emit(EventPayload::DecisionQueued(DecisionQueued {
            id: decision_id.clone(),
            trigger,
        }));
        decision_id
    }

    fn raise_interrupt(
        &mut self,
        origin: InterruptOrigin,
        interrupt_id: String,
        reason: String,
        payload: serde_json::Value,
    ) {
        if self.head_parked() {
            return;
        }
        let anchor = self.head_id.clone();
        self.emit(EventPayload::SessionInterrupted(SessionInterrupted {
            interrupt_id,
            origin,
            reason,
            payload,
            anchor: anchor.clone(),
        }));
        self.void_llm_calls_for_interrupt(anchor);
    }

    fn deferred_turn(
        &self,
        target: &TurnTarget,
        queue: bool,
        payload: &ClientPayload,
    ) -> Result<Option<String>, SessionError> {
        let deferrable = queue
            && matches!(
                payload,
                ClientPayload::Message(_) | ClientPayload::Append(_)
            );
        let TurnTarget::Open(turn_id) = target else {
            return Ok(None);
        };
        if !deferrable || self.phase.turn_id().is_none() {
            return Ok(None);
        }
        self.turn_taken(turn_id)?;
        Ok(Some(turn_id.clone()))
    }

    fn turn_taken(&self, turn_id: &str) -> Result<(), SessionError> {
        if self.completed_turn_ids.iter().any(|t| t == turn_id) {
            return Err(SessionError::TurnAlreadyCompleted {
                turn_id: turn_id.to_string(),
            });
        }
        if self.phase.turn_id() == Some(turn_id) || self.queued_turn_ids().any(|t| t == turn_id) {
            return Err(SessionError::TurnAlreadyActive {
                turn_id: turn_id.to_string(),
            });
        }
        Ok(())
    }

    fn queued_turn_ids(&self) -> impl Iterator<Item = &str> {
        self.queued_decisions()
            .into_iter()
            .filter_map(|e| e.decision()?.trigger.deferred_turn_id())
    }

    fn begin_turn(&mut self, target: TurnTarget) -> Result<(), SessionError> {
        let turn_id = match target {
            TurnTarget::Detached => return Ok(()),
            TurnTarget::Continue(_) if matches!(self.phase, TurnPhase::Active { .. }) => {
                return Ok(())
            }
            TurnTarget::Continue(id) | TurnTarget::Open(id) => id,
        };
        self.turn_taken(&turn_id)?;
        match &self.phase {
            TurnPhase::Active { turn_id: active } => {
                return Err(SessionError::TurnAlreadyActive {
                    turn_id: active.clone(),
                })
            }
            TurnPhase::Finalizing(_) => {
                self.then(|s| s.finalize_run(None));
            }
            TurnPhase::Idle => {}
        }
        self.emit(EventPayload::TurnStarted(TurnStarted { turn_id }));
        Ok(())
    }

    fn void_llm_calls_for_interrupt(&mut self, node: Option<String>) {
        let at = self.at(node.as_deref());
        let ids: Vec<String> = self
            .effects_of(EffectKind::LlmCall)
            .filter(|e| {
                matches!(
                    e.tracking.status(),
                    EffectStatus::Queued | EffectStatus::Pending
                )
            })
            .filter(|e| at.anchor_on_path(e.anchor.as_deref()))
            .map(|e| e.id.clone())
            .collect();
        for id in ids {
            self.emit(EventPayload::CallVoided(CallVoided {
                kind: EffectKind::LlmCall,
                id,
            }));
        }
    }

    fn apply_actions(&mut self, actions: Vec<Action>, decision_id: &str) {
        let system = Caller::System {
            tenant_id: self
                .owner
                .as_ref()
                .map(|o| o.tenant_id.clone())
                .unwrap_or_default(),
        };
        for action in actions {
            let _ = match action {
                Action::SendMessage {
                    session_id,
                    message,
                } => {
                    self.emit(EventPayload::SessionMessageRequested(
                        SessionMessageRequested {
                            target_session_id: session_id,
                            message,
                        },
                    ));
                    Ok(())
                }
                Action::Interrupt {
                    interrupt_id,
                    reason,
                    payload,
                } => {
                    self.raise_interrupt(InterruptOrigin::Frontend, interrupt_id, reason, payload);
                    Ok(())
                }
                Action::ResolveInterrupt {
                    interrupt_id,
                    payload,
                } => {
                    let allowed = self.open_interrupt(&interrupt_id).is_none_or(|open| {
                        InterruptOrigin::Machine.privilege() >= open.origin.privilege()
                    });
                    if !allowed {
                        tracing::warn!(
                            %interrupt_id,
                            "interrupt.resolve refused: the interrupt outranks the worker"
                        );
                        continue;
                    }
                    self.run_active(
                        CommandPayload::ResumeInterrupt {
                            interrupt_id,
                            payload,
                        },
                        &system,
                    )
                }
                Action::SyncConnector { path: id } => {
                    let named = self
                        .at_head()
                        .resolve_agent_for()
                        .is_some_and(|c| self.state.servers_for(&c).iter().any(|m| m.path == id));
                    let settled = self
                        .tracking(EffectKind::ConnectorSync, &id.to_string())
                        .is_some_and(|t| !t.is_in_flight());
                    if !named || !settled {
                        tracing::warn!(
                            connection = %id,
                            named,
                            settled,
                            "connector.sync refused"
                        );
                        continue;
                    }
                    let config = self.state.retry_config();
                    self.emit(EventPayload::ConnectorSyncRequested(
                        ConnectorSyncRequested {
                            path: id,
                            attempt: 0,
                            retry: RetryPolicy::resolve(
                                None,
                                config.as_ref(),
                                RetryTarget::ConnectorSync,
                            ),
                        },
                    ));
                    Ok(())
                }
                Action::Done { data } => {
                    let cmd = if self.phase.finalizing().is_some() {
                        CommandPayload::CompleteTurn
                    } else {
                        CommandPayload::FinishTurn { data }
                    };
                    self.run_active(cmd, &system)
                }
                other => match other.into_command(decision_id) {
                    Some(cmd) => self.run_active(cmd, &system),
                    None => Ok(()),
                },
            };
        }
    }

    pub fn run(&mut self, cmd: CommandPayload, caller: &Caller) -> Result<(), SessionError> {
        let cancelling = matches!(cmd, CommandPayload::CancelSession);

        match (self.agent_id.is_some(), cmd) {
            (
                false,
                CommandPayload::CreateSession {
                    agent_id,
                    owner,
                    ancestry,
                    worker_retry,
                    agent,
                    worker,
                },
            ) => {
                SessionState::ensure_tenant_matches(caller, &owner.tenant_id)?;
                if matches!(caller, Caller::Frontend { .. }) && !caller.owns(&owner) {
                    return Err(SessionError::SessionAccessDenied);
                }
                if agent.is_some() || worker.as_ref().is_some_and(|w| w.url.is_some()) {
                    SessionState::ensure_operator_or_system(caller)?;
                }
                self.emit(EventPayload::SessionCreated(Box::new(SessionCreated {
                    agent_id,
                    identity: owner,
                    ancestry,
                    worker_retry,
                    worker,
                })));
                self.then(|s| s.agent_config_events(agent));
                self.queue_decision(Trigger::SessionStart);
            }
            (true, CommandPayload::CreateSession { .. }) => {
                return Err(SessionError::SessionAlreadyCreated)
            }
            (false, _) => return Err(SessionError::SessionNotCreated),
            (true, cmd) => self.run_active(cmd, caller)?,
        }
        if !cancelling {
            self.schedule();
        }
        Ok(())
    }

    fn run_active(&mut self, cmd: CommandPayload, caller: &Caller) -> Result<(), SessionError> {
        if let Some(owner) = self.owner.as_ref() {
            SessionState::ensure_tenant_matches(caller, &owner.tenant_id)?;
        }
        match cmd {
            CommandPayload::CreateSession { .. } => Err(SessionError::SessionAlreadyCreated),

            CommandPayload::SubmitClientPayload {
                payload,
                turn,
                queue,
            } => {
                self.ensure_owns_session(caller)?;
                let deferred = self.deferred_turn(&turn, queue, &payload)?;
                if deferred.is_none() {
                    self.begin_turn(turn)?;
                }
                self.try_then(|s| s.client_payload_events(payload, caller, deferred))
                    .map(|_| ())
            }

            CommandPayload::SendMessage {
                message,
                stream: _,
                turn_id,
                parent_id: _,
            } => {
                SessionState::ensure_internal(caller)?;
                if self.head_parked() && message.role == Role::User {
                    return Err(SessionError::SessionInterrupted);
                }
                self.begin_turn(turn_id.map_or(TurnTarget::Detached, TurnTarget::Open))?;
                if message.role == Role::User {
                    self.queue_decision(Trigger::ClientMessage {
                        messages: vec![message],
                        client: ClientContext::default(),
                        turn_id: None,
                    });
                }
                Ok(())
            }

            CommandPayload::RequestLlmCall {
                call_id,
                llm,
                request,
                stream,
                retry,
                handler,
                format,
            } => self
                .try_then(|s| {
                    effects::llm::request(
                        s, call_id, llm, request, stream, retry, handler, format, caller,
                    )
                })
                .map(|_| ()),

            CommandPayload::SettleEffect {
                kind,
                id,
                attempt,
                outcome,
            } => self.settle_effect(kind, id, attempt, outcome, caller),

            CommandPayload::RequestToolCall {
                tool_call_id,
                name,
                arguments,
                retry,
            } => self
                .try_then(|s| {
                    effects::tool::request(s, tool_call_id, name, arguments, retry, caller)
                })
                .map(|_| ()),

            CommandPayload::RequestSubagent {
                session_id,
                agent_id,
                tool_call_id,
                message,
                retry,
                decision_id,
                mode,
            } => self
                .try_then(|s| {
                    effects::subagent::request(
                        s,
                        effects::subagent::Spawn {
                            tool_call_id,
                            agent_id,
                            session_id,
                            message,
                            retry,
                            decision_id,
                            mode,
                        },
                        caller,
                    )
                })
                .map(|_| ()),

            CommandPayload::CompleteSubagentTurn {
                session_id,
                turn_id,
                data,
                cost,
                token_usage,
                error,
                ..
            } => {
                SessionState::ensure_internal(caller)?;
                if let Some((id, sa)) = self.state.subagent_awaiting(&session_id) {
                    let (id, agent_id) = (id.to_string(), sa.agent_id.clone());
                    return match error {
                        Some(error) => self.settle_effect(
                            EffectKind::Subagent,
                            id,
                            None,
                            SettleError::new(error, false).into(),
                            caller,
                        ),
                        None => {
                            let events = effects::subagent::complete_turn(
                                id,
                                session_id,
                                agent_id,
                                data,
                                cost,
                                token_usage,
                            );
                            self.then(|_| events);
                            Ok(())
                        }
                    };
                }
                if self.state.detached_turns.contains(&turn_id) {
                    return Ok(());
                }
                let Some((id, sa)) = self.state.subagent_detached(&session_id) else {
                    return Ok(());
                };
                let (id, agent_id) = (id.to_string(), sa.agent_id.clone());
                let completed = SubagentTurnCompleted {
                    id,
                    cost,
                    token_usage,
                    data,
                    turn_id: Some(turn_id),
                    error,
                };
                let notice = effects::subagent::notice(&session_id, &agent_id, &completed);
                let mut events = vec![EventPayload::SubagentTurnCompleted(completed)];
                let (mut messages, mut sessions, notice_turn) =
                    match self.state.queued_subagent_notice() {
                        Some(prior) => {
                            let queued = prior.decision_id.to_string();
                            let held = (
                                prior.messages.to_vec(),
                                prior.sessions.to_vec(),
                                prior.turn_id.to_string(),
                            );
                            events.push(EventPayload::DecisionDropped(DecisionDropped {
                                id: queued,
                            }));
                            held
                        }
                        None => (Vec::new(), Vec::new(), new_call_id()),
                    };
                messages.push(notice);
                sessions.push(session_id);
                events.push(decision_queued(Trigger::SubagentNotice {
                    messages,
                    sessions,
                    turn_id: notice_turn,
                }));
                self.then(|_| events);
                Ok(())
            }

            CommandPayload::Interrupt {
                interrupt_id,
                reason,
                payload,
            } => {
                self.ensure_owns_session(caller)?;
                let origin = SessionState::caller_interrupt_origin(caller);
                self.raise_interrupt(origin, interrupt_id, reason, payload);
                Ok(())
            }

            CommandPayload::ResumeInterrupt {
                interrupt_id,
                payload,
            } => {
                self.ensure_owns_session(caller)?;
                let Some(open) = self.open_interrupt(&interrupt_id) else {
                    return Ok(());
                };
                if SessionState::caller_interrupt_origin(caller).privilege()
                    < open.origin.privilege()
                {
                    return Err(SessionError::SessionAccessDenied);
                }
                let parked_head = self.at_head().anchor_on_path(open.anchor.as_deref());
                self.emit(EventPayload::InterruptResumed(InterruptResumed {
                    interrupt_id: interrupt_id.clone(),
                    payload: payload.clone(),
                }));
                if parked_head {
                    self.queue_decision(Trigger::InterruptResumed {
                        interrupt_id,
                        payload,
                    });
                }
                Ok(())
            }

            CommandPayload::SubmitWorkerDecision {
                decision_id,
                transcript,
                actions,
                state,
                agent,
                channels,
            } => {
                SessionState::ensure_worker_or_system(caller)?;
                match self
                    .tracking(EffectKind::Decision, &decision_id)
                    .map(|t| t.status())
                {
                    Some(EffectStatus::Pending) => {}
                    _ => return Ok(()),
                }
                let decision = self.worker_decision(&decision_id);
                let is_action =
                    decision.is_some_and(|d| matches!(d.trigger, Trigger::ClientAction { .. }));
                let finishes_turn =
                    decision.is_some_and(|d| matches!(d.trigger, Trigger::TurnFinished { .. }));
                self.then(|_| {
                    vec![EventPayload::DecisionCompleted(DecisionCompleted {
                        id: decision_id.clone(),
                    })]
                })
                .then(|s| s.reconcile_transcript(transcript).0)
                .then(|s| s.void_stranded_work(s.head_id.as_deref()))
                .then(|s| s.worker_state_events(state))
                .then(|s| s.agent_config_events(agent));

                let starts_work = actions.iter().any(|a| {
                    matches!(
                        a,
                        Action::CallLlm { .. }
                            | Action::CallTool { .. }
                            | Action::SpawnSubagent { .. }
                    )
                });
                if is_action && starts_work && !matches!(self.phase, TurnPhase::Active { .. }) {
                    if self.phase.finalizing().is_some() {
                        self.then(|s| s.finalize_run(None));
                    }
                    self.emit(EventPayload::TurnStarted(TurnStarted {
                        turn_id: format!("action:{decision_id}"),
                    }));
                }

                if !channels.is_empty() {
                    self.emit(EventPayload::ChannelsUpdated(ChannelsUpdated {
                        decision_id: decision_id.clone(),
                        finishes_turn,
                        channels,
                    }));
                }

                self.apply_actions(actions, &decision_id);
                Ok(())
            }

            CommandPayload::CancelSession => {
                SessionState::ensure_operator_or_system(caller)?;
                if matches!(self.status, SessionStatus::Done) {
                    return Ok(());
                }
                self.then(|_| vec![EventPayload::SessionCancelled])
                    .then(|s| {
                        s.void_effects(|kind, tracking, _| {
                            tracking.is_open()
                                || (tracking.status() == EffectStatus::Running
                                    && kind.spec().voids_when_running())
                        })
                    });
                Ok(())
            }

            CommandPayload::FinishTurn { data } => self
                .try_then(|s| s.finish_turn_events(data, caller))
                .map(|_| ()),

            CommandPayload::CompleteTurn => self
                .try_then(|s| {
                    SessionState::ensure_internal(caller)?;
                    Ok(s.finalize_run(None))
                })
                .map(|_| ()),

            CommandPayload::Wake { now } => {
                SessionState::ensure_internal(caller)?;
                self.run_wake(now);
                Ok(())
            }

            CommandPayload::ReconcileDispatch => {
                SessionState::ensure_internal(caller)?;
                self.run_reconcile_dispatch()
            }
        }
    }

    fn run_reconcile_dispatch(&mut self) -> Result<(), SessionError> {
        const LOST: &str = "dispatch lost on engine restart";
        let lost = || Outcome::Error(SettleError::new(ErrorInfo::internal(LOST), true));
        for (id, terminal) in self.pending_decisions() {
            self.then(|s| EffectKind::Decision.spec().settle(s, &id, lost()));
            if terminal {
                return Ok(());
            }
        }
        let orphaned: Vec<String> = self
            .effects_of(EffectKind::LlmCall)
            .filter(|e| {
                e.llm().is_some_and(|c| c.handler == LlmHandler::Server)
                    && e.tracking.status() == EffectStatus::Pending
            })
            .map(|e| e.id.clone())
            .collect();
        for id in orphaned {
            self.then(|s| EffectKind::LlmCall.spec().settle(s, &id, lost()));
        }
        Ok(())
    }

    fn run_wake(&mut self, now: DateTime<Utc>) {
        self.plan_now = now;
    }
}

mod authorize;
mod transcript;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod props;

#[cfg(test)]
mod traces;
