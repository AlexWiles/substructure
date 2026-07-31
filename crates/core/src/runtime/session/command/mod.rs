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
use crate::protocol::{
    AgentConfig, ClientContext, ClientPayload, DraftMessage, EffectStatus, InterruptOrigin,
    LlmFormat, LlmRequest, RetryPolicy, Role, SessionOwner, WorkerState,
};
use crate::runtime::Caller;

/// The settle vocabulary lives with the effect table; every settler names it
/// through the command layer, so it is re-exported here.
pub use super::effects::{Outcome, SettleError};

#[derive(Debug, Clone)]
pub enum CommandPayload {
    CreateSession {
        agent_id: String,
        owner: SessionOwner,
        ancestry: Vec<String>,
        worker_retry: RetryPolicy,
    },
    SubmitClientPayload {
        payload: ClientPayload,
        turn: TurnTarget,
        /// Hold the payload for the next turn instead of refusing it, when a
        /// turn is already running. Only message-shaped payloads defer.
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
    /// One effect's answer, whatever kind produced it: the success shapes are
    /// kind-specific, a failure is one shape for every kind.
    SettleEffect {
        kind: EffectKind,
        id: String,
        /// `None` = settle the current attempt; `Some` fences a stale executor.
        attempt: Option<u32>,
        outcome: Outcome,
    },
    RequestToolCall {
        tool_call_id: String,
        name: String,
        arguments: String,
        retry: RetryPolicy,
    },
    RequestSubAgent {
        session_id: String,
        agent_id: String,
        tool_call_id: String,
        retry: RetryPolicy,
    },
    CompleteSubAgentTurn {
        session_id: String,
        agent_id: String,
        turn_id: String,
        data: serde_json::Value,
        cost: Decimal,
        token_usage: BTreeMap<String, u64>,
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
        /// `None` = no opinion, keep the current state.
        state: Option<WorkerState>,
        /// `None` = no opinion, keep the current agent config.
        agent: Option<AgentConfig>,
    },
    CancelSession,
    /// The agent finished its turn: begin finalization by notifying the worker
    /// (`turn.finished`), deferring completion.
    FinishTurn {
        data: serde_json::Value,
    },
    /// The finalizer settled: complete the turn (emit `TurnCompleted` + `SessionDone`).
    CompleteTurn,
    Wake {
        now: DateTime<Utc>,
    },
    /// Boot-time recovery: fail in-flight work whose executor died with the
    /// process — pending worker decisions (reply rides the severed dispatch) and
    /// pending server-handled LLM calls (the awaiting future is gone) — so their
    /// retry policies re-issue now instead of at the deadline.
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

/// Which turn a submitted payload belongs to.
///
/// A session runs one turn at a time, so opening one is the case that can be
/// refused. The other two exist because not all client input is a request for
/// work: some continues a turn already running, and some belongs to no turn of
/// its own.
#[derive(Debug, Clone, PartialEq)]
pub enum TurnTarget {
    /// Open `id` as a new turn. Refused with
    /// [`TurnAlreadyActive`](SessionError::TurnAlreadyActive) when another turn
    /// is already running.
    Open(String),
    /// Continue the turn already running, opening `id` only if none is. For
    /// input that continues a turn rather than requesting one — an interrupt
    /// resume that also revises the view, which AG-UI requires to arrive as one
    /// input while an interrupt is open.
    Continue(String),
    /// Belongs to no turn of its own: recorded against whatever is running, or
    /// against nothing.
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
    /// The command an action maps to, for actions that are commands verbatim.
    /// `None` for the three that are not: `SendMessage` (a direct event),
    /// `Interrupt` (raised inline), and `Done` (conditional on finalization).
    fn into_command(self) -> Option<CommandPayload> {
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
            Action::SpawnSubAgent {
                session_id,
                agent_id,
                tool_call_id,
                retry,
            } => Some(CommandPayload::RequestSubAgent {
                session_id,
                agent_id,
                tool_call_id,
                retry,
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
                code,
                detail,
            } => Some(CommandPayload::settle(
                EffectKind::LlmCall,
                id,
                attempt,
                Outcome::from(SettleError::new(error, retryable).with_detail(code, detail)),
            )),
            Action::SendMessage { .. } | Action::Interrupt { .. } | Action::Done { .. } => None,
        }
    }
}

/// A tool message in a submitted view that answers a still-pending client call.
struct Completion {
    /// Position in the normalized view (used to line up with the reconcile plan).
    index: usize,
    tool_call_id: String,
    name: String,
    result: String,
}

impl SessionState {
    /// Whether the decision's effect is anchored off the path to `leaf`.
    fn stale_decision(&self, leaf: Option<&str>, trigger: &Trigger) -> bool {
        !self.anchor_on_path(leaf, self.trigger_anchor(trigger))
    }

    /// Void events for every effect matching `stranded`. Fetches and decisions
    /// are spared: a fetch belongs to its connection, and a decision drops
    /// rather than voids.
    pub(in crate::runtime::session) fn void_effects(
        &self,
        stranded: impl Fn(&EffectTracking, Option<&str>) -> bool,
    ) -> Vec<EventPayload> {
        self.effects
            .values()
            .filter(|e| {
                matches!(
                    e.kind(),
                    EffectKind::ToolCall | EffectKind::LlmCall | EffectKind::SubAgent
                )
            })
            .filter(|e| stranded(&e.tracking, e.anchor.as_deref()))
            .map(|e| {
                EventPayload::CallVoided(CallVoided {
                    kind: e.kind(),
                    id: e.id.clone(),
                })
            })
            .collect()
    }

    /// Void work and drop undelivered decisions anchored off the retained path.
    fn void_stranded_work(&self, leaf: Option<&str>) -> Vec<EventPayload> {
        let mut events = self.void_effects(|tracking, anchor| {
            matches!(
                tracking.status(),
                EffectStatus::Queued | EffectStatus::Pending | EffectStatus::RetryScheduled
            ) && !self.anchor_on_path(leaf, anchor)
        });
        for e in self.effects_of(EffectKind::Decision) {
            let Some(wd) = e.decision() else { continue };
            if matches!(
                e.tracking.status(),
                EffectStatus::Queued | EffectStatus::RetryScheduled
            ) && self.stale_decision(leaf, &wd.trigger)
            {
                events.push(EventPayload::DecisionDropped(DecisionDropped {
                    id: e.id.clone(),
                }));
            }
        }
        events
    }

    /// The agent's turn is done. For an active, not-yet-finished turn, begin
    /// finalization: queue `turn.finished` with the frozen output and defer
    /// completion (the queued trigger moves the phase to `Finalizing`, see the `DecisionQueued`
    /// apply). Otherwise settle immediately.
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
            // Nothing to finalize: no turn is running, or one already is and
            // pass 2 arrives as `CompleteTurn`.
            TurnPhase::Idle | TurnPhase::Finalizing(_) => {
                Ok(vec![EventPayload::SessionDone(SessionDone {})])
            }
        }
    }

    /// The worker's state write, anchored on the head. Empty when the worker
    /// has no opinion or the state is unchanged.
    fn worker_state_events(&self, state: Option<WorkerState>) -> Vec<EventPayload> {
        let Some(state) = state else {
            return Vec::new();
        };
        if state == self.resolve_state_for(self.head_id.as_deref()) {
            return Vec::new();
        }
        vec![EventPayload::WorkerStateUpdated(WorkerStateUpdated {
            state,
            anchor: self.head_id.clone(),
        })]
    }

    /// The worker's config write and the fetches it triggers; the run tail parks
    /// the next decision behind them. Empty when the worker has no opinion or the
    /// config is unchanged.
    fn agent_config_events(&self, config: Option<AgentConfig>) -> Vec<EventPayload> {
        let Some(config) = config else {
            return Vec::new();
        };
        if Some(&config) == self.resolve_agent_for(self.head_id.as_deref()).as_ref() {
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

/// A command's working context: the loaded state plus every event the command
/// has emitted so far, applied as emitted. Reads through `Deref` always see
/// the events already in `events` — there is no pre-batch/in-batch split.
///
/// The scratch state is discarded after the command; `commit` re-applies the
/// returned events to the canonical state (`apply` is deterministic, so the
/// two converge). An `Err` from a handler discards the whole context, which
/// is why handlers validate fully before their first `emit`.
pub struct Working {
    state: SessionState,
    seq: u64,
    /// The command's single clock: every emitted event and every deadline
    /// computed while handling it uses this instant.
    now: DateTime<Utc>,
    /// The instant the planner reads for deadlines and backoffs. Equal to `now`
    /// except under a `Wake`, which carries the scheduler's own instant — the
    /// one the due-work query was answered against. Kept apart from `now` so a
    /// wake cannot restamp the deadlines of what it dispatches.
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

    /// Scheduling invariants, checked after every emit. A violation asserts at
    /// the exact event that caused it, with the batch so far in hand.
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

    /// Run one step of a command: a pure function of the world the steps before
    /// it left. Chain them to read a command as the sequence of writes it makes.
    fn then(&mut self, step: impl FnOnce(&SessionState) -> Vec<EventPayload>) -> &mut Self {
        let events = step(&self.state);
        self.emit_all(events);
        self
    }

    /// [`then`](Self::then) for a step that validates: `Err` leaves the batch
    /// untouched and discards the whole command, `Ok(vec![])` writes nothing.
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

    // ── The schedule tail ───────────────────────────────────────────────
    //
    // The only dispatch path in the engine. Policy lives in `schedule.rs`,
    // per-kind behaviour in `effects/`; what is here is execution — turning a
    // step into the events it writes, without naming a kind.

    /// Ask the planner what the state licenses, take one step, and look again.
    /// Dispatch is a function of `(state, now)`, so the plan is re-derived
    /// rather than remembered: a step that changes what is ready is seen by the
    /// next pass, and nothing has to predict it.
    ///
    /// At most one sweep step per command. A session that has been idle past
    /// several deadlines emits one settle and reschedules, rather than a batch
    /// whose size is set by how long nobody looked.
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

    /// One planned step, as the events it writes.
    fn execute(&mut self, step: ScheduleStep) {
        match step {
            ScheduleStep::RequestFetch { connection_id } => {
                self.emit(EventPayload::ConnectorSyncRequested(
                    ConnectorSyncRequested {
                        id: connection_id,
                        attempt: 0,
                        retry: RetryPolicy::connector_default(),
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

    /// Start one queued effect: the kind's dispatch marker, then — for a
    /// worker-run one — its execute trigger. Two steps on purpose: the trigger
    /// is built from the post-dispatch effect, whose deadline is stamped and
    /// whose spec now carries the tools in force.
    fn dispatch(&mut self, kind: EffectKind, id: String) {
        let spec = kind.spec();
        self.then(|s| spec.dispatch(s, &id));
        if let Some(trigger) = spec.execute_trigger(&self.state, &id) {
            self.emit(decision_queued(trigger));
        }
    }

    /// Fail one effect past its deadline through the same path as a reported
    /// failure — including folding the failure back into the loop when
    /// terminal. No caller to authorize: the clock is the engine's own.
    fn time_out(&mut self, kind: EffectKind, id: String) {
        let spec = kind.spec();
        self.then(|s| spec.settle(s, &id, Outcome::Error(spec.timeout_error())));
    }

    /// The one settle seam, for every kind: authorize the caller, fence the
    /// answer against the effect it names, record it. Nothing here knows which
    /// kind it is holding.
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

    /// Queue a decision: the queued event carries the trigger, its single
    /// stored copy. Promotion is the run tail's job, never the queueing site's.
    fn queue_decision(&mut self, trigger: Trigger) -> String {
        let decision_id = new_call_id();
        self.emit(EventPayload::DecisionQueued(DecisionQueued {
            id: decision_id.clone(),
            trigger,
        }));
        decision_id
    }

    /// Raise an interrupt on the current head. Idempotent per branch: one
    /// open interrupt parks a path, so a second raise is a no-op.
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

    /// The turn a submit holds for later, when it asked to queue and the phase
    /// is busy with another one. `Ok(None)` means the ordinary path: open the
    /// turn now, or be refused for it.
    ///
    /// The submit is accepted and its decision parks until the phase is free,
    /// so the caller gets an answer to "did you take this" without waiting for
    /// the agent. Redelivery still refuses: a turn id already running, already
    /// queued, or already completed names work the session has, not new work.
    fn deferred_turn(
        &self,
        target: &TurnTarget,
        queue: bool,
        payload: &ClientPayload,
    ) -> Result<Option<String>, SessionError> {
        // Only the message shapes defer: both lower to the late-bound
        // `client.message` trigger, which materializes against the tree as it
        // stands when the turn starts. A full view settles pending client calls
        // at submit time and an action is not a request for a turn, so neither
        // can be frozen behind one — they meet `begin_turn` as before.
        let deferrable = queue
            && matches!(
                payload,
                ClientPayload::Message(_) | ClientPayload::Append(_)
            );
        let TurnTarget::Open(turn_id) = target else {
            return Ok(None);
        };
        // Nothing holds the phase: this turn opens now.
        if !deferrable || self.phase.turn_id().is_none() {
            return Ok(None);
        }
        // The id may be the one holding the phase, in which case this is a
        // redelivery rather than a turn to hold for later.
        self.turn_taken(turn_id)?;
        Ok(Some(turn_id.clone()))
    }

    /// Whether the session already knows this turn id — dedup, in one place for
    /// every path that opens a turn. A completed turn is refused as completed.
    /// One running, finalizing, or queued is refused as active, because each has
    /// its run or still owes one: resubmitting a finalizing turn is the
    /// idempotent case, and a queued turn will start on its own.
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

    /// The turns queued decisions will open when they dispatch.
    fn queued_turn_ids(&self) -> impl Iterator<Item = &str> {
        self.queued_decisions()
            .into_iter()
            .filter_map(|e| e.decision()?.trigger.deferred_turn_id())
    }

    /// Open the turn `target` names, if it names one: an id the session already
    /// holds is rejected, and a turn running when another is asked for is either
    /// continued or rejected depending on what the caller asked for.
    fn begin_turn(&mut self, target: TurnTarget) -> Result<(), SessionError> {
        let turn_id = match target {
            TurnTarget::Detached => return Ok(()),
            // Already working: this input joins the turn rather than opening a
            // second one. Only a running turn can be continued — a finalizing
            // one has produced its answer and falls through to be closed below.
            TurnTarget::Continue(_) if matches!(self.phase, TurnPhase::Active { .. }) => {
                return Ok(())
            }
            TurnTarget::Continue(id) | TurnTarget::Open(id) => id,
        };
        // Dedup first, so what follows is only about the phase: this id is new,
        // and the question left is whether something else holds the session.
        self.turn_taken(&turn_id)?;
        match &self.phase {
            // A running turn cannot be preempted: a session holds one turn at a
            // time, and the caller is told which one holds it. Redirecting a
            // working agent is an interrupt followed by a submit — two steps,
            // because stopping the work and replacing it are two decisions.
            TurnPhase::Active { turn_id: active } => {
                return Err(SessionError::TurnAlreadyActive {
                    turn_id: active.clone(),
                })
            }
            // A finalizing turn is not running: it has produced its answer and
            // only awaits the worker's finalizer. Its terminal is owed either
            // way and its queued `TurnEnd` would deliver it at the next plan, so
            // deliver it now rather than bounce a caller off a hook it cannot
            // see. Only a different turn reaches here — its own resubmit is the
            // idempotent case, refused above.
            TurnPhase::Finalizing(_) => {
                self.then(|s| s.finalize_run(None));
            }
            TurnPhase::Idle => {}
        }
        self.emit(EventPayload::TurnStarted(TurnStarted { turn_id }));
        Ok(())
    }

    /// Void queued and pending LLM calls on the path to `leaf`. Tools and
    /// sub-agents are spared: their async settles may still arrive and are
    /// worth keeping.
    fn void_llm_calls_for_interrupt(&mut self, leaf: Option<String>) {
        let ids: Vec<String> = self
            .effects_of(EffectKind::LlmCall)
            .filter(|e| {
                matches!(
                    e.tracking.status(),
                    EffectStatus::Queued | EffectStatus::Pending
                )
            })
            .filter(|e| self.anchor_on_path(leaf.as_deref(), e.anchor.as_deref()))
            .map(|e| e.id.clone())
            .collect();
        for id in ids {
            self.emit(EventPayload::CallVoided(CallVoided {
                kind: EffectKind::LlmCall,
                id,
            }));
        }
    }

    /// Run a worker's actions in order. Each is a sub-command against the live
    /// batch, so it sees everything the actions before it emitted.
    fn apply_actions(&mut self, actions: Vec<Action>) {
        let system = Caller::System {
            tenant_id: self
                .owner
                .as_ref()
                .map(|o| o.tenant_id.clone())
                .unwrap_or_default(),
        };
        for action in actions {
            // An action that validates away contributes nothing: every
            // dispatched arm validates before its first emit. A settle for work
            // this batch voided resolves to `Settle::Drop` (the void is
            // applied, and the caller here is the engine), so it drops without
            // a special case.
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
                // A `done` while finalizing completes the turn; otherwise it
                // ends the agent's turn and starts finalization.
                Action::Done { data } => {
                    let cmd = if self.phase.finalizing().is_some() {
                        CommandPayload::CompleteTurn
                    } else {
                        CommandPayload::FinishTurn { data }
                    };
                    self.run_active(cmd, &system)
                }
                other => match other.into_command() {
                    Some(cmd) => self.run_active(cmd, &system),
                    None => Ok(()),
                },
            };
        }
    }

    /// Handle a command: validate, emit facts and writes in order against
    /// live state, then schedule. Every emitted event is applied immediately,
    /// so each step reads the world with all earlier steps in force. An `Err`
    /// discards the whole context.
    pub fn run(&mut self, cmd: CommandPayload, caller: &Caller) -> Result<(), SessionError> {
        // Cancellation is terminal: nothing queued may go live behind it.
        let cancelling = matches!(cmd, CommandPayload::CancelSession);

        match (self.agent_id.is_some(), cmd) {
            (
                false,
                CommandPayload::CreateSession {
                    agent_id,
                    owner,
                    ancestry,
                    worker_retry,
                },
            ) => {
                SessionState::ensure_tenant_matches(caller, &owner.tenant_id)?;
                if let Caller::Frontend { user_id, .. } = caller {
                    if owner.id.as_deref() != Some(user_id.as_str()) {
                        return Err(SessionError::SessionAccessDenied);
                    }
                }
                self.emit(EventPayload::SessionCreated(Box::new(SessionCreated {
                    agent_id,
                    identity: owner,
                    ancestry,
                    worker_retry,
                })));
                // The session's first decision: the worker declares its
                // identity before any client input.
                self.queue_decision(Trigger::SessionStart);
            }
            (true, CommandPayload::CreateSession { .. }) => {
                return Err(SessionError::SessionAlreadyCreated)
            }
            (false, _) => return Err(SessionError::SessionNotCreated),
            (true, cmd) => self.run_active(cmd, caller)?,
        }
        // The schedule tail: after every command, dispatch whatever the queue
        // allows. Idempotent — an arm that already dispatched (a due retry, a
        // resumed interrupt) leaves its gate closed, and the walk reads
        // applied state.
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
                // `begin_turn` writes, so it sits between the two: the payload
                // is authorized before a turn is opened for it.
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

            CommandPayload::RequestSubAgent {
                session_id,
                agent_id,
                tool_call_id,
                retry,
            } => self
                .try_then(|s| {
                    effects::sub_agent::request(
                        s,
                        session_id,
                        agent_id,
                        tool_call_id,
                        retry,
                        caller,
                    )
                })
                .map(|_| ()),

            CommandPayload::CompleteSubAgentTurn {
                session_id,
                data,
                cost,
                token_usage,
                ..
            } => self
                .try_then(|s| {
                    effects::sub_agent::complete_turn(
                        s,
                        session_id,
                        data,
                        cost,
                        token_usage,
                        caller,
                    )
                })
                .map(|_| ()),

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
                let parked_head =
                    self.anchor_on_path(self.head_id.as_deref(), open.anchor.as_deref());
                self.emit(EventPayload::InterruptResumed(InterruptResumed {
                    interrupt_id: interrupt_id.clone(),
                    payload: payload.clone(),
                }));
                // Trigger only when the interrupt parked the head path. Queuing
                // is all this does: the planner sorts a resume decision ahead
                // of whatever queued up behind its interrupt, so the queue jump
                // is an ordering rule rather than a dispatch from here.
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
            } => {
                SessionState::ensure_machine_or_system(caller)?;
                match self
                    .tracking(EffectKind::Decision, &decision_id)
                    .map(|t| t.status())
                {
                    Some(EffectStatus::Pending) => {}
                    _ => return Ok(()),
                }
                // Each step is a pure function of the world the steps before it
                // left. The transcript moves `head_id`, so the reap and the two
                // writes all anchor against the post-reconcile head.
                self.then(|_| {
                    vec![EventPayload::DecisionCompleted(DecisionCompleted {
                        id: decision_id,
                    })]
                })
                .then(|s| s.reconcile_transcript(transcript).0)
                .then(|s| s.void_stranded_work(s.head_id.as_deref()))
                .then(|s| s.worker_state_events(state))
                .then(|s| s.agent_config_events(agent));

                // Actions are sub-commands, not steps: each re-enters the
                // handler and must see what the actions before it emitted.
                self.apply_actions(actions);
                Ok(())
            }

            CommandPayload::CancelSession => {
                SessionState::ensure_machine_or_system(caller)?;
                if matches!(self.status, SessionStatus::Done) {
                    return Ok(());
                }
                // Cancelling voids pending decisions as it applies, so the sweep
                // reads the post-cancel statuses.
                self.then(|_| vec![EventPayload::SessionCancelled])
                    .then(|s| {
                        s.void_effects(|tracking, _| {
                            matches!(
                                tracking.status(),
                                EffectStatus::Queued
                                    | EffectStatus::Pending
                                    | EffectStatus::RetryScheduled
                            )
                        })
                    });
                Ok(())
            }

            CommandPayload::FinishTurn { data } => self
                .try_then(|s| s.finish_turn_events(data, caller))
                .map(|_| ()),

            // The finalizer settled: emit the run terminal from the frozen output.
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

    /// Fail every in-flight effect orphaned by a process death. Both go through
    /// their kind's settle, so recovery behaves exactly like a reported failure.
    /// Worker tool calls are never touched: their async settle may still arrive.
    fn run_reconcile_dispatch(&mut self) -> Result<(), SessionError> {
        const LOST: &str = "dispatch lost on engine restart";
        let lost = || Outcome::Error(SettleError::new(LOST, true));
        for (id, terminal) in self.pending_decisions() {
            self.then(|s| EffectKind::Decision.spec().settle(s, &id, lost()));
            // A terminal one already ended the run; nothing after it can run.
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

    /// A wake carries no work of its own. It hands the planner the instant the
    /// due-work query was answered against; the run tail does the rest, and
    /// takes one sweep step from it.
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
