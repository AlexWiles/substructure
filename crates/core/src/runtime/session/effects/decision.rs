//! One turn of the worker's loop.
//!
//! ```text
//! Queued ─dispatch→ Pending ─submitted───────────→ (removed)
//!                      │ ─error[retries left]────→ RetryScheduled ─due→ Pending
//!                      │ ─error[exhausted]───────→ (removed) ⇒ the run fails
//!                      └ ─interrupt/void────────→ (removed)
//! ```
//!
//! At most one decision is live at a time — a second would let two transcript
//! writers fork the tree — so every other decision waits on the slot
//! ([`Dep::DecisionSettled`](super::super::schedule::Dep)). A decision leaves the
//! queue at its first dispatch, so a retry redelivers through the planner's
//! `RedispatchDecision` under the same gates rather than re-entering the walk.
//!
//! A terminal failure ends the run. `TurnCompleted` is the only terminal every
//! consumer watches, so a failure that never reaches it strands the turn open
//! forever: no reply, no error, no timeout.

use super::{fail, mismatched, resolve_pending, KindSpec, Outcome, Settle, SettleError};
use crate::protocol::EffectStatus;
use crate::runtime::session::command::SessionError;
use crate::runtime::session::decision::Trigger;
use crate::runtime::session::events::*;
use crate::runtime::session::schedule::Dep;
use crate::runtime::session::state::{DecisionPark, EffectKind, QueueEntry, SessionState};
use crate::runtime::Caller;

pub struct DecisionSpec;

impl KindSpec for DecisionSpec {
    fn kind(&self) -> EffectKind {
        EffectKind::Decision
    }

    fn authorize(
        &self,
        _state: &SessionState,
        _id: &str,
        caller: &Caller,
    ) -> Result<(), SessionError> {
        SessionState::ensure_machine_or_system(caller)
    }

    /// No attempt to fence and no external claim on the answer: a late failure
    /// report for a settled decision is dropped, not rejected.
    fn resolve(
        &self,
        state: &SessionState,
        id: &str,
        _attempt: Option<u32>,
        _caller: &Caller,
    ) -> Result<Settle, SessionError> {
        Ok(resolve_pending(state.tracking(EffectKind::Decision, id)))
    }

    /// Only failure settles here. A submitted decision is its own command: it
    /// carries a transcript and actions, which no other kind has.
    fn settle(&self, state: &SessionState, id: &str, outcome: Outcome) -> Vec<EventPayload> {
        match outcome {
            Outcome::Error(e) => fail(self, state, id, &e),
            other => mismatched(self.kind(), &other),
        }
    }

    fn errored(&self, _state: &SessionState, id: &str, e: &SettleError) -> Option<EventPayload> {
        Some(EventPayload::DecisionErrored(DecisionErrored {
            id: id.to_string(),
            error: e.error.clone(),
            retryable: e.retryable,
        }))
    }

    /// A turn whose driving decision is dead can never settle on its own, so
    /// the run ends. A failed `turn.finished` still completes its turn: the
    /// output is durable, the finalizer is not.
    fn terminal(&self, state: &SessionState, _id: &str, e: &SettleError) -> Vec<EventPayload> {
        state.fail_run(e)
    }

    /// A decision left the queue at its first dispatch, so the walk cannot see
    /// it: its redelivery is named directly by the planner instead.
    fn requeues_on_retry(&self) -> bool {
        false
    }

    /// A decision with no record drops rather than voids: it was never work in
    /// flight, only a delivery that lost its subject.
    fn voids_when_missing(&self) -> bool {
        false
    }

    /// The loop is what an interrupt pauses and what a turn occupies, so this
    /// is the kind that parks. The same judgment spares tools and sub-agents
    /// from interrupt voiding: a client still answers its tool call
    /// mid-interrupt.
    fn park(&self, state: &SessionState, id: &str) -> Option<Dep> {
        let park = state
            .worker_decision(id)
            .and_then(|d| state.decision_park(&d.trigger))?;
        Some(match park {
            DecisionPark::Interrupt(i) => Dep::InterruptResumed {
                interrupt_id: i.interrupt_id.clone(),
            },
            DecisionPark::Turn(turn_id) => Dep::TurnSettled {
                turn_id: turn_id.to_string(),
            },
        })
    }

    /// The slot (at most one live decision — a second would let two transcript
    /// writers fork the tree), the `session.start` prerequisite (its decision
    /// writes the config that parameterizes every other turn; only its own
    /// retry passes), and owed fetches.
    fn deps(&self, state: &SessionState, entry: &QueueEntry) -> Vec<Dep> {
        let mut deps: Vec<Dep> = state
            .effects_of(EffectKind::Decision)
            .filter(|e| {
                let Some(wd) = e.decision() else { return false };
                e.tracking.status() == EffectStatus::Pending
                    || (matches!(wd.trigger, Trigger::SessionStart)
                        && e.tracking.status() == EffectStatus::RetryScheduled
                        && e.id != entry.id)
            })
            .map(|e| Dep::DecisionSettled {
                decision_id: e.id.clone(),
            })
            .collect();
        deps.extend(super::connector::owed(state, state.head_id.as_deref()));
        deps.sort_by_key(Dep::label);
        deps.dedup();
        deps
    }

    /// A deferred decision opens its turn here: the park held it until the
    /// phase was free, so the turn starts where the work does.
    fn dispatch(&self, state: &SessionState, id: &str) -> Vec<EventPayload> {
        // A decision with no record drops rather than voids: it was never work
        // in flight, only a delivery that lost its subject.
        let Some(decision) = state.worker_decision(id) else {
            return vec![EventPayload::DecisionDropped(DecisionDropped {
                id: id.to_string(),
            })];
        };
        let mut events = Vec::new();
        if let Some(turn_id) = decision.trigger.deferred_turn_id() {
            if state.completed_turn_ids.iter().any(|t| t == turn_id) {
                return vec![EventPayload::DecisionDropped(DecisionDropped {
                    id: id.to_string(),
                })];
            }
            events.push(EventPayload::TurnStarted(TurnStarted {
                turn_id: turn_id.to_string(),
            }));
        }
        events.push(EventPayload::DecisionDispatched(DecisionDispatched {
            id: id.to_string(),
        }));
        events
    }
}

impl SessionState {
    /// The run is over, so nothing queued can still run: drop them all rather
    /// than promote a decision whose turn already ended. After a terminally-failed
    /// `session.start` this is also the only correct move — no config was ever
    /// resolved, so a promoted turn would settle as a silent no-op.
    ///
    /// Decisions holding a turn of their own are spared: this turn failing must
    /// not un-ask the next turn's question. Cancellation still voids them.
    pub(in crate::runtime::session) fn drop_queued_decisions(&self) -> Vec<EventPayload> {
        self.queued_decisions()
            .into_iter()
            .filter(|e| {
                e.decision()
                    .is_none_or(|d| d.trigger.deferred_turn_id().is_none())
            })
            .map(|e| EventPayload::DecisionDropped(DecisionDropped { id: e.id.clone() }))
            .collect()
    }

    /// Every decision the engine still owes a delivery, with whether failing it
    /// now would be terminal. Boot-time recovery reads this: a pull-dispatched
    /// decision is not provably orphaned, but failing one is safe — a late
    /// submit no-ops — at worst re-running the handler.
    pub(in crate::runtime::session) fn pending_decisions(&self) -> Vec<(String, bool)> {
        self.effects_of(EffectKind::Decision)
            .filter(|e| e.tracking.status() == EffectStatus::Pending)
            .map(|e| (e.id.clone(), e.tracking.is_terminal_failure(true)))
            .collect()
    }
}
