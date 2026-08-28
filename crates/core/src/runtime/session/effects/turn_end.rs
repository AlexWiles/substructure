//! The end of a turn.
//!
//! ```text
//! Queued ─dispatch→ (removed) ⇒ emit turn.completed + session.done
//! ```
//!
//! Not work, and not tracked: a turn's end has no record in the effect table
//! and no deadline of its own. It is a kind because it queues, waits, and
//! dispatches like one — the queue is what orders it after the finalizer.
//!
//! Ending a turn takes two passes. `FinishTurn` queues `turn.finished` with the
//! turn's frozen output, which is what puts this entry in the queue; the turn
//! completes only once that finalizer settles
//! ([`Dep::DecisionSettled`](super::super::schedule::Dep)). A `done` echo from
//! the worker emits `TurnCompleted` itself and removes the entry before the plan
//! reaches it; this entry covers a finalizer that settles any other way.

use super::{KindSpec, Outcome, SettleError};
use crate::protocol::EffectStatus;
use crate::runtime::session::decision::Trigger;
use crate::runtime::session::events::*;
use crate::runtime::session::schedule::Dep;
use crate::runtime::session::state::{EffectKind, QueueEntry, SessionState, TurnPhase};

pub struct TurnEndSpec;

impl KindSpec for TurnEndSpec {
    fn kind(&self) -> EffectKind {
        EffectKind::TurnEnd
    }

    /// Never: there is no effect to settle, and no deadline to sweep.
    fn settle(&self, _state: &SessionState, _id: &str, _outcome: Outcome) -> Vec<EventPayload> {
        Vec::new()
    }

    /// A turn's end has no record of its own, so an entry for one is never
    /// inconsistent state.
    fn voids_when_missing(&self) -> bool {
        false
    }

    /// The turn completes when its `turn.finished` finalizer settles.
    fn deps(&self, state: &SessionState, entry: &QueueEntry) -> Vec<Dep> {
        state
            .effects_of(EffectKind::Decision)
            .find(|e| {
                matches!(&e.decision().map(|d| &d.trigger),
                    Some(Trigger::TurnFinished { turn_id, .. }) if *turn_id == entry.id)
            })
            .map(|e| {
                vec![Dep::DecisionSettled {
                    decision_id: e.id.clone(),
                }]
            })
            .unwrap_or_default()
    }

    fn dispatch(&self, state: &SessionState, _id: &str) -> Vec<EventPayload> {
        state.finalize_run(None)
    }
}

impl SessionState {
    /// The run terminal from the in-flight `finalizing` output: `TurnCompleted`
    /// (carrying `error` when the finalizer failed terminally) + `SessionDone`.
    /// Falls back to a bare `SessionDone` if nothing is finalizing.
    pub(in crate::runtime::session) fn finalize_run(
        &self,
        error: Option<&SettleError>,
    ) -> Vec<EventPayload> {
        match self.phase.finalizing().cloned() {
            Some(f) => vec![
                EventPayload::TurnCompleted(TurnCompleted {
                    turn_id: f.turn_id,
                    data: f.data,
                    turn_cost: f.cost,
                    turn_token_usage: f.usage,
                    error: error.map(|e| e.error.clone()),
                }),
                EventPayload::SessionDone(SessionDone {}),
            ],
            None => vec![EventPayload::SessionDone(SessionDone {})],
        }
    }

    /// End the turn as a failed run: void what is in flight, drop what is
    /// queued, then report the failure against the turn that was running.
    pub(in crate::runtime::session) fn fail_run(&self, error: &SettleError) -> Vec<EventPayload> {
        // The finalizer itself failed: the frozen output is already held, and
        // `finalize_run` reports it against the turn that produced it.
        if self.phase.finalizing().is_some() {
            return self.finalize_run(Some(error));
        }
        let mut events = self.void_effects(|_, tracking, _| {
            matches!(
                tracking.status(),
                EffectStatus::Queued | EffectStatus::Pending | EffectStatus::RetryScheduled
            )
        });
        events.extend(self.drop_queued_decisions());
        // No turn to end — a `session.start` that failed before any client input.
        // The decision's error event is the whole record. A completed turn is
        // `Idle` and a finalizing one returned above, so `Active` is exactly the
        // case that still owes a terminal.
        let TurnPhase::Active { turn_id } = &self.phase else {
            return events;
        };
        events.push(EventPayload::TurnCompleted(TurnCompleted {
            turn_id: turn_id.clone(),
            data: serde_json::Value::Null,
            turn_cost: self.turn_cost,
            turn_token_usage: self.turn_token_usage.clone(),
            error: Some(error.error.clone()),
        }));
        events.push(EventPayload::SessionDone(SessionDone {}));
        events
    }
}
