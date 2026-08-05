//! The kind registry: one module per kind of work, one spec each.
//!
//! # Laws
//!
//! **Kinds are registered, not wired in.** A kind is a [`KindSpec`] impl and a
//! line in [`EffectKind::spec`]. The seams that act on effects — dispatch,
//! settle, timeout, retry, the queue gates — hold no arm per kind: they ask the
//! spec. Adding a kind is a new file here plus the edge it waits on, written in
//! the file of the kind that waits.
//!
//! Three places still name kinds on purpose, and each is policy rather than
//! behaviour: the sweep order and the decision slot in
//! [`schedule`](super::schedule) (both are "what runs first", which only makes
//! sense stated in one list), and which kinds are abandoned on a fork or an
//! interrupt in [`command`](super::command) (a judgment about branches, not
//! about work).
//!
//! **Effects are data.** Every method here is a pure function of `(state, id)`
//! returning events. No IO, no clock: a deadline is stamped by the `apply` of
//! the dispatch event, from the command's own instant.
//!
//! **One lifecycle.** Every kind moves through the same
//! [`EffectTracking`](super::state::EffectTracking) statechart. What differs is
//! which events record the moves and what a terminal failure folds back into
//! the loop; each module opens with its own transition table.
//!
//! # The shape of a kind
//!
//! Three seams call through a spec, and only through a spec:
//!
//! - **dispatch** — the planner licensed a queued effect to start.
//!   [`dispatch`](KindSpec::dispatch) writes the marker;
//!   [`execute_trigger`](KindSpec::execute_trigger) hands a worker-run effect to
//!   the worker, built from post-dispatch state.
//! - **settle** — an answer arrived, or a deadline passed.
//!   [`authorize`](KindSpec::authorize) then [`resolve`](KindSpec::resolve) fence
//!   it, [`settle`](KindSpec::settle) records it. A failure is
//!   [`errored`](KindSpec::errored) plus, when the retry policy is exhausted,
//!   [`terminal`](KindSpec::terminal).
//! - **retry** — a backoff elapsed; [`retry`](KindSpec::retry) re-issues it.

use super::command::SessionError;
use super::decision::Trigger;
use super::events::{CallVoided, DecisionQueued, EventPayload};
use super::schedule::Dep;
use super::state::{
    new_call_id, EffectKind, EffectPayload, EffectTracking, QueueEntry, SessionState,
};
use crate::connectors::RemoteTool;
use crate::protocol::{EffectStatus, ErrorCode, ErrorInfo, LlmResponse};
use crate::runtime::Caller;

pub mod connector;
pub mod decision;
pub mod llm;
pub mod sub_agent;
pub mod tool;
pub mod turn_end;

/// What an effect answered with. One success variant per kind that produces
/// data, and one failure shape shared by every kind — a settle differs by what
/// it carries, not by which command names it.
#[derive(Debug, Clone)]
pub enum Outcome {
    Llm(Box<LlmResponse>),
    Tool {
        result: String,
    },
    /// The child session is running; its turn result arrives separately.
    SubAgentStarted,
    Connector {
        prefix: Option<String>,
        tools: Vec<RemoteTool>,
    },
    Error(SettleError),
}

/// A failed settle: the failure itself, plus what the engine decided about
/// this attempt. `needs_reauth` is connector-only, absent for every other kind
/// rather than modelled as its own command.
#[derive(Debug, Clone)]
pub struct SettleError {
    pub error: ErrorInfo,
    pub retryable: bool,
    pub needs_reauth: bool,
}

impl SettleError {
    pub fn new(error: ErrorInfo, retryable: bool) -> Self {
        Self {
            error,
            retryable,
            needs_reauth: false,
        }
    }

    pub fn reauth(mut self, needs_reauth: bool) -> Self {
        self.needs_reauth = needs_reauth;
        self
    }
}

impl From<SettleError> for Outcome {
    fn from(e: SettleError) -> Self {
        Outcome::Error(e)
    }
}

/// The error a swept effect settles with.
pub const DEADLINE: &str = "deadline exceeded";

/// One kind of work, as data the engine reads rather than arms it branches on.
///
/// Every method is a pure function of the state its predecessors left, so a
/// spec composes with the command batch exactly like a
/// [`then`](super::command::Working) step does.
pub trait KindSpec: Sync {
    fn kind(&self) -> EffectKind;

    // ── Settling ────────────────────────────────────────────────────────

    /// Who may settle this kind's effects. Runs before the fence, so a caller
    /// with no claim on the effect is refused whatever state it is in.
    ///
    /// Most kinds are engine-internal; the two a worker or client can answer
    /// (tool calls, LLM calls) check the handler the call was frozen with.
    fn authorize(
        &self,
        _state: &SessionState,
        _id: &str,
        caller: &Caller,
    ) -> Result<(), SessionError> {
        SessionState::ensure_internal(caller)
    }

    /// Fence the settle against the effect it names. The shared rule
    /// ([`resolve_settle`]) unless the kind has no attempt to fence.
    fn resolve(
        &self,
        state: &SessionState,
        id: &str,
        attempt: Option<u32>,
        caller: &Caller,
    ) -> Result<Settle, SessionError> {
        resolve_settle(state.tracking(self.kind(), id), attempt, caller)
    }

    /// What recording this answer writes. Called only for a live settle.
    fn settle(&self, state: &SessionState, id: &str, outcome: Outcome) -> Vec<EventPayload>;

    /// The kind's `*.errored` event. `None` for a kind that cannot fail — a
    /// turn's end is bookkeeping, not work.
    fn errored(&self, _state: &SessionState, _id: &str, _e: &SettleError) -> Option<EventPayload> {
        None
    }

    /// What a terminal failure folds back into the loop — for a call, the
    /// `*.finished` trigger that lets the model see the error as the result.
    /// Empty when a failure of this kind ends there.
    fn terminal(&self, _state: &SessionState, _id: &str, _e: &SettleError) -> Vec<EventPayload> {
        Vec::new()
    }

    /// How this kind reports a deadline. A timeout settles through the same
    /// path as a reported failure, so nothing special-cases it.
    ///
    /// An attempt that lapsed is retryable — the next one may well land. A whole
    /// effect that lapsed is not: the bound covers every attempt and the backoff
    /// between them, so there is no budget left to retry into.
    fn timeout_error(&self, total: bool) -> SettleError {
        SettleError::new(
            ErrorInfo::new(ErrorCode::DeadlineExceeded, DEADLINE),
            !total,
        )
    }

    // ── Starting ────────────────────────────────────────────────────────

    /// What starting one queued effect writes.
    fn dispatch(&self, state: &SessionState, id: &str) -> Vec<EventPayload>;

    /// The `*.execute` trigger a worker-run effect needs. Built from
    /// post-dispatch state on purpose: the deadline is stamped and the spec
    /// carries the tools in force only once the dispatch event has applied, so
    /// this cannot be folded into [`dispatch`](Self::dispatch).
    fn execute_trigger(&self, _state: &SessionState, _id: &str) -> Option<Trigger> {
        None
    }

    /// Re-issue an effect whose backoff has elapsed. Calls re-enter the queue
    /// and re-dispatch at the run tail.
    fn retry(&self, _state: &SessionState, _id: &str) -> Vec<EventPayload> {
        Vec::new()
    }

    /// Whether a due retry re-enters the queue and re-dispatches through the
    /// walk. False for a kind that left the queue at its first dispatch and
    /// redelivers under its own gates instead.
    fn requeues_on_retry(&self) -> bool {
        true
    }

    // ── Queueing ────────────────────────────────────────────────────────

    /// Whether a queue entry naming no live effect is inconsistent state. True
    /// for work with a record of its own — voiding the entry lets the walk
    /// advance instead of spinning.
    fn voids_when_missing(&self) -> bool {
        true
    }

    /// The region holding this entry, named as the dep it waits on — `None` when
    /// nothing holds it. A park skips the walk instead of blocking it, so a
    /// parked region never stops the queue: an interrupt pauses the loop, not
    /// effect execution, and a running turn holds only the turns behind it.
    /// Answering with the dep keeps the gate and its explanation one lookup.
    fn park(&self, _state: &SessionState, _id: &str) -> Option<Dep> {
        None
    }

    /// The entry's unmet prerequisites — this kind's edges of the dependency
    /// graph, derived from live state on every plan and never stored. Empty ⇒
    /// dispatchable, position permitting.
    ///
    /// Every dep must be bounded: the effect it names carries a deadline and
    /// always settles — failed if nothing else — so a dep can delay an entry
    /// but never strand one.
    fn deps(&self, _state: &SessionState, _entry: &QueueEntry) -> Vec<Dep> {
        Vec::new()
    }
}

impl EffectKind {
    /// The kind's spec. The one match on the kind enum.
    pub fn spec(self) -> &'static dyn KindSpec {
        match self {
            EffectKind::LlmCall => &llm::LlmSpec,
            EffectKind::ToolCall => &tool::ToolSpec,
            EffectKind::SubAgent => &sub_agent::SubAgentSpec,
            EffectKind::ConnectorSync => &connector::ConnectorSpec,
            EffectKind::Decision => &decision::DecisionSpec,
            EffectKind::TurnEnd => &turn_end::TurnEndSpec,
        }
    }
}

impl EffectPayload {
    pub fn spec(&self) -> &'static dyn KindSpec {
        self.kind().spec()
    }
}

/// The shared failure shape: record the error, and when the retry policy is
/// exhausted let the kind fold it back into the loop. Read before the errored
/// event applies — afterwards the tracking no longer answers the same question.
pub(super) fn fail(
    spec: &dyn KindSpec,
    state: &SessionState,
    id: &str,
    e: &SettleError,
) -> Vec<EventPayload> {
    let Some(tracking) = state.tracking(spec.kind(), id) else {
        return Vec::new();
    };
    let terminal = tracking.is_terminal_failure(e.retryable);
    let Some(errored) = spec.errored(state, id, e) else {
        return Vec::new();
    };
    let mut events = vec![errored];
    if terminal {
        events.extend(spec.terminal(state, id, e));
    }
    events
}

/// A kind cannot answer in another kind's shape. Callers are typed; an external
/// one is rejected at the wire seam long before it gets here.
pub(super) fn mismatched(kind: EffectKind, outcome: &Outcome) -> Vec<EventPayload> {
    debug_assert!(false, "{kind:?} cannot settle with {outcome:?}");
    Vec::new()
}

/// Queue a decision. The queued event carries the trigger, its single stored
/// copy; promotion is the run tail's job, never the queueing site's.
pub(super) fn decision_queued(trigger: Trigger) -> EventPayload {
    EventPayload::DecisionQueued(DecisionQueued {
        id: new_call_id(),
        trigger,
    })
}

/// Void one effect, so a plan that met inconsistent state still advances.
pub(super) fn void_events(kind: EffectKind, id: String) -> Vec<EventPayload> {
    vec![EventPayload::CallVoided(CallVoided { kind, id })]
}

/// What a settle resolves to against the effect it names.
pub enum Settle {
    /// Live: record it.
    Live,
    /// Stale, and the caller cannot act on the answer — drop it silently.
    Drop,
}

/// The shared fence: a settle is live only for a pending effect on the attempt
/// it names.
///
/// A stale settle is unavoidable — the deadline fires while the provider or
/// connection is still in flight, and its answer arrives for an effect that
/// already settled. Taking it late would fork reality: `*.finished` already
/// carries the timeout and the worker may have acted on it. So it is always
/// dropped; only the reply differs.
///
/// The engine's own executors settle as [`Caller::System`] and do nothing with
/// an error but log it (`llm/executor.rs`, `connector/executor.rs`), so for them
/// the drop is silent — the race is expected, not a fault. Every other settler
/// is external, holds a result, and needs the `409` to know it was discarded.
pub fn resolve_settle(
    tracking: Option<&EffectTracking>,
    attempt: Option<u32>,
    caller: &Caller,
) -> Result<Settle, SessionError> {
    match tracking {
        Some(t)
            if t.status() == EffectStatus::Pending
                && attempt.is_none_or(|a| a == t.retry.attempts) =>
        {
            Ok(Settle::Live)
        }
        _ if matches!(caller, Caller::System { .. }) => Ok(Settle::Drop),
        None => Err(SessionError::EffectNotFound),
        Some(t) if t.status() != EffectStatus::Pending => Err(SessionError::EffectNotPending),
        Some(_) => Err(SessionError::EffectAttemptMismatch),
    }
}

/// A settle with nothing to fence: live while the effect is pending, dropped
/// otherwise, never rejected. For the kinds no external party holds an answer
/// to — a late report is the engine talking to itself.
pub(super) fn resolve_pending(tracking: Option<&EffectTracking>) -> Settle {
    match tracking {
        Some(t) if t.status() == EffectStatus::Pending => Settle::Live,
        _ => Settle::Drop,
    }
}
