//! The planner: what runs next, and why not.
//!
//! # Laws
//!
//! **Effects are data.** [`plan`] returns descriptions. It performs no IO,
//! reads no clock, and mutates nothing — `now` arrives as an argument. The
//! caller turns steps into events.
//!
//! **Level-triggered only.** Every dispatch in the engine derives from
//! `(state, now)` through this one function. There is no "I know this just
//! became ready" path: a transition writes state, and the next plan sees it.
//! Transitions mutate state; they never store tasks.
//!
//! **Kinds are registered, not wired in.** A new prerequisite is a new [`Dep`]
//! variant plus an edge in the waiting kind's own
//! [`KindSpec::deps`](super::effects::KindSpec::deps); a new kind of work is an
//! [`EffectKind`] and a spec. This file holds no arm per kind — only the
//! ordering policy below, which is a list on purpose.
//!
//! # Ordering policy
//!
//! One place, so "what runs when, and why not" has a one-file answer:
//!
//! 1. **Owed fetches first, and alone.** A fetch is a prerequisite: the queue
//!    waits on it, and a prerequisite that queued behind its dependent would
//!    deadlock it.
//! 2. **Then the schedule queue**, in arrival order, with resume decisions
//!    pulled to the front — a resume answers the interrupt the caller just
//!    resolved, so it runs before whatever queued up behind that interrupt.
//!    Parked entries are skipped; the first blocked entry stops the walk, and
//!    nothing behind it starts.
//!
//! # Blocked and parked
//!
//! Two different gates, and the difference is what the entry is waiting for.
//!
//! A **dep** is queue-shaped: something ahead in this queue must move. It
//! blocks, head-of-line — an entry that cannot start holds everything behind it,
//! because the order is the contract.
//!
//! A **park** is region-shaped: a region elsewhere holds the entry while the
//! queue flows past it. It defers, never blocks. The two regions that park are
//! the interrupt overlay (an open interrupt on the entry's branch) and the turn
//! phase (a running turn, for an entry that opens a different one) — see
//! [`DecisionPark`].
//! 3. **Then, only when nothing above is ready, one sweep step**: the first
//!    effect past its deadline, else the first due retry, else a due decision
//!    redelivery. The caller takes at most one sweep per command, so a session
//!    with a pile of overdue work makes bounded progress and `wake_at`
//!    reschedules until nothing is due.

use std::collections::HashMap;

use chrono::{DateTime, Utc};

use super::decision::Trigger;
use super::state::{
    EffectKind, EffectPayload, EffectTracking, QueueEntry, SessionState, SessionStatus,
};
use crate::protocol::EffectStatus;

/// One unmet prerequisite of a queue entry — an edge of the dependency graph,
/// derived from live state by [`deps`], never stored. Each variant names the
/// effect that must move before the entry can start; a new prerequisite kind is
/// a new variant, not a new scheduler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dep {
    /// A connection's fetch must settle (completed or terminally failed).
    ConnectorSettled { connection_id: String },
    /// Another decision must settle first: the live slot, or the
    /// `session.start` prerequisite.
    DecisionSettled { decision_id: String },
    /// The blocked entry ahead in program order must dispatch first.
    EntryDispatched { kind: EffectKind, id: String },
    /// The interrupt parking the entry's branch must resume.
    InterruptResumed { interrupt_id: String },
    /// The turn holding the phase must end before the entry's turn can open.
    TurnSettled { turn_id: String },
}

impl Dep {
    /// The `waiting_on` label, the wire form of the edge.
    pub fn label(&self) -> String {
        match self {
            Dep::ConnectorSettled { connection_id } => format!("connector_sync:{connection_id}"),
            Dep::DecisionSettled { decision_id } => format!("decision:{decision_id}"),
            Dep::EntryDispatched { kind, id } => format!("queued_behind:{}:{id}", kind.label()),
            Dep::InterruptResumed { interrupt_id } => format!("interrupt:{interrupt_id}"),
            Dep::TurnSettled { turn_id } => format!("turn:{turn_id}"),
        }
    }
}

/// One unit of work the schedule licenses. A description, not an action: the
/// caller decides how many to take and turns each into events.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScheduleStep {
    /// Fetch a connection's tool list — a prerequisite the config in force owes.
    RequestFetch { connection_id: String },
    /// Start the queued entry.
    Dispatch { kind: EffectKind, id: String },
    /// A queue entry naming work that no longer exists. Voiding it lets the
    /// walk advance instead of spinning on inconsistent state.
    VoidPhantom { kind: EffectKind, id: String },
    /// An in-flight effect is past its deadline.
    TimeOut { kind: EffectKind, id: String },
    /// A failed effect's backoff has elapsed; re-issue it.
    Retry { kind: EffectKind, id: String },
    /// A decision left the queue at its first dispatch, so the walk cannot see
    /// it; its redelivery is named directly.
    RedispatchDecision { decision_id: String },
}

impl ScheduleStep {
    /// The deadline-and-retry family. The caller takes at most one per command:
    /// sweeping one item at a time bounds the batch a long-idle session emits.
    pub fn is_sweep(&self) -> bool {
        matches!(
            self,
            ScheduleStep::TimeOut { .. }
                | ScheduleStep::Retry { .. }
                | ScheduleStep::RedispatchDecision { .. }
        )
    }
}

/// Everything the schedule licenses at `now`, in the order it should be taken.
///
/// Pure: same state and same clock, same plan. Executing a step changes the
/// state, so the caller re-plans rather than remembering — see the module
/// ordering policy for what each pass offers.
pub fn plan(state: &SessionState, now: DateTime<Utc>) -> Vec<ScheduleStep> {
    let fetches = owed_fetches(state);
    if !fetches.is_empty() {
        return fetches;
    }
    let dispatches = walk(state);
    if !dispatches.is_empty() {
        return dispatches;
    }
    sweep(state, now).into_iter().collect()
}

/// Fetch every connection the config in force names but has never fetched.
fn owed_fetches(state: &SessionState) -> Vec<ScheduleStep> {
    state
        .unsynced_connectors(state.head_id.as_deref())
        .into_iter()
        .map(|connection_id| ScheduleStep::RequestFetch { connection_id })
        .collect()
}

/// The queue in the order the walk considers it: arrival order, with resume
/// decisions pulled to the front.
fn walk_order(state: &SessionState) -> Vec<&QueueEntry> {
    let (resumes, rest): (Vec<&QueueEntry>, Vec<&QueueEntry>) = state
        .schedule_queue
        .iter()
        .partition(|e| is_resume_decision(state, e));
    resumes.into_iter().chain(rest).collect()
}

fn is_resume_decision(state: &SessionState, entry: &QueueEntry) -> bool {
    entry.kind == EffectKind::Decision
        && state
            .worker_decision(&entry.id)
            .is_some_and(|d| matches!(d.trigger, Trigger::InterruptResumed { .. }))
}

/// The dispatchable prefix of the queue.
fn walk(state: &SessionState) -> Vec<ScheduleStep> {
    let mut steps = Vec::new();
    // A dispatched decision takes the slot, which blocks every decision behind
    // it. This is the only coupling between the steps of one plan — every other
    // gate reads state the steps do not touch — so it is the only thing the
    // walk has to model.
    let mut slot_taken = state.has_pending_worker_decision();
    for entry in walk_order(state) {
        if entry_parked(state, entry) {
            continue;
        }
        let takes_slot =
            entry.kind == EffectKind::Decision && state.has_effect(EffectKind::Decision, &entry.id);
        if takes_slot && slot_taken {
            break;
        }
        if entry_blocked(state, entry) {
            break;
        }
        steps.push(step_for(state, entry));
        slot_taken |= takes_slot;
    }
    steps
}

/// Whether the entry names work that still exists. A missing record is
/// inconsistent state for the kinds that keep one; the kinds that do not say so
/// themselves.
fn step_for(state: &SessionState, entry: &QueueEntry) -> ScheduleStep {
    let id = entry.id.clone();
    let kind = entry.kind;
    match kind.spec().voids_when_missing() && !state.has_effect(kind, &id) {
        true => ScheduleStep::VoidPhantom { kind, id },
        false => ScheduleStep::Dispatch { kind, id },
    }
}

/// One overdue item, families in a fixed order. `wake_at` reschedules until
/// nothing is due, so a pile of overdue work drains a step at a time.
fn sweep(state: &SessionState, now: DateTime<Utc>) -> Option<ScheduleStep> {
    if let Some((kind, id)) = first_timed_out(state, now) {
        return Some(ScheduleStep::TimeOut { kind, id });
    }
    if let Some((kind, id)) = first_due_retry(state, now) {
        return Some(ScheduleStep::Retry { kind, id });
    }
    due_decision(state, now).map(|decision_id| ScheduleStep::RedispatchDecision { decision_id })
}

/// The first effect whose tracking satisfies `due`, in `(kind, id)` order — the
/// table is sorted, and the planner must be deterministic.
fn first_due(
    state: &SessionState,
    kinds: &[EffectKind],
    due: impl Fn(&EffectTracking) -> bool,
) -> Option<(EffectKind, String)> {
    kinds.iter().find_map(|kind| {
        state
            .effects_of(*kind)
            .find(|e| due(&e.tracking))
            .map(|e| (*kind, e.id.clone()))
    })
}

/// Every kind a sweep considers, in a fixed order. One of the few lists that
/// names kinds: "which overdue item goes first" is a policy that only reads as
/// a policy when it is written as an order.
const SWEEP_ORDER: [EffectKind; 5] = [
    EffectKind::LlmCall,
    EffectKind::ToolCall,
    EffectKind::ConnectorSync,
    EffectKind::SubAgent,
    EffectKind::Decision,
];

/// The first effect past its deadline.
fn first_timed_out(state: &SessionState, now: DateTime<Utc>) -> Option<(EffectKind, String)> {
    first_due(state, &SWEEP_ORDER, |t| {
        t.status() == EffectStatus::Pending && t.deadline.is_some_and(|d| d <= now)
    })
}

/// The first effect whose retry is due, among the kinds that re-enter the queue
/// on one. A kind that redelivers under its own gates is excluded and sweeps
/// through its own path instead.
fn first_due_retry(state: &SessionState, now: DateTime<Utc>) -> Option<(EffectKind, String)> {
    let kinds: Vec<EffectKind> = SWEEP_ORDER
        .into_iter()
        .filter(|k| k.spec().requeues_on_retry())
        .collect();
    first_due(state, &kinds, |t| {
        t.status() == EffectStatus::RetryScheduled && t.retry.next_at.is_some_and(|at| at <= now)
    })
}

/// A decision whose backoff has elapsed and whose gates are open. Its own
/// `session.start` may be the one holding the prerequisite gate, which [`deps`]
/// excludes by id.
fn due_decision(state: &SessionState, now: DateTime<Utc>) -> Option<String> {
    let mut due: Vec<QueueEntry> = state
        .effects_of(EffectKind::Decision)
        .filter(|e| {
            e.tracking.status() == EffectStatus::RetryScheduled
                && e.tracking.retry.next_at.is_some_and(|at| at <= now)
        })
        .filter_map(|e| {
            Some(QueueEntry {
                seq: e.decision()?.source_event_sequence,
                kind: EffectKind::Decision,
                id: e.id.clone(),
            })
        })
        .collect();
    due.sort_by(|a, b| (a.seq, &a.id).cmp(&(b.seq, &b.id)));
    due.into_iter()
        .find(|entry| !entry_parked(state, entry) && !entry_blocked(state, entry))
        .map(|entry| entry.id)
}

/// Whether a region elsewhere holds this entry. Parked entries are skipped by
/// the walk, never blocking — see the module's "Blocked and parked". Which
/// kinds park is each kind's own answer.
pub fn entry_parked(state: &SessionState, entry: &QueueEntry) -> bool {
    entry_park(state, entry).is_some()
}

/// What holds a parked entry, as the dep that names it. The walk and
/// [`waiting_on`] read this same answer, so the gate and its explanation cannot
/// drift — and there is no second lookup to disagree with.
pub fn entry_park(state: &SessionState, entry: &QueueEntry) -> Option<Dep> {
    entry.kind.spec().park(state, &entry.id)
}

/// Whether a queue entry's prerequisites hold it.
pub fn entry_blocked(state: &SessionState, entry: &QueueEntry) -> bool {
    !deps(state, entry).is_empty()
}

/// The unmet prerequisites of a queue entry — the dependency graph, one node at
/// a time. Each kind draws its own edges ([`KindSpec::deps`]); this is only the
/// lookup. Derived from live state on every plan, so a config rewrite or head
/// move re-draws the graph instantly; nothing is stored. Empty ⇒ dispatchable,
/// position permitting. The walk and [`waiting_on`] read this same enumeration,
/// so the gate and its explanation cannot drift.
pub fn deps(state: &SessionState, entry: &QueueEntry) -> Vec<Dep> {
    entry.kind.spec().deps(state, entry)
}

/// Why each queued entry has not dispatched, keyed by `(kind, id)`. Walked in
/// the planner's own order, so the explanation matches the gate: the first
/// blocked entry names its deps; everything behind it names that entry
/// (`queued_behind:*` — strict order). A parked entry names the region holding
/// it — the interrupt on its branch, or the turn ahead of its own.
/// Dispatchable entries carry no reasons: they exist only between a queueing
/// event and the same batch's plan.
pub fn waiting_on(state: &SessionState) -> HashMap<(EffectKind, String), Vec<String>> {
    let mut out = HashMap::new();
    let mut head_of_line: Option<Dep> = None;
    for entry in walk_order(state) {
        let key = (entry.kind, entry.id.clone());
        if let Some(park) = entry_park(state, entry) {
            out.insert(key, vec![park.label()]);
            continue;
        }
        if let Some(front) = &head_of_line {
            out.insert(key, vec![front.label()]);
            continue;
        }
        let deps = deps(state, entry);
        if !deps.is_empty() {
            head_of_line = Some(Dep::EntryDispatched {
                kind: entry.kind,
                id: entry.id.clone(),
            });
            out.insert(key, deps.iter().map(Dep::label).collect());
        }
    }
    out
}

/// The next instant the session has work at. `Some(now)` when the plan already
/// has a step: the run tail leaves nothing ready, so this is belt and braces
/// against a state that somehow got ahead of its schedule.
pub fn wake_at(state: &SessionState, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
    if matches!(state.status, SessionStatus::Done) {
        return None;
    }
    if !plan(state, now).is_empty() {
        return Some(now);
    }
    // A parked effect contributes no wake: its branch is paused. A fetch is
    // unanchored, so never parked — a hung one must still time out, or the
    // decisions waiting on it park forever.
    state
        .effects
        .values()
        .filter(|e| match &e.payload {
            EffectPayload::ConnectorSync(_) => true,
            EffectPayload::Decision(d) => !state.decision_parked(&d.trigger),
            _ => !state.anchor_parked(e.anchor.as_deref()),
        })
        .filter_map(|e| e.tracking.earliest_wake())
        .min()
}

#[cfg(test)]
mod tests {
    //! The planner against synthetic states: policy asserted directly, rather
    //! than inferred from the event stream of a full command choreography.

    use super::*;
    use crate::protocol::{
        AgentConfig, InterruptOrigin, McpServer, Message, NewMessage, RetryPolicy, Role,
    };
    use crate::runtime::session::state::{
        AgentVersion, ConnectorSyncState, EffectPayload, EffectState, EffectTracking, LlmCallSpec,
        LlmCallState, Logged, OpenInterrupt, QueueEntry, ToolCallState, TurnPhase,
        WorkerDecisionState,
    };

    fn epoch() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc)
    }

    fn state() -> SessionState {
        let mut s = SessionState::new("sess-1".to_string());
        s.status = SessionStatus::Idle;
        s
    }

    fn tracking(status: EffectStatus) -> EffectTracking {
        let mut t = EffectTracking::new_queued(RetryPolicy::no_retry());
        match status {
            EffectStatus::Queued => {}
            EffectStatus::Pending => t.dispatch(epoch()),
            EffectStatus::Completed => {
                t.dispatch(epoch());
                t.complete();
            }
            EffectStatus::RetryScheduled | EffectStatus::Failed => {
                t.dispatch(epoch());
                t.record_error(status == EffectStatus::RetryScheduled, epoch());
            }
        }
        t
    }

    fn queue(s: &mut SessionState, kind: EffectKind, id: &str) {
        let seq = s.schedule_queue.len() as u64 + 1;
        s.schedule_queue.push(QueueEntry {
            seq,
            kind,
            id: id.to_string(),
        });
    }

    fn add_tool(s: &mut SessionState, id: &str, status: EffectStatus) {
        s.put_effect(EffectState::new(
            id,
            tracking(status),
            EffectPayload::ToolCall(ToolCallState {
                name: "t".to_string(),
                handler: crate::runtime::session::decision::ToolHandler::Worker,
                target: None,
                arguments: "{}".to_string(),
                result: None,
                is_error: false,
            }),
        ));
    }

    fn add_llm(s: &mut SessionState, id: &str, status: EffectStatus) {
        s.put_effect(EffectState::new(
            id,
            tracking(status),
            EffectPayload::LlmCall(LlmCallState {
                prompt: vec![],
                spec: LlmCallSpec {
                    model: "m".to_string(),
                    tools: None,
                    temperature: None,
                    max_completion_tokens: None,
                    reasoning: None,
                },
                stream: false,
                handler: crate::runtime::session::decision::LlmHandler::Server,
                format: None,
            }),
        ));
    }

    fn add_decision(s: &mut SessionState, id: &str, status: EffectStatus, trigger: Trigger) {
        let seq = s.effects_of(EffectKind::Decision).count() as u64 + 1;
        s.put_effect(EffectState::new(
            id,
            tracking(status),
            EffectPayload::Decision(WorkerDecisionState {
                trigger,
                source_event_sequence: seq,
            }),
        ));
    }

    fn deadline(s: &mut SessionState, kind: EffectKind, id: &str, at: DateTime<Utc>) {
        s.effect_mut(kind, id).unwrap().tracking.deadline = Some(at);
    }

    fn client_message() -> Trigger {
        Trigger::ClientMessage {
            messages: vec![],
            client: Default::default(),
            turn_id: None,
        }
    }

    /// A client message holding a turn of its own until the phase is free.
    fn deferred(turn_id: &str) -> Trigger {
        Trigger::ClientMessage {
            messages: vec![],
            client: Default::default(),
            turn_id: Some(turn_id.to_string()),
        }
    }

    fn config_with_mcp(connection: &str) -> AgentConfig {
        AgentConfig {
            model: "m".to_string(),
            system: None,
            stream: false,
            handler: None,
            format: None,
            retry: None,
            tools: Vec::new(),
            sub_agents: Vec::new(),
            mcp: vec![McpServer {
                id: connection.to_string(),
                tools: None,
            }],
        }
    }

    fn set_config(s: &mut SessionState, config: AgentConfig) {
        s.agent_versions.push(Logged {
            seq: 0,
            entry: AgentVersion {
                value: config,
                anchor: None,
            },
        });
    }

    fn dispatch(kind: EffectKind, id: &str) -> ScheduleStep {
        ScheduleStep::Dispatch {
            kind,
            id: id.to_string(),
        }
    }

    #[test]
    fn the_queue_dispatches_in_arrival_order() {
        let mut s = state();
        add_tool(&mut s, "tc-1", EffectStatus::Queued);
        add_tool(&mut s, "tc-2", EffectStatus::Queued);
        queue(&mut s, EffectKind::ToolCall, "tc-1");
        queue(&mut s, EffectKind::ToolCall, "tc-2");

        assert_eq!(
            plan(&s, epoch()),
            vec![
                dispatch(EffectKind::ToolCall, "tc-1"),
                dispatch(EffectKind::ToolCall, "tc-2"),
            ]
        );
    }

    #[test]
    fn a_blocked_entry_stops_everything_behind_it() {
        let mut s = state();
        // A config naming a connection that has never been fetched: the LLM
        // call ahead of the tool call cannot start, so neither does the tool.
        set_config(&mut s, config_with_mcp("conn-1"));
        s.put_effect(EffectState::new(
            "conn-1",
            tracking(EffectStatus::Pending),
            EffectPayload::ConnectorSync(ConnectorSyncState {
                tools: vec![],
                prefix: None,
                error: None,
                needs_reauth: false,
            }),
        ));
        add_llm(&mut s, "call-1", EffectStatus::Queued);
        add_tool(&mut s, "tc-1", EffectStatus::Queued);
        queue(&mut s, EffectKind::LlmCall, "call-1");
        queue(&mut s, EffectKind::ToolCall, "tc-1");

        assert_eq!(plan(&s, epoch()), vec![], "strict head of line");
        assert_eq!(
            waiting_on(&s)[&(EffectKind::LlmCall, "call-1".to_string())],
            vec!["connector_sync:conn-1".to_string()],
            "the head names its own dep"
        );
        assert_eq!(
            waiting_on(&s)[&(EffectKind::ToolCall, "tc-1".to_string())],
            vec!["queued_behind:llm_call:call-1".to_string()],
            "and everything behind names the head"
        );
    }

    #[test]
    fn an_owed_fetch_is_planned_first_and_alone() {
        let mut s = state();
        set_config(&mut s, config_with_mcp("conn-1"));
        add_tool(&mut s, "tc-1", EffectStatus::Queued);
        queue(&mut s, EffectKind::ToolCall, "tc-1");

        assert_eq!(
            plan(&s, epoch()),
            vec![ScheduleStep::RequestFetch {
                connection_id: "conn-1".to_string()
            }],
            "a prerequisite cannot queue behind its dependent"
        );
    }

    #[test]
    fn only_one_decision_goes_live_per_plan() {
        let mut s = state();
        add_decision(&mut s, "d-1", EffectStatus::Queued, client_message());
        add_decision(&mut s, "d-2", EffectStatus::Queued, client_message());
        queue(&mut s, EffectKind::Decision, "d-1");
        queue(&mut s, EffectKind::Decision, "d-2");

        assert_eq!(
            plan(&s, epoch()),
            vec![dispatch(EffectKind::Decision, "d-1")],
            "the slot admits one; the second waits for it to settle"
        );
    }

    #[test]
    fn a_parked_decision_is_skipped_not_blocking() {
        let mut s = state();
        s.nodes.push(Logged {
            seq: 1,
            entry: NewMessage {
                message: Message {
                    id: "u1".to_string(),
                    role: Role::User,
                    content: None,
                    tool_calls: vec![],
                    tool_call_id: None,
                    name: None,
                },
                parent_id: None,
            },
        });
        s.head_id = Some("u1".to_string());
        s.open_interrupts.push(OpenInterrupt {
            interrupt_id: "int-1".to_string(),
            origin: InterruptOrigin::Frontend,
            reason: "pause".to_string(),
            payload: serde_json::Value::Null,
            anchor: Some("u1".to_string()),
        });
        add_decision(&mut s, "d-1", EffectStatus::Queued, client_message());
        add_tool(&mut s, "tc-1", EffectStatus::Queued);
        queue(&mut s, EffectKind::Decision, "d-1");
        queue(&mut s, EffectKind::ToolCall, "tc-1");

        assert_eq!(
            plan(&s, epoch()),
            vec![dispatch(EffectKind::ToolCall, "tc-1")],
            "interrupts park branches, not the session"
        );
        assert_eq!(
            waiting_on(&s)[&(EffectKind::Decision, "d-1".to_string())],
            vec!["interrupt:int-1".to_string()]
        );
    }

    #[test]
    fn a_decision_for_another_turn_is_skipped_not_blocking() {
        let mut s = state();
        s.phase = TurnPhase::Active {
            turn_id: "turn-1".to_string(),
        };
        add_decision(&mut s, "d-2", EffectStatus::Queued, deferred("turn-2"));
        add_tool(&mut s, "tc-1", EffectStatus::Queued);
        queue(&mut s, EffectKind::Decision, "d-2");
        queue(&mut s, EffectKind::ToolCall, "tc-1");

        assert_eq!(
            plan(&s, epoch()),
            vec![dispatch(EffectKind::ToolCall, "tc-1")],
            "the running turn's own work flows past the turn queued behind it"
        );
        assert_eq!(
            waiting_on(&s)[&(EffectKind::Decision, "d-2".to_string())],
            vec!["turn:turn-1".to_string()],
            "and the park names the turn holding the phase"
        );

        s.phase = TurnPhase::Idle;
        assert_eq!(
            plan(&s, epoch()),
            vec![
                dispatch(EffectKind::Decision, "d-2"),
                dispatch(EffectKind::ToolCall, "tc-1"),
            ],
            "an idle phase unparks it"
        );
    }

    #[test]
    fn a_deferred_decision_does_not_park_against_its_own_turn() {
        // Its dispatch started the turn, so the phase now holds the id the
        // trigger names: a redelivery must see an open gate, not itself.
        let mut s = state();
        s.phase = TurnPhase::Active {
            turn_id: "turn-2".to_string(),
        };
        add_decision(&mut s, "d-2", EffectStatus::Queued, deferred("turn-2"));
        let at = epoch() + chrono::Duration::seconds(5);
        let t = &mut s.effect_mut(EffectKind::Decision, "d-2").unwrap().tracking;
        t.retry_policy = RetryPolicy::worker_default();
        t.dispatch(epoch());
        t.record_error(true, epoch());
        t.retry.next_at = Some(at);

        assert_eq!(wake_at(&s, epoch()), Some(at), "the redelivery still wakes");
        assert_eq!(
            plan(&s, at),
            vec![ScheduleStep::RedispatchDecision {
                decision_id: "d-2".to_string()
            }]
        );
    }

    #[test]
    fn a_resume_decision_jumps_the_queue() {
        let mut s = state();
        add_decision(&mut s, "d-1", EffectStatus::Queued, client_message());
        add_decision(
            &mut s,
            "d-resume",
            EffectStatus::Queued,
            Trigger::InterruptResumed {
                interrupt_id: "int-1".to_string(),
                payload: serde_json::Value::Null,
            },
        );
        queue(&mut s, EffectKind::Decision, "d-1");
        queue(&mut s, EffectKind::Decision, "d-resume");

        assert_eq!(
            plan(&s, epoch()),
            vec![dispatch(EffectKind::Decision, "d-resume")],
            "the resume answers the interrupt the caller just resolved"
        );
    }

    #[test]
    fn a_queue_entry_with_no_effect_voids_so_the_walk_advances() {
        let mut s = state();
        queue(&mut s, EffectKind::ToolCall, "gone");

        assert_eq!(
            plan(&s, epoch()),
            vec![ScheduleStep::VoidPhantom {
                kind: EffectKind::ToolCall,
                id: "gone".to_string(),
            }]
        );
    }

    #[test]
    fn a_sweep_is_planned_only_when_nothing_is_ready() {
        let mut s = state();
        add_tool(&mut s, "tc-1", EffectStatus::Queued);
        queue(&mut s, EffectKind::ToolCall, "tc-1");
        add_llm(&mut s, "call-1", EffectStatus::Pending);
        deadline(&mut s, EffectKind::LlmCall, "call-1", epoch());

        assert_eq!(
            plan(&s, epoch()),
            vec![dispatch(EffectKind::ToolCall, "tc-1")],
            "ready work first"
        );

        s.schedule_queue.clear();
        s.effects
            .remove(&(EffectKind::ToolCall, "tc-1".to_string()));
        assert_eq!(
            plan(&s, epoch()),
            vec![ScheduleStep::TimeOut {
                kind: EffectKind::LlmCall,
                id: "call-1".to_string(),
            }]
        );
    }

    #[test]
    fn a_sweep_offers_one_item_at_a_time() {
        let mut s = state();
        for id in ["call-1", "call-2"] {
            add_llm(&mut s, id, EffectStatus::Pending);
            deadline(&mut s, EffectKind::LlmCall, id, epoch());
        }
        let planned = plan(&s, epoch());
        assert_eq!(planned.len(), 1, "one sweep item per plan; got {planned:?}");
    }

    #[test]
    fn a_deadline_in_the_future_plans_nothing_and_sets_the_wake() {
        let mut s = state();
        let later = epoch() + chrono::Duration::seconds(30);
        add_llm(&mut s, "call-1", EffectStatus::Pending);
        deadline(&mut s, EffectKind::LlmCall, "call-1", later);

        assert_eq!(plan(&s, epoch()), vec![]);
        assert_eq!(wake_at(&s, epoch()), Some(later));
    }

    #[test]
    fn a_done_session_never_wakes() {
        let mut s = state();
        s.status = SessionStatus::Done;
        add_llm(&mut s, "call-1", EffectStatus::Pending);
        deadline(&mut s, EffectKind::LlmCall, "call-1", epoch());
        assert_eq!(wake_at(&s, epoch()), None);
    }

    #[test]
    fn ready_work_wakes_immediately() {
        let mut s = state();
        add_tool(&mut s, "tc-1", EffectStatus::Queued);
        queue(&mut s, EffectKind::ToolCall, "tc-1");
        assert_eq!(wake_at(&s, epoch()), Some(epoch()));
    }
}
