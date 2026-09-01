use std::collections::HashMap;

use chrono::{DateTime, Utc};

use super::decision::Trigger;
use super::state::{
    EffectKind, EffectPayload, EffectTracking, QueueEntry, SessionState, SessionStatus,
};
use crate::connectors::registry::ConnectionPath;
use crate::protocol::EffectStatus;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dep {
    ConnectorSettled { connection_id: ConnectionPath },
    DecisionSettled { decision_id: String },
    EntryDispatched { kind: EffectKind, id: String },
    InterruptResumed { interrupt_id: String },
    TurnSettled { turn_id: String },
}

impl Dep {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScheduleStep {
    RequestFetch { connection_id: ConnectionPath },
    Dispatch { kind: EffectKind, id: String },
    VoidPhantom { kind: EffectKind, id: String },
    TimeOut { kind: EffectKind, id: String },
    Retry { kind: EffectKind, id: String },
    RedispatchDecision { decision_id: String },
}

impl ScheduleStep {
    pub fn is_sweep(&self) -> bool {
        matches!(
            self,
            ScheduleStep::TimeOut { .. }
                | ScheduleStep::Retry { .. }
                | ScheduleStep::RedispatchDecision { .. }
        )
    }
}

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

fn owed_fetches(state: &SessionState) -> Vec<ScheduleStep> {
    state
        .at_head()
        .unsynced_connectors()
        .into_iter()
        .map(|connection_id| ScheduleStep::RequestFetch { connection_id })
        .collect()
}

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

fn walk(state: &SessionState) -> Vec<ScheduleStep> {
    let mut steps = Vec::new();
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

fn step_for(state: &SessionState, entry: &QueueEntry) -> ScheduleStep {
    let id = entry.id.clone();
    let kind = entry.kind;
    match kind.spec().voids_when_missing() && !state.has_effect(kind, &id) {
        true => ScheduleStep::VoidPhantom { kind, id },
        false => ScheduleStep::Dispatch { kind, id },
    }
}

fn sweep(state: &SessionState, now: DateTime<Utc>) -> Option<ScheduleStep> {
    if let Some((kind, id)) = first_timed_out(state, now) {
        return Some(ScheduleStep::TimeOut { kind, id });
    }
    if let Some((kind, id)) = first_due_retry(state, now) {
        return Some(ScheduleStep::Retry { kind, id });
    }
    due_decision(state, now).map(|decision_id| ScheduleStep::RedispatchDecision { decision_id })
}

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

const SWEEP_ORDER: [EffectKind; 5] = [
    EffectKind::LlmCall,
    EffectKind::ToolCall,
    EffectKind::ConnectorSync,
    EffectKind::Subagent,
    EffectKind::Decision,
];

fn first_timed_out(state: &SessionState, now: DateTime<Utc>) -> Option<(EffectKind, String)> {
    first_due(state, &SWEEP_ORDER, |t| {
        t.expiry().is_some_and(|d| d <= now)
    })
}

fn first_due_retry(state: &SessionState, now: DateTime<Utc>) -> Option<(EffectKind, String)> {
    let kinds: Vec<EffectKind> = SWEEP_ORDER
        .into_iter()
        .filter(|k| k.spec().requeues_on_retry())
        .collect();
    first_due(state, &kinds, |t| {
        t.status() == EffectStatus::RetryScheduled && t.retry.next_at.is_some_and(|at| at <= now)
    })
}

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

pub fn entry_parked(state: &SessionState, entry: &QueueEntry) -> bool {
    entry_park(state, entry).is_some()
}

pub fn entry_park(state: &SessionState, entry: &QueueEntry) -> Option<Dep> {
    entry.kind.spec().park(state, &entry.id)
}

pub fn entry_blocked(state: &SessionState, entry: &QueueEntry) -> bool {
    !deps(state, entry).is_empty()
}

pub fn deps(state: &SessionState, entry: &QueueEntry) -> Vec<Dep> {
    entry.kind.spec().deps(state, entry)
}

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

pub fn wake_at(state: &SessionState, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
    if matches!(state.status, SessionStatus::Done) {
        return None;
    }
    if !plan(state, now).is_empty() {
        return Some(now);
    }
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

    use super::*;
    use crate::protocol::{
        AgentConfig, InterruptOrigin, McpServer, Message, NewMessage, RetryPolicy, Role,
    };
    use crate::runtime::retry::RetryTarget;
    use crate::runtime::session::state::{
        AgentVersion, ConnectorSyncState, EffectPayload, EffectState, EffectTracking, LlmCallSpec,
        LlmCallState, Logged, OpenInterrupt, QueueEntry, SubagentCallState, ToolCallState,
        TurnPhase, WorkerDecisionState,
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
            EffectStatus::Running => {
                t.dispatch(epoch());
                t.run();
            }
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
                defer_tools_strategy: Default::default(),
                context_ids: Vec::new(),
                format: None,
                llm: "claude".to_string(),
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

    fn add_running_subagent(s: &mut SessionState, id: &str, total: Option<u32>) {
        let mut t = EffectTracking::new_queued(RetryPolicy {
            queue_timeout_secs: None,
            run_timeout_secs: None,
            total_timeout_secs: total,
            max_attempts: 3,
            backoff_base_secs: 1,
            backoff_max_secs: 1,
        });
        t.dispatch(epoch());
        t.run();
        s.put_effect(EffectState::new(
            id,
            t,
            EffectPayload::Subagent(SubagentCallState {
                agent_id: "child".to_string(),
                session_id: "child-1".to_string(),
                message: None,
                result: None,
                is_error: false,
                mode: Default::default(),
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

    fn deferred(turn_id: &str) -> Trigger {
        Trigger::ClientMessage {
            messages: vec![],
            client: Default::default(),
            turn_id: Some(turn_id.to_string()),
        }
    }

    fn config_with_mcp(connection: &str) -> AgentConfig {
        AgentConfig {
            llm: Some("claude".to_string()),
            model: "m".to_string(),
            mcp: vec![McpServer {
                id: connection.to_string(),
                tools: None,
                auth_failure: Default::default(),
                tool_sync_failure: Default::default(),
                approve: Default::default(),
            }],
            ..Default::default()
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
        set_config(&mut s, config_with_mcp("conn-1"));
        s.put_effect(EffectState::new(
            "mcp.conn-1",
            tracking(EffectStatus::Pending),
            EffectPayload::ConnectorSync(ConnectorSyncState {
                tools: vec![],
                prefix: None,
                error: None,
                auth: None,
                instructions: None,
            }),
        ));
        add_llm(&mut s, "call-1", EffectStatus::Queued);
        add_tool(&mut s, "tc-1", EffectStatus::Queued);
        queue(&mut s, EffectKind::LlmCall, "call-1");
        queue(&mut s, EffectKind::ToolCall, "tc-1");

        assert_eq!(plan(&s, epoch()), vec![], "strict head of line");
        assert_eq!(
            waiting_on(&s)[&(EffectKind::LlmCall, "call-1".to_string())],
            vec!["connector_sync:mcp.conn-1".to_string()],
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
                connection_id: ConnectionPath::Mcp("conn-1".into())
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
                    reasoning: None,
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
        let mut s = state();
        s.phase = TurnPhase::Active {
            turn_id: "turn-2".to_string(),
        };
        add_decision(&mut s, "d-2", EffectStatus::Queued, deferred("turn-2"));
        let at = epoch() + chrono::Duration::seconds(5);
        let t = &mut s.effect_mut(EffectKind::Decision, "d-2").unwrap().tracking;
        t.retry_policy = RetryPolicy::default_for(RetryTarget::Decision);
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
    fn a_running_subagent_is_swept_once_its_total_lapses() {
        let mut s = state();
        add_running_subagent(&mut s, "child-1", Some(60));
        let due = epoch() + chrono::Duration::seconds(60);

        assert_eq!(
            plan(&s, due - chrono::Duration::seconds(1)),
            vec![],
            "a child turn may legitimately run long; only the total settles it"
        );
        assert_eq!(
            plan(&s, due),
            vec![ScheduleStep::TimeOut {
                kind: EffectKind::Subagent,
                id: "child-1".to_string(),
            }],
            "the whole-effect bound is the only thing that recovers a dead child"
        );
    }

    #[test]
    fn a_running_subagent_wakes_at_its_total() {
        let mut s = state();
        add_running_subagent(&mut s, "child-1", Some(60));
        assert_eq!(
            wake_at(&s, epoch()),
            Some(epoch() + chrono::Duration::seconds(60)),
            "nothing else would ever wake the parent"
        );
    }

    #[test]
    fn a_running_subagent_with_no_total_is_never_swept() {
        let mut s = state();
        add_running_subagent(&mut s, "child-1", None);
        let far = epoch() + chrono::Duration::days(365);
        assert_eq!(plan(&s, far), vec![], "unbounded by declaration");
        assert_eq!(wake_at(&s, far), None);
    }

    #[test]
    fn the_total_deadline_is_not_pushed_out_by_a_retry() {
        let mut s = state();
        add_running_subagent(&mut s, "child-1", Some(60));
        let started = epoch();
        let t = &mut s
            .effect_mut(EffectKind::Subagent, "child-1")
            .unwrap()
            .tracking;
        t.requeue();
        t.dispatch(started + chrono::Duration::seconds(60));

        assert_eq!(
            t.total_deadline(),
            Some(started + chrono::Duration::seconds(60)),
            "measured from the first dispatch, so retries cannot buy more time"
        );
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
