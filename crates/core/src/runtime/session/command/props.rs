//! A property test over random command sequences.
//!
//! Two ingredients make this cheap: `apply` is deterministic, and the schedule
//! has invariants. So for any sequence of commands the aggregate accepts:
//!
//! 1. the scheduling invariants hold after every commit, and
//! 2. replaying the emitted events into a fresh state converges with the
//!    aggregate's own state, byte for byte.
//!
//! (2) is what catches a representation change that example-based tests miss:
//! it compares the whole persisted snapshot, not the fields a test thought to
//! look at.

use crate::protocol::StoredResult;
use std::collections::HashMap;

use chrono::{DateTime, Duration, Utc};
use proptest::prelude::*;
use proptest::test_runner::FileFailurePersistence;

use crate::protocol::LlmResponse;

use super::super::aggregate::{CommitContext, SessionAggregate};
use super::super::events::EventPayload;
use super::super::state::ApplyContext;
use super::*;
use crate::protocol::{
    AgentConfig, ClientMessage, ClientPayload, Content, DraftMessage, McpServer, OwnerKind, Role,
};
use crate::runtime::span::SpanContext;
use crate::runtime::Caller;

const TENANT: &str = "tenant-a";
const USER: &str = "user-1";

fn system() -> Caller {
    Caller::System {
        tenant_id: TENANT.to_string(),
    }
}

fn frontend() -> Caller {
    Caller::Frontend {
        tenant_id: TENANT.to_string(),
        user_id: USER.to_string(),
        attrs: HashMap::new(),
    }
}

/// Short deadlines and shallow retries, so a handful of wakes walks an effect
/// all the way from in-flight to terminally failed.
fn policy() -> RetryPolicy {
    RetryPolicy {
        attempt_timeout_secs: Some(5),
        total_timeout_secs: None,
        max_attempts: 2,
        backoff_base_secs: 1,
        backoff_max_secs: 1,
    }
}

/// What a settled decision asks for next.
#[derive(Debug, Clone)]
enum ActOp {
    CallLlm { worker: bool },
    CallTool,
    SpawnSubAgent,
    Interrupt,
    Done,
}

/// One command against the live session. Every index is taken modulo the
/// candidates in force, so an op is meaningful whatever state it lands in.
#[derive(Debug, Clone)]
enum Op {
    ClientMessage {
        /// Ask to be held for the next turn instead of refused mid-turn.
        queue: bool,
    },
    Decide {
        acts: Vec<ActOp>,
        append: bool,
        set_config: bool,
    },
    FailDecision {
        retryable: bool,
    },
    SettleLlm {
        index: usize,
        ok: bool,
        retryable: bool,
    },
    SettleTool {
        index: usize,
        ok: bool,
        retryable: bool,
    },
    SettleSubAgent {
        index: usize,
        ok: bool,
        retryable: bool,
    },
    SettleConnector {
        index: usize,
        ok: bool,
        retryable: bool,
    },
    Interrupt,
    Resume {
        index: usize,
    },
    Wake {
        advance_secs: u32,
    },
    ReconcileDispatch,
    Cancel,
}

fn act_op() -> impl Strategy<Value = ActOp> {
    prop_oneof![
        any::<bool>().prop_map(|worker| ActOp::CallLlm { worker }),
        Just(ActOp::CallTool),
        Just(ActOp::SpawnSubAgent),
        Just(ActOp::Interrupt),
        Just(ActOp::Done),
    ]
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        // Weighted towards the two commands that drive the loop; the rest keep
        // their default weight of 1.
        4 => (proptest::collection::vec(act_op(), 0..3), any::<bool>(), any::<bool>()).prop_map(
            |(acts, append, set_config)| Op::Decide { acts, append, set_config }
        ),
        3 => any::<bool>().prop_map(|queue| Op::ClientMessage { queue }),
        2 => (any::<usize>(), any::<bool>(), any::<bool>())
            .prop_map(|(index, ok, retryable)| Op::SettleLlm { index, ok, retryable }),
        2 => (any::<usize>(), any::<bool>(), any::<bool>())
            .prop_map(|(index, ok, retryable)| Op::SettleTool { index, ok, retryable }),
        2 => (any::<usize>(), any::<bool>(), any::<bool>())
            .prop_map(|(index, ok, retryable)| Op::SettleSubAgent { index, ok, retryable }),
        2 => (any::<usize>(), any::<bool>(), any::<bool>())
            .prop_map(|(index, ok, retryable)| Op::SettleConnector { index, ok, retryable }),
        1 => any::<bool>().prop_map(|retryable| Op::FailDecision { retryable }),
        1 => Just(Op::Interrupt),
        1 => any::<usize>().prop_map(|index| Op::Resume { index }),
        2 => (0u32..12).prop_map(|advance_secs| Op::Wake { advance_secs }),
        1 => Just(Op::ReconcileDispatch),
        // Terminal, so rare: it ends every sequence that reaches it.
        1 => Just(Op::Cancel),
    ]
}

/// The session under test, plus every event it has committed.
struct World {
    agg: SessionAggregate,
    now: DateTime<Utc>,
    log: Vec<(u64, DateTime<Utc>, EventPayload)>,
    next_id: u32,
    trace: Vec<String>,
    /// The instant the planner reads. Mirrors `Working`: the command clock,
    /// except under a wake, which supplies the scheduler's own.
    plan_now: DateTime<Utc>,
    /// Set once a cancel lands. Cancellation deliberately skips the schedule
    /// tail and leaves queued decisions in place, so the idempotence law stops
    /// describing the session from that point on.
    cancelled: bool,
}

impl World {
    fn new() -> Self {
        let mut world = World {
            agg: SessionAggregate::new(
                "sess-1".to_string(),
                TENANT.to_string(),
                SessionState::new("sess-1".to_string()),
            ),
            now: DateTime::parse_from_rfc3339("2026-01-01T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            log: Vec::new(),
            next_id: 0,
            trace: Vec::new(),
            plan_now: DateTime::parse_from_rfc3339("2026-01-01T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            cancelled: false,
        };
        world.run(
            CommandPayload::CreateSession {
                agent_id: "agent-1".to_string(),
                owner: SessionOwner {
                    kind: OwnerKind::Frontend,
                    audience: Default::default(),
                    tenant_id: TENANT.to_string(),
                    id: Some(USER.to_string()),
                    metadata: HashMap::new(),
                },
                ancestry: vec![],
                worker_retry: policy(),
            },
            &system(),
        );
        world
    }

    fn mint(&mut self) -> String {
        self.next_id += 1;
        format!("e-{}", self.next_id)
    }

    /// Handle and commit one command. A rejected command writes nothing, which
    /// is the behaviour under test as much as an accepted one is.
    fn run(&mut self, cmd: CommandPayload, caller: &Caller) {
        self.now += Duration::milliseconds(1);
        self.plan_now = match &cmd {
            CommandPayload::Wake { now } => *now,
            _ => self.now,
        };
        self.trace.push(format!("{cmd:?}"));
        let Ok(events) = self.agg.handle(cmd, caller, self.now) else {
            return;
        };
        let committed = self.agg.commit(
            events,
            &CommitContext {
                span: SpanContext::root(),
                occurred_at: self.now,
            },
        );
        for event in committed {
            self.log.push((event.seq, event.occurred_at, event.payload));
        }
    }

    /// Apply the whole log into a fresh state; it must equal the aggregate's.
    fn replayed(&self) -> SessionState {
        let mut state = SessionState::new("sess-1".to_string());
        for (seq, occurred_at, payload) in &self.log {
            state.apply(
                payload,
                &ApplyContext {
                    occurred_at: *occurred_at,
                    sequence: *seq,
                },
            );
        }
        state
    }

    /// Ids of the effects a settle can name, sorted so a shrunk sequence
    /// reproduces exactly.
    fn pending<'a>(&self, ids: impl Iterator<Item = &'a str>) -> Vec<String> {
        let mut ids: Vec<String> = ids.map(str::to_string).collect();
        ids.sort();
        ids
    }

    fn pending_llm(&self) -> Vec<String> {
        self.pending(
            self.agg
                .state
                .effects_of(EffectKind::LlmCall)
                .filter(|c| c.tracking.status() == EffectStatus::Pending)
                .map(|c| c.id.as_str()),
        )
    }

    fn pending_tool(&self) -> Vec<String> {
        self.pending(
            self.agg
                .state
                .effects_of(EffectKind::ToolCall)
                .filter(|c| c.tracking.status() == EffectStatus::Pending)
                .map(|c| c.id.as_str()),
        )
    }

    fn pending_sub_agent(&self) -> Vec<String> {
        self.pending(
            self.agg
                .state
                .effects_of(EffectKind::SubAgent)
                .filter(|c| c.tracking.status() == EffectStatus::Pending)
                .map(|c| c.id.as_str()),
        )
    }

    fn pending_connector(&self) -> Vec<String> {
        self.pending(
            self.agg
                .state
                .effects_of(EffectKind::ConnectorSync)
                .filter(|c| c.tracking.status() == EffectStatus::Pending)
                .map(|c| c.id.as_str()),
        )
    }

    fn live_decision(&self) -> Option<String> {
        let mut ids = self.pending(
            self.agg
                .state
                .effects_of(EffectKind::Decision)
                .filter(|d| d.tracking.status() == EffectStatus::Pending)
                .map(|d| d.id.as_str()),
        );
        ids.pop()
    }

    /// The active branch as a client would echo it back.
    fn recorded(&self) -> Vec<DraftMessage> {
        let tree = self.agg.state.message_tree();
        match tree.head_id.as_deref() {
            Some(head) => tree
                .path_to(head)
                .into_iter()
                .map(DraftMessage::from)
                .collect(),
            None => Vec::new(),
        }
    }

    fn message(&mut self, role: Role) -> DraftMessage {
        let text = self.mint();
        DraftMessage {
            id: None,
            role,
            content: Some(Content::Text(text)),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning: None,
        }
    }

    fn actions(&mut self, acts: Vec<ActOp>) -> Vec<Action> {
        acts.into_iter()
            .map(|a| match a {
                ActOp::CallLlm { worker } => Action::CallLlm {
                    llm: "claude".to_string(),
                    format: None,
                    id: self.mint(),
                    request: LlmRequest {
                        model: "test-model".to_string(),
                        messages: vec![],
                        tools: None,
                        temperature: None,
                        max_completion_tokens: None,
                        reasoning: None,
                    },
                    stream: false,
                    retry: policy(),
                    handler: if worker {
                        LlmHandler::Worker
                    } else {
                        LlmHandler::Server
                    },
                },
                ActOp::CallTool => Action::CallTool {
                    id: self.mint(),
                    name: "a_tool".to_string(),
                    arguments: "{}".to_string(),
                    retry: Some(policy().as_override()),
                },
                ActOp::SpawnSubAgent => {
                    let session_id = self.mint();
                    let tool_call_id = self.mint();
                    Action::SpawnSubAgent {
                        session_id,
                        agent_id: "child".to_string(),
                        tool_call_id,
                        message: None,
                        retry: policy(),
                    }
                }
                ActOp::Interrupt => Action::Interrupt {
                    interrupt_id: self.mint(),
                    reason: "pause".to_string(),
                    payload: serde_json::Value::Null,
                },
                ActOp::Done => Action::Done {
                    data: serde_json::Value::Null,
                },
            })
            .collect()
    }

    /// A config naming one connection, rewritten each time so the write always
    /// lands; the connection id is constant, so only the first costs a fetch.
    fn config(&mut self) -> AgentConfig {
        let model = self.mint();
        AgentConfig {
            llm: Some("claude".to_string()),
            model,
            system: None,
            retry: None,
            tools: Vec::new(),
            sub_agents: Vec::new(),
            mcp: vec![McpServer {
                id: "conn-1".to_string(),
                tools: None,
                auth_failure: Default::default(),
                approve: Default::default(),
            }],
            defer_tools: None,
            announce_mcp: Default::default(),
            plugins: Vec::new(),
            effort: None,
        }
    }

    fn step(&mut self, op: Op) {
        match op {
            Op::ClientMessage { queue } => {
                let message = self.message(Role::User);
                let turn_id = self.mint();
                self.run(
                    CommandPayload::SubmitClientPayload {
                        payload: ClientPayload::Message(ClientMessage {
                            message,
                            stream: false,
                        }),
                        turn: TurnTarget::Open(turn_id),
                        queue,
                    },
                    &frontend(),
                );
            }
            Op::Decide {
                acts,
                append,
                set_config,
            } => {
                let Some(decision_id) = self.live_decision() else {
                    return;
                };
                let transcript = if append {
                    let mut t = self.recorded();
                    let msg = self.message(Role::Assistant);
                    t.push(msg);
                    t
                } else {
                    Vec::new()
                };
                let actions = self.actions(acts);
                let agent = set_config.then(|| self.config());
                self.run(
                    CommandPayload::SubmitWorkerDecision {
                        decision_id,
                        transcript,
                        actions,
                        state: None,
                        agent,
                        channels: Default::default(),
                    },
                    &system(),
                );
            }
            Op::FailDecision { retryable } => {
                let Some(decision_id) = self.live_decision() else {
                    return;
                };
                self.run(
                    CommandPayload::settle(
                        EffectKind::Decision,
                        decision_id,
                        None,
                        SettleError::new(ErrorInfo::internal("boom"), retryable),
                    ),
                    &system(),
                );
            }
            Op::SettleLlm {
                index,
                ok,
                retryable,
            } => {
                let ids = self.pending_llm();
                let Some(call_id) = pick(&ids, index) else {
                    return;
                };
                let cmd = if ok {
                    CommandPayload::settle(
                        EffectKind::LlmCall,
                        call_id,
                        None,
                        Outcome::Llm(Box::new(LlmResponse {
                            model: "test-model".to_string(),
                            content: Some("ok".to_string()),
                            tool_calls: vec![],
                            finish_reason: None,
                            usage: None,
                            cost: None,
                            images: vec![],
                            reasoning: None,
                        })),
                    )
                } else {
                    CommandPayload::settle(
                        EffectKind::LlmCall,
                        call_id,
                        None,
                        SettleError::new(ErrorInfo::internal("boom"), retryable),
                    )
                };
                self.run(cmd, &system());
            }
            Op::SettleTool {
                index,
                ok,
                retryable,
            } => {
                let ids = self.pending_tool();
                let Some(tool_call_id) = pick(&ids, index) else {
                    return;
                };
                let cmd = if ok {
                    CommandPayload::settle(
                        EffectKind::ToolCall,
                        tool_call_id,
                        None,
                        Outcome::Tool {
                            result: StoredResult::text("ok".to_string()),
                        },
                    )
                } else {
                    CommandPayload::settle(
                        EffectKind::ToolCall,
                        tool_call_id,
                        None,
                        SettleError::new(ErrorInfo::internal("boom"), retryable),
                    )
                };
                self.run(cmd, &system());
            }
            Op::SettleSubAgent {
                index,
                ok,
                retryable,
            } => {
                let ids = self.pending_sub_agent();
                let Some(session_id) = pick(&ids, index) else {
                    return;
                };
                let cmd = if ok {
                    CommandPayload::CompleteSubAgentTurn {
                        session_id,
                        agent_id: "child".to_string(),
                        turn_id: "t".to_string(),
                        data: serde_json::json!("done"),
                        cost: Default::default(),
                        token_usage: Default::default(),
                    }
                } else {
                    CommandPayload::settle(
                        EffectKind::SubAgent,
                        session_id,
                        None,
                        SettleError::new(ErrorInfo::internal("boom"), retryable),
                    )
                };
                self.run(cmd, &system());
            }
            Op::SettleConnector {
                index,
                ok,
                retryable,
            } => {
                let ids = self.pending_connector();
                let Some(connection_id) = pick(&ids, index) else {
                    return;
                };
                let cmd = if ok {
                    CommandPayload::settle(
                        EffectKind::ConnectorSync,
                        connection_id,
                        None,
                        Outcome::Connector {
                            prefix: None,
                            tools: vec![],
                            instructions: None,
                        },
                    )
                } else {
                    CommandPayload::settle(
                        EffectKind::ConnectorSync,
                        connection_id,
                        None,
                        SettleError::new(ErrorInfo::internal("boom"), retryable),
                    )
                };
                self.run(cmd, &system());
            }
            Op::Interrupt => {
                let interrupt_id = self.mint();
                self.run(
                    CommandPayload::Interrupt {
                        interrupt_id,
                        reason: "pause".to_string(),
                        payload: serde_json::Value::Null,
                    },
                    &frontend(),
                );
            }
            Op::Resume { index } => {
                let ids = self.pending(
                    self.agg
                        .state
                        .open_interrupts
                        .iter()
                        .map(|i| i.interrupt_id.as_str()),
                );
                let Some(interrupt_id) = pick(&ids, index) else {
                    return;
                };
                self.run(
                    CommandPayload::ResumeInterrupt {
                        interrupt_id,
                        payload: serde_json::Value::Null,
                    },
                    &system(),
                );
            }
            Op::Wake { advance_secs } => {
                self.now += Duration::seconds(i64::from(advance_secs));
                let now = self.now;
                self.run(CommandPayload::Wake { now }, &system());
            }
            Op::ReconcileDispatch => {
                self.run(CommandPayload::ReconcileDispatch, &system());
            }
            Op::Cancel => {
                self.run(CommandPayload::CancelSession, &system());
                self.cancelled = true;
            }
        }
    }
}

/// `None` when nothing is a candidate; otherwise an index into the candidates.
fn pick(ids: &[String], index: usize) -> Option<String> {
    if ids.is_empty() {
        return None;
    }
    Some(ids[index % ids.len()].clone())
}

/// A snapshot round trip: what the store would write, read back.
fn round_trip(state: &SessionState) -> SessionState {
    let json = serde_json::to_string(state).expect("a session state serializes");
    serde_json::from_str(&json).expect("and deserializes")
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 96,
        failure_persistence: Some(Box::new(FileFailurePersistence::SourceParallel(
            "tests/proptest-regressions"
        ))),
        ..ProptestConfig::default()
    })]

    /// Whatever the sequence, the schedule stays consistent, the event log is
    /// the whole truth, and a snapshot round trip changes nothing.
    #[test]
    fn random_command_sequences_hold_every_law(
        ops in proptest::collection::vec(op(), 1..24)
    ) {
        let mut world = World::new();
        for op in ops {
            world.step(op);
            let state = &world.agg.state;
            let commands = world.trace.join("\n");

            // The invariants hold on committed state, not only mid-command.
            if let Err(violation) = state.check_invariants() {
                prop_assert!(false, "{violation}\ncommands:\n{commands}");
            }

            // Replay convergence: the log rebuilds the state exactly.
            prop_assert_eq!(
                serde_json::to_value(world.replayed()).unwrap(),
                serde_json::to_value(state).unwrap(),
                "replay diverged\ncommands:\n{}", commands
            );

            // Snapshot round trip: what is persisted is all the state there is.
            let reloaded = round_trip(state);
            prop_assert_eq!(
                serde_json::to_value(&reloaded).unwrap(),
                serde_json::to_value(state).unwrap(),
                "snapshot round trip changed the state\ncommands:\n{}", commands
            );

            // Law — the plan is state-only: it reads nothing but the state and
            // the clock, so a round trip through the store cannot change it.
            let live_plan = schedule::plan(state, world.plan_now);
            prop_assert_eq!(
                &live_plan, &schedule::plan(&reloaded, world.plan_now),
                "the plan read differently after a round trip\ncommands:\n{}", commands
            );

            // Law — idempotence: every command ends by running the schedule
            // tail, so a second look at the same state must find no work ready.
            // Sweep steps are the exception by policy, not by accident: the
            // tail takes one per command, so more overdue items may remain and
            // `wake_at` reschedules until none do. `CancelSession` is the one
            // command that deliberately skips the tail, so it is the one state
            // this law does not describe.
            if !world.cancelled {
                let ready: Vec<&ScheduleStep> =
                    live_plan.iter().filter(|s| !s.is_sweep()).collect();
                prop_assert!(
                    ready.is_empty(),
                    "the schedule tail left work ready: {ready:?}\ncommands:\n{commands}"
                );
            }
        }
    }
}
