//! Event-trace snapshots.
//!
//! Each canonical flow is driven through the aggregate and its ordered event
//! names asserted verbatim. The traces read as stories, and every later stage
//! must leave them identical — a diff here is a behaviour change, not a test to
//! update. (Stage 2 renames events; there the diff is a reviewed rename, and
//! nothing else.)
//!
//! [`traces_cover_every_command_arm`] keeps the set honest: `CommandPayload` is
//! matched exhaustively, so a new command fails to compile until a trace
//! exercises it.

use std::collections::{BTreeSet, HashMap};

use chrono::{DateTime, Duration, Utc};

use crate::protocol::{ErrorCode, LlmResponse};

use super::super::aggregate::{CommitContext, SessionAggregate};
use super::super::decision::Trigger;
use super::super::events::EventPayload;
use super::*;
use crate::connectors::{AuthNeed, RemoteTool};
use crate::protocol::{
    AgentConfig, AgentTool, ClientMessage, ClientPayload, Content, DraftMessage, Handler,
    McpServer, Role, ToolCall, ToolCallFunction,
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

/// The serde `"type"` tag — the name a trace is written in.
fn name(payload: &EventPayload) -> String {
    serde_json::to_value(payload)
        .ok()
        .and_then(|v| v.get("type")?.as_str().map(str::to_string))
        .unwrap_or_else(|| "unknown".to_string())
}

/// Which command a payload is. Matched exhaustively on purpose: a new
/// `CommandPayload` arm breaks this build until a trace covers it.
fn arm(cmd: &CommandPayload) -> &'static str {
    match cmd {
        CommandPayload::CreateSession { .. } => "CreateSession",
        CommandPayload::SubmitClientPayload { .. } => "SubmitClientPayload",
        CommandPayload::SendMessage { .. } => "SendMessage",
        CommandPayload::RequestLlmCall { .. } => "RequestLlmCall",
        // One command, but coverage stays per outcome: a settle that no trace
        // exercises is as much of a hole as an uncovered command was.
        CommandPayload::SettleEffect { kind, outcome, .. } => match (kind, outcome) {
            (EffectKind::LlmCall, Outcome::Error(_)) => "FailLlmCall",
            (EffectKind::LlmCall, _) => "CompleteLlmCall",
            (EffectKind::ToolCall, Outcome::Error(_)) => "FailToolCall",
            (EffectKind::ToolCall, _) => "CompleteToolCall",
            (EffectKind::ConnectorSync, Outcome::Error(_)) => "FailConnectorSync",
            (EffectKind::ConnectorSync, _) => "CompleteConnectorSync",
            (EffectKind::SubAgent, Outcome::Error(_)) => "FailSubAgent",
            (EffectKind::SubAgent, _) => "StartSubAgent",
            (EffectKind::Decision, _) => "FailWorkerDecision",
            (EffectKind::TurnEnd, _) => "SettleTurnEnd",
        },
        CommandPayload::RequestToolCall { .. } => "RequestToolCall",
        CommandPayload::RequestSubAgent { .. } => "RequestSubAgent",
        CommandPayload::CompleteSubAgentTurn { .. } => "CompleteSubAgentTurn",
        CommandPayload::Interrupt { .. } => "Interrupt",
        CommandPayload::ResumeInterrupt { .. } => "ResumeInterrupt",
        CommandPayload::SubmitWorkerDecision { .. } => "SubmitWorkerDecision",
        CommandPayload::CancelSession => "CancelSession",
        CommandPayload::FinishTurn { .. } => "FinishTurn",
        CommandPayload::CompleteTurn => "CompleteTurn",
        CommandPayload::Wake { .. } => "Wake",
        CommandPayload::ReconcileDispatch => "ReconcileDispatch",
    }
}

fn user_message(text: &str) -> DraftMessage {
    DraftMessage {
        id: None,
        role: Role::User,
        content: Some(Content::Text(text.to_string())),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }
}

fn llm_request() -> LlmRequest {
    LlmRequest {
        model: "test-model".to_string(),
        messages: vec![],
        tools: None,
        temperature: None,
        max_completion_tokens: None,
        reasoning: None,
    }
}

fn llm_response(tool_calls: Vec<ToolCall>) -> LlmResponse {
    LlmResponse {
        model: "test-model".to_string(),
        content: None,
        tool_calls,
        finish_reason: None,
        usage: None,
        cost: None,
        images: vec![],
    }
}

fn tool_call(id: &str, name: &str) -> ToolCall {
    ToolCall {
        id: id.to_string(),
        call_type: "function".to_string(),
        function: ToolCallFunction {
            name: name.to_string(),
            arguments: "{}".to_string(),
        },
    }
}

fn config() -> AgentConfig {
    AgentConfig {
        llm: Some("claude".to_string()),
        model: "test-model".to_string(),
        system: None,
        retry: None,
        tools: Vec::new(),
        sub_agents: Vec::new(),
        mcp: Vec::new(),
        tool_discovery: None,
    }
}

fn config_with_client_tool(tool: &str) -> AgentConfig {
    AgentConfig {
        tools: vec![AgentTool {
            name: tool.to_string(),
            description: "d".to_string(),
            input: None,
            output: None,
            handler: Some(Handler::Client),
        }],
        ..config()
    }
}

fn config_with_mcp(connection: &str) -> AgentConfig {
    AgentConfig {
        mcp: vec![McpServer {
            id: connection.to_string(),
            tools: None,
            auth_failure: Default::default(),
        }],
        ..config()
    }
}

/// A worker-decision policy with room for one redelivery.
fn retrying() -> RetryPolicy {
    RetryPolicy {
        attempt_timeout_secs: Some(60),
        total_timeout_secs: None,
        max_attempts: 2,
        backoff_base_secs: 1,
        backoff_max_secs: 1,
    }
}

/// A session driven command by command, accumulating the names of every event
/// it commits and of every command it runs.
struct Trace {
    agg: SessionAggregate,
    names: Vec<String>,
    arms: BTreeSet<&'static str>,
    now: DateTime<Utc>,
}

impl Trace {
    /// A fresh session with its `session.start` decision live.
    fn create() -> Self {
        Self::create_with(RetryPolicy::no_retry())
    }

    fn create_with(worker_retry: RetryPolicy) -> Self {
        let mut trace = Trace {
            agg: SessionAggregate::new(
                "sess-1".to_string(),
                TENANT.to_string(),
                SessionState::new("sess-1".to_string()),
            ),
            names: Vec::new(),
            arms: BTreeSet::new(),
            // A fixed epoch: traces must not depend on the wall clock.
            now: DateTime::parse_from_rfc3339("2026-01-01T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
        };
        trace.run(
            CommandPayload::CreateSession {
                agent_id: "agent-1".to_string(),
                owner: SessionOwner {
                    tenant_id: TENANT.to_string(),
                    id: Some(USER.to_string()),
                    metadata: HashMap::new(),
                },
                ancestry: vec![],
                worker_retry,
            },
            &system(),
        );
        trace
    }

    /// Handle and commit one command, exactly as production `execute` does.
    #[track_caller]
    fn run(&mut self, cmd: CommandPayload, caller: &Caller) -> Vec<EventPayload> {
        self.now += Duration::milliseconds(1);
        self.arms.insert(arm(&cmd));
        let events = self
            .agg
            .handle(cmd, caller, self.now)
            .expect("command rejected");
        self.agg.commit(
            events.clone(),
            &CommitContext {
                span: SpanContext::root(),
                occurred_at: self.now,
            },
        );
        self.names.extend(events.iter().map(name));
        events
    }

    /// A command expected to be rejected: it records the arm but writes nothing.
    #[track_caller]
    fn reject(&mut self, cmd: CommandPayload, caller: &Caller) -> SessionError {
        self.arms.insert(arm(&cmd));
        self.agg
            .handle(cmd, caller, self.now)
            .expect_err("command should have been rejected")
    }

    /// Move the clock without emitting anything, so the next `Wake` is due.
    fn advance(&mut self, secs: i64) {
        self.now += Duration::seconds(secs);
    }

    fn wake(&mut self) {
        let now = self.now + Duration::milliseconds(1);
        self.run(CommandPayload::Wake { now }, &system());
    }

    fn submit(&mut self, text: &str, turn_id: Option<&str>) {
        self.run(self.submission(text, turn_id, false), &frontend());
    }

    /// A submit that asks to be held for the next turn rather than refused.
    fn submit_queued(&mut self, text: &str, turn_id: &str) {
        self.run(self.submission(text, Some(turn_id), true), &frontend());
    }

    fn submission(&self, text: &str, turn_id: Option<&str>, queue: bool) -> CommandPayload {
        CommandPayload::SubmitClientPayload {
            payload: ClientPayload::Message(ClientMessage {
                message: user_message(text),
                stream: false,
            }),
            turn: turn_id.map_or(TurnTarget::Detached, |t| TurnTarget::Open(t.to_string())),
            queue,
        }
    }

    /// The one live decision. The slot admits at most one, so this is unambiguous.
    #[track_caller]
    fn live_decision(&self) -> String {
        let mut live = self
            .agg
            .state
            .effects_of(EffectKind::Decision)
            .filter(|d| d.tracking.status() == EffectStatus::Pending);
        let id = live.next().map(|d| d.id.clone()).expect("a live decision");
        assert!(live.next().is_none(), "more than one live decision");
        id
    }

    #[track_caller]
    fn live_trigger(&self) -> Trigger {
        let id = self.live_decision();
        self.agg.state.worker_decision(&id).unwrap().trigger.clone()
    }

    /// The recorded conversation on the active branch, as a client would echo
    /// it back. A transcript built on this shares the tree's prefix, so only
    /// what the test appends is written.
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

    /// Settle the live decision with a transcript and actions.
    fn decide(&mut self, transcript: Vec<DraftMessage>, actions: Vec<Action>) {
        self.decide_with(transcript, actions, None);
    }

    #[track_caller]
    fn decide_with(
        &mut self,
        transcript: Vec<DraftMessage>,
        actions: Vec<Action>,
        agent: Option<AgentConfig>,
    ) {
        let decision_id = self.live_decision();
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

    /// Settle the live decision, appending `message` to what is recorded.
    fn decide_appending(&mut self, message: DraftMessage, actions: Vec<Action>) {
        let mut transcript = self.recorded();
        transcript.push(message);
        self.decide(transcript, actions);
    }

    #[track_caller]
    fn assert_trace(&self, expected: &[&str]) {
        if self.names != expected {
            panic!(
                "event trace changed.\nactual:\n{}\n",
                self.names
                    .iter()
                    .map(|n| format!("        \"{n}\","))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
        }
    }
}

fn call_llm(id: &str, retry: RetryPolicy) -> Action {
    Action::CallLlm {
        llm: "claude".to_string(),
        format: None,
        id: id.to_string(),
        request: llm_request(),
        stream: false,
        retry,
        handler: LlmHandler::Server,
    }
}

fn call_tool(id: &str, name: &str) -> Action {
    Action::CallTool {
        id: id.to_string(),
        name: name.to_string(),
        arguments: "{}".to_string(),
        retry: None,
    }
}

fn done() -> Action {
    Action::Done {
        data: serde_json::Value::Null,
    }
}

// ── The flows ───────────────────────────────────────────────────────────────
//
// Each is a function, so `traces_cover_every_command_arm` can run them all and
// union the commands they issue.

/// create → session.start → client payload → LLM → tool → `*.finished` → done
/// → finalize → SessionDone.
fn flow_full_loop() -> Trace {
    let mut t = Trace::create();
    // The worker declares its identity; nothing else happens yet.
    t.decide(vec![], vec![]);

    // A user message opens a turn and wakes the loop.
    t.submit("hi", Some("turn-1"));

    // The worker records the user turn and asks for a model call.
    t.decide_appending(
        user_message("hi"),
        vec![call_llm("call-1", RetryPolicy::no_retry())],
    );

    // The provider answers with a tool call.
    t.run(
        CommandPayload::settle(
            EffectKind::LlmCall,
            "call-1".to_string(),
            Some(0),
            Outcome::Llm(Box::new(llm_response(vec![tool_call(
                "tc-1",
                "get_weather",
            )]))),
        ),
        &system(),
    );

    // `llm.finished`: the worker records the assistant node and calls the tool.
    assert!(matches!(
        t.live_trigger(),
        Trigger::LlmFinished { ok: true, .. }
    ));
    let assistant = DraftMessage {
        id: Some("call-1".to_string()),
        role: Role::Assistant,
        content: None,
        tool_calls: Some(vec![tool_call("tc-1", "get_weather")]),
        tool_call_id: None,
        name: None,
    };
    t.decide_appending(assistant, vec![call_tool("tc-1", "get_weather")]);

    // `tool.execute`: the worker runs the tool and reports the result.
    assert!(matches!(t.live_trigger(), Trigger::ToolExecute { .. }));
    t.decide(
        vec![],
        vec![Action::ToolResult {
            id: "tc-1".to_string(),
            attempt: Some(0),
            result: "sunny".to_string(),
        }],
    );

    // `tool.finished`: the worker records the answer and ends the turn.
    assert!(matches!(
        t.live_trigger(),
        Trigger::ToolFinished { ok: true, .. }
    ));
    let tool_answer = DraftMessage {
        id: None,
        role: Role::Tool,
        content: Some(Content::Text("sunny".to_string())),
        tool_calls: None,
        tool_call_id: Some("tc-1".to_string()),
        name: None,
    };
    t.decide_appending(
        tool_answer,
        vec![Action::Done {
            data: serde_json::json!({"ok": true}),
        }],
    );

    // `turn.finished`: the finalizer echoes `done` and the run completes.
    assert!(matches!(t.live_trigger(), Trigger::TurnFinished { .. }));
    t.decide(vec![], vec![done()]);
    t
}

#[test]
fn trace_full_loop() {
    let t = flow_full_loop();
    assert_eq!(
        t.agg.state.data,
        serde_json::json!({"ok": true}),
        "the turn's output survives finalization"
    );
    t.assert_trace(&[
        "session.created",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "turn.started",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "message.new",
        "llm.call.requested",
        "llm.call.dispatched",
        "llm.call.completed",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "message.new",
        "tool.call.requested",
        "tool.call.dispatched",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "tool.call.completed",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "message.new",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "turn.completed",
        "session.done",
    ]);
}

/// A connector fetch gates the LLM call queued behind it: the call waits in the
/// queue until the fetch settles, then dispatch merges the fetched tools in.
fn flow_connector_gating() -> Trace {
    let mut t = Trace::create();
    // The worker's config names a connection, so settling the start decision
    // requests the fetch.
    t.decide_with(vec![], vec![], Some(config_with_mcp("conn-1")));
    assert!(
        t.agg
            .state
            .tracking(EffectKind::ConnectorSync, "conn-1")
            .unwrap()
            .is_in_flight(),
        "the fetch is in flight"
    );

    t.submit("hi", Some("turn-1"));

    // The client's decision is queued behind the fetch, not dispatched.
    assert!(
        !t.agg.state.has_pending_worker_decision(),
        "the decision waits for the fetch"
    );
    assert_eq!(
        super::schedule::waiting_on(&t.agg.state).values().next(),
        Some(&vec!["connector_sync:conn-1".to_string()]),
        "and says so"
    );

    // The fetch settles; the queued decision goes live at the same walk.
    t.run(
        CommandPayload::settle(
            EffectKind::ConnectorSync,
            "conn-1".to_string(),
            Some(0),
            Outcome::Connector {
                prefix: None,
                tools: vec![RemoteTool {
                    name: "lookup".to_string(),
                    description: "d".to_string(),
                    input: None,
                    output: None,
                    annotations: Default::default(),
                }],
                instructions: None,
            },
        ),
        &system(),
    );

    t.decide_appending(
        user_message("hi"),
        vec![call_llm("call-1", RetryPolicy::no_retry())],
    );
    t
}

#[test]
fn trace_connector_fetch_gates_an_llm_call() {
    let t = flow_connector_gating();
    let merged = t
        .agg
        .state
        .llm_call("call-1")
        .unwrap()
        .spec
        .tools
        .clone()
        .unwrap_or_default();
    assert_eq!(
        merged.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
        vec!["lookup"],
        "dispatch merges the connector tools in force"
    );
    t.assert_trace(&[
        "session.created",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "connector.sync.requested",
        "agent.updated",
        "turn.started",
        "decision.queued",
        "connector.sync.completed",
        "decision.dispatched",
        "decision.completed",
        "message.new",
        "llm.call.requested",
        "llm.call.dispatched",
    ]);
}

/// A terminally failed fetch is settled, so it stops gating: the turn goes
/// ahead without those tools.
fn flow_connector_failure_releases_the_gate() -> Trace {
    let mut t = Trace::create_with(retrying());
    t.decide_with(vec![], vec![], Some(config_with_mcp("conn-1")));
    t.submit("hi", Some("turn-1"));
    t.run(
        CommandPayload::settle(
            EffectKind::ConnectorSync,
            "conn-1".to_string(),
            Some(0),
            SettleError::new(ErrorInfo::internal("unreachable"), false)
                .auth(Some(AuthNeed::Reauthorize)),
        ),
        &system(),
    );
    t
}

#[test]
fn trace_a_failed_fetch_releases_the_gate() {
    let t = flow_connector_failure_releases_the_gate();
    assert!(
        t.agg.state.has_pending_worker_decision(),
        "the decision the fetch was holding goes live"
    );
    t.assert_trace(&[
        "session.created",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "connector.sync.requested",
        "agent.updated",
        "turn.started",
        "decision.queued",
        "connector.sync.errored",
        "decision.dispatched",
    ]);
}

/// interrupt raise → LLM voiding → resume → the resume decision jumps the queue
/// ahead of what parked behind the interrupt.
fn flow_interrupt_and_resume() -> Trace {
    let mut t = Trace::create();
    t.decide_with(vec![], vec![], Some(config_with_client_tool("ask_user")));
    t.submit("hi", Some("turn-1"));

    // One client-handled tool call (the client owes an answer) and one LLM call
    // (which the interrupt will void).
    t.decide_appending(
        user_message("hi"),
        vec![
            call_tool("tc-1", "ask_user"),
            call_llm("call-1", RetryPolicy::no_retry()),
        ],
    );

    t.run(
        CommandPayload::Interrupt {
            interrupt_id: "int-1".to_string(),
            reason: "approval".to_string(),
            payload: serde_json::Value::Null,
        },
        &frontend(),
    );

    // A user message on the parked branch is refused; nothing is written.
    let err = t.reject(
        CommandPayload::SubmitClientPayload {
            payload: ClientPayload::Message(ClientMessage {
                message: user_message("again"),
                stream: false,
            }),
            turn: TurnTarget::Detached,
            queue: false,
        },
        &frontend(),
    );
    assert!(matches!(err, SessionError::SessionInterrupted));

    // The client answers its tool call anyway — effects still settle while
    // parked — so a `tool.finished` decision queues behind the interrupt.
    t.run(
        CommandPayload::settle(
            EffectKind::ToolCall,
            "tc-1".to_string(),
            Some(0),
            Outcome::Tool {
                result: "yes".to_string(),
            },
        ),
        &frontend(),
    );
    assert!(
        !t.agg.state.has_pending_worker_decision(),
        "the parked branch dispatches nothing"
    );

    t.run(
        CommandPayload::ResumeInterrupt {
            interrupt_id: "int-1".to_string(),
            payload: serde_json::json!({"approved": true}),
        },
        &frontend(),
    );
    t
}

#[test]
fn trace_interrupt_and_resume() {
    let t = flow_interrupt_and_resume();
    assert!(
        matches!(t.live_trigger(), Trigger::InterruptResumed { .. }),
        "the resume answers first, ahead of the tool.finished queued before it"
    );
    t.assert_trace(&[
        "session.created",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "agent.updated",
        "turn.started",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "message.new",
        "tool.call.requested",
        "llm.call.requested",
        "tool.call.dispatched",
        "llm.call.dispatched",
        "session.interrupted",
        "call.voided",
        "tool.call.completed",
        "decision.queued",
        "session.interrupt_resumed",
        "decision.queued",
        "decision.dispatched",
    ]);
}

/// A second message arrives mid-turn and asks to queue: it is accepted, parks
/// as a decision, and becomes the next turn the moment the first one completes.
fn flow_queued_turn() -> Trace {
    let mut t = Trace::create();
    t.decide(vec![], vec![]);
    t.submit("hi", Some("turn-1"));

    // Mid-turn, and taken: no turn starts, nothing dispatches.
    t.submit_queued("and another thing", "turn-2");
    assert_eq!(
        t.agg.state.phase.turn_id(),
        Some("turn-1"),
        "the running turn keeps the phase"
    );

    // Redelivery of either turn is refused, so a retrying transport cannot ask
    // the same question twice.
    assert!(matches!(
        t.reject(t.submission("hi", Some("turn-1"), true), &frontend()),
        SessionError::TurnAlreadyActive { .. }
    ));
    assert!(matches!(
        t.reject(
            t.submission("and another thing", Some("turn-2"), true),
            &frontend()
        ),
        SessionError::TurnAlreadyActive { .. }
    ));

    // The first turn runs to its end. Completing it releases the phase, and the
    // same batch starts the queued turn.
    t.decide_appending(user_message("hi"), vec![done()]);
    t.decide(vec![], vec![done()]);
    t
}

#[test]
fn trace_queued_turn() {
    let t = flow_queued_turn();
    assert_eq!(
        t.agg.state.phase.turn_id(),
        Some("turn-2"),
        "the queued turn is the one running now"
    );
    assert!(
        matches!(t.live_trigger(), Trigger::ClientMessage { .. }),
        "delivered as an ordinary client message"
    );
    t.assert_trace(&[
        "session.created",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "turn.started",
        "decision.queued",
        "decision.dispatched",
        // The queued submit writes one event and starts nothing.
        "decision.queued",
        "decision.completed",
        "message.new",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "turn.completed",
        "session.done",
        // One batch: the first turn ends and the queued one takes the phase.
        "turn.started",
        "decision.dispatched",
    ]);
}

/// timeout → retry → exhaustion → a terminal `llm.finished` carrying the error.
fn flow_timeout_to_exhaustion() -> Trace {
    let mut t = Trace::create();
    t.decide(vec![], vec![]);
    t.submit("hi", Some("turn-1"));

    // One retry, a one-second deadline: the first wake times it out, the second
    // re-issues it, the third exhausts the policy.
    t.decide_appending(
        user_message("hi"),
        vec![call_llm(
            "call-1",
            RetryPolicy {
                attempt_timeout_secs: Some(1),
                total_timeout_secs: None,
                max_attempts: 2,
                backoff_base_secs: 1,
                backoff_max_secs: 1,
            },
        )],
    );

    t.advance(2);
    t.wake(); // deadline exceeded → errored, retry scheduled
    t.advance(2);
    t.wake(); // retry due → re-requested and re-dispatched
    t.advance(2);
    t.wake(); // deadline exceeded again → terminal, llm.finished queued

    assert!(matches!(
        t.live_trigger(),
        Trigger::LlmFinished { ok: false, .. }
    ));
    // The worker sees the failure and ends the turn.
    t.decide(vec![], vec![done()]);
    t.decide(vec![], vec![done()]);
    t
}

#[test]
fn trace_timeout_retry_then_exhaustion() {
    let t = flow_timeout_to_exhaustion();
    t.assert_trace(&[
        "session.created",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "turn.started",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "message.new",
        "llm.call.requested",
        "llm.call.dispatched",
        "llm.call.errored",
        "llm.call.requested",
        "llm.call.dispatched",
        "llm.call.errored",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "turn.completed",
        "session.done",
    ]);
}

/// A decision failure: a retryable one is redelivered at its backoff; the next
/// exhausts the policy and ends the run.
fn flow_decision_failure() -> Trace {
    let mut t = Trace::create_with(retrying());
    t.decide(vec![], vec![]);
    t.submit("hi", Some("turn-1"));

    let decision_id = t.live_decision();
    t.run(
        CommandPayload::settle(
            EffectKind::Decision,
            decision_id.clone(),
            None,
            SettleError::new(ErrorInfo::internal("worker down".to_string()), true),
        ),
        &system(),
    );
    assert!(
        !t.agg.state.has_pending_worker_decision(),
        "a failed decision leaves the slot"
    );

    // The backoff elapses and the same decision is redelivered.
    t.advance(2);
    t.wake();
    assert_eq!(t.live_decision(), decision_id, "the same decision, retried");

    t.run(
        CommandPayload::settle(
            EffectKind::Decision,
            decision_id,
            None,
            SettleError::new(ErrorInfo::internal("worker down".to_string()), true),
        ),
        &system(),
    );
    t
}

#[test]
fn trace_decision_retry_then_terminal_failure() {
    let t = flow_decision_failure();
    assert!(
        t.agg
            .state
            .effects_of(EffectKind::Decision)
            .next()
            .is_none(),
        "a terminally failed decision leaves the table"
    );
    t.assert_trace(&[
        "session.created",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "turn.started",
        "decision.queued",
        "decision.dispatched",
        "decision.errored",
        "decision.dispatched",
        "decision.errored",
        "turn.completed",
        "session.done",
    ]);
}

/// A parallel fan-out: two tool calls settle, and the first `tool.finished`
/// decision is told its sibling's result is still unrecorded, so it waits
/// instead of prompting against an incomplete transcript.
fn flow_parallel_fan_out() -> Trace {
    let mut t = Trace::create();
    t.decide(vec![], vec![]);
    t.submit("hi", Some("turn-1"));

    t.decide_appending(
        user_message("hi"),
        vec![call_tool("tc-1", "a"), call_tool("tc-2", "b")],
    );

    // Two `tool.execute` decisions queued; the slot admits one at a time.
    assert!(matches!(t.live_trigger(), Trigger::ToolExecute { .. }));
    t.decide(
        vec![],
        vec![Action::ToolResult {
            id: "tc-1".to_string(),
            attempt: Some(0),
            result: "one".to_string(),
        }],
    );
    t.decide(
        vec![],
        vec![Action::ToolResult {
            id: "tc-2".to_string(),
            attempt: Some(0),
            result: "two".to_string(),
        }],
    );
    t
}

#[test]
fn trace_parallel_fan_out_holds_a_sibling_result() {
    let t = flow_parallel_fan_out();
    let live = t.live_decision();
    assert!(matches!(t.live_trigger(), Trigger::ToolFinished { .. }));
    assert_eq!(
        t.agg.state.event_meta(t.now).pending_work(&live),
        1,
        "the sibling's tool.finished is queued and unrecorded"
    );
    t.assert_trace(&[
        "session.created",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "turn.started",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "message.new",
        "tool.call.requested",
        "tool.call.requested",
        "tool.call.dispatched",
        "decision.queued",
        "tool.call.dispatched",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "tool.call.completed",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "tool.call.completed",
        "decision.queued",
        "decision.dispatched",
    ]);
}

/// A worker that answers with an earlier prefix forks the branch: the head
/// rebases and work anchored off the retained path is voided.
fn flow_fork_voids_stranded_work() -> Trace {
    let mut t = Trace::create();
    t.decide(vec![], vec![]);
    t.submit("hi", Some("turn-1"));

    // u1 recorded, one model call against it.
    t.decide_appending(
        user_message("hi"),
        vec![call_llm("call-1", RetryPolicy::no_retry())],
    );
    t.run(
        CommandPayload::settle(
            EffectKind::LlmCall,
            "call-1".to_string(),
            Some(0),
            Outcome::Llm(Box::new(llm_response(vec![tool_call("tc-1", "a")]))),
        ),
        &system(),
    );

    // The assistant node is recorded and a tool call anchored on it dispatches.
    let assistant = DraftMessage {
        id: Some("call-1".to_string()),
        role: Role::Assistant,
        content: None,
        tool_calls: Some(vec![tool_call("tc-1", "a")]),
        tool_call_id: None,
        name: None,
    };
    t.decide_appending(assistant, vec![call_tool("tc-1", "a")]);

    // The worker now answers with only the first message: the head rebases to
    // u1 and the tool call anchored on the assistant node is stranded.
    let root: Vec<DraftMessage> = t.recorded().into_iter().take(1).collect();
    t.decide(root, vec![]);
    t
}

#[test]
fn trace_fork_voids_stranded_work() {
    let t = flow_fork_voids_stranded_work();
    assert_eq!(
        t.agg
            .state
            .effect(EffectKind::ToolCall, "tc-1")
            .unwrap()
            .tracking
            .status(),
        EffectStatus::Failed,
        "the stranded call is voided"
    );
    t.assert_trace(&[
        "session.created",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "turn.started",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "message.new",
        "llm.call.requested",
        "llm.call.dispatched",
        "llm.call.completed",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "message.new",
        "tool.call.requested",
        "tool.call.dispatched",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "head.moved",
        "call.voided",
    ]);
}

/// A delegation to a child session, both outcomes.
fn flow_sub_agent_delegation() -> Trace {
    let mut t = Trace::create();
    t.decide(vec![], vec![]);
    t.submit("hi", Some("turn-1"));

    t.decide_appending(
        user_message("hi"),
        vec![Action::SpawnSubAgent {
            session_id: "child-1".to_string(),
            agent_id: "child".to_string(),
            tool_call_id: "tc-1".to_string(),
            message: None,
            retry: RetryPolicy::no_retry(),
        }],
    );
    t.run(
        CommandPayload::settle(
            EffectKind::SubAgent,
            "child-1".to_string(),
            None,
            Outcome::SubAgentStarted,
        ),
        &system(),
    );
    t.run(
        CommandPayload::CompleteSubAgentTurn {
            session_id: "child-1".to_string(),
            agent_id: "child".to_string(),
            turn_id: "child-turn".to_string(),
            data: serde_json::json!("answer"),
            cost: Default::default(),
            token_usage: Default::default(),
        },
        &system(),
    );

    // A second delegation — requested directly this time — that never starts,
    // then fails.
    assert!(matches!(
        t.live_trigger(),
        Trigger::SubAgentFinished { ok: true, .. }
    ));
    t.decide(vec![], vec![]);
    t.run(
        CommandPayload::RequestSubAgent {
            session_id: "child-2".to_string(),
            agent_id: "child".to_string(),
            tool_call_id: "tc-2".to_string(),
            message: None,
            retry: RetryPolicy::no_retry(),
        },
        &system(),
    );
    t.run(
        CommandPayload::settle(
            EffectKind::SubAgent,
            "child-2".to_string(),
            None,
            SettleError::new(ErrorInfo::internal("spawn failed".to_string()), false),
        ),
        &system(),
    );
    t
}

#[test]
fn trace_sub_agent_delegation() {
    let t = flow_sub_agent_delegation();
    assert!(matches!(
        t.live_trigger(),
        Trigger::SubAgentFinished { ok: false, .. }
    ));
    t.assert_trace(&[
        "session.created",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "turn.started",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "message.new",
        "sub_agent.requested",
        "sub_agent.dispatched",
        "sub_agent.started",
        "sub_agent.turn_completed",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "sub_agent.requested",
        "sub_agent.dispatched",
        "sub_agent.errored",
        "decision.queued",
        "decision.dispatched",
    ]);
}

/// The engine restarted with work in flight: the pending decision is failed
/// (its reply rode the severed dispatch) and so is the server-handled LLM call
/// whose awaiting future died with the process.
fn flow_reconcile_dispatch() -> Trace {
    let mut t = Trace::create_with(retrying());
    t.decide(vec![], vec![]);
    t.submit("hi", Some("turn-1"));

    t.decide_appending(
        user_message("hi"),
        vec![call_llm("call-1", RetryPolicy::no_retry())],
    );
    // A second message opens another decision, so a decision is live too.
    t.run(
        CommandPayload::SendMessage {
            message: user_message("more"),
            stream: false,
            turn_id: None,
            parent_id: None,
        },
        &system(),
    );
    assert!(t.agg.state.has_pending_worker_decision());

    t.run(CommandPayload::ReconcileDispatch, &system());
    t
}

#[test]
fn trace_reconcile_dispatch_after_a_crash() {
    let t = flow_reconcile_dispatch();
    assert!(matches!(
        t.live_trigger(),
        Trigger::LlmFinished { ok: false, .. }
    ));
    t.assert_trace(&[
        "session.created",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "turn.started",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "message.new",
        "llm.call.requested",
        "llm.call.dispatched",
        "decision.queued",
        "decision.dispatched",
        "decision.errored",
        "llm.call.errored",
        "decision.queued",
        "decision.dispatched",
    ]);
}

/// The commands the engine issues to itself: a direct call request, a reported
/// failure, and the two-pass turn ending. Then a cancel, which voids what is
/// still in flight.
fn flow_direct_commands_then_cancel() -> Trace {
    let mut t = Trace::create();
    t.decide(vec![], vec![]);
    t.submit("hi", Some("turn-1"));
    t.decide_appending(user_message("hi"), vec![]);

    t.run(
        CommandPayload::RequestLlmCall {
            llm: "claude".to_string(),
            call_id: "call-1".to_string(),
            request: llm_request(),
            stream: false,
            retry: RetryPolicy::no_retry(),
            handler: LlmHandler::Server,
            format: None,
        },
        &system(),
    );
    t.run(
        CommandPayload::settle(
            EffectKind::LlmCall,
            "call-1".to_string(),
            Some(0),
            SettleError::new(
                ErrorInfo::new(ErrorCode::ProviderError, "provider down"),
                false,
            ),
        ),
        &system(),
    );
    t.decide(vec![], vec![]);

    t.run(
        CommandPayload::RequestToolCall {
            tool_call_id: "tc-1".to_string(),
            name: "a".to_string(),
            arguments: "{}".to_string(),
            retry: None,
        },
        &system(),
    );
    // The worker-handled call's execute decision is live; fail the call itself.
    t.run(
        CommandPayload::settle(
            EffectKind::ToolCall,
            "tc-1".to_string(),
            Some(0),
            SettleError::new(ErrorInfo::internal("tool down".to_string()), false),
        ),
        &system(),
    );

    // Pass 1: the turn's output is frozen and the finalizer runs.
    t.run(
        CommandPayload::FinishTurn {
            data: serde_json::json!("out"),
        },
        &system(),
    );
    // Pass 2, out of band: the run terminal is emitted from the frozen output.
    t.run(CommandPayload::CompleteTurn, &system());

    // Nothing queued may go live behind a cancel.
    t.submit("again", Some("turn-2"));
    t.run(CommandPayload::CancelSession, &system());
    t
}

#[test]
fn trace_direct_commands_then_cancel() {
    let t = flow_direct_commands_then_cancel();
    assert!(
        matches!(t.agg.state.status, SessionStatus::Done),
        "a cancelled session is done"
    );
    t.assert_trace(&[
        "session.created",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "turn.started",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "message.new",
        "llm.call.requested",
        "llm.call.dispatched",
        "llm.call.errored",
        "decision.queued",
        "decision.dispatched",
        "decision.completed",
        "tool.call.requested",
        "tool.call.dispatched",
        "decision.queued",
        "decision.dispatched",
        "tool.call.errored",
        "decision.queued",
        "decision.queued",
        "turn.completed",
        "session.done",
        "turn.started",
        "decision.queued",
        "session.cancelled",
    ]);
}

/// Every `CommandPayload` arm is exercised by some trace. `arm` matches
/// exhaustively, so a new command cannot be added without landing here.
#[test]
fn traces_cover_every_command_arm() {
    const ALL: &[&str] = &[
        "CancelSession",
        "CompleteConnectorSync",
        "CompleteLlmCall",
        "CompleteSubAgentTurn",
        "CompleteToolCall",
        "CompleteTurn",
        "CreateSession",
        "FailConnectorSync",
        "FailLlmCall",
        "FailSubAgent",
        "FailToolCall",
        "FailWorkerDecision",
        "FinishTurn",
        "Interrupt",
        "ReconcileDispatch",
        "RequestLlmCall",
        "RequestSubAgent",
        "RequestToolCall",
        "ResumeInterrupt",
        "SendMessage",
        "StartSubAgent",
        "SubmitClientPayload",
        "SubmitWorkerDecision",
        "Wake",
    ];

    let flows: Vec<Trace> = vec![
        flow_full_loop(),
        flow_connector_gating(),
        flow_connector_failure_releases_the_gate(),
        flow_interrupt_and_resume(),
        flow_queued_turn(),
        flow_timeout_to_exhaustion(),
        flow_decision_failure(),
        flow_parallel_fan_out(),
        flow_fork_voids_stranded_work(),
        flow_sub_agent_delegation(),
        flow_reconcile_dispatch(),
        flow_direct_commands_then_cancel(),
    ];
    let covered: BTreeSet<&'static str> =
        flows.iter().flat_map(|t| t.arms.iter().copied()).collect();

    let missing: Vec<&&str> = ALL.iter().filter(|a| !covered.contains(**a)).collect();
    assert!(missing.is_empty(), "commands with no trace: {missing:?}");
    let unlisted: Vec<&&str> = covered.iter().filter(|a| !ALL.contains(*a)).collect();
    assert!(
        unlisted.is_empty(),
        "commands missing from ALL: {unlisted:?}"
    );
}
