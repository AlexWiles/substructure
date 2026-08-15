//! The session's canonical state, and the reducer that builds it.
//!
//! # Laws
//!
//! **A transition writes state; it never stores a task.** `apply` records what
//! happened and nothing else. What runs next is derived from the result by
//! [`schedule::plan`](super::schedule::plan), every time. There is no queue of
//! pending work here beyond `schedule_queue`, which is position — an arrival
//! order — not a work list. The reflex to "enqueue the follow-up while we know
//! about it" is the thing this design exists to prevent: it puts the same
//! decision in two places, and they drift.
//!
//! **One table, one lifecycle.** Every tracked effect lives in `effects`, keyed
//! by `(kind, id)`, and carries an [`EffectTracking`] whose status only its own
//! transitions may write. Deadlines, retries, voiding, the wire projection and
//! the queue invariant are each one pass over that table, so adding a kind adds
//! no sweep.
//!
//! **Orthogonal regions.** The turn's phase, each effect's lifecycle, and the
//! interrupt overlay are independent. An interrupt is projected from
//! `open_interrupts` at read time and never stored as a status; `Interrupted`
//! is a [`SessionStatus`], never an [`EffectStatus`].

use std::collections::{BTreeMap, HashMap};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::decision::{LlmHandler, ToolHandler, Trigger};
use super::events::*;
use super::prompt_context;
use super::tool_contract::classify_arguments;
use crate::connectors::{filter, AuthNeed, RemoteTool};
use crate::protocol::{AgentTool, ConnectorTool, DeferToolsStrategy, Handler, McpServer};

/// One connection as the engine's own tools see it: what the agent declared,
/// what the connection offered, and what it said it is for.
#[derive(Debug, Clone, PartialEq)]
pub struct Source {
    pub server: McpServer,
    pub offered: Vec<RemoteTool>,
    pub instructions: Option<String>,
}

/// One connection as an announcement gives it to the model.
///
/// A struct, and not a `json!` map: a map sorts its keys, which would put a
/// server's own words ahead of the name that says whose they are.
#[derive(serde::Serialize)]
struct Summary<'a> {
    mcp_server: &'a str,
    tools: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    about: Option<&'a str>,
}

/// What a `call_tool` addresses. Any source, because deferral is a property of
/// a tool and not of where it came from.
#[derive(Debug, Clone, PartialEq)]
pub enum CallTarget {
    Connector(ConnectorTool),
    Declared(AgentTool),
}

impl CallTarget {
    pub fn input(&self) -> Option<serde_json::Value> {
        match self {
            CallTarget::Connector(t) => t.input.clone(),
            CallTarget::Declared(t) => t.input.clone(),
        }
    }
}

/// The tool's own arguments, out of the `call_tool` wrapper.
///
/// A model that sent them as a JSON string meant the object, and we hold the
/// schema that says so. Refusing it would teach the model nothing.
pub(in crate::runtime::session) fn inner_arguments(raw: &serde_json::Value) -> String {
    match raw.get("arguments") {
        Some(serde_json::Value::String(text)) => text.clone(),
        Some(value) => value.to_string(),
        None => "{}".to_string(),
    }
}

/// What the engine answers for one of its own connector tools.
#[derive(Debug, Clone, PartialEq)]
pub enum LocalAnswer {
    Result(String),
    Error(String),
}

/// One skill call as the executor answers it.
#[derive(Debug, Clone, PartialEq)]
pub struct SkillCall {
    pub leaf: Option<String>,
    pub plugin_id: String,
    pub arguments: String,
}
use rust_decimal::Decimal;

pub use crate::protocol::EffectKind;
use crate::protocol::{
    AgentConfig, DraftMessage, Effect, EffectStatus, InterruptOrigin, LlmFormat, LlmRequest,
    LlmTool, Message, MessageTree, NewMessage, ReasoningConfig, RetryConfig, RetryPolicy, Role,
    SessionOwner, Usage, WorkerState,
};
use crate::runtime::retry::RetryState;

pub struct ApplyContext {
    pub occurred_at: DateTime<Utc>,
    pub sequence: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    /// Waiting for external input: LLM responses, tool results, worker decisions.
    Idle,
    /// Paused for external input. Never stored — projected from `open_interrupts`.
    Interrupted {
        interrupt_id: String,
        origin: InterruptOrigin,
        reason: String,
    },
    /// Agent loop finished. Waiting for next user input.
    Done,
}

/// One effect's lifecycle, the same for every kind.
///
/// ```text
/// Queued ─dispatch→ Pending ─complete──────────────→ Completed
///                      │ ─error[retries left]──────→ RetryScheduled ─requeue→ Queued
///                      │ ─error[exhausted]─────────→ Failed
///                      └ ─void────────────────────→ Failed
/// ```
///
/// `status` is only ever written by the transitions below, each of which asserts
/// the move is legal. Nothing else assigns it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectTracking {
    status: EffectStatus,
    pub retry: RetryState,
    pub retry_policy: RetryPolicy,
    /// The current attempt's deadline. Cleared once the effect is `Running`:
    /// the attempt landed, and how long the work then takes is its own business.
    pub deadline: Option<DateTime<Utc>>,
    /// When the effect first left the queue. Never reset, so the whole-effect
    /// bound covers every attempt and the backoff between them — a retry cannot
    /// buy more time by restarting the clock.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<DateTime<Utc>>,
}

impl EffectTracking {
    /// A fetch: requested and in flight at once, never queued.
    pub fn new(retry_policy: RetryPolicy, now: DateTime<Utc>) -> Self {
        let deadline = retry_policy.attempt_deadline(now);
        Self {
            status: EffectStatus::Pending,
            retry: RetryState::default(),
            retry_policy,
            deadline,
            started_at: Some(now),
        }
    }

    pub fn new_queued(retry_policy: RetryPolicy) -> Self {
        Self {
            status: EffectStatus::Queued,
            retry: RetryState::default(),
            retry_policy,
            deadline: None,
            started_at: None,
        }
    }

    pub fn status(&self) -> EffectStatus {
        self.status
    }

    fn move_to(&mut self, next: EffectStatus, from: &[EffectStatus]) {
        debug_assert!(
            from.contains(&self.status),
            "illegal effect transition {:?} → {next:?}",
            self.status
        );
        self.status = next;
    }

    /// Start a settled effect over, with its retry budget whole again. Not a
    /// retry: the cause of every failed attempt has been corrected.
    ///
    /// `Completed` is the usual start. A fetch that succeeded proves only that
    /// the credential was good then. Only a connector fetch takes this.
    pub fn restart(&mut self, now: DateTime<Utc>) {
        self.move_to(
            EffectStatus::Pending,
            &[EffectStatus::Failed, EffectStatus::Completed],
        );
        self.retry = RetryState::default();
        self.deadline = self.retry_policy.attempt_deadline(now);
        self.started_at = Some(now);
    }

    /// Re-armed for another attempt: a retry, or a re-request of work still queued.
    pub fn requeue(&mut self) {
        self.move_to(
            EffectStatus::Queued,
            &[
                EffectStatus::Queued,
                EffectStatus::Pending,
                EffectStatus::Running,
                EffectStatus::RetryScheduled,
                EffectStatus::Failed,
            ],
        );
    }

    /// Started: the deadline clock runs from here.
    pub fn dispatch(&mut self, now: DateTime<Utc>) {
        self.move_to(
            EffectStatus::Pending,
            &[
                EffectStatus::Queued,
                EffectStatus::Pending,
                EffectStatus::RetryScheduled,
            ],
        );
        self.deadline = self.retry_policy.attempt_deadline(now);
        self.started_at.get_or_insert(now);
        self.retry.next_at = None;
    }

    /// Alive and working: the spawn landed, so the *attempt* clock stops
    /// applying — a child turn may legitimately run far longer than the call
    /// that started it. The whole-effect bound stays, so a dead child still
    /// settles instead of stalling its parent forever.
    pub fn run(&mut self) {
        self.move_to(EffectStatus::Running, &[EffectStatus::Pending]);
        self.deadline = None;
    }

    /// The whole-effect deadline, measured from the first dispatch. None until
    /// the effect has started, or when the policy sets no total.
    pub fn total_deadline(&self) -> Option<DateTime<Utc>> {
        self.started_at
            .and_then(|at| self.retry_policy.total_deadline(at))
    }

    /// When this effect is past due, whichever bound lapses first. `Pending`
    /// answers to both clocks; `Running` only to the total, having already
    /// cleared its attempt.
    pub fn expiry(&self) -> Option<DateTime<Utc>> {
        let total = self.total_deadline();
        match self.status {
            EffectStatus::Pending => match (self.deadline, total) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            },
            EffectStatus::Running => total,
            _ => None,
        }
    }

    /// Whether the whole-effect bound is what lapsed. A retry cannot help once
    /// it has: the budget covers every attempt, so the failure is terminal
    /// however retryable it looks on its own.
    pub fn total_expired(&self, now: DateTime<Utc>) -> bool {
        self.total_deadline().is_some_and(|d| d <= now)
    }

    pub fn complete(&mut self) {
        self.move_to(
            EffectStatus::Completed,
            &[EffectStatus::Pending, EffectStatus::Running],
        );
    }

    /// Abandoned mid-flight — its branch was forked away, or the session ended.
    pub fn void(&mut self) {
        self.move_to(
            EffectStatus::Failed,
            &[
                EffectStatus::Queued,
                EffectStatus::Pending,
                EffectStatus::Running,
                EffectStatus::RetryScheduled,
                EffectStatus::Failed,
            ],
        );
    }

    /// Whether recording a failure now would be terminal (no further attempt):
    /// pure, computed from the pre-failure state so callers can branch before
    /// mutating. `record_error` reuses it to keep the two in lockstep.
    pub fn is_terminal_failure(&self, retryable: bool) -> bool {
        self.retry_policy.exhausted(&self.retry, retryable)
    }

    pub fn record_error(&mut self, retryable: bool, now: DateTime<Utc>) {
        let terminal = self.is_terminal_failure(retryable);
        self.retry = self.retry_policy.record_failure(&self.retry, now);
        let next = if terminal {
            self.retry.next_at = None;
            EffectStatus::Failed
        } else {
            EffectStatus::RetryScheduled
        };
        self.move_to(
            next,
            &[
                EffectStatus::Queued,
                EffectStatus::Pending,
                EffectStatus::RetryScheduled,
            ],
        );
    }

    /// Whether a connector offer is usable — settled, and settled successfully.
    pub fn is_ready(&self) -> bool {
        self.status == EffectStatus::Completed
    }

    /// Whether the effect is still going to produce something.
    pub fn is_in_flight(&self) -> bool {
        matches!(
            self.status,
            EffectStatus::Pending | EffectStatus::RetryScheduled
        )
    }

    /// Recorded but not started.
    pub fn is_queued(&self) -> bool {
        self.status == EffectStatus::Queued
    }

    /// Not settled: recorded, running, or waiting to run again.
    pub fn is_open(&self) -> bool {
        self.is_queued() || self.is_in_flight()
    }

    pub fn earliest_wake(&self) -> Option<DateTime<Utc>> {
        match self.status {
            EffectStatus::Pending | EffectStatus::Running => self.expiry(),
            EffectStatus::RetryScheduled => self.retry.next_at,
            _ => None,
        }
    }
}

/// One tracked effect: the envelope every kind shares, plus its own data.
///
/// `id` is the id that kind's own events name it by — a tool call, an LLM call,
/// a child session, a connection, a decision — so `(kind, id)` keys the table
/// and nothing needs a second identifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectState {
    pub id: String,
    pub tracking: EffectTracking,
    /// The active head when the effect was first requested; retries keep it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
    pub payload: EffectPayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EffectPayload {
    LlmCall(LlmCallState),
    ToolCall(ToolCallState),
    SubAgent(SubAgentCallState),
    ConnectorSync(ConnectorSyncState),
    Decision(WorkerDecisionState),
}

impl EffectPayload {
    pub fn kind(&self) -> EffectKind {
        match self {
            EffectPayload::LlmCall(_) => EffectKind::LlmCall,
            EffectPayload::ToolCall(_) => EffectKind::ToolCall,
            EffectPayload::SubAgent(_) => EffectKind::SubAgent,
            EffectPayload::ConnectorSync(_) => EffectKind::ConnectorSync,
            EffectPayload::Decision(_) => EffectKind::Decision,
        }
    }
}

/// `payload()` reads a kind's data off an effect of that kind; `None` means the
/// caller asked the wrong kind, which the table's key makes impossible in
/// practice. One pair per kind, written out rather than generated so the
/// matches stay greppable.
macro_rules! payload_views {
    ($($get:ident, $get_mut:ident => $variant:ident($ty:ty)),* $(,)?) => {
        impl EffectState {
            pub fn kind(&self) -> EffectKind {
                self.payload.kind()
            }

            $(
                pub fn $get(&self) -> Option<&$ty> {
                    match &self.payload {
                        EffectPayload::$variant(v) => Some(v),
                        _ => None,
                    }
                }
                pub fn $get_mut(&mut self) -> Option<&mut $ty> {
                    match &mut self.payload {
                        EffectPayload::$variant(v) => Some(v),
                        _ => None,
                    }
                }
            )*
        }
    };
}

impl EffectState {
    pub fn new(id: impl Into<String>, tracking: EffectTracking, payload: EffectPayload) -> Self {
        Self {
            id: id.into(),
            tracking,
            anchor: None,
            payload,
        }
    }

    pub fn at(mut self, anchor: Option<&str>) -> Self {
        self.anchor = anchor.map(str::to_string);
        self
    }
}

payload_views! {
    llm, llm_mut => LlmCall(LlmCallState),
    tool, tool_mut => ToolCall(ToolCallState),
    sub_agent, sub_agent_mut => SubAgent(SubAgentCallState),
    connector, connector_mut => ConnectorSync(ConnectorSyncState),
    decision, decision_mut => Decision(WorkerDecisionState),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCallState {
    /// The verbatim prompt the worker sent, stored for retries.
    #[serde(default)]
    pub prompt: Vec<Message>,
    /// The `[llm.*]` block the call names, kept for retries and for the
    /// executor to pick its client by.
    pub llm: String,
    pub spec: LlmCallSpec,
    pub stream: bool,
    #[serde(default)]
    pub handler: LlmHandler,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<LlmFormat>,
    /// How a deferred tool reaches the model, frozen from the block at request
    /// time so a replay lowers the same way.
    #[serde(default)]
    pub defer_tools_strategy: DeferToolsStrategy,
    /// Ids of the engine-derived context this call's prompt already carries.
    /// The record a retry checks, and the record a later call of this path
    /// reads to know what the model has already been told.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub context_ids: Vec<String>,
}

/// An `LlmRequest` without its message list; the prompt is stored alongside.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCallSpec {
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<LlmTool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
}

impl LlmCallSpec {
    pub fn to_request(&self, messages: Vec<Message>) -> LlmRequest {
        self.to_wire_request(messages.into_iter().map(DraftMessage::from).collect())
    }

    pub fn to_wire_request(&self, messages: Vec<DraftMessage>) -> LlmRequest {
        LlmRequest {
            model: self.model.clone(),
            messages,
            tools: self.tools.clone(),
            temperature: self.temperature,
            max_completion_tokens: self.max_completion_tokens,
            reasoning: self.reasoning.clone(),
        }
    }
}

impl From<&LlmRequest> for LlmCallSpec {
    fn from(r: &LlmRequest) -> Self {
        Self {
            model: r.model.clone(),
            tools: r.tools.clone(),
            temperature: r.temperature,
            max_completion_tokens: r.max_completion_tokens,
            reasoning: r.reasoning.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallState {
    pub name: String,
    #[serde(default)]
    pub handler: ToolHandler,
    /// The connection and remote name for a `Server` call; `None` otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<ConnectorTarget>,
    /// Original arguments, stored for retries and crash recovery.
    #[serde(default)]
    pub arguments: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    #[serde(default)]
    pub is_error: bool,
}

/// A delegation, keyed by the child session it runs — the id every sub-agent
/// event names it by.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubAgentCallState {
    pub agent_id: String,
    /// The model tool-call id this delegation answers.
    #[serde(default)]
    pub tool_call_id: String,
    /// The child's opening message, held until the spawn creates its session.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<DraftMessage>,
    /// The child's turn result (or error); `Some` once the turn is terminal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    #[serde(default)]
    pub is_error: bool,
}

/// One connection's fetched tool list.
///
/// Unanchored, and deliberately not rewound with the tree: what a connection
/// offered at a sequence is a fact about the remote, not about a branch. So a
/// fork back to an older head reuses the offer instead of refetching, the same
/// way the call maps stay current across a rewind.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorSyncState {
    /// Every tool the connection offered, unfiltered. Empty until it settles.
    #[serde(default)]
    pub tools: Vec<RemoteTool>,
    /// The prefix its tools expand under, frozen at fetch time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix: Option<String>,
    /// What the server said it is for. The only description of a connection
    /// that the engine does not have to write itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// Why the last attempt failed; `Some` only while the fetch is unsettled or
    /// terminally failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth: Option<AuthNeed>,
}

/// One anchored document write. Both typed channels — worker state and agent
/// config — are versioned and resolved newest-on-path identically, so they are
/// one type over what they carry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Versioned<T> {
    pub value: T,
    pub anchor: Option<String>,
}

/// Anchored worker-state write; current state is resolved, not stored.
pub type StateVersion = Versioned<WorkerState>;

/// Anchored agent-config write, the sibling channel for agent identity.
pub type AgentVersion = Versioned<AgentConfig>;

/// An unresumed interrupt: parks paths through its anchor (`None` = every path).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenInterrupt {
    pub interrupt_id: String,
    pub origin: InterruptOrigin,
    pub reason: String,
    pub payload: serde_json::Value,
    pub anchor: Option<String>,
}

/// What holds a parked decision. A park is region-shaped: the entry is skipped
/// while a region elsewhere holds it, and the queue flows past. Each variant
/// names the region — a branch under an interrupt, or the phase under a turn.
#[derive(Debug, Clone, Copy)]
pub enum DecisionPark<'a> {
    /// An open interrupt parks the branch the decision lands on.
    Interrupt(&'a OpenInterrupt),
    /// The named turn holds the phase; this decision opens a different one.
    Turn(&'a str),
}

/// A history-log entry tagged with the seq of the event that appended it —
/// the as-of cursor is the event's own seq, nothing stamped elsewhere.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Logged<T> {
    pub seq: u64,
    #[serde(flatten)]
    pub entry: T,
}

impl<T> std::ops::Deref for Logged<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.entry
    }
}

/// Truncate an append-only log to entries at or before `seq`.
fn rewind_log<T>(log: &mut Vec<Logged<T>>, seq: u64) {
    debug_assert!(log.windows(2).all(|w| w[0].seq <= w[1].seq));
    log.truncate(log.partition_point(|e| e.seq <= seq));
}

/// A versioned, anchored document write. Both worker state and agent config
/// resolve newest-on-path identically; this shares that logic without
/// touching either struct's serialized shape.
pub trait Anchored {
    fn anchor(&self) -> Option<&str>;
}

impl<T: Anchored> Anchored for Logged<T> {
    fn anchor(&self) -> Option<&str> {
        self.entry.anchor()
    }
}

impl<T> Anchored for Versioned<T> {
    fn anchor(&self) -> Option<&str> {
        self.anchor.as_deref()
    }
}

impl Anchored for OpenInterrupt {
    fn anchor(&self) -> Option<&str> {
        self.anchor.as_deref()
    }
}

/// Newest version whose anchor is on `on_path`; an unanchored version matches any path.
pub fn resolve_on_path<'a, V: Anchored>(
    versions: &'a [V],
    on_path: &std::collections::HashSet<&str>,
) -> Option<&'a V> {
    versions.iter().rev().find(|v| match v.anchor() {
        None => true,
        Some(a) => on_path.contains(a),
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerDecisionState {
    pub trigger: Trigger,
    #[serde(default)]
    pub source_event_sequence: u64,
}

/// One entry of the schedule queue: work recorded but not yet dispatched.
/// `seq` is the queued event's own sequence — arrival order is log order.
/// The payload stays in its kind's map; the entry is position, not data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueEntry {
    pub seq: u64,
    pub kind: EffectKind,
    pub id: String,
}

/// The frozen turn output held while its `turn.finished` finalizer runs, emitted as
/// `TurnCompleted` once the finalizer settles (pass 2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnEnd {
    pub turn_id: String,
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub data: serde_json::Value,
    #[serde(default)]
    pub cost: Decimal,
    #[serde(default)]
    pub usage: Usage,
}

/// Where the session is in a turn.
///
/// ```text
/// Idle ─turn.started→ Active{turn_id} ─turn.finished queued→ Finalizing(TurnEnd)
///  ↑                                                              │
///  └────────────────────── turn.completed ────────────────────────┘
/// ```
///
/// One of three orthogonal regions a session lives in — turn phase ∥ effect
/// lifecycles ∥ the interrupt overlay — so it says nothing about what work is in
/// flight or whether a branch is parked. There is deliberately no `Interrupted`
/// variant: an interrupt parks a branch of the tree, and a turn can be
/// interrupted in any phase. `SessionStatus` is the same story from the other
/// side — stored `Idle`/`Done`, with `Interrupted` projected from the overlay.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum TurnPhase {
    /// No turn is running. The session may still hold settled work and a tree.
    #[default]
    Idle,
    /// The agent is working.
    Active { turn_id: String },
    /// The agent's turn is over and its output is frozen; the worker's
    /// `turn.finished` finalizer is in flight. Holds what `TurnCompleted` will
    /// carry, and is the pass-1/pass-2 discriminator.
    Finalizing(TurnEnd),
}

impl TurnPhase {
    /// The turn in progress, in either working phase.
    pub fn turn_id(&self) -> Option<&str> {
        match self {
            TurnPhase::Idle => None,
            TurnPhase::Active { turn_id } => Some(turn_id),
            TurnPhase::Finalizing(end) => Some(&end.turn_id),
        }
    }

    /// The frozen output, once the turn has one.
    pub fn finalizing(&self) -> Option<&TurnEnd> {
        match self {
            TurnPhase::Finalizing(end) => Some(end),
            _ => None,
        }
    }
}

/// Per-event stamp: scalars plus bounded in-flight status. History-sized
/// state (tree, versions, prompts) lives in the store; `head_id` +
/// `node_count` locate the as-of-event tree prefix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventMeta {
    pub status: SessionStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wake_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<SessionOwner>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ancestry: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    #[serde(default)]
    pub cost: Decimal,
    #[serde(default)]
    pub sub_agent_cost: Decimal,
    /// The active branch at this event; with the event's `seq`, the full
    /// as-of cursor for `SessionState::rewind`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub calls: Vec<Effect>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub decisions: Vec<MetaDecision>,
}

/// A Pending/Queued decision at stamp time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaDecision {
    pub decision_id: String,
    pub status: EffectStatus,
    /// Trigger is `ToolFinished`/`SubAgentFinished`: an unrecorded result.
    pub finished: bool,
    pub attempts: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deadline: Option<DateTime<Utc>>,
    pub source_event_sequence: u64,
}

impl EventMeta {
    /// Outstanding parallel work the decision `decision_id` must wait for before
    /// it prompts: in-flight tool/sub-agent calls, plus other `*.finished`
    /// decisions still queued or pending — results not yet folded into the
    /// transcript. Without the latter, every finished decision of a parallel
    /// fan-out sees zero (all calls already completed) and prompts, so each
    /// sibling issues a follow-up call against an incomplete transcript.
    pub fn pending_work(&self, decision_id: &str) -> usize {
        let in_flight_calls = self
            .calls
            .iter()
            .filter(|e| {
                matches!(e.kind, EffectKind::ToolCall | EffectKind::SubAgent)
                    && matches!(
                        e.status,
                        EffectStatus::Queued
                            | EffectStatus::Pending
                            | EffectStatus::Running
                            | EffectStatus::RetryScheduled
                    )
            })
            .count();
        let unrecorded_results = self
            .decisions
            .iter()
            .filter(|d| d.decision_id != decision_id)
            .filter(|d| {
                matches!(d.status, EffectStatus::Pending | EffectStatus::Queued) && d.finished
            })
            .count();
        in_flight_calls + unrecorded_results
    }
}

/// The effect table serializes as a flat list: its key is `(payload.kind(), id)`,
/// both of which every entry already carries, and JSON has no tuple keys.
mod effect_table {
    use super::{EffectKind, EffectState};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::BTreeMap;

    pub fn serialize<S: Serializer>(
        table: &BTreeMap<(EffectKind, String), EffectState>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        table.values().collect::<Vec<_>>().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<BTreeMap<(EffectKind, String), EffectState>, D::Error> {
        Ok(Vec::<EffectState>::deserialize(d)?
            .into_iter()
            .map(|e| ((e.kind(), e.id.clone()), e))
            .collect())
    }
}

pub fn new_call_id() -> String {
    Uuid::now_v7().to_string()
}

pub fn new_message_id() -> String {
    Uuid::now_v7().to_string()
}

/// A JSON string passes through; anything else is serialized.
pub fn json_to_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    pub session_id: String,
    pub status: SessionStatus,
    pub agent_id: Option<String>,
    pub owner: Option<SessionOwner>,
    #[serde(default)]
    pub token_usage: Usage,

    /// Accumulated cost across all LLM calls in this session.
    #[serde(default)]
    pub cost: Decimal,

    /// Accumulated cost from sub-agent sessions.
    #[serde(default)]
    pub sub_agent_cost: Decimal,

    /// Cost accumulated in the current turn only.
    #[serde(default)]
    pub turn_cost: Decimal,

    /// Token usage accumulated in the current turn only.
    #[serde(default)]
    pub turn_token_usage: Usage,

    /// Token usage accumulated from sub-agent sessions.
    #[serde(default)]
    pub sub_agent_token_usage: Usage,

    #[serde(default)]
    pub state_versions: Vec<Logged<StateVersion>>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub agent_versions: Vec<Logged<AgentVersion>>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ancestry: Vec<String>,

    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub data: serde_json::Value,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker_retry: Option<RetryPolicy>,

    /// Every tracked effect, keyed by `(kind, id)`. One table so that every
    /// generic sweep — deadlines, retries, voiding, the wire projection, the
    /// queue invariant — is one iteration rather than one per kind.
    ///
    /// Connector syncs live here too, keyed by connection id. They are keyed on
    /// the connection rather than on the agent version that asked for one, so a
    /// config rewritten for unrelated reasons — a client tool appearing, a
    /// branch switch — costs no round trip.
    #[serde(with = "effect_table", default = "BTreeMap::new")]
    pub effects: BTreeMap<(EffectKind, String), EffectState>,

    /// Queued work in arrival order — every effect and decision waits here
    /// between its queued event and its dispatch event. Maintained solely by
    /// `apply`: queued events push, dispatch/settle/void/drop events remove.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub schedule_queue: Vec<QueueEntry>,

    /// Where the session is in a turn — the whole turn region, in one field.
    #[serde(default)]
    pub phase: TurnPhase,

    /// The seq of the running turn's `turn.started` event.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_started_seq: Option<u64>,

    /// Turn IDs that have completed, used for idempotency checks.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub completed_turn_ids: Vec<String>,

    /// The active branch's leaf; advances to each appended message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_id: Option<String>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nodes: Vec<Logged<NewMessage>>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub open_interrupts: Vec<OpenInterrupt>,

    /// `session.start` failed terminally: no config can ever be resolved, so
    /// user payloads are refused and queued decisions are dropped. Cleared by a
    /// re-queued `session.start`.
    ///
    /// Its own region, not a [`TurnPhase`] variant: a start failure outlives the
    /// turn it happened in — the re-queued start rides the *next* user message,
    /// which opens a new turn — so a phase that held it would be cleared by the
    /// very turn meant to recover from it.
    #[serde(default)]
    pub session_start_failed: bool,
}

impl SessionState {
    pub fn new(session_id: String) -> Self {
        SessionState {
            session_id,
            status: SessionStatus::Done,
            agent_id: None,
            owner: None,
            token_usage: Usage::default(),
            cost: Decimal::ZERO,
            sub_agent_cost: Decimal::ZERO,
            turn_cost: Decimal::ZERO,
            turn_token_usage: Usage::default(),
            sub_agent_token_usage: Usage::default(),
            state_versions: Vec::new(),
            agent_versions: Vec::new(),
            ancestry: Vec::new(),
            data: serde_json::Value::Null,
            worker_retry: None,
            effects: BTreeMap::new(),
            schedule_queue: Vec::new(),
            phase: TurnPhase::default(),
            turn_started_seq: None,
            completed_turn_ids: Vec::new(),
            head_id: None,
            nodes: Vec::new(),
            open_interrupts: Vec::new(),
            session_start_failed: false,
        }
    }

    /// The turn events belong to: the live one, or — once it has completed —
    /// the one that just finished. The tail matters: `TurnCompleted` is stamped
    /// after it applies, so a turn-scoped stream would otherwise miss its own
    /// terminal event.
    pub fn turn_id(&self) -> Option<&str> {
        self.phase
            .turn_id()
            .or_else(|| self.completed_turn_ids.last().map(String::as_str))
    }

    pub fn effect(&self, kind: EffectKind, id: &str) -> Option<&EffectState> {
        self.effects.get(&(kind, id.to_string()))
    }

    pub fn effect_mut(&mut self, kind: EffectKind, id: &str) -> Option<&mut EffectState> {
        self.effects.get_mut(&(kind, id.to_string()))
    }

    /// Record an effect verbatim. For fixtures and store hydration; `apply` uses
    /// [`SessionState::insert_effect`], which anchors at the head.
    pub fn put_effect(&mut self, effect: EffectState) {
        self.effects
            .insert((effect.kind(), effect.id.clone()), effect);
    }

    pub fn has_effect(&self, kind: EffectKind, id: &str) -> bool {
        self.effects.contains_key(&(kind, id.to_string()))
    }

    pub fn tracking(&self, kind: EffectKind, id: &str) -> Option<&EffectTracking> {
        self.effect(kind, id).map(|e| &e.tracking)
    }

    /// Every effect of one kind, in id order.
    pub fn effects_of(&self, kind: EffectKind) -> impl Iterator<Item = &EffectState> {
        self.effects.values().filter(move |e| e.kind() == kind)
    }

    pub fn llm_call(&self, id: &str) -> Option<&LlmCallState> {
        self.effect(EffectKind::LlmCall, id).and_then(|e| e.llm())
    }

    pub fn tool_call(&self, id: &str) -> Option<&ToolCallState> {
        self.effect(EffectKind::ToolCall, id).and_then(|e| e.tool())
    }

    pub fn sub_agent(&self, id: &str) -> Option<&SubAgentCallState> {
        self.effect(EffectKind::SubAgent, id)
            .and_then(|e| e.sub_agent())
    }

    pub fn connector_sync(&self, id: &str) -> Option<&ConnectorSyncState> {
        self.effect(EffectKind::ConnectorSync, id)
            .and_then(|e| e.connector())
    }

    /// Record an effect anchored at the current head — where it was requested.
    /// A fetch anchors nowhere: it belongs to the connection, not to a branch.
    fn insert_effect(&mut self, id: &str, tracking: EffectTracking, payload: EffectPayload) {
        let anchor = match payload {
            EffectPayload::ConnectorSync(_) => None,
            _ => self.head_id.clone(),
        };
        self.effects.insert(
            (payload.kind(), id.to_string()),
            EffectState {
                id: id.to_string(),
                tracking,
                anchor,
                payload,
            },
        );
    }

    fn remove_effect(&mut self, kind: EffectKind, id: &str) -> Option<EffectState> {
        self.effects.remove(&(kind, id.to_string()))
    }

    /// Push a queue entry; idempotent per (kind, id) so a retry re-request of
    /// an entry still queued re-arms in place instead of duplicating it.
    fn enqueue(&mut self, seq: u64, kind: EffectKind, id: &str) {
        if !self
            .schedule_queue
            .iter()
            .any(|e| e.kind == kind && e.id == id)
        {
            self.schedule_queue.push(QueueEntry {
                seq,
                kind,
                id: id.to_string(),
            });
        }
    }

    fn dequeue(&mut self, kind: EffectKind, id: &str) {
        self.schedule_queue
            .retain(|e| !(e.kind == kind && e.id == id));
    }

    /// A (re-)issued call: re-arm an existing one, else record it. Applies after
    /// the same batch's `NewMessage` events, so `head_id` is post-reconcile.
    fn apply_llm_requested(&mut self, payload: &LlmCallRequested, ctx: &ApplyContext) {
        let prompt: Vec<Message> = payload
            .request
            .messages
            .iter()
            .cloned()
            .map(DraftMessage::record)
            .collect();
        let spec = LlmCallSpec::from(&payload.request);
        if let Some(existing) = self.effect_mut(EffectKind::LlmCall, &payload.id) {
            existing.tracking.requeue();
            if let Some(call) = existing.llm_mut() {
                call.prompt = prompt;
                call.llm = payload.llm.clone();
                call.spec = spec;
                call.stream = payload.stream;
                call.handler = payload.handler;
                call.format = payload.format;
                call.defer_tools_strategy = payload.defer_tools_strategy;
            }
        } else {
            self.insert_effect(
                &payload.id,
                EffectTracking::new_queued(payload.retry.clone()),
                EffectPayload::LlmCall(LlmCallState {
                    prompt,
                    llm: payload.llm.clone(),
                    spec,
                    stream: payload.stream,
                    handler: payload.handler,
                    format: payload.format,
                    context_ids: Vec::new(),
                    defer_tools_strategy: payload.defer_tools_strategy,
                }),
            );
        }
        self.enqueue(ctx.sequence, EffectKind::LlmCall, &payload.id);
    }

    /// The queued call clears its gates: merge the connector tools in force
    /// (dispatch is when derived context is current), start the deadline clock,
    /// leave the queue.
    fn apply_llm_dispatched(&mut self, payload: &LlmCallDispatched, now: DateTime<Utc>) {
        let leaf = self
            .effect(EffectKind::LlmCall, &payload.id)
            .and_then(|e| e.anchor.clone())
            .or_else(|| self.head_id.clone());
        let connector_tools = self.connector_tools(leaf.as_deref()).tools;
        let owed = prompt_context::owed(self, leaf.as_deref(), &payload.id);
        let system_open = self.system_prefix_open(leaf.as_deref(), &payload.id);
        if let Some(effect) = self.effect_mut(EffectKind::LlmCall, &payload.id) {
            effect.tracking.dispatch(now);
            if let Some(call) = effect.llm_mut() {
                if !connector_tools.is_empty() {
                    let tools = call.spec.tools.get_or_insert_with(Vec::new);
                    for tool in connector_tools {
                        if !tools.iter().any(|t| t.name == tool.name) {
                            tools.push(tool.to_llm_tool());
                        }
                    }
                }
                prompt_context::merge(
                    &mut call.prompt,
                    &mut call.context_ids,
                    &payload.id,
                    system_open,
                    owed,
                );
            }
        }
        self.dequeue(EffectKind::LlmCall, &payload.id);
    }

    /// A (re-)issued tool call: re-arm an existing one, else record it.
    fn apply_tool_requested(&mut self, payload: &ToolCallRequested, ctx: &ApplyContext) {
        if let Some(existing) = self.effect_mut(EffectKind::ToolCall, &payload.id) {
            existing.tracking.requeue();
            if let Some(tc) = existing.tool_mut() {
                tc.arguments = payload.arguments.clone();
            }
        } else {
            self.insert_effect(
                &payload.id,
                EffectTracking::new_queued(payload.retry.clone()),
                EffectPayload::ToolCall(ToolCallState {
                    name: payload.name.clone(),
                    handler: payload.handler,
                    target: payload.target.clone(),
                    arguments: payload.arguments.clone(),
                    result: None,
                    is_error: false,
                }),
            );
        }
        self.enqueue(ctx.sequence, EffectKind::ToolCall, &payload.id);
    }

    /// A (re-)issued delegation: re-arm an existing one, else record it.
    fn apply_sub_agent_requested(&mut self, payload: &SubAgentRequested, ctx: &ApplyContext) {
        if let Some(existing) = self.effect_mut(EffectKind::SubAgent, &payload.id) {
            existing.tracking.requeue();
        } else {
            self.insert_effect(
                &payload.id,
                EffectTracking::new_queued(payload.retry.clone()),
                EffectPayload::SubAgent(SubAgentCallState {
                    agent_id: payload.agent_id.clone(),
                    tool_call_id: payload.tool_call_id.clone(),
                    message: payload.message.clone(),
                    result: None,
                    is_error: false,
                }),
            );
        }
        self.enqueue(ctx.sequence, EffectKind::SubAgent, &payload.id);
    }

    /// One open interrupt per id. Pending decisions void so late submissions
    /// no-op; calls void via their own `CallVoided` events.
    fn apply_interrupted(&mut self, payload: &SessionInterrupted) {
        if !self
            .open_interrupts
            .iter()
            .any(|i| i.interrupt_id == payload.interrupt_id)
        {
            self.open_interrupts.push(OpenInterrupt {
                interrupt_id: payload.interrupt_id.clone(),
                origin: payload.origin,
                reason: payload.reason.clone(),
                payload: payload.payload.clone(),
                anchor: payload.anchor.clone(),
            });
        }
        self.effects.retain(|(kind, _), e| {
            *kind != EffectKind::Decision || e.tracking.status() != EffectStatus::Pending
        });
    }

    /// A terminally-failed decision leaves the map, and a failed `session.start`
    /// poisons the session until another start is queued.
    fn apply_decision_errored(&mut self, p: &DecisionErrored, now: DateTime<Utc>) {
        let failed = self
            .effect_mut(EffectKind::Decision, &p.id)
            .map(|e| {
                e.tracking.record_error(p.retryable, now);
                e.tracking.status == EffectStatus::Failed
            })
            .unwrap_or(false);
        if failed {
            let started = self
                .remove_effect(EffectKind::Decision, &p.id)
                .and_then(|e| e.decision().map(|d| d.trigger.clone()))
                .is_some_and(|t| matches!(t, Trigger::SessionStart));
            self.session_start_failed |= started;
            self.dequeue(EffectKind::Decision, &p.id);
        }
    }

    /// Queuing `turn.finished` (pass 1) captures the frozen output; the turn is
    /// finalizing until `TurnCompleted` (pass 2). The turn's completion queues as
    /// a dependent of the finalizer decision, so a finalizer that settles without
    /// a `done` echo still completes the turn at the next walk.
    fn apply_decision_queued(&mut self, p: &DecisionQueued, ctx: &ApplyContext) {
        // A re-queued start clears the poison: the session gets another chance
        // at a config, so client input is accepted again.
        if matches!(p.trigger, Trigger::SessionStart) {
            self.session_start_failed = false;
        }
        let retry_policy = self.worker_retry.clone().unwrap_or(RetryPolicy::no_retry());
        self.insert_effect(
            &p.id,
            EffectTracking::new_queued(retry_policy),
            EffectPayload::Decision(WorkerDecisionState {
                trigger: p.trigger.clone(),
                source_event_sequence: ctx.sequence,
            }),
        );
        self.enqueue(ctx.sequence, EffectKind::Decision, &p.id);
        if let Trigger::TurnFinished {
            turn_id,
            data,
            cost,
            usage,
        } = &p.trigger
        {
            self.phase = TurnPhase::Finalizing(TurnEnd {
                turn_id: turn_id.clone(),
                data: data.clone(),
                cost: *cost,
                usage: usage.clone(),
            });
            let turn_id = turn_id.clone();
            self.enqueue(ctx.sequence, EffectKind::TurnEnd, &turn_id);
        }
    }

    /// A voided effect fails in place and leaves the queue.
    ///
    /// A fetch is the exception: it belongs to the connection, not to the branch
    /// that asked for it, so a fork never abandons one. The queue removal is a
    /// guard — fetches never queue in practice.
    fn apply_call_voided(&mut self, p: &CallVoided) {
        let kind = p.kind;
        if kind != EffectKind::ConnectorSync {
            if let Some(effect) = self.effect_mut(kind, &p.id) {
                effect.tracking.void();
            }
        }
        self.dequeue(kind, &p.id);
    }

    pub fn apply(&mut self, event: &EventPayload, ctx: &ApplyContext) {
        let now = ctx.occurred_at;
        match event {
            EventPayload::SessionCreated(payload) => {
                self.status = SessionStatus::Idle;
                self.agent_id = Some(payload.agent_id.clone());
                self.owner = Some(payload.identity.clone());
                self.ancestry = payload.ancestry.clone();
                self.worker_retry = Some(payload.worker_retry.clone());
            }
            EventPayload::NewMessage(payload) => {
                if payload.message.role == Role::User {
                    self.status = SessionStatus::Idle;
                }
                self.head_id = Some(payload.message.id.clone());
                self.nodes.push(Logged {
                    seq: ctx.sequence,
                    entry: payload.clone(),
                });
            }
            EventPayload::HeadMoved(payload) => {
                self.head_id = Some(payload.head_id.clone());
            }
            EventPayload::LlmCallRequested(payload) => self.apply_llm_requested(payload, ctx),
            EventPayload::LlmCallDispatched(payload) => self.apply_llm_dispatched(payload, now),
            EventPayload::LlmCallCompleted(payload) => {
                self.track_usage(&payload.response.usage);
                self.track_turn_usage(&payload.response.usage);
                if let Some(c) = payload.response.cost {
                    self.cost += c;
                    self.turn_cost += c;
                }
                if let Some(e) = self.effect_mut(EffectKind::LlmCall, &payload.id) {
                    e.tracking.complete();
                }
            }
            EventPayload::LlmCallErrored(payload) => {
                if let Some(e) = self.effect_mut(EffectKind::LlmCall, &payload.id) {
                    e.tracking.record_error(payload.retryable, now);
                }
            }
            EventPayload::ToolCallRequested(payload) => self.apply_tool_requested(payload, ctx),
            EventPayload::ToolCallDispatched(payload) => {
                if let Some(e) = self.effect_mut(EffectKind::ToolCall, &payload.id) {
                    e.tracking.dispatch(now);
                }
                self.dequeue(EffectKind::ToolCall, &payload.id);
            }
            EventPayload::ToolCallCompleted(payload) => {
                if let Some(e) = self.effect_mut(EffectKind::ToolCall, &payload.id) {
                    e.tracking.complete();
                    if let Some(tc) = e.tool_mut() {
                        tc.result = Some(payload.result.rendered());
                        tc.is_error = false;
                    }
                }
            }
            EventPayload::ToolCallErrored(payload) => {
                if let Some(e) = self.effect_mut(EffectKind::ToolCall, &payload.id) {
                    e.tracking.record_error(payload.retryable, now);
                    let failed = e.tracking.status == EffectStatus::Failed;
                    if let Some(tc) = e.tool_mut().filter(|_| failed) {
                        tc.result = Some(payload.error.message.clone());
                        tc.is_error = true;
                    }
                }
            }
            EventPayload::SubAgentRequested(payload) => {
                self.apply_sub_agent_requested(payload, ctx)
            }
            EventPayload::SubAgentDispatched(payload) => {
                if let Some(e) = self.effect_mut(EffectKind::SubAgent, &payload.id) {
                    e.tracking.dispatch(now);
                }
                self.dequeue(EffectKind::SubAgent, &payload.id);
            }
            // The spawn landed; the delegation is not done. It stays in flight
            // until its turn returns, so a sibling finishing first cannot
            // re-prompt the model with this one's result still missing.
            EventPayload::SubAgentStarted(payload) => {
                if let Some(e) = self.effect_mut(EffectKind::SubAgent, &payload.id) {
                    e.tracking.run();
                }
            }
            EventPayload::SubAgentErrored(payload) => {
                if let Some(e) = self.effect_mut(EffectKind::SubAgent, &payload.id) {
                    e.tracking.record_error(payload.retryable, now);
                    let failed = e.tracking.status == EffectStatus::Failed;
                    if let Some(sa) = e.sub_agent_mut().filter(|_| failed) {
                        sa.result = Some(payload.error.message.clone());
                        sa.is_error = true;
                    }
                }
            }
            EventPayload::SessionInterrupted(payload) => self.apply_interrupted(payload),
            EventPayload::InterruptResumed(p) => {
                self.open_interrupts
                    .retain(|i| i.interrupt_id != p.interrupt_id);
            }
            // Dispatch marker: the decision (seeded by DecisionQueued)
            // goes live and leaves the queue.
            EventPayload::DecisionDispatched(p) => {
                if let Some(e) = self.effect_mut(EffectKind::Decision, &p.id) {
                    e.tracking.dispatch(now);
                }
                self.dequeue(EffectKind::Decision, &p.id);
                self.status = SessionStatus::Idle;
            }
            // Settled decisions leave the table: absent reads as not-Pending, so
            // late submissions still no-op and stored triggers don't accumulate.
            EventPayload::DecisionCompleted(p) => {
                self.remove_effect(EffectKind::Decision, &p.id);
                self.dequeue(EffectKind::Decision, &p.id);
            }
            EventPayload::DecisionErrored(p) => self.apply_decision_errored(p, now),
            EventPayload::SessionMessageRequested(_) => {}
            EventPayload::DecisionQueued(p) => self.apply_decision_queued(p, ctx),
            EventPayload::DecisionDropped(p) => {
                // Voided like an interrupted decision: out of the queue, never delivered.
                self.remove_effect(EffectKind::Decision, &p.id);
                self.dequeue(EffectKind::Decision, &p.id);
            }
            EventPayload::CallVoided(p) => self.apply_call_voided(p),
            EventPayload::ConnectorSyncRequested(p) => {
                if !self.has_effect(EffectKind::ConnectorSync, &p.id) {
                    self.insert_effect(
                        &p.id,
                        EffectTracking::new(p.retry.clone(), now),
                        EffectPayload::ConnectorSync(ConnectorSyncState {
                            tools: Vec::new(),
                            prefix: None,
                            instructions: None,
                            error: None,
                            auth: None,
                        }),
                    );
                }
                // A retry re-arms the same entry rather than starting a new one:
                // one connection, one fetch, however many attempts. A settled
                // effect starts over, because its attempts are spent.
                if let Some(e) = self.effect_mut(EffectKind::ConnectorSync, &p.id) {
                    match e.tracking.status() {
                        EffectStatus::Failed | EffectStatus::Completed => e.tracking.restart(now),
                        _ => e.tracking.dispatch(now),
                    }
                }
            }
            EventPayload::ConnectorSyncCompleted(p) => {
                if let Some(e) = self.effect_mut(EffectKind::ConnectorSync, &p.id) {
                    e.tracking.complete();
                    if let Some(sync) = e.connector_mut() {
                        sync.tools = p.tools.clone();
                        sync.prefix = p.prefix.clone();
                        sync.instructions = p.instructions.clone();
                        sync.error = None;
                        sync.auth = None;
                    }
                }
            }
            EventPayload::ConnectorSyncErrored(p) => {
                if let Some(e) = self.effect_mut(EffectKind::ConnectorSync, &p.id) {
                    e.tracking.record_error(p.retryable, now);
                    if let Some(sync) = e.connector_mut() {
                        sync.error = Some(p.error.message.clone());
                        sync.auth = p.auth;
                    }
                }
            }
            // The fetch does not change. Only the credential is not valid.
            EventPayload::ConnectorAuthFailed(p) => {
                if let Some(sync) = self
                    .effect_mut(EffectKind::ConnectorSync, &p.id)
                    .and_then(EffectState::connector_mut)
                {
                    sync.auth = Some(p.auth);
                }
            }
            // Append-only, like the tree: `resolve_on_path` scans newest-first,
            // so superseded same-anchor writes never win resolution.
            EventPayload::WorkerStateUpdated(p) => {
                self.state_versions.push(Logged {
                    seq: ctx.sequence,
                    entry: StateVersion {
                        value: p.state.clone(),
                        anchor: p.anchor.clone(),
                    },
                });
            }
            EventPayload::AgentConfigUpdated(p) => {
                self.agent_versions.push(Logged {
                    seq: ctx.sequence,
                    entry: AgentVersion {
                        value: p.config.clone(),
                        anchor: p.anchor.clone(),
                    },
                });
            }
            EventPayload::SessionCancelled => {
                self.status = SessionStatus::Done;
                // Terminal: void pending decisions so late submissions no-op; calls void via CallVoided events.
                self.effects.retain(|(kind, _), e| {
                    *kind != EffectKind::Decision || e.tracking.status() != EffectStatus::Pending
                });
            }
            EventPayload::SessionDone(_) => {
                if !self.ancestry.is_empty() {
                    self.status = SessionStatus::Done;
                } else {
                    self.status = SessionStatus::Idle;
                }
                // A run terminal leaves nothing to finalize.
                self.schedule_queue
                    .retain(|e| e.kind != EffectKind::TurnEnd);
            }
            EventPayload::SubAgentTurnCompleted(payload) => {
                self.sub_agent_cost += payload.cost;
                self.turn_cost += payload.cost;
                self.sub_agent_token_usage.add(&payload.token_usage);
                self.turn_token_usage.add(&payload.token_usage);
                if let Some(e) = self.effect_mut(EffectKind::SubAgent, &payload.id) {
                    if let Some(sa) = e.sub_agent_mut() {
                        sa.result = Some(json_to_string(&payload.data));
                        sa.is_error = false;
                    }
                    // The result is what settles a delegation.
                    if e.tracking.status() == EffectStatus::Running {
                        e.tracking.complete();
                    }
                }
            }
            EventPayload::ChannelsUpdated(_) => {}
            EventPayload::TurnStarted(p) => {
                self.phase = TurnPhase::Active {
                    turn_id: p.turn_id.clone(),
                };
                self.turn_started_seq = Some(ctx.sequence);
                self.turn_cost = Decimal::ZERO;
                self.turn_token_usage = Usage::default();
            }
            EventPayload::TurnCompleted(payload) => {
                // The event names the turn it ends, and every emitter builds it
                // from the live phase, so the log needs no second source.
                self.completed_turn_ids.push(payload.turn_id.clone());
                self.data = payload.data.clone();
                self.turn_cost = Decimal::ZERO;
                self.turn_token_usage = Usage::default();
                self.phase = TurnPhase::Idle;
                self.turn_started_seq = None;
                self.dequeue(EffectKind::TurnEnd, &payload.turn_id);
            }
        }
    }

    /// The scheduling invariants, as one enumeration. `Working` asserts these
    /// after every emit (so a violation names the event that caused it) and the
    /// property tests assert them on committed state; both read this, so the
    /// two can't drift.
    pub(crate) fn check_invariants(&self) -> Result<(), String> {
        let live = self
            .effects_of(EffectKind::Decision)
            .filter(|e| e.tracking.status == EffectStatus::Pending)
            .count();
        if live > 1 {
            return Err(format!("decision slot violated: {live} live decisions"));
        }
        if live > 0 && self.has_pending_connector_sync(self.head_id.as_deref()) {
            return Err(
                "a decision is live while a fetch its config owes is in flight".to_string(),
            );
        }
        // Queue and statuses agree: Queued status ⟺ a queue entry. Fetches are
        // excluded on both sides — they never queue.
        let queued_statuses = self
            .effects
            .values()
            .filter(|e| e.kind() != EffectKind::ConnectorSync)
            .filter(|e| e.tracking.status == EffectStatus::Queued)
            .count();
        // TurnEnd entries carry no status of their own: their payload is the
        // finalizing phase, so they are counted against it instead.
        let effect_entries = self
            .schedule_queue
            .iter()
            .filter(|e| e.kind != EffectKind::TurnEnd)
            .count();
        if queued_statuses != effect_entries {
            return Err(format!(
                "queue ({effect_entries}) and Queued statuses ({queued_statuses}) disagree"
            ));
        }
        let turn_ends = self
            .schedule_queue
            .iter()
            .filter(|e| e.kind == EffectKind::TurnEnd)
            .count();
        if turn_ends > usize::from(self.phase.finalizing().is_some()) {
            return Err("a TurnEnd entry exists without a finalizing turn".to_string());
        }
        Ok(())
    }

    /// Newest open interrupt on the path to `leaf`; unanchored matches any path.
    pub fn active_interrupt_for(&self, leaf: Option<&str>) -> Option<&OpenInterrupt> {
        let on_path = leaf.map(|l| self.path_ids(l)).unwrap_or_default();
        resolve_on_path(&self.open_interrupts, &on_path)
    }

    /// All open interrupts on the path to `leaf`, oldest first.
    pub fn interrupts_for(&self, leaf: Option<&str>) -> Vec<&OpenInterrupt> {
        let on_path = leaf.map(|l| self.path_ids(l)).unwrap_or_default();
        self.open_interrupts
            .iter()
            .filter(|i| match i.anchor.as_deref() {
                None => true,
                Some(a) => on_path.contains(a),
            })
            .collect()
    }

    /// An open interrupt by id, on any path.
    pub fn open_interrupt(&self, id: &str) -> Option<&OpenInterrupt> {
        self.open_interrupts.iter().find(|i| i.interrupt_id == id)
    }

    /// Whether an open interrupt parks the active branch.
    pub fn head_parked(&self) -> bool {
        self.active_interrupt_for(self.head_id.as_deref()).is_some()
    }

    /// Stored status, overridden when an interrupt parks the head path.
    pub fn projected_status(&self) -> SessionStatus {
        match self.active_interrupt_for(self.head_id.as_deref()) {
            Some(i) => SessionStatus::Interrupted {
                interrupt_id: i.interrupt_id.clone(),
                origin: i.origin,
                reason: i.reason.clone(),
            },
            None => self.status.clone(),
        }
    }

    /// The anchor of the effect a decision settles.
    pub fn trigger_anchor(&self, trigger: &Trigger) -> Option<&str> {
        match trigger {
            Trigger::ToolFinished { id, .. } | Trigger::ToolExecute { id, .. } => {
                self.effect(EffectKind::ToolCall, id)
            }
            Trigger::SubAgentFinished { session_id, .. } => {
                self.effect(EffectKind::SubAgent, session_id)
            }
            Trigger::LlmFinished { id, .. } | Trigger::LlmExecute { id, .. } => {
                self.effect(EffectKind::LlmCall, id)
            }
            _ => None,
        }
        .and_then(|e| e.anchor.as_deref())
    }

    /// A decision's stored data, by id.
    pub fn worker_decision(&self, id: &str) -> Option<&WorkerDecisionState> {
        self.effect(EffectKind::Decision, id)
            .and_then(|e| e.decision())
    }

    /// Whether work at `anchor` lands on a parked branch: on-head anchors gate
    /// on the head, off-head anchors on their own path.
    pub fn anchor_parked(&self, anchor: Option<&str>) -> bool {
        match anchor {
            Some(a) if !self.anchor_on_path(self.head_id.as_deref(), Some(a)) => {
                self.active_interrupt_for(Some(a)).is_some()
            }
            _ => self.head_parked(),
        }
    }

    /// Whether a decision waits on a region elsewhere.
    pub fn decision_parked(&self, trigger: &Trigger) -> bool {
        self.decision_park(trigger).is_some()
    }

    /// The region holding a decision, if one does. Two members: an interrupt
    /// parks the branch the decision lands on, and a running turn parks a
    /// decision that opens a different one.
    pub fn decision_park(&self, trigger: &Trigger) -> Option<DecisionPark<'_>> {
        if let Some(interrupt) = self.decision_park_interrupt(trigger) {
            return Some(DecisionPark::Interrupt(interrupt));
        }
        // A different turn holds the phase — never "any turn is running": once
        // this decision dispatches, its own `TurnStarted` makes the phase Active
        // with its id, and a redelivery must not park against itself.
        match (trigger.deferred_turn_id(), self.phase.turn_id()) {
            (Some(turn_id), Some(active)) if active != turn_id => Some(DecisionPark::Turn(active)),
            _ => None,
        }
    }

    /// The open interrupt parking a decision's landing branch, if any:
    /// transcripts gate at their landing leaf, effect settles at their anchor.
    /// Actions are exempt: a click may be what answers the prompt.
    fn decision_park_interrupt(&self, trigger: &Trigger) -> Option<&OpenInterrupt> {
        if matches!(trigger, Trigger::ClientAction { .. }) {
            return None;
        }
        if let Trigger::ClientTranscript { messages, .. } = trigger {
            let messages = self.normalize_client_view(messages.clone());
            let known: std::collections::HashSet<&str> =
                self.nodes.iter().map(|n| n.message.id.as_str()).collect();
            let plan = super::reconcile::plan_reconcile(&known, &messages);
            let landing = super::reconcile::landing_leaf(&messages, &plan);
            return self.active_interrupt_for(landing.as_deref());
        }
        let anchor = self.trigger_anchor(trigger);
        match anchor {
            Some(a) if !self.anchor_on_path(self.head_id.as_deref(), Some(a)) => {
                self.active_interrupt_for(Some(a))
            }
            _ => self.active_interrupt_for(self.head_id.as_deref()),
        }
    }

    pub fn has_pending_worker_decision(&self) -> bool {
        self.decisions_with(EffectStatus::Pending).next().is_some()
    }

    /// All queued decisions in arrival order.
    pub fn queued_decisions(&self) -> Vec<&EffectState> {
        let mut queued: Vec<&EffectState> = self.decisions_with(EffectStatus::Queued).collect();
        queued.sort_by_key(|e| e.decision().map(|d| d.source_event_sequence));
        queued
    }

    pub fn has_queued_worker_decision(&self) -> bool {
        self.decisions_with(EffectStatus::Queued).next().is_some()
    }

    fn decisions_with(&self, status: EffectStatus) -> impl Iterator<Item = &EffectState> {
        self.effects_of(EffectKind::Decision)
            .filter(move |e| e.tracking.status == status)
    }

    /// All message ids on the root→`leaf` chain.
    pub fn path_ids<'a>(&'a self, leaf: &'a str) -> std::collections::HashSet<&'a str> {
        let by_id: HashMap<&str, &Logged<NewMessage>> = self
            .nodes
            .iter()
            .map(|n| (n.message.id.as_str(), n))
            .collect();
        let mut ids = std::collections::HashSet::new();
        let mut cursor = Some(leaf);
        while let Some(id) = cursor {
            let Some(node) = by_id.get(id) else { break };
            if !ids.insert(id) {
                break; // parent cycle guard
            }
            cursor = node.parent_id.as_deref();
        }
        ids
    }

    /// Whether `anchor` lies on the root→`leaf` path. A `None` anchor matches any path.
    pub fn anchor_on_path(&self, leaf: Option<&str>, anchor: Option<&str>) -> bool {
        match anchor {
            None => true,
            Some(a) => leaf.is_some_and(|l| self.path_ids(l).contains(a)),
        }
    }

    /// Newest worker state whose anchor is on the path to `leaf`; unanchored matches any path.
    pub fn resolve_state_for(&self, leaf: Option<&str>) -> WorkerState {
        let on_path = leaf.map(|l| self.path_ids(l)).unwrap_or_default();
        resolve_on_path(&self.state_versions, &on_path)
            .map(|v| v.value.clone())
            .unwrap_or_default()
    }

    /// Where a call to `name` runs, per the config in force on the current path.
    ///
    /// Derived rather than declared: a tool the config marks `handler: client`
    /// runs on the client, a name a connector resolved runs on the engine, and
    /// everything else — including a name nothing declares — runs on the worker,
    /// which is where an undeclared name gets its contract error.
    ///
    /// A declared tool is checked first, matching `merge`: a name the config
    /// claims is never taken by a connector.
    pub fn tool_handler_for(&self, name: &str) -> ToolHandler {
        let config = self.resolve_agent_for(self.head_id.as_deref());
        match config.as_ref().and_then(|c| c.tool(name)) {
            Some(t) => ToolHandler::declared(t.handler),
            None if self.connector_tool_for(name).is_some() => ToolHandler::Server,
            None => ToolHandler::Worker,
        }
    }

    /// Every model tool call this session took on, settled ones included.
    pub fn dispatched_calls(&self) -> Vec<String> {
        self.effects
            .values()
            .filter_map(|e| match e.kind() {
                EffectKind::ToolCall => Some(e.id.clone()),
                // The key is the child session; the call is inside.
                EffectKind::SubAgent => e.sub_agent().map(|s| s.tool_call_id.clone()),
                _ => None,
            })
            .collect()
    }

    /// The connector tool `name` resolves to on the current path, if any. The
    /// executor reads `connector`/`remote_name` off this to place the call.
    pub fn connector_tool_for(&self, name: &str) -> Option<ConnectorTool> {
        self.connector_tools(self.head_id.as_deref())
            .tools
            .into_iter()
            .find(|t| t.name == name)
    }

    /// The connector tools in force on the path to `leaf`, derived by filtering
    /// each fetched offer through the config's `McpServer` entry.
    ///
    /// Pure, and recomputed rather than stored: the offer is the recorded fact,
    /// so editing a filter or flipping `prefix_tools` re-derives without another
    /// round trip. `collisions` reports every name dropped for clashing with a
    /// declared tool, a sub-agent, or another connector.
    pub fn connector_tools(&self, leaf: Option<&str>) -> filter::Merged {
        let Some(config) = self.resolve_agent_for(leaf) else {
            return filter::Merged {
                tools: Vec::new(),
                collisions: Vec::new(),
            };
        };
        self.connector_tools_for_config(&config)
    }

    pub fn connector_tools_for_config(&self, config: &AgentConfig) -> filter::Merged {
        let servers = self.servers_for(config);
        let mut resolutions: Vec<filter::Resolution> = servers
            .iter()
            .filter_map(|connector| {
                let effect = self.effect(EffectKind::ConnectorSync, &connector.id)?;
                let sync = effect.connector()?;
                effect.tracking.is_ready().then(|| {
                    filter::resolve(
                        connector,
                        &sync.tools,
                        sync.prefix.as_deref(),
                        filter::defers(connector, config.defers_tools()),
                    )
                })
            })
            .collect();
        // From the config alone: a fetch that has not settled must not decide
        // whether a tool definition exists. An agent that says `search` thus
        // gets these from its first turn, and a connection added later moves
        // no definition. A plugin's servers count for the same reason.
        let defers = config.defers_tools()
            || config.tools.iter().any(|t| t.defer == Some(true))
            || config.mcp.iter().any(|c| filter::defers(c, false))
            || config.plugins.iter().any(|p| {
                p.servers
                    .iter()
                    .any(|id| filter::defers(&p.server(id), false))
            });
        if defers {
            resolutions.push(filter::Resolution::of(filter::search_tools(
                config.defer_strategy(),
            )));
        }
        if !config.plugins.is_empty() {
            resolutions.push(filter::Resolution::of(vec![filter::skill_tool()]));
        }
        let taken: Vec<&str> = config
            .tools
            .iter()
            .map(|t| t.name.as_str())
            .chain(config.sub_agents.iter().map(|s| s.id.as_str()))
            .collect();
        filter::merge(resolutions, taken)
    }

    /// The connection as the config at `leaf` names it, with the offer this
    /// session recorded.
    fn connector_source(&self, connector_id: &str, leaf: Option<&str>) -> Option<Source> {
        let config = self.resolve_agent_for(leaf)?;
        let server = self
            .servers_for(&config)
            .into_iter()
            .find(|c| c.id == connector_id)?;
        let sync = self.connector_sync(connector_id)?;
        Some(Source {
            server,
            offered: sync.tools.clone(),
            instructions: sync.instructions.clone(),
        })
    }

    /// Every connection of the agent whose offer has arrived, with that offer.
    /// What the pair of search tools answers over.
    ///
    /// Every connection, not only the searched ones. A connection on `all` is
    /// listed up front *and* findable, so one search covers the agent and an
    /// answer of nothing means nothing is there.
    ///
    /// Readiness belongs here and not in the tool list: a connection that
    /// arrives during a session joins the next answer, and no definition moves.
    fn searchable_connectors(&self, leaf: Option<&str>) -> Vec<Source> {
        let Some(config) = self.resolve_agent_for(leaf) else {
            return Vec::new();
        };
        self.servers_for(&config)
            .iter()
            .filter(|c| {
                self.effect(EffectKind::ConnectorSync, &c.id)
                    .is_some_and(|e| e.tracking.is_ready())
            })
            .filter_map(|c| self.connector_source(&c.id, leaf))
            .collect()
    }

    /// The engine's answer to one of its own tools, or `None` when the call is
    /// the connection's. Read from state, so a replay answers the same.
    pub fn local_connector_answer(&self, tool_call_id: &str) -> Option<LocalAnswer> {
        let effect = self.effect(EffectKind::ToolCall, tool_call_id)?;
        let leaf = effect.anchor.clone();
        let tc = effect.tool()?;
        let target = tc.target.as_ref()?;
        super::engine_tools::answer(self, target.kind, leaf.as_deref(), &tc.arguments)
    }

    /// A skill call's branch, plugin, and arguments, for the executor that
    /// answers it from the bundle. `None` for every other call. The plugin is
    /// the routing frozen on the call, not a fresh read of the arguments.
    pub fn skill_call(&self, tool_call_id: &str) -> Option<SkillCall> {
        let effect = self.effect(EffectKind::ToolCall, tool_call_id)?;
        let tc = effect.tool()?;
        let target = tc.target.as_ref()?;
        (target.kind == crate::protocol::ConnectorToolKind::Skill).then(|| SkillCall {
            leaf: effect.anchor.clone(),
            plugin_id: target.connector.clone(),
            arguments: tc.arguments.clone(),
        })
    }

    /// Every tool the agent can reach, deferred or not, as the model would see
    /// it. What a search answers over.
    ///
    /// Each source, not only the connections: deferral is a property of a tool,
    /// so a search that skipped one source would report an absence that is not
    /// real. The engine's own two are left out — they are how you search, not
    /// something to find — and so are the sub-agents, which a `call_tool`
    /// cannot place.
    pub fn searchable_tools(&self, leaf: Option<&str>) -> Vec<LlmTool> {
        let Some(config) = self.resolve_agent_for(leaf) else {
            return Vec::new();
        };
        let declared = config
            .tools
            .iter()
            .map(|t| t.to_llm_tool(config.defers_tools()));
        let connector = self
            .connector_tools_for_config(&config)
            .tools
            .into_iter()
            .filter(|t| t.kind.is_remote())
            .map(|t| t.to_llm_tool())
            .collect::<Vec<_>>();
        declared.chain(connector).collect()
    }

    /// What one connection is, for an announcement: its size, and its own
    /// words. `None` while the connection has not settled, so a notice never
    /// claims a server the engine cannot yet reach.
    pub fn connection_summary(&self, id: &str, leaf: Option<&str>) -> Option<String> {
        let source = self
            .searchable_connectors(leaf)
            .into_iter()
            .find(|source| source.server.id == id)?;
        serde_json::to_string(&Summary {
            mcp_server: &source.server.id,
            tools: filter::callable(&source.server, &source.offered).len(),
            about: source.instructions.as_deref(),
        })
        .ok()
    }

    /// Context ids that an earlier call of this path already carried.
    ///
    /// Read from the effects on the path, so a fork that never held a call does
    /// not inherit what it said, and a rewind that removed one lets it be said
    /// again.
    pub fn context_ids_on_path(
        &self,
        leaf: Option<&str>,
        exclude: &str,
    ) -> std::collections::HashSet<String> {
        let path = self.path_set(leaf);
        self.effects
            .values()
            .filter(|e| e.id != exclude && Self::anchored_on(&path, e.anchor.as_deref()))
            .filter_map(EffectState::llm)
            .flat_map(|call| call.context_ids.iter().cloned())
            .collect()
    }

    /// Whether the system prefix is still free to write.
    ///
    /// It is free until a call commits it. A provider caches the prefix, and
    /// the Anthropic wire gathers every system message into it whatever the
    /// position, so a system message added later rewrites what is cached.
    pub fn system_prefix_open(&self, leaf: Option<&str>, exclude: &str) -> bool {
        let path = self.path_set(leaf);
        !self.effects.values().any(|e| {
            e.id != exclude
                && matches!(e.payload, EffectPayload::LlmCall(_))
                && Self::anchored_on(&path, e.anchor.as_deref())
                && e.tracking.status() != EffectStatus::Queued
        })
    }

    /// The ids from the root to `leaf`, walked once. A caller that asked per
    /// effect would walk the whole path again for each one.
    fn path_set<'a>(&'a self, leaf: Option<&'a str>) -> std::collections::HashSet<&'a str> {
        leaf.map(|l| self.path_ids(l)).unwrap_or_default()
    }

    /// An unanchored effect belongs to every path, the same way an unanchored
    /// agent version does.
    fn anchored_on(path: &std::collections::HashSet<&str>, anchor: Option<&str>) -> bool {
        anchor.is_none_or(|a| path.contains(a))
    }

    /// The tool a `call_tool` names, and where a call to it runs.
    pub fn call_tool_target(&self, named: &str, leaf: Option<&str>) -> Option<CallTarget> {
        let config = self.resolve_agent_for(leaf)?;
        if let Some(declared) = config.tool(named) {
            return Some(CallTarget::Declared(declared.clone()));
        }
        self.connector_tool_for(named)
            .filter(|t| t.kind.is_remote())
            .map(CallTarget::Connector)
    }

    /// Why a `call_tool` cannot be placed, if it cannot: an unknown name, or
    /// arguments that break that tool's schema.
    ///
    /// Each fault carries what the model needs to fix it. The engine holds the
    /// schema the provider never received, so a fault that withheld it would
    /// leave the model to guess or to search again.
    ///
    /// One function for two callers. The route freezes an empty remote name
    /// when this answers `Some`, and the answer reads the same fault again to
    /// tell the model which of the three it was.
    pub fn call_tool_fault(&self, arguments: &str, leaf: Option<&str>) -> Option<String> {
        let raw: serde_json::Value = serde_json::from_str(arguments).unwrap_or_default();
        let named = raw
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();

        let Some(target) = self.call_tool_target(&named, leaf) else {
            // The name is the query: a wrong name is usually a near miss, and
            // the same search the model should have run ranks the neighbours.
            let tools = self.searchable_tools(leaf);
            let cap = self
                .resolve_agent_for(leaf)
                .map(|c| c.defer_settings())
                .unwrap_or_default()
                .max_matches
                .get();
            let near: Vec<&str> = filter::find(&tools, &named)
                .into_iter()
                .take(cap)
                .map(|t| t.name.as_str())
                .collect();
            return Some(if near.is_empty() {
                format!(
                    "no tool `{named}` for this agent. Call `{}` with an empty query for every \
                     tool.",
                    filter::TOOL_SEARCH
                )
            } else {
                format!(
                    "no tool `{named}` for this agent. The closest are: {}. Call `{}` for the \
                     schema of one.",
                    near.join(", "),
                    filter::TOOL_SEARCH
                )
            });
        };
        // The provider never received this tool's schema, so it checked
        // nothing. The engine holds one, so the engine checks it — and hands it
        // back, because the model is the party that can act on it.
        let input = target.input();
        classify_arguments(&inner_arguments(&raw), input.as_ref())
            .error()
            .map(|e| match &input {
                Some(schema) => format!("`{named}`: {e}. Its input schema is: {schema}"),
                None => format!("`{named}`: {e}"),
            })
    }

    /// Connections the config in force on `leaf` names but has never fetched.
    /// Each needs a `connector.sync.requested` before the model can be prompted.
    pub fn unsynced_connectors(&self, leaf: Option<&str>) -> Vec<String> {
        let Some(config) = self.resolve_agent_for(leaf) else {
            return Vec::new();
        };
        self.servers_for(&config)
            .iter()
            .filter(|c| !self.has_effect(EffectKind::ConnectorSync, &c.id))
            .map(|c| c.id.clone())
            .collect()
    }

    /// Whether a fetch the config on `leaf` depends on is still unsettled. A
    /// decision that would prompt the model parks behind this, the same way it
    /// parks behind an unsettled `session.start` — the config names a connection
    /// whose tools are not known yet, so the turn cannot be authored against it.
    ///
    /// A terminally failed fetch is settled, so it never parks anything: the
    /// engine unblocks and the worker decides whether a missing connector is
    /// fatal.
    pub fn has_pending_connector_sync(&self, leaf: Option<&str>) -> bool {
        let Some(config) = self.resolve_agent_for(leaf) else {
            return false;
        };
        self.servers_for(&config).iter().any(|c| {
            self.tracking(EffectKind::ConnectorSync, &c.id)
                .is_some_and(EffectTracking::is_in_flight)
        })
    }

    /// Newest agent config whose anchor is on the path to `leaf`; unanchored matches
    /// any path. `None` when no config was ever written on the path.
    /// The retry policies the config in force declares, if any. None ⇒ nothing
    /// declared, and every kind falls to the engine's own default.
    pub fn retry_config(&self) -> Option<RetryConfig> {
        self.resolve_agent_for(self.head_id.as_deref())
            .and_then(|c| c.retry)
            .map(|b| *b)
    }

    pub fn resolve_agent_for(&self, leaf: Option<&str>) -> Option<AgentConfig> {
        let on_path = leaf.map(|l| self.path_ids(l)).unwrap_or_default();
        resolve_on_path(&self.agent_versions, &on_path).map(|v| v.value.clone())
    }

    /// Every server connection `config` reaches: its `mcp` entries, then each
    /// plugin's servers under the plugin's own policy. The one place a
    /// plugin's servers join the `mcp` machinery.
    ///
    /// Naming a plugin is what turns it on, so the config in force is the
    /// whole answer — a plugin written in mid-session is reached here, and the
    /// schedule owes its fetch like any connection's.
    pub fn servers_for(&self, config: &AgentConfig) -> Vec<McpServer> {
        let plugin_servers = config
            .plugins
            .iter()
            .flat_map(|p| p.servers.iter().map(|id| p.server(id)));
        config.mcp.iter().cloned().chain(plugin_servers).collect()
    }

    pub fn message_tree(&self) -> MessageTree {
        MessageTree {
            nodes: self.nodes.iter().map(|n| n.entry.clone()).collect(),
            head_id: self.head_id.clone(),
        }
    }

    /// The effects still open: queued, in flight, or awaiting a retry.
    ///
    /// A fetch is surfaced too, so a worker sees one in flight rather than
    /// inferring it from a decision that has not arrived.
    /// Nothing to do until the caller answers: every outstanding call is one
    /// only the client can settle.
    pub fn waiting_on_client(&self) -> bool {
        waiting_on_client(&self.effects())
    }

    pub fn effects(&self) -> Vec<Effect> {
        let mut effects: Vec<Effect> = self
            .effects
            .values()
            .filter(|e| e.kind() != EffectKind::Decision)
            .filter(|e| {
                matches!(
                    e.tracking.status(),
                    EffectStatus::Queued
                        | EffectStatus::Pending
                        | EffectStatus::Running
                        | EffectStatus::RetryScheduled
                )
            })
            .map(|e| {
                let mut wire = Effect {
                    id: e.id.clone(),
                    kind: e.kind(),
                    status: e.tracking.status(),
                    attempt: e.tracking.retry.attempts,
                    deadline: e.tracking.deadline,
                    anchor: e.anchor.clone(),
                    name: None,
                    arguments: None,
                    handler: None,
                    stream: None,
                    agent_id: None,
                    tool_call_id: None,
                };
                match &e.payload {
                    EffectPayload::ToolCall(c) => {
                        wire.name = Some(c.name.clone());
                        wire.arguments = Some(c.arguments.clone());
                        wire.handler = Some(c.handler.into());
                    }
                    EffectPayload::SubAgent(c) => {
                        wire.agent_id = Some(c.agent_id.clone());
                        wire.tool_call_id = Some(c.tool_call_id.clone());
                    }
                    EffectPayload::LlmCall(c) => {
                        wire.handler = Some(c.handler.into());
                        wire.stream = Some(c.stream);
                    }
                    // The connection being fetched; unanchored, since a fetch
                    // belongs to the connection, not to a branch.
                    EffectPayload::ConnectorSync(_) => wire.name = Some(e.id.clone()),
                    EffectPayload::Decision(_) => unreachable!(),
                }
                wire
            })
            .collect();
        effects.sort_by(|a, b| a.id.cmp(&b.id));
        effects
    }

    /// A call is open while any of its tool calls lacks a recorded answer on the
    /// active path. The tree is frozen during a pending decision, so the call
    /// answering the last `tool.finished` is still open when that decision is
    /// projected — exactly when its re-issue proposal is derived.
    pub(crate) fn open_llm_calls(&self, tree: &MessageTree) -> HashMap<String, EffectState> {
        let Some(head) = tree.head_id.as_deref() else {
            return HashMap::new();
        };
        let path = tree.path_to(head);
        let answered: std::collections::HashSet<&str> = path
            .iter()
            .filter_map(|m| m.tool_call_id.as_deref())
            .collect();
        path.iter()
            .filter(|m| {
                m.tool_calls
                    .iter()
                    .any(|c| !answered.contains(c.id.as_str()))
            })
            .filter_map(|m| {
                self.effect(EffectKind::LlmCall, &m.id)
                    .map(|e| (m.id.clone(), e.clone()))
            })
            .collect()
    }

    pub fn event_meta(&self, now: DateTime<Utc>) -> EventMeta {
        let mut decisions: Vec<MetaDecision> = self
            .effects_of(EffectKind::Decision)
            .filter(|e| {
                matches!(
                    e.tracking.status(),
                    EffectStatus::Pending | EffectStatus::Queued
                )
            })
            .filter_map(|e| {
                let d = e.decision()?;
                Some(MetaDecision {
                    decision_id: e.id.clone(),
                    status: e.tracking.status(),
                    finished: matches!(
                        d.trigger,
                        Trigger::ToolFinished { .. } | Trigger::SubAgentFinished { .. }
                    ),
                    attempts: e.tracking.retry.attempts,
                    deadline: e.tracking.deadline,
                    source_event_sequence: d.source_event_sequence,
                })
            })
            .collect();

        decisions.sort_by_key(|d| d.source_event_sequence);

        EventMeta {
            status: self.projected_status(),
            wake_at: super::schedule::wake_at(self, now),
            owner: self.owner.clone(),
            agent_id: self.agent_id.clone(),
            ancestry: self.ancestry.clone(),
            turn_id: self.turn_id().map(str::to_string),
            cost: self.cost,
            sub_agent_cost: self.sub_agent_cost,
            head_id: self.head_id.clone(),
            calls: self.effects(),
            decisions,
        }
    }

    /// Rewind to an event: every history-shaped log is append-only and
    /// seq-tagged, so the state at the event is the prefix of entries the
    /// event's own seq admits. Call maps stay current (entries immutable,
    /// path picks the as-of subset); path-dependent resolution after a
    /// rewind is exact.
    #[must_use]
    pub fn rewind(mut self, seq: u64, head_id: Option<&str>) -> Self {
        rewind_log(&mut self.nodes, seq);
        rewind_log(&mut self.state_versions, seq);
        rewind_log(&mut self.agent_versions, seq);
        self.head_id = head_id.map(str::to_string);
        self
    }

    fn track_usage(&mut self, usage: &Option<Usage>) {
        if let Some(u) = usage {
            self.token_usage.add(u);
        }
    }

    fn track_turn_usage(&mut self, usage: &Option<Usage>) {
        if let Some(u) = usage {
            self.turn_token_usage.add(u);
        }
    }
}

#[cfg(test)]
mod open_llm_calls_tests {
    use super::*;
    use crate::protocol::{NewMessage, ToolCall, ToolCallFunction};

    fn node(message: Message, parent_id: Option<&str>) -> Logged<NewMessage> {
        Logged {
            seq: 0,
            entry: NewMessage {
                message,
                parent_id: parent_id.map(str::to_string),
            },
        }
    }

    fn user(id: &str) -> Message {
        Message {
            id: id.to_string(),
            role: Role::User,
            content: None,
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
            reasoning: None,
        }
    }

    fn assistant_calling(id: &str, tool_call_id: &str) -> Message {
        Message {
            id: id.to_string(),
            role: Role::Assistant,
            content: None,
            tool_calls: vec![ToolCall {
                id: tool_call_id.to_string(),
                call_type: "function".to_string(),
                function: ToolCallFunction {
                    name: "t".to_string(),
                    arguments: "{}".to_string(),
                },
            }],
            tool_call_id: None,
            name: None,
            reasoning: None,
        }
    }

    fn tool_answer(id: &str, tool_call_id: &str) -> Message {
        Message {
            id: id.to_string(),
            role: Role::Tool,
            content: None,
            tool_calls: vec![],
            tool_call_id: Some(tool_call_id.to_string()),
            name: None,
            reasoning: None,
        }
    }

    fn call_state(call_id: &str) -> EffectState {
        EffectState::new(
            call_id,
            EffectTracking::new(RetryPolicy::no_retry(), Utc::now()),
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
                handler: LlmHandler::Server,
            }),
        )
    }

    #[test]
    fn a_call_is_open_until_its_tool_answer_is_recorded() {
        let mut s = SessionState::new("sess-1".to_string());
        s.nodes.push(node(user("u1"), None));
        s.nodes
            .push(node(assistant_calling("call-1", "tc-1"), Some("u1")));
        s.head_id = Some("call-1".to_string());
        s.put_effect(call_state("call-1"));

        assert!(
            s.open_llm_calls(&s.message_tree()).contains_key("call-1"),
            "unanswered tool call keeps the parent open"
        );

        s.nodes
            .push(node(tool_answer("t1", "tc-1"), Some("call-1")));
        s.head_id = Some("t1".to_string());
        assert!(
            s.open_llm_calls(&s.message_tree()).is_empty(),
            "recorded answer closes the call"
        );
    }
}

#[cfg(test)]
mod state_version_tests {
    use serde_json::json;

    use super::*;
    use crate::protocol::NewMessage;

    fn message_node(id: &str, parent_id: Option<&str>) -> Logged<NewMessage> {
        Logged {
            seq: 0,
            entry: NewMessage {
                message: Message {
                    id: id.to_string(),
                    role: Role::User,
                    content: None,
                    tool_calls: vec![],
                    tool_call_id: None,
                    name: None,
                    reasoning: None,
                },
                parent_id: parent_id.map(str::to_string),
            },
        }
    }

    /// A session with the tree  u1 → a1 → u2  and a fork leaf  u1 → x1.
    fn forked_session() -> SessionState {
        let mut s = SessionState::new("sess-1".to_string());
        s.nodes.push(message_node("u1", None));
        s.nodes.push(message_node("a1", Some("u1")));
        s.nodes.push(message_node("u2", Some("a1")));
        s.nodes.push(message_node("x1", Some("u1")));
        s
    }

    fn version(state: serde_json::Value, anchor: Option<&str>) -> Logged<StateVersion> {
        Logged {
            seq: 0,
            entry: StateVersion {
                value: WorkerState(state),
                anchor: anchor.map(str::to_string),
            },
        }
    }

    #[test]
    fn empty_versions_resolve_to_default() {
        let s = forked_session();
        assert_eq!(s.resolve_state_for(Some("u2")), WorkerState::default());
        assert_eq!(s.resolve_state_for(None), WorkerState::default());
    }

    #[test]
    fn linear_resolution_latest_on_path_wins() {
        let mut s = forked_session();
        s.state_versions.push(version(json!({"v": 1}), Some("u1")));
        s.state_versions.push(version(json!({"v": 2}), Some("u2")));
        assert_eq!(s.resolve_state_for(Some("u2")).0, json!({"v": 2}));
    }

    #[test]
    fn fork_resolves_as_of_the_fork_point() {
        let mut s = forked_session();
        s.state_versions.push(version(json!({"v": 1}), Some("u1")));
        s.state_versions.push(version(json!({"v": 2}), Some("u2")));
        // u2's version is off the u1→x1 path; the branch sees state as-of u1.
        assert_eq!(s.resolve_state_for(Some("x1")).0, json!({"v": 1}));
    }

    #[test]
    fn unanchored_version_matches_any_path() {
        let mut s = forked_session();
        s.state_versions.push(version(json!({"v": 0}), None));
        s.state_versions.push(version(json!({"v": 2}), Some("u2")));
        assert_eq!(s.resolve_state_for(Some("x1")).0, json!({"v": 0}));
        assert_eq!(s.resolve_state_for(Some("u2")).0, json!({"v": 2}));
        assert_eq!(s.resolve_state_for(None).0, json!({"v": 0}));
    }

    #[test]
    fn same_anchor_versions_resolve_to_the_newest() {
        let mut s = forked_session();
        s.head_id = Some("u2".to_string());
        let ctx = ApplyContext {
            occurred_at: Utc::now(),
            sequence: 1,
        };
        for (i, anchor) in [(1, Some("u1")), (2, Some("u2")), (3, Some("u2"))] {
            s.apply(
                &EventPayload::WorkerStateUpdated(WorkerStateUpdated {
                    state: WorkerState(json!({ "v": i })),
                    anchor: anchor.map(str::to_string),
                }),
                &ctx,
            );
        }
        // Append-only: superseded same-anchor writes stay but never win.
        assert_eq!(s.state_versions.len(), 3);
        assert_eq!(s.resolve_state_for(s.head_id.as_deref()).0, json!({"v": 3}));
        assert_eq!(s.resolve_state_for(Some("x1")).0, json!({"v": 1}));
    }

    #[test]
    fn a_version_serializes_flat_under_its_log_tag() {
        // Guard: the log tag flattens alongside the version rather than nesting.
        let v = version(json!({"v": 1}), Some("u1"));
        let value = serde_json::to_value(&v).expect("serializes");
        assert_eq!(value, json!({"seq": 0, "value": {"v": 1}, "anchor": "u1"}));
    }

    /// Rewinding by an event's seq restores every history log to its prefix.
    #[test]
    fn rewind_restores_each_log_to_the_events_prefix() {
        let mut s = SessionState::new("sess-1".to_string());
        let apply = |s: &mut SessionState, seq: u64, payload: EventPayload| {
            s.apply(
                &payload,
                &ApplyContext {
                    occurred_at: Utc::now(),
                    sequence: seq,
                },
            );
        };
        let msg = |id: &str, parent: Option<&str>| {
            EventPayload::NewMessage(NewMessage {
                message: Message {
                    id: id.to_string(),
                    role: Role::User,
                    content: None,
                    tool_calls: vec![],
                    tool_call_id: None,
                    name: None,
                    reasoning: None,
                },
                parent_id: parent.map(str::to_string),
            })
        };
        let state = |v: serde_json::Value, anchor: &str| {
            EventPayload::WorkerStateUpdated(WorkerStateUpdated {
                state: WorkerState(v),
                anchor: Some(anchor.to_string()),
            })
        };

        apply(&mut s, 1, msg("u1", None));
        apply(&mut s, 2, state(json!({"v": 1}), "u1"));
        apply(&mut s, 3, msg("u2", Some("u1")));
        apply(&mut s, 4, state(json!({"v": 2}), "u2"));

        let s = s.rewind(2, Some("u1"));
        assert_eq!(s.nodes.len(), 1);
        assert_eq!(s.state_versions.len(), 1);
        assert_eq!(s.head_id.as_deref(), Some("u1"));
        assert_eq!(s.resolve_state_for(s.head_id.as_deref()).0, json!({"v": 1}));
    }
}

#[cfg(test)]
mod agent_version_tests {
    use super::*;
    use crate::protocol::NewMessage;

    fn message_node(id: &str, parent_id: Option<&str>) -> Logged<NewMessage> {
        Logged {
            seq: 0,
            entry: NewMessage {
                message: Message {
                    id: id.to_string(),
                    role: Role::User,
                    content: None,
                    tool_calls: vec![],
                    tool_call_id: None,
                    name: None,
                    reasoning: None,
                },
                parent_id: parent_id.map(str::to_string),
            },
        }
    }

    /// A session with the tree  u1 → a1 → u2  and a fork leaf  u1 → x1.
    fn forked_session() -> SessionState {
        let mut s = SessionState::new("sess-1".to_string());
        s.nodes.push(message_node("u1", None));
        s.nodes.push(message_node("a1", Some("u1")));
        s.nodes.push(message_node("u2", Some("a1")));
        s.nodes.push(message_node("x1", Some("u1")));
        s
    }

    fn config(model: &str) -> AgentConfig {
        AgentConfig {
            llm: Some("claude".to_string()),
            model: model.to_string(),
            system: None,
            retry: None,
            tools: Vec::new(),
            sub_agents: Vec::new(),
            mcp: Vec::new(),
            defer_tools: None,
            announce_mcp: Default::default(),
            plugins: Vec::new(),
            effort: None,
        }
    }

    fn version(model: &str, anchor: Option<&str>) -> Logged<AgentVersion> {
        Logged {
            seq: 0,
            entry: AgentVersion {
                value: config(model),
                anchor: anchor.map(str::to_string),
            },
        }
    }

    #[test]
    fn empty_versions_resolve_to_none() {
        let s = forked_session();
        assert_eq!(s.resolve_agent_for(Some("u2")), None);
        assert_eq!(s.resolve_agent_for(None), None);
    }

    #[test]
    fn linear_resolution_latest_on_path_wins() {
        let mut s = forked_session();
        s.agent_versions.push(version("m1", Some("u1")));
        s.agent_versions.push(version("m2", Some("u2")));
        assert_eq!(s.resolve_agent_for(Some("u2")), Some(config("m2")));
    }

    #[test]
    fn fork_resolves_as_of_the_fork_point() {
        let mut s = forked_session();
        s.agent_versions.push(version("m1", Some("u1")));
        s.agent_versions.push(version("m2", Some("u2")));
        // m2 is off the u1→x1 path; the branch sees config as-of u1.
        assert_eq!(s.resolve_agent_for(Some("x1")), Some(config("m1")));
    }

    /// The summary answers at a place in the tree, and not at the head.
    ///
    /// A call built where a connection existed must be able to describe that
    /// connection, whatever the head has done since. The head is for choosing
    /// what to do next; a call already in flight reads its own anchor.
    #[test]
    fn a_connection_is_summarised_where_the_work_was_authored() {
        let with_sentry = |model: &str, anchor: Option<&str>| Logged {
            seq: 0,
            entry: AgentVersion {
                value: AgentConfig {
                    mcp: vec![McpServer {
                        id: "sentry".to_string(),
                        tools: None,
                        auth_failure: Default::default(),
                        approve: Default::default(),
                    }],
                    ..config(model)
                },
                anchor: anchor.map(str::to_string),
            },
        };
        let mut s = forked_session();
        s.head_id = Some("u2".to_string());
        // u1 has the connection; u2 drops it. x1 forks from u1 and keeps it.
        s.agent_versions.push(with_sentry("m1", Some("u1")));
        s.agent_versions.push(version("m2", Some("u2")));
        s.insert_effect(
            "sentry",
            EffectTracking::new_queued(RetryPolicy::no_retry()),
            EffectPayload::ConnectorSync(ConnectorSyncState {
                prefix: Some("sentry".to_string()),
                tools: vec![RemoteTool {
                    name: "search_issues".to_string(),
                    description: String::new(),
                    input: None,
                    output: None,
                    annotations: Default::default(),
                }],
                instructions: None,
                error: None,
                auth: None,
            }),
        );
        if let Some(e) = s.effect_mut(EffectKind::ConnectorSync, "sentry") {
            e.tracking.dispatch(Utc::now());
            e.tracking.complete();
        }

        assert!(
            s.connection_summary("sentry", Some("x1")).is_some(),
            "the fork still holds the connection, so a call there can describe it"
        );
        assert!(
            s.connection_summary("sentry", Some("u2")).is_none(),
            "u2 dropped it, so a call there has nothing to describe"
        );
    }

    #[test]
    fn unanchored_config_is_the_universal_fallback() {
        let mut s = forked_session();
        s.agent_versions.push(version("m0", None));
        s.agent_versions.push(version("m2", Some("u2")));
        assert_eq!(s.resolve_agent_for(Some("x1")), Some(config("m0")));
        assert_eq!(s.resolve_agent_for(Some("u2")), Some(config("m2")));
        assert_eq!(s.resolve_agent_for(None), Some(config("m0")));
    }

    #[test]
    fn same_anchor_configs_resolve_to_the_newest() {
        let mut s = forked_session();
        s.head_id = Some("u2".to_string());
        let ctx = ApplyContext {
            occurred_at: Utc::now(),
            sequence: 1,
        };
        for (model, anchor) in [("m1", Some("u1")), ("m2", Some("u2")), ("m3", Some("u2"))] {
            s.apply(
                &EventPayload::AgentConfigUpdated(AgentConfigUpdated {
                    config: config(model),
                    anchor: anchor.map(str::to_string),
                }),
                &ctx,
            );
        }
        // Append-only: superseded same-anchor writes stay but never win.
        assert_eq!(s.agent_versions.len(), 3);
        assert_eq!(
            s.resolve_agent_for(s.head_id.as_deref()),
            Some(config("m3"))
        );
        assert_eq!(s.resolve_agent_for(Some("x1")), Some(config("m1")));
    }
}

#[cfg(test)]
mod effect_tests {
    use super::*;

    #[test]
    fn effect_serializes_flat_tagged_and_round_trips() {
        let e = Effect {
            id: "call_1".to_string(),
            kind: EffectKind::ToolCall,
            status: EffectStatus::Pending,
            attempt: 0,
            deadline: None,
            anchor: Some("n1".to_string()),
            name: Some("get_weather".to_string()),
            arguments: Some("{}".to_string()),
            handler: Some(crate::protocol::Handler::Worker),
            stream: None,
            agent_id: None,
            tool_call_id: None,
        };
        let json = serde_json::to_value(&e).unwrap();
        // Envelope, tag, and kind-specific fields all on one flat object.
        assert_eq!(json["id"], "call_1");
        assert_eq!(json["status"], "pending");
        assert_eq!(json["kind"], "tool_call");
        assert_eq!(json["name"], "get_weather");
        assert_eq!(json["handler"], "worker");
        // Absent kinds' fields are omitted, not null.
        assert!(json.get("stream").is_none());
        assert!(json.get("agent_id").is_none());
        let back: Effect = serde_json::from_value(json).unwrap();
        assert_eq!(back, e);
    }
}

#[cfg(test)]
mod node_wire_compat_tests {
    use crate::protocol::NewMessage;

    #[test]
    fn legacy_kind_tagged_node_json_still_deserializes() {
        // Trees persisted before the Node union was removed carry `kind`.
        let node: NewMessage = serde_json::from_value(serde_json::json!({
            "kind": "message",
            "message": {"id": "m1", "role": "user", "content": "hi", "tool_calls": []},
            "parent_id": null,
        }))
        .expect("unknown fields are ignored");
        assert_eq!(node.message.id, "m1");
    }
}

/// Nothing to do until the caller answers: every outstanding call is one only
/// the client can settle.
pub fn waiting_on_client(calls: &[Effect]) -> bool {
    !calls.is_empty() && calls.iter().all(|c| c.handler == Some(Handler::Client))
}
