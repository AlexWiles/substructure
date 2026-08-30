use std::collections::{BTreeMap, BTreeSet, HashMap};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::decision::{LlmHandler, ToolHandler, Trigger};
use super::events::*;
use super::prompt_context;
use super::tool_contract::classify_arguments;
use crate::connectors::registry::ConnectionPath;
use crate::connectors::{filter, AuthNeed, RemoteTool};
use crate::protocol::{
    AgentTool, ConnectorTool, ConnectorToolKind, DeferToolsStrategy, Handler, McpServer,
    StoredResult, SubagentToolsStrategy,
};

#[derive(Debug, Clone, PartialEq)]
pub struct Source {
    pub server: McpServer,
    pub offered: Vec<RemoteTool>,
    pub instructions: Option<String>,
}

#[derive(serde::Serialize)]
struct Summary<'a> {
    mcp_server: &'a ConnectionPath,
    tools: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    about: Option<&'a str>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CallTarget {
    Connector(ConnectorTool),
    Declared(AgentTool),
    Subagent(ConnectorTool),
}

impl CallTarget {
    pub fn input(&self) -> Option<serde_json::Value> {
        match self {
            CallTarget::Connector(t) | CallTarget::Subagent(t) => t.input.clone(),
            CallTarget::Declared(t) => t.input.clone(),
        }
    }
}

pub(in crate::runtime::session) fn inner_arguments(raw: &serde_json::Value) -> String {
    match raw.get("arguments") {
        Some(serde_json::Value::String(text)) => text.clone(),
        Some(value) => value.to_string(),
        None => "{}".to_string(),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SkillCall {
    pub node: Option<String>,
    pub plugin_id: String,
    pub arguments: String,
}
use rust_decimal::Decimal;

pub use crate::protocol::EffectKind;
use crate::protocol::{
    AgentConfig, DraftMessage, Effect, EffectStatus, InterruptOrigin, LlmFormat, LlmRequest,
    LlmTool, Message, MessageTree, NewMessage, ReasoningConfig, RetryConfig, RetryPolicy, Role,
    SessionOwner, SpawnMode, SubagentTools, Usage, WorkerState,
};
use crate::runtime::retry::RetryState;

pub struct ApplyContext {
    pub occurred_at: DateTime<Utc>,
    pub sequence: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    Idle,
    Interrupted {
        interrupt_id: String,
        origin: InterruptOrigin,
        reason: String,
    },
    Done,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectTracking {
    status: EffectStatus,
    pub retry: RetryState,
    pub retry_policy: RetryPolicy,
    pub deadline: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<DateTime<Utc>>,
}

impl EffectTracking {
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

    pub fn restart(&mut self, now: DateTime<Utc>) {
        self.move_to(
            EffectStatus::Pending,
            &[EffectStatus::Failed, EffectStatus::Completed],
        );
        self.retry = RetryState::default();
        self.deadline = self.retry_policy.attempt_deadline(now);
        self.started_at = Some(now);
    }

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

    pub fn run(&mut self) {
        self.move_to(EffectStatus::Running, &[EffectStatus::Pending]);
        self.deadline = None;
    }

    pub fn total_deadline(&self) -> Option<DateTime<Utc>> {
        self.started_at
            .and_then(|at| self.retry_policy.total_deadline(at))
    }

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

    pub fn total_expired(&self, now: DateTime<Utc>) -> bool {
        self.total_deadline().is_some_and(|d| d <= now)
    }

    pub fn complete(&mut self) {
        self.move_to(
            EffectStatus::Completed,
            &[EffectStatus::Pending, EffectStatus::Running],
        );
    }

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

    pub fn is_ready(&self) -> bool {
        self.status == EffectStatus::Completed
    }

    pub fn is_in_flight(&self) -> bool {
        matches!(
            self.status,
            EffectStatus::Pending | EffectStatus::RetryScheduled
        )
    }

    pub fn is_queued(&self) -> bool {
        self.status == EffectStatus::Queued
    }

    pub fn is_open(&self) -> bool {
        self.is_queued() || self.is_in_flight()
    }

    pub fn is_unsettled(&self) -> bool {
        self.is_open() || self.status == EffectStatus::Running
    }

    pub fn earliest_wake(&self) -> Option<DateTime<Utc>> {
        match self.status {
            EffectStatus::Pending | EffectStatus::Running => self.expiry(),
            EffectStatus::RetryScheduled => self.retry.next_at,
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectState {
    pub id: String,
    pub tracking: EffectTracking,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
    pub payload: EffectPayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EffectPayload {
    LlmCall(LlmCallState),
    ToolCall(ToolCallState),
    Subagent(SubagentCallState),
    ConnectorSync(ConnectorSyncState),
    Decision(WorkerDecisionState),
}

impl EffectPayload {
    pub fn kind(&self) -> EffectKind {
        match self {
            EffectPayload::LlmCall(_) => EffectKind::LlmCall,
            EffectPayload::ToolCall(_) => EffectKind::ToolCall,
            EffectPayload::Subagent(_) => EffectKind::Subagent,
            EffectPayload::ConnectorSync(_) => EffectKind::ConnectorSync,
            EffectPayload::Decision(_) => EffectKind::Decision,
        }
    }
}

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
    subagent, subagent_mut => Subagent(SubagentCallState),
    connector, connector_mut => ConnectorSync(ConnectorSyncState),
    decision, decision_mut => Decision(WorkerDecisionState),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCallState {
    #[serde(default)]
    pub prompt: Vec<Message>,
    pub llm: String,
    pub spec: LlmCallSpec,
    pub stream: bool,
    #[serde(default)]
    pub handler: LlmHandler,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<LlmFormat>,
    #[serde(default)]
    pub defer_tools_strategy: DeferToolsStrategy,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub context_ids: Vec<String>,
}

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<ConnectorTarget>,
    #[serde(default)]
    pub arguments: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    #[serde(default)]
    pub is_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentCallState {
    pub agent_id: String,
    #[serde(default)]
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<DraftMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    #[serde(default)]
    pub is_error: bool,
    #[serde(default)]
    pub mode: SpawnMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorSyncState {
    #[serde(default)]
    pub tools: Vec<RemoteTool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth: Option<AuthNeed>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Versioned<T> {
    pub value: T,
    pub anchor: Option<String>,
}

pub type StateVersion = Versioned<WorkerState>;

pub type AgentVersion = Versioned<AgentConfig>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenInterrupt {
    pub interrupt_id: String,
    pub origin: InterruptOrigin,
    pub reason: String,
    pub payload: serde_json::Value,
    pub anchor: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct QueuedNotice<'a> {
    pub decision_id: &'a str,
    pub messages: &'a [DraftMessage],
    pub sessions: &'a [String],
    pub turn_id: &'a str,
}

#[derive(Debug, Clone, Copy)]
pub enum DecisionPark<'a> {
    Interrupt(&'a OpenInterrupt),
    Turn(&'a str),
}

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

fn rewind_log<T>(log: &mut Vec<Logged<T>>, seq: u64) {
    debug_assert!(log.windows(2).all(|w| w[0].seq <= w[1].seq));
    log.truncate(log.partition_point(|e| e.seq <= seq));
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueEntry {
    pub seq: u64,
    pub kind: EffectKind,
    pub id: String,
}

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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum TurnPhase {
    #[default]
    Idle,
    Active {
        turn_id: String,
    },
    Finalizing(TurnEnd),
}

impl TurnPhase {
    pub fn turn_id(&self) -> Option<&str> {
        match self {
            TurnPhase::Idle => None,
            TurnPhase::Active { turn_id } => Some(turn_id),
            TurnPhase::Finalizing(end) => Some(&end.turn_id),
        }
    }

    pub fn finalizing(&self) -> Option<&TurnEnd> {
        match self {
            TurnPhase::Finalizing(end) => Some(end),
            _ => None,
        }
    }
}

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
    pub subagent_cost: Decimal,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub calls: Vec<Effect>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub decisions: Vec<MetaDecision>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaDecision {
    pub decision_id: String,
    pub status: EffectStatus,
    pub finished: bool,
    pub attempts: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deadline: Option<DateTime<Utc>>,
    pub source_event_sequence: u64,
}

impl EventMeta {
    pub fn pending_work(&self, decision_id: &str) -> usize {
        let in_flight_calls = self
            .calls
            .iter()
            .filter(|e| {
                matches!(e.kind, EffectKind::ToolCall | EffectKind::Subagent)
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

    #[serde(default)]
    pub cost: Decimal,

    #[serde(default)]
    pub subagent_cost: Decimal,

    #[serde(default)]
    pub turn_cost: Decimal,

    #[serde(default)]
    pub turn_token_usage: Usage,

    #[serde(default)]
    pub subagent_token_usage: Usage,

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

    #[serde(with = "effect_table", default = "BTreeMap::new")]
    pub effects: BTreeMap<(EffectKind, String), EffectState>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub schedule_queue: Vec<QueueEntry>,

    #[serde(default)]
    pub phase: TurnPhase,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_started_seq: Option<u64>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub completed_turn_ids: Vec<String>,

    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub detached_turns: BTreeSet<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_id: Option<String>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nodes: Vec<Logged<NewMessage>>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub open_interrupts: Vec<OpenInterrupt>,

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
            subagent_cost: Decimal::ZERO,
            turn_cost: Decimal::ZERO,
            turn_token_usage: Usage::default(),
            subagent_token_usage: Usage::default(),
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
            detached_turns: BTreeSet::new(),
            head_id: None,
            nodes: Vec::new(),
            open_interrupts: Vec::new(),
            session_start_failed: false,
        }
    }

    pub fn at<'a, 'n>(&'a self, node: Option<&'n str>) -> SessionStateAtNode<'a, 'n> {
        SessionStateAtNode { state: self, node }
    }

    pub fn at_head(&self) -> SessionStateAtNode<'_, '_> {
        self.at(self.head_id.as_deref())
    }

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

    pub fn effects_of(&self, kind: EffectKind) -> impl Iterator<Item = &EffectState> {
        self.effects.values().filter(move |e| e.kind() == kind)
    }

    pub fn llm_call(&self, id: &str) -> Option<&LlmCallState> {
        self.effect(EffectKind::LlmCall, id).and_then(|e| e.llm())
    }

    pub fn tool_call(&self, id: &str) -> Option<&ToolCallState> {
        self.effect(EffectKind::ToolCall, id).and_then(|e| e.tool())
    }

    pub fn subagent(&self, id: &str) -> Option<&SubagentCallState> {
        self.effect(EffectKind::Subagent, id)
            .and_then(|e| e.subagent())
    }

    pub fn subagent_awaiting(&self, session_id: &str) -> Option<(&str, &SubagentCallState)> {
        self.effects_of(EffectKind::Subagent)
            .filter(|e| e.tracking.is_unsettled())
            .find_map(|e| {
                let sa = e.subagent()?;
                (sa.session_id == session_id).then_some((e.id.as_str(), sa))
            })
    }

    pub fn subagent_session(&self, session_id: &str) -> Option<&SubagentCallState> {
        self.effects_of(EffectKind::Subagent)
            .find_map(|e| e.subagent().filter(|sa| sa.session_id == session_id))
    }

    pub fn subagent_detached(&self, session_id: &str) -> Option<(&str, &SubagentCallState)> {
        let mut answered = None;
        for e in self.effects_of(EffectKind::Subagent) {
            let Some(sa) = e
                .subagent()
                .filter(|sa| sa.mode == SpawnMode::Detached && sa.session_id == session_id)
            else {
                continue;
            };
            match sa.result {
                None => return Some((e.id.as_str(), sa)),
                Some(_) => answered = answered.or(Some((e.id.as_str(), sa))),
            }
        }
        answered
    }

    pub fn connector_sync(&self, path: &ConnectionPath) -> Option<&ConnectorSyncState> {
        self.effect(EffectKind::ConnectorSync, &path.to_string())
            .and_then(|e| e.connector())
    }

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

    fn apply_llm_dispatched(&mut self, payload: &LlmCallDispatched, now: DateTime<Utc>) {
        let node = self
            .effect(EffectKind::LlmCall, &payload.id)
            .and_then(|e| e.anchor.clone())
            .or_else(|| self.head_id.clone());
        let at = self.at(node.as_deref());
        let connector_tools = at.connector_tools().tools;
        let owed = prompt_context::owed(at, &payload.id);
        let system_open = at.system_prefix_open(&payload.id);
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

    fn apply_subagent_requested(&mut self, payload: &SubagentRequested, ctx: &ApplyContext) {
        if let Some(existing) = self.effect_mut(EffectKind::Subagent, &payload.id) {
            existing.tracking.requeue();
        } else {
            self.insert_effect(
                &payload.id,
                EffectTracking::new_queued(payload.retry.clone()),
                EffectPayload::Subagent(SubagentCallState {
                    agent_id: payload.agent_id.clone(),
                    session_id: payload.session_id.clone(),
                    message: payload.message.clone(),
                    result: None,
                    is_error: false,
                    mode: payload.mode,
                }),
            );
        }
        self.enqueue(ctx.sequence, EffectKind::Subagent, &payload.id);
    }

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

    fn apply_decision_queued(&mut self, p: &DecisionQueued, ctx: &ApplyContext) {
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
            EventPayload::SubagentRequested(payload) => self.apply_subagent_requested(payload, ctx),
            EventPayload::SubagentDispatched(payload) => {
                if let Some(e) = self.effect_mut(EffectKind::Subagent, &payload.id) {
                    e.tracking.dispatch(now);
                }
                self.dequeue(EffectKind::Subagent, &payload.id);
            }
            EventPayload::SubagentStarted(payload) => {
                if let Some(e) = self.effect_mut(EffectKind::Subagent, &payload.id) {
                    match e
                        .subagent()
                        .is_some_and(|sa| sa.mode == SpawnMode::Detached)
                    {
                        true => e.tracking.complete(),
                        false => e.tracking.run(),
                    }
                }
            }
            EventPayload::SubagentErrored(payload) => {
                if let Some(e) = self.effect_mut(EffectKind::Subagent, &payload.id) {
                    e.tracking.record_error(payload.retryable, now);
                    let failed = e.tracking.status == EffectStatus::Failed;
                    if let Some(sa) = e.subagent_mut().filter(|_| failed) {
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
            EventPayload::DecisionDispatched(p) => {
                if let Some(e) = self.effect_mut(EffectKind::Decision, &p.id) {
                    e.tracking.dispatch(now);
                }
                self.dequeue(EffectKind::Decision, &p.id);
                self.status = SessionStatus::Idle;
            }
            EventPayload::DecisionCompleted(p) => {
                self.remove_effect(EffectKind::Decision, &p.id);
                self.dequeue(EffectKind::Decision, &p.id);
            }
            EventPayload::DecisionErrored(p) => self.apply_decision_errored(p, now),
            EventPayload::SessionMessageRequested(_) => {}
            EventPayload::DecisionQueued(p) => self.apply_decision_queued(p, ctx),
            EventPayload::DecisionDropped(p) => {
                self.remove_effect(EffectKind::Decision, &p.id);
                self.dequeue(EffectKind::Decision, &p.id);
            }
            EventPayload::CallVoided(p) => self.apply_call_voided(p),
            EventPayload::ConnectorSyncRequested(p) => {
                if !self.has_effect(EffectKind::ConnectorSync, &p.path.to_string()) {
                    self.insert_effect(
                        &p.path.to_string(),
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
                if let Some(e) = self.effect_mut(EffectKind::ConnectorSync, &p.path.to_string()) {
                    match e.tracking.status() {
                        EffectStatus::Failed | EffectStatus::Completed => e.tracking.restart(now),
                        _ => e.tracking.dispatch(now),
                    }
                }
            }
            EventPayload::ConnectorSyncCompleted(p) => {
                if let Some(e) = self.effect_mut(EffectKind::ConnectorSync, &p.path.to_string()) {
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
                if let Some(e) = self.effect_mut(EffectKind::ConnectorSync, &p.path.to_string()) {
                    e.tracking.record_error(p.retryable, now);
                    if let Some(sync) = e.connector_mut() {
                        sync.error = Some(p.error.message.clone());
                        sync.auth = p.auth;
                    }
                }
            }
            EventPayload::ConnectorAuthFailed(p) => {
                if let Some(sync) = self
                    .effect_mut(EffectKind::ConnectorSync, &p.path.to_string())
                    .and_then(EffectState::connector_mut)
                {
                    sync.auth = Some(p.auth);
                }
            }
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
                self.schedule_queue
                    .retain(|e| e.kind != EffectKind::TurnEnd);
            }
            EventPayload::SubagentTurnCompleted(payload) => {
                self.subagent_cost += payload.cost;
                self.turn_cost += payload.cost;
                self.subagent_token_usage.add(&payload.token_usage);
                self.turn_token_usage.add(&payload.token_usage);
                if let Some(turn_id) = &payload.turn_id {
                    self.detached_turns.insert(turn_id.clone());
                }
                if let Some(e) = self.effect_mut(EffectKind::Subagent, &payload.id) {
                    if let Some(sa) = e.subagent_mut() {
                        sa.result = Some(match &payload.error {
                            Some(error) => error.message.clone(),
                            None => json_to_string(&payload.data),
                        });
                        sa.is_error = payload.error.is_some();
                    }
                    match e.tracking.status() {
                        EffectStatus::Queued | EffectStatus::RetryScheduled => {
                            e.tracking.dispatch(now);
                            e.tracking.complete();
                        }
                        EffectStatus::Pending | EffectStatus::Running => e.tracking.complete(),
                        _ => {}
                    }
                }
                self.dequeue(EffectKind::Subagent, &payload.id);
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

    pub(crate) fn check_invariants(&self) -> Result<(), String> {
        let live = self
            .effects_of(EffectKind::Decision)
            .filter(|e| e.tracking.status == EffectStatus::Pending)
            .count();
        if live > 1 {
            return Err(format!("decision slot violated: {live} live decisions"));
        }
        if live > 0 && self.at_head().has_pending_connector_sync() {
            return Err(
                "a decision is live while a fetch its config owes is in flight".to_string(),
            );
        }
        let queued_statuses = self
            .effects
            .values()
            .filter(|e| e.kind() != EffectKind::ConnectorSync)
            .filter(|e| e.tracking.status == EffectStatus::Queued)
            .count();
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

    pub fn open_interrupt(&self, id: &str) -> Option<&OpenInterrupt> {
        self.open_interrupts.iter().find(|i| i.interrupt_id == id)
    }

    pub fn head_parked(&self) -> bool {
        self.at_head().active_interrupt_for().is_some()
    }

    pub fn projected_status(&self) -> SessionStatus {
        match self.at_head().active_interrupt_for() {
            Some(i) => SessionStatus::Interrupted {
                interrupt_id: i.interrupt_id.clone(),
                origin: i.origin,
                reason: i.reason.clone(),
            },
            None => self.status.clone(),
        }
    }

    pub fn trigger_anchor(&self, trigger: &Trigger) -> Option<&str> {
        match trigger {
            Trigger::ToolFinished { id, .. } | Trigger::ToolExecute { id, .. } => {
                self.effect(EffectKind::ToolCall, id)
            }
            Trigger::SubagentFinished { id, .. } => self.effect(EffectKind::Subagent, id),
            Trigger::LlmFinished { id, .. } | Trigger::LlmExecute { id, .. } => {
                self.effect(EffectKind::LlmCall, id)
            }
            _ => None,
        }
        .and_then(|e| e.anchor.as_deref())
    }

    pub fn worker_decision(&self, id: &str) -> Option<&WorkerDecisionState> {
        self.effect(EffectKind::Decision, id)
            .and_then(|e| e.decision())
    }

    pub fn anchor_parked(&self, anchor: Option<&str>) -> bool {
        match anchor {
            Some(a) if !self.at_head().anchor_on_path(Some(a)) => {
                self.at(Some(a)).active_interrupt_for().is_some()
            }
            _ => self.head_parked(),
        }
    }

    pub fn decision_parked(&self, trigger: &Trigger) -> bool {
        self.decision_park(trigger).is_some()
    }

    pub fn decision_park(&self, trigger: &Trigger) -> Option<DecisionPark<'_>> {
        if let Some(interrupt) = self.decision_park_interrupt(trigger) {
            return Some(DecisionPark::Interrupt(interrupt));
        }
        match (trigger.deferred_turn_id(), self.phase.turn_id()) {
            (Some(turn_id), Some(active)) if active != turn_id => Some(DecisionPark::Turn(active)),
            _ => None,
        }
    }

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
            return self.at(landing.as_deref()).active_interrupt_for();
        }
        let at = match self.trigger_anchor(trigger) {
            Some(a) if !self.at_head().anchor_on_path(Some(a)) => self.at(Some(a)),
            _ => self.at_head(),
        };
        at.active_interrupt_for()
    }

    pub fn queued_subagent_notice(&self) -> Option<QueuedNotice<'_>> {
        self.decisions_with(EffectStatus::Queued)
            .find_map(|e| match &e.decision()?.trigger {
                Trigger::SubagentNotice {
                    messages,
                    sessions,
                    turn_id,
                } => Some(QueuedNotice {
                    decision_id: &e.id,
                    messages,
                    sessions,
                    turn_id,
                }),
                _ => None,
            })
    }

    pub fn has_pending_worker_decision(&self) -> bool {
        self.decisions_with(EffectStatus::Pending).next().is_some()
    }

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

    pub fn path_ids<'a>(&'a self, node: &'a str) -> std::collections::HashSet<&'a str> {
        let by_id: HashMap<&str, &Logged<NewMessage>> = self
            .nodes
            .iter()
            .map(|n| (n.message.id.as_str(), n))
            .collect();
        let mut ids = std::collections::HashSet::new();
        let mut cursor = Some(node);
        while let Some(id) = cursor {
            let Some(node) = by_id.get(id) else { break };
            if !ids.insert(id) {
                break; // parent cycle guard
            }
            cursor = node.parent_id.as_deref();
        }
        ids
    }

    pub fn tool_handler_for(&self, name: &str) -> ToolHandler {
        let config = self.at_head().resolve_agent_for();
        let connector = || {
            self.connector_tool_for(name)
                .is_some_and(|t| t.kind != ConnectorToolKind::Subagent)
        };
        match config.as_ref().and_then(|c| c.tool(name)) {
            Some(t) => ToolHandler::declared(t.handler),
            None if connector() => ToolHandler::Server,
            None => ToolHandler::Worker,
        }
    }

    pub fn dispatched_calls(&self) -> Vec<String> {
        self.effects
            .values()
            .filter_map(|e| match e.kind() {
                EffectKind::ToolCall | EffectKind::Subagent => Some(e.id.clone()),
                _ => None,
            })
            .collect()
    }

    pub fn connector_tool_for(&self, name: &str) -> Option<ConnectorTool> {
        self.at_head()
            .connector_tools()
            .tools
            .into_iter()
            .find(|t| t.name == name)
    }

    pub fn connector_tools_for_config(&self, config: &AgentConfig) -> filter::Merged {
        let servers = self.servers_for(config);
        let may_spawn = config.may_spawn_subagent(self.depth());
        let mut resolutions: Vec<filter::Resolution> = Vec::new();
        if may_spawn {
            resolutions.push(filter::subagent_tools(
                &config.subagents,
                config.defers_tools(),
                config.subagent_strategy(),
                SubagentTools::wait_of(config.subagent_tools),
            ));
        }
        resolutions.extend(servers.iter().filter_map(|connector| {
            let effect = self.effect(EffectKind::ConnectorSync, &connector.path.to_string())?;
            let sync = effect.connector()?;
            effect.tracking.is_ready().then(|| {
                filter::resolve(
                    connector,
                    &sync.tools,
                    sync.prefix.as_deref(),
                    filter::defers(connector, config.defers_tools()),
                )
            })
        }));
        let defers = config.defers_tools()
            || config.tools.iter().any(|t| t.defer == Some(true))
            || (may_spawn
                && config.subagent_strategy() == SubagentToolsStrategy::PerAgent
                && config.subagents.iter().any(|s| s.defer == Some(true)))
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
        let taken: Vec<&str> = config.tools.iter().map(|t| t.name.as_str()).collect();
        filter::merge(resolutions, taken)
    }

    pub fn local_connector_answer(&self, tool_call_id: &str) -> Option<StoredResult> {
        let effect = self.effect(EffectKind::ToolCall, tool_call_id)?;
        let node = effect.anchor.clone();
        let tc = effect.tool()?;
        let target = tc.target.as_ref()?;
        super::engine_tools::answer(self.at(node.as_deref()), target.kind(), &tc.arguments)
    }

    pub fn skill_call(&self, tool_call_id: &str) -> Option<SkillCall> {
        let effect = self.effect(EffectKind::ToolCall, tool_call_id)?;
        let tc = effect.tool()?;
        let ConnectorTarget::Skill { plugin, .. } = tc.target.as_ref()? else {
            return None;
        };
        Some(SkillCall {
            node: effect.anchor.clone(),
            plugin_id: plugin.clone(),
            arguments: tc.arguments.clone(),
        })
    }

    fn anchored_on(path: &std::collections::HashSet<&str>, anchor: Option<&str>) -> bool {
        anchor.is_none_or(|a| path.contains(a))
    }

    pub fn retry_config(&self) -> Option<RetryConfig> {
        self.at_head()
            .resolve_agent_for()
            .and_then(|c| c.retry)
            .map(|b| *b)
    }

    pub fn depth(&self) -> u32 {
        self.ancestry.len() as u32
    }

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
                    session_id: None,
                };
                match &e.payload {
                    EffectPayload::ToolCall(c) => {
                        wire.name = Some(c.name.clone());
                        wire.arguments = Some(c.arguments.clone());
                        wire.handler = Some(c.handler.into());
                    }
                    EffectPayload::Subagent(c) => {
                        wire.agent_id = Some(c.agent_id.clone());
                        wire.session_id = Some(c.session_id.clone());
                    }
                    EffectPayload::LlmCall(c) => {
                        wire.handler = Some(c.handler.into());
                        wire.stream = Some(c.stream);
                    }
                    EffectPayload::ConnectorSync(_) => wire.name = Some(e.id.clone()),
                    EffectPayload::Decision(_) => unreachable!(),
                }
                wire
            })
            .collect();
        effects.sort_by(|a, b| a.id.cmp(&b.id));
        effects
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
                        Trigger::ToolFinished { .. } | Trigger::SubagentFinished { .. }
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
            subagent_cost: self.subagent_cost,
            head_id: self.head_id.clone(),
            calls: self.effects(),
            decisions,
        }
    }

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

#[derive(Clone, Copy)]
pub struct SessionStateAtNode<'a, 'n> {
    state: &'a SessionState,
    node: Option<&'n str>,
}

impl<'a, 'n> SessionStateAtNode<'a, 'n> {
    pub fn state(&self) -> &'a SessionState {
        self.state
    }

    fn path_set(&self) -> std::collections::HashSet<&str> {
        self.node
            .map(|n| self.state.path_ids(n))
            .unwrap_or_default()
    }

    pub fn active_interrupt_for(&self) -> Option<&'a OpenInterrupt> {
        resolve_on_path(&self.state.open_interrupts, &self.path_set())
    }

    pub fn interrupts_for(&self) -> Vec<&'a OpenInterrupt> {
        let on_path = self.path_set();
        self.state
            .open_interrupts
            .iter()
            .filter(|i| match i.anchor.as_deref() {
                None => true,
                Some(a) => on_path.contains(a),
            })
            .collect()
    }

    pub fn anchor_on_path(&self, anchor: Option<&str>) -> bool {
        SessionState::anchored_on(&self.path_set(), anchor)
    }

    pub fn resolve_state_for(&self) -> WorkerState {
        resolve_on_path(&self.state.state_versions, &self.path_set())
            .map(|v| v.value.clone())
            .unwrap_or_default()
    }

    pub fn resolve_agent_for(&self) -> Option<AgentConfig> {
        resolve_on_path(&self.state.agent_versions, &self.path_set()).map(|v| v.value.clone())
    }

    pub fn connector_tools(&self) -> filter::Merged {
        let Some(config) = self.resolve_agent_for() else {
            return filter::Merged {
                tools: Vec::new(),
                collisions: Vec::new(),
            };
        };
        self.state.connector_tools_for_config(&config)
    }

    fn connector_source(&self, connector_id: &ConnectionPath) -> Option<Source> {
        let config = self.resolve_agent_for()?;
        let server = self
            .state
            .servers_for(&config)
            .into_iter()
            .find(|c| c.path == *connector_id)?;
        let sync = self.state.connector_sync(connector_id)?;
        Some(Source {
            server,
            offered: sync.tools.clone(),
            instructions: sync.instructions.clone(),
        })
    }

    fn searchable_connectors(&self) -> Vec<Source> {
        let Some(config) = self.resolve_agent_for() else {
            return Vec::new();
        };
        self.state
            .servers_for(&config)
            .iter()
            .filter(|c| {
                self.state
                    .effect(EffectKind::ConnectorSync, &c.path.to_string())
                    .is_some_and(|e| e.tracking.is_ready())
            })
            .filter_map(|c| self.connector_source(&c.path))
            .collect()
    }

    pub fn searchable_tools(&self) -> Vec<LlmTool> {
        let Some(config) = self.resolve_agent_for() else {
            return Vec::new();
        };
        let declared = config
            .tools
            .iter()
            .map(|t| t.to_llm_tool(config.defers_tools()));
        let connector = self
            .state
            .connector_tools_for_config(&config)
            .tools
            .into_iter()
            .filter(|t| {
                matches!(
                    t.kind,
                    ConnectorToolKind::Remote | ConnectorToolKind::Subagent
                )
            })
            .map(|t| t.to_llm_tool())
            .collect::<Vec<_>>();
        declared.chain(connector).collect()
    }

    pub fn connection_summary(&self, path: &ConnectionPath) -> Option<String> {
        let source = self
            .searchable_connectors()
            .into_iter()
            .find(|source| source.server.path == *path)?;
        serde_json::to_string(&Summary {
            mcp_server: &source.server.path,
            tools: filter::callable(&source.server, &source.offered).len(),
            about: source.instructions.as_deref(),
        })
        .ok()
    }

    pub fn unavailable_connectors(
        &self,
        config: &AgentConfig,
    ) -> Vec<(ConnectionPath, Option<AuthNeed>)> {
        self.state
            .servers_for(config)
            .into_iter()
            .filter(|c| c.tool_sync_failure.warns())
            .filter_map(|c| {
                let effect = self
                    .state
                    .effect(EffectKind::ConnectorSync, &c.path.to_string())?;
                if effect.tracking.status() != EffectStatus::Failed {
                    return None;
                }
                Some((c.path.clone(), effect.connector().and_then(|s| s.auth)))
            })
            .collect()
    }

    pub fn unavailable_connector_paths(&self, config: &AgentConfig) -> Vec<ConnectionPath> {
        self.unavailable_connectors(config)
            .into_iter()
            .map(|(path, _)| path)
            .collect()
    }

    pub fn context_ids_on_path(&self, exclude: &str) -> std::collections::HashSet<String> {
        let path = self.path_set();
        self.state
            .effects
            .values()
            .filter(|e| e.id != exclude && SessionState::anchored_on(&path, e.anchor.as_deref()))
            .filter_map(EffectState::llm)
            .flat_map(|call| call.context_ids.iter().cloned())
            .collect()
    }

    pub fn system_prefix_open(&self, exclude: &str) -> bool {
        let path = self.path_set();
        !self.state.effects.values().any(|e| {
            e.id != exclude
                && matches!(e.payload, EffectPayload::LlmCall(_))
                && SessionState::anchored_on(&path, e.anchor.as_deref())
                && e.tracking.status() != EffectStatus::Queued
        })
    }

    pub fn call_tool_target(&self, named: &str) -> Option<CallTarget> {
        let config = self.resolve_agent_for()?;
        if let Some(declared) = config.tool(named) {
            return Some(CallTarget::Declared(declared.clone()));
        }
        self.state
            .connector_tool_for(named)
            .and_then(|t| match t.kind {
                ConnectorToolKind::Remote => Some(CallTarget::Connector(t)),
                ConnectorToolKind::Subagent => Some(CallTarget::Subagent(t)),
                _ => None,
            })
    }

    pub fn call_tool_fault(&self, arguments: &str) -> Option<String> {
        let raw: serde_json::Value = serde_json::from_str(arguments).unwrap_or_default();
        let named = raw
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();

        let Some(target) = self.call_tool_target(&named) else {
            let config = self.resolve_agent_for();
            let tools = self.searchable_tools();
            let cap = config
                .as_ref()
                .map(AgentConfig::defer_settings)
                .unwrap_or_default()
                .max_matches
                .get();
            let near: Vec<&str> = filter::find(&tools, &named)
                .into_iter()
                .take(cap)
                .map(|t| t.name.as_str())
                .collect();
            let unavailable = config
                .as_ref()
                .map(|c| self.unavailable_connector_paths(c))
                .unwrap_or_default();
            return Some(crate::copy::no_such_tool(&named, &near, &unavailable));
        };
        let input = target.input();
        classify_arguments(&inner_arguments(&raw), input.as_ref())
            .error()
            .map(|e| crate::copy::bad_arguments(&named, e, input.as_ref()))
    }

    pub fn unsynced_connectors(&self) -> Vec<ConnectionPath> {
        let Some(config) = self.resolve_agent_for() else {
            return Vec::new();
        };
        self.state
            .servers_for(&config)
            .iter()
            .filter(|c| {
                !self
                    .state
                    .has_effect(EffectKind::ConnectorSync, &c.path.to_string())
            })
            .map(|c| c.path.clone())
            .collect()
    }

    pub fn has_pending_connector_sync(&self) -> bool {
        let Some(config) = self.resolve_agent_for() else {
            return false;
        };
        self.state.servers_for(&config).iter().any(|c| {
            self.state
                .tracking(EffectKind::ConnectorSync, &c.path.to_string())
                .is_some_and(EffectTracking::is_in_flight)
        })
    }

    pub fn transcript(&self) -> Vec<Message> {
        match self.node {
            Some(node) => self.state.message_tree().path_to(node),
            None => Vec::new(),
        }
    }

    pub(crate) fn open_llm_calls(&self) -> HashMap<String, EffectState> {
        let path = self.transcript();
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
                self.state
                    .effect(EffectKind::LlmCall, &m.id)
                    .map(|e| (m.id.clone(), e.clone()))
            })
            .collect()
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
            s.at_head().open_llm_calls().contains_key("call-1"),
            "unanswered tool call keeps the parent open"
        );

        s.nodes
            .push(node(tool_answer("t1", "tc-1"), Some("call-1")));
        s.head_id = Some("t1".to_string());
        assert!(
            s.at_head().open_llm_calls().is_empty(),
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
        assert_eq!(s.at(Some("u2")).resolve_state_for(), WorkerState::default());
        assert_eq!(s.at(None).resolve_state_for(), WorkerState::default());
    }

    #[test]
    fn linear_resolution_latest_on_path_wins() {
        let mut s = forked_session();
        s.state_versions.push(version(json!({"v": 1}), Some("u1")));
        s.state_versions.push(version(json!({"v": 2}), Some("u2")));
        assert_eq!(s.at(Some("u2")).resolve_state_for().0, json!({"v": 2}));
    }

    #[test]
    fn fork_resolves_as_of_the_fork_point() {
        let mut s = forked_session();
        s.state_versions.push(version(json!({"v": 1}), Some("u1")));
        s.state_versions.push(version(json!({"v": 2}), Some("u2")));
        assert_eq!(s.at(Some("x1")).resolve_state_for().0, json!({"v": 1}));
    }

    #[test]
    fn unanchored_version_matches_any_path() {
        let mut s = forked_session();
        s.state_versions.push(version(json!({"v": 0}), None));
        s.state_versions.push(version(json!({"v": 2}), Some("u2")));
        assert_eq!(s.at(Some("x1")).resolve_state_for().0, json!({"v": 0}));
        assert_eq!(s.at(Some("u2")).resolve_state_for().0, json!({"v": 2}));
        assert_eq!(s.at(None).resolve_state_for().0, json!({"v": 0}));
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
        assert_eq!(s.state_versions.len(), 3);
        assert_eq!(s.at_head().resolve_state_for().0, json!({"v": 3}));
        assert_eq!(s.at(Some("x1")).resolve_state_for().0, json!({"v": 1}));
    }

    #[test]
    fn a_version_serializes_flat_under_its_log_tag() {
        let v = version(json!({"v": 1}), Some("u1"));
        let value = serde_json::to_value(&v).expect("serializes");
        assert_eq!(value, json!({"seq": 0, "value": {"v": 1}, "anchor": "u1"}));
    }

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
        assert_eq!(s.at_head().resolve_state_for().0, json!({"v": 1}));
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
            ..Default::default()
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
        assert_eq!(s.at(Some("u2")).resolve_agent_for(), None);
        assert_eq!(s.at(None).resolve_agent_for(), None);
    }

    #[test]
    fn linear_resolution_latest_on_path_wins() {
        let mut s = forked_session();
        s.agent_versions.push(version("m1", Some("u1")));
        s.agent_versions.push(version("m2", Some("u2")));
        assert_eq!(s.at(Some("u2")).resolve_agent_for(), Some(config("m2")));
    }

    #[test]
    fn fork_resolves_as_of_the_fork_point() {
        let mut s = forked_session();
        s.agent_versions.push(version("m1", Some("u1")));
        s.agent_versions.push(version("m2", Some("u2")));
        assert_eq!(s.at(Some("x1")).resolve_agent_for(), Some(config("m1")));
    }

    #[test]
    fn a_connection_is_summarised_where_the_work_was_authored() {
        let with_sentry = |model: &str, anchor: Option<&str>| Logged {
            seq: 0,
            entry: AgentVersion {
                value: AgentConfig {
                    mcp: vec![McpServer {
                        path: ConnectionPath::Mcp("sentry".into()),
                        tools: None,
                        auth_failure: Default::default(),
                        tool_sync_failure: Default::default(),
                        approve: Default::default(),
                    }],
                    ..config(model)
                },
                anchor: anchor.map(str::to_string),
            },
        };
        let mut s = forked_session();
        s.head_id = Some("u2".to_string());
        s.agent_versions.push(with_sentry("m1", Some("u1")));
        s.agent_versions.push(version("m2", Some("u2")));
        s.insert_effect(
            "mcp.sentry",
            EffectTracking::new_queued(RetryPolicy::no_retry()),
            EffectPayload::ConnectorSync(ConnectorSyncState {
                prefix: Some("sentry".to_string()),
                tools: vec![RemoteTool {
                    name: "search_issues".to_string(),
                    title: None,
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
        if let Some(e) = s.effect_mut(EffectKind::ConnectorSync, "mcp.sentry") {
            e.tracking.dispatch(Utc::now());
            e.tracking.complete();
        }

        assert!(
            s.at(Some("x1"))
                .connection_summary(&ConnectionPath::Mcp("sentry".into()))
                .is_some(),
            "the fork still holds the connection, so a call there can describe it"
        );
        assert!(
            s.at(Some("u2"))
                .connection_summary(&ConnectionPath::Mcp("sentry".into()))
                .is_none(),
            "u2 dropped it, so a call there has nothing to describe"
        );
    }

    #[test]
    fn unanchored_config_is_the_universal_fallback() {
        let mut s = forked_session();
        s.agent_versions.push(version("m0", None));
        s.agent_versions.push(version("m2", Some("u2")));
        assert_eq!(s.at(Some("x1")).resolve_agent_for(), Some(config("m0")));
        assert_eq!(s.at(Some("u2")).resolve_agent_for(), Some(config("m2")));
        assert_eq!(s.at(None).resolve_agent_for(), Some(config("m0")));
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
        assert_eq!(s.agent_versions.len(), 3);
        assert_eq!(s.at_head().resolve_agent_for(), Some(config("m3")));
        assert_eq!(s.at(Some("x1")).resolve_agent_for(), Some(config("m1")));
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
            session_id: None,
        };
        let json = serde_json::to_value(&e).unwrap();
        assert_eq!(json["id"], "call_1");
        assert_eq!(json["status"], "pending");
        assert_eq!(json["kind"], "tool_call");
        assert_eq!(json["name"], "get_weather");
        assert_eq!(json["handler"], "worker");
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
        let node: NewMessage = serde_json::from_value(serde_json::json!({
            "kind": "message",
            "message": {"id": "m1", "role": "user", "content": "hi", "tool_calls": []},
            "parent_id": null,
        }))
        .expect("unknown fields are ignored");
        assert_eq!(node.message.id, "m1");
    }
}

pub fn waiting_on_client(calls: &[Effect]) -> bool {
    !calls.is_empty() && calls.iter().all(|c| c.handler == Some(Handler::Client))
}
