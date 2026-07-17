use std::collections::{BTreeMap, HashMap};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::decision::{LlmHandler, ToolHandler, Trigger};
use super::events::*;
use rust_decimal::Decimal;

use crate::protocol::{
    AgentConfig, DraftMessage, Effect, EffectKind, EffectStatus, InterruptOrigin, LlmFormat,
    LlmRequest, LlmTool, Message, MessageTree, NewMessage, ReasoningConfig, RetryPolicy, Role,
    SessionOwner, WorkerState,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectTracking {
    pub status: EffectStatus,
    pub retry: RetryState,
    pub retry_policy: RetryPolicy,
    pub deadline: Option<DateTime<Utc>>,
}

impl EffectTracking {
    pub fn new(retry_policy: RetryPolicy, now: DateTime<Utc>) -> Self {
        let deadline = retry_policy.deadline(now);
        Self {
            status: EffectStatus::Pending,
            retry: RetryState::default(),
            retry_policy,
            deadline,
        }
    }

    pub fn new_queued(retry_policy: RetryPolicy) -> Self {
        Self {
            status: EffectStatus::Queued,
            retry: RetryState::default(),
            retry_policy,
            deadline: None,
        }
    }

    pub fn reset_pending(&mut self, now: DateTime<Utc>) {
        self.status = EffectStatus::Pending;
        self.deadline = self.retry_policy.deadline(now);
        self.retry.next_at = None;
    }

    pub fn complete(&mut self) {
        self.status = EffectStatus::Completed;
    }

    pub fn record_error(&mut self, retryable: bool, now: DateTime<Utc>) {
        self.retry = self.retry_policy.record_failure(&self.retry, now);
        if retryable && self.retry.next_at.is_some() {
            self.status = EffectStatus::RetryScheduled;
        } else {
            self.status = EffectStatus::Failed;
            self.retry.next_at = None;
        }
    }

    pub fn earliest_wake(&self) -> Option<DateTime<Utc>> {
        match self.status {
            EffectStatus::Pending => self.deadline,
            EffectStatus::RetryScheduled => self.retry.next_at,
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCallState {
    pub call_id: String,
    pub tracking: EffectTracking,
    /// The verbatim prompt the worker sent, stored for retries.
    #[serde(default)]
    pub prompt: Vec<Message>,
    pub spec: LlmCallSpec,
    pub stream: bool,
    #[serde(default)]
    pub handler: LlmHandler,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<LlmFormat>,
    /// The active head when the effect was first requested; retries keep it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
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
    pub tool_call_id: String,
    pub name: String,
    pub tracking: EffectTracking,
    #[serde(default)]
    pub handler: ToolHandler,
    /// Original arguments, stored for retries and crash recovery.
    #[serde(default)]
    pub arguments: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    #[serde(default)]
    pub is_error: bool,
    /// The active head when the effect was first requested; retries keep it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubAgentCallState {
    pub session_id: String,
    pub agent_id: String,
    pub tracking: EffectTracking,
    /// The model tool-call id this delegation answers.
    #[serde(default)]
    pub tool_call_id: String,
    /// The child's turn result (or error); `Some` once the turn is terminal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    #[serde(default)]
    pub is_error: bool,
    /// The active head when the effect was first requested; retries keep it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
}

/// Anchored worker-state write; current state is resolved, not stored.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateVersion {
    pub state: WorkerState,
    pub anchor: Option<String>,
}

/// Anchored agent-config write; the sibling of [`StateVersion`] for the typed
/// agent-identity channel. Kept as its own struct (not a generic over
/// `StateVersion`) so the persisted `state` field name never changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentVersion {
    pub agent: AgentConfig,
    pub anchor: Option<String>,
}

/// An unresumed interrupt: parks paths through its anchor (`None` = every path).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenInterrupt {
    pub interrupt_id: String,
    pub origin: InterruptOrigin,
    pub reason: String,
    pub payload: serde_json::Value,
    pub anchor: Option<String>,
}

/// A versioned, anchored document write. Both worker state and agent config
/// resolve newest-on-path and compact by anchor identically; this shares that
/// logic without touching either struct's serialized shape.
pub trait Anchored {
    fn anchor(&self) -> Option<&str>;
}

impl Anchored for StateVersion {
    fn anchor(&self) -> Option<&str> {
        self.anchor.as_deref()
    }
}

impl Anchored for AgentVersion {
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

/// Append `new`, dropping any prior version at the same anchor: a superseded
/// same-anchor version can never win resolution.
pub fn compact_push<V: Anchored>(versions: &mut Vec<V>, new: V) {
    versions.retain(|v| v.anchor() != new.anchor());
    versions.push(new);
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerDecisionState {
    pub decision_id: String,
    pub tracking: EffectTracking,
    pub trigger: Trigger,
    #[serde(default)]
    pub source_event_sequence: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedState {
    pub status: SessionStatus,
    pub wake_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub calls: Vec<Effect>,
    pub owner: Option<SessionOwner>,
    pub agent_id: Option<String>,
    #[serde(default)]
    pub worker_state: WorkerState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_config: Option<AgentConfig>,
    #[serde(default)]
    pub message_tree: MessageTree,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ancestry: Vec<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub sub_agent_calls: HashMap<String, SubAgentCallState>,
    /// Llm calls whose assistant message has tool calls not yet answered on the
    /// active path — the parents a `tool.finished` proposal may re-issue. Pruned
    /// to that window so stored prompts don't accumulate in per-event state.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub open_llm_calls: HashMap<String, LlmCallState>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub worker_decisions: HashMap<String, WorkerDecisionState>,
    pub turn_id: Option<String>,
    #[serde(default)]
    pub cost: Decimal,
    #[serde(default)]
    pub sub_agent_cost: Decimal,
    #[serde(default)]
    pub turn_cost: Decimal,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub turn_token_usage: BTreeMap<String, u64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub token_usage: BTreeMap<String, u64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub sub_agent_token_usage: BTreeMap<String, u64>,
}

impl DerivedState {
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
                        EffectStatus::Pending | EffectStatus::RetryScheduled
                    )
            })
            .count();
        let unrecorded_results = self
            .worker_decisions
            .values()
            .filter(|d| d.decision_id != decision_id)
            .filter(|d| {
                matches!(
                    d.tracking.status,
                    EffectStatus::Pending | EffectStatus::Queued
                )
            })
            .filter(|d| {
                matches!(
                    d.trigger,
                    Trigger::ToolFinished { .. } | Trigger::SubAgentFinished { .. }
                )
            })
            .count();
        in_flight_calls + unrecorded_results
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
    pub token_usage: BTreeMap<String, u64>,

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
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub turn_token_usage: BTreeMap<String, u64>,

    /// Token usage accumulated from sub-agent sessions.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub sub_agent_token_usage: BTreeMap<String, u64>,

    #[serde(default)]
    pub state_versions: Vec<StateVersion>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub agent_versions: Vec<AgentVersion>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ancestry: Vec<String>,

    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub data: serde_json::Value,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker_retry: Option<RetryPolicy>,

    pub llm_calls: HashMap<String, LlmCallState>,
    pub tool_calls: HashMap<String, ToolCallState>,
    pub sub_agent_calls: HashMap<String, SubAgentCallState>,
    pub worker_decisions: HashMap<String, WorkerDecisionState>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,

    /// Turn IDs that have completed, used for idempotency checks.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub completed_turn_ids: Vec<String>,

    /// The active branch's leaf; advances to each appended message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_id: Option<String>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nodes: Vec<NewMessage>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub open_interrupts: Vec<OpenInterrupt>,
}

impl SessionState {
    pub fn new(session_id: String) -> Self {
        SessionState {
            session_id,
            status: SessionStatus::Done,
            agent_id: None,
            owner: None,
            token_usage: BTreeMap::new(),
            cost: Decimal::ZERO,
            sub_agent_cost: Decimal::ZERO,
            turn_cost: Decimal::ZERO,
            turn_token_usage: BTreeMap::new(),
            sub_agent_token_usage: BTreeMap::new(),
            state_versions: Vec::new(),
            agent_versions: Vec::new(),
            ancestry: Vec::new(),
            data: serde_json::Value::Null,
            worker_retry: None,
            llm_calls: HashMap::new(),
            tool_calls: HashMap::new(),
            sub_agent_calls: HashMap::new(),
            worker_decisions: HashMap::new(),
            turn_id: None,
            completed_turn_ids: Vec::new(),
            head_id: None,
            nodes: Vec::new(),
            open_interrupts: Vec::new(),
        }
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
                self.nodes.push(payload.clone());
            }
            EventPayload::HeadMoved(payload) => {
                self.head_id = Some(payload.head_id.clone());
            }
            EventPayload::LlmCallRequested(payload) => {
                self.status = SessionStatus::Idle;
                let prompt: Vec<Message> = payload
                    .request
                    .messages
                    .iter()
                    .cloned()
                    .map(DraftMessage::record)
                    .collect();
                let spec = LlmCallSpec::from(&payload.request);
                if let Some(existing) = self.llm_calls.get_mut(&payload.call_id) {
                    existing.tracking.reset_pending(now);
                    existing.prompt = prompt;
                    existing.spec = spec;
                    existing.stream = payload.stream;
                    existing.handler = payload.handler;
                    existing.format = payload.format;
                } else {
                    // Applies after the same batch's NewMessage events: head_id is post-reconcile.
                    self.llm_calls.insert(
                        payload.call_id.clone(),
                        LlmCallState {
                            call_id: payload.call_id.clone(),
                            tracking: EffectTracking::new(payload.retry.clone(), now),
                            prompt,
                            spec,
                            stream: payload.stream,
                            handler: payload.handler,
                            format: payload.format,
                            anchor: self.head_id.clone(),
                        },
                    );
                }
            }
            EventPayload::LlmCallCompleted(payload) => {
                self.track_usage(&payload.response.usage);
                self.track_turn_usage(&payload.response.usage);
                if let Some(c) = payload.response.cost {
                    self.cost += c;
                    self.turn_cost += c;
                }
                if let Some(call) = self.llm_calls.get_mut(&payload.call_id) {
                    call.tracking.complete();
                }
                self.status = SessionStatus::Idle;
            }
            EventPayload::LlmCallErrored(payload) => {
                if let Some(call) = self.llm_calls.get_mut(&payload.call_id) {
                    call.tracking.record_error(payload.retryable, now);
                }
                self.status = SessionStatus::Idle;
            }
            EventPayload::ToolCallRequested(payload) => {
                if let Some(existing) = self.tool_calls.get_mut(&payload.tool_call_id) {
                    existing.tracking.reset_pending(now);
                    existing.arguments = payload.arguments.clone();
                } else {
                    self.tool_calls.insert(
                        payload.tool_call_id.clone(),
                        ToolCallState {
                            tool_call_id: payload.tool_call_id.clone(),
                            name: payload.name.clone(),
                            tracking: EffectTracking::new(payload.retry.clone(), now),
                            handler: payload.handler,
                            arguments: payload.arguments.clone(),
                            result: None,
                            is_error: false,
                            anchor: self.head_id.clone(),
                        },
                    );
                }
                self.status = SessionStatus::Idle;
            }
            EventPayload::ToolCallCompleted(payload) => {
                if let Some(tc) = self.tool_calls.get_mut(&payload.tool_call_id) {
                    tc.tracking.complete();
                    tc.result = Some(payload.result.clone());
                    tc.is_error = false;
                }
            }
            EventPayload::ToolCallErrored(payload) => {
                if let Some(tc) = self.tool_calls.get_mut(&payload.tool_call_id) {
                    tc.tracking.record_error(payload.retryable, now);
                    if tc.tracking.status == EffectStatus::Failed {
                        tc.result = Some(payload.error.clone());
                        tc.is_error = true;
                    }
                }
            }
            EventPayload::SubAgentRequested(payload) => {
                let sid = payload.session_id.clone();
                if let Some(existing) = self.sub_agent_calls.get_mut(&sid) {
                    existing.tracking.reset_pending(now);
                } else {
                    self.sub_agent_calls.insert(
                        sid.clone(),
                        SubAgentCallState {
                            session_id: sid,
                            agent_id: payload.agent_id.clone(),
                            tracking: EffectTracking::new(payload.retry.clone(), now),
                            tool_call_id: payload.tool_call_id.clone(),
                            result: None,
                            is_error: false,
                            anchor: self.head_id.clone(),
                        },
                    );
                }
                self.status = SessionStatus::Idle;
            }
            EventPayload::SubAgentStarted(payload) => {
                if let Some(sa) = self.sub_agent_calls.get_mut(&payload.session_id) {
                    sa.tracking.complete();
                }
            }
            EventPayload::SubAgentErrored(payload) => {
                if let Some(sa) = self.sub_agent_calls.get_mut(&payload.session_id) {
                    sa.tracking.record_error(payload.retryable, now);
                    if sa.tracking.status == EffectStatus::Failed {
                        sa.result = Some(payload.error.clone());
                        sa.is_error = true;
                    }
                }
            }
            EventPayload::SessionInterrupted(payload) => {
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
                // Void pending decisions so late submissions no-op; calls void via CallVoided events.
                for wd in self.worker_decisions.values_mut() {
                    if wd.tracking.status == EffectStatus::Pending {
                        wd.tracking.status = EffectStatus::Failed;
                    }
                }
            }
            EventPayload::InterruptResumed(p) => {
                self.open_interrupts
                    .retain(|i| i.interrupt_id != p.interrupt_id);
            }
            EventPayload::WorkerDecisionRequested(p) => {
                let retry_policy = self.worker_retry.clone().unwrap_or(RetryPolicy::no_retry());
                if let Some(existing) = self.worker_decisions.get_mut(&p.decision_id) {
                    existing.tracking.reset_pending(now);
                } else {
                    self.worker_decisions.insert(
                        p.decision_id.clone(),
                        WorkerDecisionState {
                            decision_id: p.decision_id.clone(),
                            tracking: EffectTracking::new(retry_policy, now),
                            trigger: p.trigger.clone(),
                            source_event_sequence: ctx.sequence,
                        },
                    );
                }
                self.status = SessionStatus::Idle;
            }
            EventPayload::WorkerDecisionCompleted(p) => {
                if let Some(wd) = self.worker_decisions.get_mut(&p.decision_id) {
                    wd.tracking.complete();
                }
            }
            EventPayload::WorkerDecisionErrored(p) => {
                if let Some(wd) = self.worker_decisions.get_mut(&p.decision_id) {
                    wd.tracking.record_error(p.retryable, now);
                }
            }
            EventPayload::SessionMessageRequested(_) => {}
            EventPayload::DecisionRequestQueued(p) => {
                let retry_policy = self.worker_retry.clone().unwrap_or(RetryPolicy::no_retry());
                self.worker_decisions.insert(
                    p.decision_id.clone(),
                    WorkerDecisionState {
                        decision_id: p.decision_id.clone(),
                        tracking: EffectTracking::new_queued(retry_policy),
                        trigger: p.trigger.clone(),
                        source_event_sequence: ctx.sequence,
                    },
                );
            }
            EventPayload::DecisionRequestDropped(p) => {
                // Voided like an interrupted decision: out of the queue, never delivered.
                if let Some(wd) = self.worker_decisions.get_mut(&p.decision_id) {
                    wd.tracking.status = EffectStatus::Failed;
                }
            }
            EventPayload::CallVoided(p) => {
                let tracking = match p.kind {
                    EffectKind::ToolCall => self.tool_calls.get_mut(&p.id).map(|c| &mut c.tracking),
                    EffectKind::LlmCall => self.llm_calls.get_mut(&p.id).map(|c| &mut c.tracking),
                    // Sub-agent calls are keyed by child session, not call id.
                    EffectKind::SubAgent => p
                        .session_id
                        .as_ref()
                        .and_then(|sid| self.sub_agent_calls.get_mut(sid))
                        .map(|c| &mut c.tracking),
                };
                if let Some(tracking) = tracking {
                    tracking.status = EffectStatus::Failed;
                }
            }
            EventPayload::WorkerStateUpdated(p) => {
                compact_push(
                    &mut self.state_versions,
                    StateVersion {
                        state: p.state.clone(),
                        anchor: p.anchor.clone(),
                    },
                );
            }
            EventPayload::AgentConfigUpdated(p) => {
                compact_push(
                    &mut self.agent_versions,
                    AgentVersion {
                        agent: p.config.clone(),
                        anchor: p.anchor.clone(),
                    },
                );
            }
            EventPayload::SessionCancelled => {
                self.status = SessionStatus::Done;
                // Terminal: void pending decisions so late submissions no-op; calls void via CallVoided events.
                for wd in self.worker_decisions.values_mut() {
                    if wd.tracking.status == EffectStatus::Pending {
                        wd.tracking.status = EffectStatus::Failed;
                    }
                }
            }
            EventPayload::SessionDone(_) => {
                if !self.ancestry.is_empty() {
                    self.status = SessionStatus::Done;
                } else {
                    self.status = SessionStatus::Idle;
                }
            }
            EventPayload::SubAgentTurnCompleted(payload) => {
                self.sub_agent_cost += payload.cost;
                self.turn_cost += payload.cost;
                for (k, v) in &payload.token_usage {
                    *self.sub_agent_token_usage.entry(k.clone()).or_insert(0) += v;
                    *self.turn_token_usage.entry(k.clone()).or_insert(0) += v;
                }
                if let Some(sa) = self.sub_agent_calls.get_mut(&payload.session_id) {
                    sa.result = Some(json_to_string(&payload.data));
                    sa.is_error = false;
                }
            }
            EventPayload::TurnStarted(p) => {
                self.turn_id = Some(p.turn_id.clone());
                self.turn_cost = Decimal::ZERO;
                self.turn_token_usage.clear();
            }
            EventPayload::TurnCompleted(payload) => {
                if let Some(tid) = self.turn_id.clone() {
                    self.completed_turn_ids.push(tid);
                }
                self.data = payload.data.clone();
                self.turn_cost = Decimal::ZERO;
                self.turn_token_usage.clear();
            }
        }
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
                self.tool_calls.get(id).and_then(|c| c.anchor.as_deref())
            }
            Trigger::SubAgentFinished { session_id, .. } => self
                .sub_agent_calls
                .get(session_id)
                .and_then(|c| c.anchor.as_deref()),
            Trigger::LlmFinished { id, .. } | Trigger::LlmExecute { id, .. } => {
                self.llm_calls.get(id).and_then(|c| c.anchor.as_deref())
            }
            _ => None,
        }
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

    /// Whether a decision's landing branch is parked: transcripts gate at
    /// their landing leaf, effect settles at their anchor.
    pub fn decision_parked(&self, trigger: &Trigger) -> bool {
        if let Trigger::ClientTranscript { messages, .. } = trigger {
            let messages = self.normalize_client_view(messages.clone());
            let known: std::collections::HashSet<&str> =
                self.nodes.iter().map(|n| n.message.id.as_str()).collect();
            let plan = super::reconcile::plan_reconcile(&known, &messages);
            let landing = super::reconcile::landing_leaf(&messages, &plan);
            return self.active_interrupt_for(landing.as_deref()).is_some();
        }
        self.anchor_parked(self.trigger_anchor(trigger))
    }

    pub fn has_pending_llm(&self) -> bool {
        self.llm_calls
            .values()
            .any(|c| c.tracking.status == EffectStatus::Pending)
    }

    pub fn has_pending_worker_decision(&self) -> bool {
        self.worker_decisions
            .values()
            .any(|wd| wd.tracking.status == EffectStatus::Pending)
    }

    /// All queued decisions in arrival order.
    pub fn queued_decisions(&self) -> Vec<&WorkerDecisionState> {
        let mut queued: Vec<&WorkerDecisionState> = self
            .worker_decisions
            .values()
            .filter(|wd| wd.tracking.status == EffectStatus::Queued)
            .collect();
        queued.sort_by_key(|wd| wd.source_event_sequence);
        queued
    }

    pub fn has_queued_worker_decision(&self) -> bool {
        self.worker_decisions
            .values()
            .any(|wd| wd.tracking.status == EffectStatus::Queued)
    }

    /// Next timer, counting only work on unparked branches.
    pub fn wake_at(&self) -> Option<DateTime<Utc>> {
        if matches!(self.status, SessionStatus::Done) {
            return None;
        }
        if !self.has_pending_worker_decision()
            && self
                .queued_decisions()
                .iter()
                .any(|d| !self.decision_parked(&d.trigger))
        {
            return Some(Utc::now());
        }
        self.llm_calls
            .values()
            .filter(|c| !self.anchor_parked(c.anchor.as_deref()))
            .filter_map(|c| c.tracking.earliest_wake())
            .chain(
                self.tool_calls
                    .values()
                    .filter(|c| !self.anchor_parked(c.anchor.as_deref()))
                    .filter_map(|c| c.tracking.earliest_wake()),
            )
            .chain(
                self.sub_agent_calls
                    .values()
                    .filter(|c| !self.anchor_parked(c.anchor.as_deref()))
                    .filter_map(|c| c.tracking.earliest_wake()),
            )
            .chain(
                self.worker_decisions
                    .values()
                    .filter(|d| !self.decision_parked(&d.trigger))
                    .filter_map(|d| d.tracking.earliest_wake()),
            )
            .min()
    }

    /// All message ids on the root→`leaf` chain.
    pub fn path_ids<'a>(&'a self, leaf: &'a str) -> std::collections::HashSet<&'a str> {
        let by_id: HashMap<&str, &NewMessage> = self
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
            .map(|v| v.state.clone())
            .unwrap_or_default()
    }

    /// Newest agent config whose anchor is on the path to `leaf`; unanchored matches
    /// any path. `None` when no config was ever written on the path.
    pub fn resolve_agent_for(&self, leaf: Option<&str>) -> Option<AgentConfig> {
        let on_path = leaf.map(|l| self.path_ids(l)).unwrap_or_default();
        resolve_on_path(&self.agent_versions, &on_path).map(|v| v.agent.clone())
    }

    pub fn message_tree(&self) -> MessageTree {
        MessageTree {
            nodes: self.nodes.clone(),
            head_id: self.head_id.clone(),
        }
    }

    /// The effects still in flight (Pending or RetryScheduled).
    pub fn effects(&self) -> Vec<Effect> {
        let in_flight =
            |s: &EffectStatus| matches!(s, EffectStatus::Pending | EffectStatus::RetryScheduled);
        let envelope =
            |id: &str, kind: EffectKind, t: &EffectTracking, anchor: &Option<String>| Effect {
                id: id.to_string(),
                kind,
                status: t.status.clone(),
                attempt: t.retry.attempts,
                deadline: t.deadline,
                anchor: anchor.clone(),
                name: None,
                arguments: None,
                handler: None,
                stream: None,
                agent_id: None,
                session_id: None,
            };

        let mut effects: Vec<Effect> = Vec::new();

        for c in self
            .tool_calls
            .values()
            .filter(|c| in_flight(&c.tracking.status))
        {
            let mut e = envelope(
                &c.tool_call_id,
                EffectKind::ToolCall,
                &c.tracking,
                &c.anchor,
            );
            e.name = Some(c.name.clone());
            e.arguments = Some(c.arguments.clone());
            e.handler = Some(c.handler.into());
            effects.push(e);
        }

        for c in self
            .sub_agent_calls
            .values()
            .filter(|c| in_flight(&c.tracking.status))
        {
            let mut e = envelope(
                &c.tool_call_id,
                EffectKind::SubAgent,
                &c.tracking,
                &c.anchor,
            );
            e.agent_id = Some(c.agent_id.clone());
            e.session_id = Some(c.session_id.clone());
            effects.push(e);
        }

        for c in self
            .llm_calls
            .values()
            .filter(|c| in_flight(&c.tracking.status))
        {
            let mut e = envelope(&c.call_id, EffectKind::LlmCall, &c.tracking, &c.anchor);
            e.handler = Some(c.handler.into());
            e.stream = Some(c.stream);
            effects.push(e);
        }

        effects.sort_by(|a, b| a.id.cmp(&b.id));
        effects
    }

    /// A call is open while any of its tool calls lacks a recorded answer on the
    /// active path. The tree is frozen during a pending decision, so the call
    /// answering the last `tool.finished` is still open when that decision is
    /// projected — exactly when its re-issue proposal is derived.
    fn open_llm_calls(&self, tree: &MessageTree) -> HashMap<String, LlmCallState> {
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
            .filter_map(|m| self.llm_calls.get(&m.id).map(|c| (m.id.clone(), c.clone())))
            .collect()
    }

    pub fn derived_state(&self) -> DerivedState {
        let message_tree = self.message_tree();
        let open_llm_calls = self.open_llm_calls(&message_tree);
        DerivedState {
            status: self.projected_status(),
            wake_at: self.wake_at(),
            calls: self.effects(),
            owner: self.owner.clone(),
            agent_id: self.agent_id.clone(),
            worker_state: self.resolve_state_for(self.head_id.as_deref()),
            agent_config: self.resolve_agent_for(self.head_id.as_deref()),
            message_tree,
            ancestry: self.ancestry.clone(),
            sub_agent_calls: self.sub_agent_calls.clone(),
            open_llm_calls,
            worker_decisions: self.worker_decisions.clone(),
            turn_id: self.turn_id.clone(),
            cost: self.cost,
            sub_agent_cost: self.sub_agent_cost,
            turn_cost: self.turn_cost,
            turn_token_usage: self.turn_token_usage.clone(),
            token_usage: self.token_usage.clone(),
            sub_agent_token_usage: self.sub_agent_token_usage.clone(),
        }
    }

    fn track_usage(&mut self, usage: &Option<serde_json::Value>) {
        let obj = match usage.as_ref().and_then(|v| v.as_object()) {
            Some(o) => o,
            None => return,
        };
        for (k, v) in obj {
            if let Some(n) = v.as_u64() {
                *self.token_usage.entry(k.clone()).or_insert(0) += n;
            }
        }
    }

    fn track_turn_usage(&mut self, usage: &Option<serde_json::Value>) {
        let obj = match usage.as_ref().and_then(|v| v.as_object()) {
            Some(o) => o,
            None => return,
        };
        for (k, v) in obj {
            if let Some(n) = v.as_u64() {
                *self.turn_token_usage.entry(k.clone()).or_insert(0) += n;
            }
        }
    }
}

#[cfg(test)]
mod open_llm_calls_tests {
    use super::*;
    use crate::protocol::{NewMessage, ToolCall, ToolCallFunction};

    fn node(message: Message, parent_id: Option<&str>) -> NewMessage {
        NewMessage {
            message,
            parent_id: parent_id.map(str::to_string),
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
        }
    }

    fn call_state(call_id: &str) -> LlmCallState {
        LlmCallState {
            format: None,
            call_id: call_id.to_string(),
            tracking: EffectTracking::new(RetryPolicy::no_retry(), Utc::now()),
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
            anchor: None,
        }
    }

    #[test]
    fn a_call_is_open_until_its_tool_answer_is_recorded() {
        let mut s = SessionState::new("sess-1".to_string());
        s.nodes.push(node(user("u1"), None));
        s.nodes
            .push(node(assistant_calling("call-1", "tc-1"), Some("u1")));
        s.head_id = Some("call-1".to_string());
        s.llm_calls
            .insert("call-1".to_string(), call_state("call-1"));

        assert!(
            s.derived_state().open_llm_calls.contains_key("call-1"),
            "unanswered tool call keeps the parent open"
        );

        s.nodes
            .push(node(tool_answer("t1", "tc-1"), Some("call-1")));
        s.head_id = Some("t1".to_string());
        assert!(
            s.derived_state().open_llm_calls.is_empty(),
            "recorded answer closes the call"
        );
    }
}

#[cfg(test)]
mod state_version_tests {
    use serde_json::json;

    use super::*;
    use crate::protocol::NewMessage;

    fn message_node(id: &str, parent_id: Option<&str>) -> NewMessage {
        NewMessage {
            message: Message {
                id: id.to_string(),
                role: Role::User,
                content: None,
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
            },
            parent_id: parent_id.map(str::to_string),
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

    fn version(state: serde_json::Value, anchor: Option<&str>) -> StateVersion {
        StateVersion {
            state: WorkerState(state),
            anchor: anchor.map(str::to_string),
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
    fn same_anchor_versions_compact_to_the_newest() {
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
        assert_eq!(s.state_versions.len(), 2);
        assert_eq!(s.resolve_state_for(s.head_id.as_deref()).0, json!({"v": 3}));
        assert_eq!(s.resolve_state_for(Some("x1")).0, json!({"v": 1}));
    }

    #[test]
    fn state_version_serializes_with_the_state_field() {
        // Guard: generalizing the store must not rename the persisted field, or
        // existing snapshots silently lose their versions on load.
        let v = version(json!({"v": 1}), Some("u1"));
        let value = serde_json::to_value(&v).expect("serializes");
        assert_eq!(value, json!({"state": {"v": 1}, "anchor": "u1"}));
    }
}

#[cfg(test)]
mod agent_version_tests {
    use super::*;
    use crate::protocol::NewMessage;

    fn message_node(id: &str, parent_id: Option<&str>) -> NewMessage {
        NewMessage {
            message: Message {
                id: id.to_string(),
                role: Role::User,
                content: None,
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
            },
            parent_id: parent_id.map(str::to_string),
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
            format: None,
            model: model.to_string(),
            system: None,
            stream: false,
            handler: None,
            retry: None,
            tools: Vec::new(),
            sub_agents: Vec::new(),
        }
    }

    fn version(model: &str, anchor: Option<&str>) -> AgentVersion {
        AgentVersion {
            agent: config(model),
            anchor: anchor.map(str::to_string),
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
    fn same_anchor_configs_compact_to_the_newest() {
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
        assert_eq!(s.agent_versions.len(), 2);
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
            session_id: None,
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
