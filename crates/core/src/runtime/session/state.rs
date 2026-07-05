use std::collections::{BTreeMap, HashMap};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::decision::DecisionTrigger;
use super::events::*;
use super::message::{Message, Role};
use rust_decimal::Decimal;

use crate::runtime::aggregate::ApplyContext;
use crate::runtime::llm::{LlmRequest, LlmTool, ReasoningConfig};
use crate::runtime::owner::SessionOwner;
use crate::runtime::retry::{RetryPolicy, RetryState};
use crate::runtime::worker::WorkerState;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    /// Waiting for external input: LLM responses, tool results, worker decisions.
    Idle,
    /// Paused for external input (e.g., human approval).
    Interrupted {
        interrupt_id: String,
        origin: InterruptOrigin,
        reason: String,
    },
    /// Agent loop finished. Waiting for next user input.
    Done,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffectStatus {
    Pending,
    Completed,
    Failed,
    RetryScheduled,
    Queued,
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

/// A worker-state write anchored to the tree position it was made at; the
/// current state is resolved, never stored (`resolve_state_for`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateVersion {
    pub state: WorkerState,
    pub anchor: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerDecisionState {
    pub decision_id: String,
    pub tracking: EffectTracking,
    pub trigger: DecisionTrigger,
    #[serde(default)]
    pub source_event_sequence: u64,
}

// In-flight effects surfaced on each worker decision, derived on read by filtering
// the effect maps to what's still outstanding (Pending or RetryScheduled).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Effect {
    pub id: String,
    pub status: EffectStatus,
    pub attempt: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deadline: Option<DateTime<Utc>>,
    /// The tree node the effect was requested at.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
    #[serde(flatten)]
    pub detail: EffectDetail,
}

/// Kind-specific fields, tagged by `kind` on the wire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EffectDetail {
    ToolCall {
        name: String,
        arguments: String,
        handler: ToolHandler,
    },
    SubAgent {
        agent_id: String,
        session_id: String,
    },
    LlmCall {
        handler: LlmHandler,
        stream: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedState {
    pub status: SessionStatus,
    pub wake_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub effects: Vec<Effect>,
    pub owner: Option<SessionOwner>,
    pub agent_id: Option<String>,
    #[serde(default)]
    pub worker_state: WorkerState,
    #[serde(default)]
    pub message_tree: MessageTree,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ancestry: Vec<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub sub_agent_calls: HashMap<String, SubAgentCallState>,
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

pub(super) fn new_call_id() -> String {
    Uuid::now_v7().to_string()
}

pub(super) fn new_message_id() -> String {
    Uuid::now_v7().to_string()
}

/// A JSON string passes through; anything else is serialized.
pub(super) fn json_to_string(v: &serde_json::Value) -> String {
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
    pub nodes: Vec<Node>,
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
                if !payload.message.id.is_empty() {
                    self.head_id = Some(payload.message.id.clone());
                }
                self.nodes.push(Node::Message(payload.clone()));
            }
            EventPayload::NewControl(payload) => {
                if !payload.control.id.is_empty() {
                    self.head_id = Some(payload.control.id.clone());
                }
                self.nodes.push(Node::Control(payload.clone()));
            }
            EventPayload::LlmCallRequested(payload) => {
                self.status = SessionStatus::Idle;
                let prompt = payload.request.messages.clone();
                let spec = LlmCallSpec::from(&payload.request);
                if let Some(existing) = self.llm_calls.get_mut(&payload.call_id) {
                    existing.tracking.reset_pending(now);
                    existing.prompt = prompt;
                    existing.spec = spec;
                    existing.stream = payload.stream;
                    existing.handler = payload.handler.clone();
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
                            handler: payload.handler.clone(),
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
                            handler: payload.handler.clone(),
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
                self.status = SessionStatus::Interrupted {
                    interrupt_id: payload.interrupt_id.clone(),
                    origin: payload.origin,
                    reason: payload.reason.clone(),
                };
                // Pending worker decisions are voided (effects void via the
                // batch's EffectVoided events): late submissions no-op via the
                // pending check, and the decision request emitted on resume
                // isn't queued behind a decision that will never complete.
                for wd in self.worker_decisions.values_mut() {
                    if wd.tracking.status == EffectStatus::Pending {
                        wd.tracking.status = EffectStatus::Failed;
                    }
                }
            }
            EventPayload::InterruptResumed(_) => {
                self.status = SessionStatus::Idle;
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
            EventPayload::EffectVoided(p) => {
                let tracking = match p.kind {
                    EffectKind::ToolCall => self.tool_calls.get_mut(&p.id).map(|c| &mut c.tracking),
                    EffectKind::LlmCall => self.llm_calls.get_mut(&p.id).map(|c| &mut c.tracking),
                    EffectKind::SubAgent => {
                        self.sub_agent_calls.get_mut(&p.id).map(|c| &mut c.tracking)
                    }
                };
                if let Some(tracking) = tracking {
                    tracking.status = EffectStatus::Failed;
                }
            }
            EventPayload::WorkerStateUpdated(p) => {
                // A superseded same-anchor version can never win resolution.
                self.state_versions.retain(|v| v.anchor != p.anchor);
                self.state_versions.push(StateVersion {
                    state: p.state.clone(),
                    anchor: p.anchor.clone(),
                });
            }
            EventPayload::SessionCancelled => {
                self.status = SessionStatus::Done;
                // Cancellation is terminal: void pending decisions so late
                // submissions no-op. Effects void via the batch's EffectVoided
                // events.
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

    pub fn active_interrupt(&self) -> Option<(&str, InterruptOrigin)> {
        match &self.status {
            SessionStatus::Interrupted {
                interrupt_id,
                origin,
                ..
            } => Some((interrupt_id, *origin)),
            _ => None,
        }
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

    pub fn wake_at(&self) -> Option<DateTime<Utc>> {
        match self.status {
            SessionStatus::Done | SessionStatus::Interrupted { .. } => return None,
            _ => {}
        }
        if self.has_queued_worker_decision() && !self.has_pending_worker_decision() {
            return Some(Utc::now());
        }
        self.llm_calls
            .values()
            .filter_map(|c| c.tracking.earliest_wake())
            .chain(
                self.tool_calls
                    .values()
                    .filter_map(|c| c.tracking.earliest_wake()),
            )
            .chain(
                self.sub_agent_calls
                    .values()
                    .filter_map(|c| c.tracking.earliest_wake()),
            )
            .chain(
                self.worker_decisions
                    .values()
                    .filter_map(|d| d.tracking.earliest_wake()),
            )
            .min()
    }

    /// All node ids (messages and controls) on the root→`leaf` chain.
    /// `MessageTree::path_to` returns only messages; anchors may be any node kind.
    fn path_ids<'a>(&'a self, leaf: &'a str) -> std::collections::HashSet<&'a str> {
        let by_id: HashMap<&str, &Node> = self.nodes.iter().map(|n| (n.id(), n)).collect();
        let mut ids = std::collections::HashSet::new();
        let mut cursor = Some(leaf);
        while let Some(id) = cursor {
            let Some(node) = by_id.get(id) else { break };
            if !ids.insert(id) {
                break; // malformed parent cycle guard, mirrors path_to
            }
            cursor = node.parent_id();
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

    /// Newest version whose anchor is on the path to `leaf`; unanchored matches any path.
    pub fn resolve_state_for(&self, leaf: Option<&str>) -> WorkerState {
        let on_path = leaf.map(|l| self.path_ids(l)).unwrap_or_default();
        self.state_versions
            .iter()
            .rev()
            .find(|v| match v.anchor.as_deref() {
                None => true,
                Some(a) => on_path.contains(a),
            })
            .map(|v| v.state.clone())
            .unwrap_or_default()
    }

    pub fn message_tree(&self) -> MessageTree {
        MessageTree {
            nodes: self.nodes.clone(),
            head_id: self.head_id.clone(),
        }
    }

    /// The effects still in flight (Pending or RetryScheduled), derived by filtering the effect maps.
    pub fn effects(&self) -> Vec<Effect> {
        let in_flight =
            |s: &EffectStatus| matches!(s, EffectStatus::Pending | EffectStatus::RetryScheduled);

        let mut effects: Vec<Effect> = Vec::new();

        for c in self
            .tool_calls
            .values()
            .filter(|c| in_flight(&c.tracking.status))
        {
            effects.push(Effect {
                id: c.tool_call_id.clone(),
                status: c.tracking.status.clone(),
                attempt: c.tracking.retry.attempts,
                deadline: c.tracking.deadline,
                anchor: c.anchor.clone(),
                detail: EffectDetail::ToolCall {
                    name: c.name.clone(),
                    arguments: c.arguments.clone(),
                    handler: c.handler.clone(),
                },
            });
        }

        for c in self
            .sub_agent_calls
            .values()
            .filter(|c| in_flight(&c.tracking.status))
        {
            effects.push(Effect {
                id: c.tool_call_id.clone(),
                status: c.tracking.status.clone(),
                attempt: c.tracking.retry.attempts,
                deadline: c.tracking.deadline,
                anchor: c.anchor.clone(),
                detail: EffectDetail::SubAgent {
                    agent_id: c.agent_id.clone(),
                    session_id: c.session_id.clone(),
                },
            });
        }

        for c in self
            .llm_calls
            .values()
            .filter(|c| in_flight(&c.tracking.status))
        {
            effects.push(Effect {
                id: c.call_id.clone(),
                status: c.tracking.status.clone(),
                attempt: c.tracking.retry.attempts,
                deadline: c.tracking.deadline,
                anchor: c.anchor.clone(),
                detail: EffectDetail::LlmCall {
                    handler: c.handler.clone(),
                    stream: c.stream,
                },
            });
        }

        effects.sort_by(|a, b| a.id.cmp(&b.id));
        effects
    }

    pub fn derived_state(&self) -> DerivedState {
        DerivedState {
            status: self.status.clone(),
            wake_at: self.wake_at(),
            effects: self.effects(),
            owner: self.owner.clone(),
            agent_id: self.agent_id.clone(),
            worker_state: self.resolve_state_for(self.head_id.as_deref()),
            message_tree: self.message_tree(),
            ancestry: self.ancestry.clone(),
            sub_agent_calls: self.sub_agent_calls.clone(),
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
mod state_version_tests {
    use serde_json::json;

    use super::*;

    fn message_node(id: &str, parent_id: Option<&str>) -> Node {
        Node::Message(NewMessage {
            message: Message {
                id: id.to_string(),
                role: Role::User,
                content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            parent_id: parent_id.map(str::to_string),
        })
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
}

#[cfg(test)]
mod effect_tests {
    use super::*;

    #[test]
    fn effect_serializes_flat_tagged_and_round_trips() {
        let e = Effect {
            id: "call_1".to_string(),
            status: EffectStatus::Pending,
            attempt: 0,
            deadline: None,
            anchor: Some("n1".to_string()),
            detail: EffectDetail::ToolCall {
                name: "get_weather".to_string(),
                arguments: "{}".to_string(),
                handler: ToolHandler::Worker,
            },
        };
        let json = serde_json::to_value(&e).unwrap();
        // Envelope, tag, and detail all flattened onto one object.
        assert_eq!(json["id"], "call_1");
        assert_eq!(json["status"], "pending");
        assert_eq!(json["kind"], "tool_call");
        assert_eq!(json["name"], "get_weather");
        // flatten + internally-tagged enum must deserialize back to the original.
        let back: Effect = serde_json::from_value(json).unwrap();
        assert_eq!(back, e);
    }
}
