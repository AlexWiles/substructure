use std::collections::{BTreeMap, HashMap};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::decision::DecisionTrigger;
use super::events::*;
use super::message::Role;
use rust_decimal::Decimal;

use crate::runtime::aggregate::ApplyContext;
use crate::runtime::identity::ClientIdentity;
use crate::runtime::llm::LlmRequest;
use crate::runtime::retry::{RetryPolicy, RetryState};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    /// Waiting for external input: LLM responses, tool results, worker decisions.
    Idle,
    /// Paused for external input (e.g., human approval).
    Interrupted { interrupt_id: String },
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
    /// Original request, stored for retries and crash recovery.
    pub request: LlmRequest,
    pub stream: bool,
    /// Which LLM provider to use (e.g. "openrouter", "mock").
    #[serde(default)]
    pub llm_client: String,
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
    /// Result content, stored for enriching ToolResult trigger.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    #[serde(default)]
    pub is_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubAgentCallState {
    pub session_id: String,
    pub agent_id: String,
    pub tracking: EffectTracking,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerDecisionState {
    pub decision_id: String,
    pub tracking: EffectTracking,
    pub trigger: DecisionTrigger,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedState {
    pub status: SessionStatus,
    pub wake_at: Option<DateTime<Utc>>,
    pub auth: Option<ClientIdentity>,
    pub agent_id: Option<String>,
    #[serde(default)]
    pub worker_state: Vec<u8>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    pub session_id: String,
    pub status: SessionStatus,
    pub agent_id: Option<String>,
    pub auth: Option<ClientIdentity>,
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
    pub worker_state: Vec<u8>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ancestry: Vec<String>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<Artifact>,

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
}

impl SessionState {
    pub fn new(session_id: String) -> Self {
        SessionState {
            session_id,
            status: SessionStatus::Done,
            agent_id: None,
            auth: None,
            token_usage: BTreeMap::new(),
            cost: Decimal::ZERO,
            sub_agent_cost: Decimal::ZERO,
            turn_cost: Decimal::ZERO,
            turn_token_usage: BTreeMap::new(),
            sub_agent_token_usage: BTreeMap::new(),
            worker_state: Vec::new(),
            ancestry: Vec::new(),
            artifacts: vec![],
            worker_retry: None,
            llm_calls: HashMap::new(),
            tool_calls: HashMap::new(),
            sub_agent_calls: HashMap::new(),
            worker_decisions: HashMap::new(),
            turn_id: None,
            completed_turn_ids: Vec::new(),
        }
    }

    pub fn apply(&mut self, event: &EventPayload, ctx: &ApplyContext) {
        let now = ctx.occurred_at;
        match event {
            EventPayload::SessionCreated(payload) => {
                self.status = SessionStatus::Idle;
                self.agent_id = Some(payload.agent_id.clone());
                self.auth = Some(payload.auth.clone());
                self.ancestry = payload.ancestry.clone();
                self.worker_retry = Some(payload.worker_retry.clone());
            }
            EventPayload::NewMessage(payload) => {
                if payload.message.role == Role::User {
                    self.status = SessionStatus::Idle;
                }
            }
            EventPayload::LlmCallRequested(payload) => {
                self.status = SessionStatus::Idle;
                if let Some(existing) = self.llm_calls.get_mut(&payload.call_id) {
                    existing.tracking.reset_pending(now);
                    existing.request = payload.request.clone();
                    existing.stream = payload.stream;
                } else {
                    self.llm_calls.insert(
                        payload.call_id.clone(),
                        LlmCallState {
                            call_id: payload.call_id.clone(),
                            tracking: EffectTracking::new(payload.retry.clone(), now),
                            request: payload.request.clone(),
                            stream: payload.stream,
                            llm_client: payload.llm_client.clone(),
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
                }
            }
            EventPayload::SessionInterrupted(payload) => {
                self.status = SessionStatus::Interrupted {
                    interrupt_id: payload.interrupt_id.clone(),
                };
                for call in self.llm_calls.values_mut() {
                    if call.tracking.status == EffectStatus::Pending {
                        call.tracking.status = EffectStatus::Failed;
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
                        },
                    );
                }
                self.status = SessionStatus::Idle;
            }
            EventPayload::WorkerDecisionCompleted(p) => {
                if let Some(wd) = self.worker_decisions.get_mut(&p.decision_id) {
                    wd.tracking.complete();
                }
                self.worker_state = p.state.clone();
            }
            EventPayload::WorkerDecisionErrored(p) => {
                if let Some(wd) = self.worker_decisions.get_mut(&p.decision_id) {
                    wd.tracking.record_error(p.retryable, now);
                }
            }
            EventPayload::SessionMessageRequested(_) => {}
            EventPayload::ToolCallResolutionRequested(_) => {}
            EventPayload::WorkerStateUpdated(p) => {
                self.worker_state = p.state.clone();
            }
            EventPayload::SessionCancelled => {
                self.status = SessionStatus::Done;
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
                self.artifacts = payload.artifacts.clone();
                self.turn_cost = Decimal::ZERO;
                self.turn_token_usage.clear();
            }
        }
    }

    pub fn active_interrupt(&self) -> Option<&str> {
        match &self.status {
            SessionStatus::Interrupted { interrupt_id } => Some(interrupt_id),
            _ => None,
        }
    }

    pub fn all_tools_resolved(&self) -> bool {
        !self.tool_calls.is_empty()
            && self.tool_calls.values().all(|tc| {
                tc.tracking.status == EffectStatus::Completed
                    || tc.tracking.status == EffectStatus::Failed
            })
    }

    pub fn has_pending_llm(&self) -> bool {
        self.llm_calls
            .values()
            .any(|c| c.tracking.status == EffectStatus::Pending)
    }

    pub fn wake_at(&self) -> Option<DateTime<Utc>> {
        match self.status {
            SessionStatus::Done | SessionStatus::Interrupted { .. } => return None,
            _ => {}
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

    pub fn derived_state(&self) -> DerivedState {
        DerivedState {
            status: self.status.clone(),
            wake_at: self.wake_at(),
            auth: self.auth.clone(),
            agent_id: self.agent_id.clone(),
            worker_state: self.worker_state.clone(),
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
