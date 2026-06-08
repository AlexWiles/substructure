use std::collections::{BTreeMap, HashMap};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::decision::{DecisionTrigger, ToolResult};
use super::events::*;
use super::message::Role;
use rust_decimal::Decimal;

use crate::runtime::aggregate::ApplyContext;
use crate::runtime::llm::LlmRequest;
use crate::runtime::owner::SessionOwner;
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
    /// Original request, stored for retries and crash recovery.
    pub request: LlmRequest,
    pub stream: bool,
    #[serde(default)]
    pub handler: LlmHandler,
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
    /// Source event sequence, for stable result ordering.
    #[serde(default)]
    pub order: u64,
    /// Result already delivered to the worker.
    #[serde(default)]
    pub delivered: bool,
}

impl ToolCallState {
    fn to_result(&self) -> ToolResult {
        ToolResult {
            tool_call_id: self.tool_call_id.clone(),
            name: self.name.clone(),
            content: self.result.clone().unwrap_or_default(),
            is_error: self.is_error,
        }
    }
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
    #[serde(default)]
    pub order: u64,
    #[serde(default)]
    pub delivered: bool,
}

impl SubAgentCallState {
    fn to_result(&self) -> ToolResult {
        ToolResult {
            tool_call_id: self.tool_call_id.clone(),
            name: self.agent_id.clone(),
            content: self.result.clone().unwrap_or_default(),
            is_error: self.is_error,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerDecisionState {
    pub decision_id: String,
    pub tracking: EffectTracking,
    pub trigger: DecisionTrigger,
    #[serde(default)]
    pub source_event_sequence: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedState {
    pub status: SessionStatus,
    pub wake_at: Option<DateTime<Utc>>,
    pub owner: Option<SessionOwner>,
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
    pub worker_state: Vec<u8>,

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
            worker_state: Vec::new(),
            ancestry: Vec::new(),
            data: serde_json::Value::Null,
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
                self.owner = Some(payload.owner.clone());
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
                    existing.handler = payload.handler.clone();
                } else {
                    self.llm_calls.insert(
                        payload.call_id.clone(),
                        LlmCallState {
                            call_id: payload.call_id.clone(),
                            tracking: EffectTracking::new(payload.retry.clone(), now),
                            request: payload.request.clone(),
                            stream: payload.stream,
                            handler: payload.handler.clone(),
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
                            order: ctx.sequence,
                            delivered: false,
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
                            order: ctx.sequence,
                            delivered: false,
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
                            source_event_sequence: ctx.sequence,
                        },
                    );
                }
                if let DecisionTrigger::EffectsComplete { results } = &p.trigger {
                    let ids: Vec<String> = results.iter().map(|r| r.tool_call_id.clone()).collect();
                    self.mark_results_delivered(&ids);
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
                if let DecisionTrigger::EffectsComplete { results } = &p.trigger {
                    let ids: Vec<String> = results.iter().map(|r| r.tool_call_id.clone()).collect();
                    self.mark_results_delivered(&ids);
                }
            }
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

    pub fn active_interrupt(&self) -> Option<&str> {
        match &self.status {
            SessionStatus::Interrupted { interrupt_id } => Some(interrupt_id),
            _ => None,
        }
    }

    /// Ordered results for the undelivered effects once every one is terminal;
    /// `None` while any is still pending or there are none.
    pub fn drainable_batch(&self) -> Option<Vec<ToolResult>> {
        let any = self.tool_calls.values().any(|t| !t.delivered)
            || self.sub_agent_calls.values().any(|s| !s.delivered);

        if !any {
            return None;
        }

        let all_terminal = self
            .tool_calls
            .values()
            .filter(|t| !t.delivered)
            .all(|t| t.result.is_some())
            && self
                .sub_agent_calls
                .values()
                .filter(|s| !s.delivered)
                .all(|s| s.result.is_some());

        if !all_terminal {
            return None;
        }

        Some(self.collect_undelivered(None))
    }

    /// `drainable_batch` for the moment a result arrives: `pending` stands in
    /// for the effect being completed, whose result is not yet in state. Drains
    /// once every *other* undelivered effect is terminal.
    pub fn drainable_with(&self, pending: ToolResult) -> Option<Vec<ToolResult>> {
        let others_terminal = self
            .tool_calls
            .values()
            .filter(|t| !t.delivered && t.tool_call_id != pending.tool_call_id)
            .all(|t| t.result.is_some())
            && self
                .sub_agent_calls
                .values()
                .filter(|s| !s.delivered && s.tool_call_id != pending.tool_call_id)
                .all(|s| s.result.is_some());

        if !others_terminal {
            return None;
        }

        Some(self.collect_undelivered(Some(&pending)))
    }

    fn collect_undelivered(&self, overlay: Option<&ToolResult>) -> Vec<ToolResult> {
        let tools = self
            .tool_calls
            .values()
            .filter(|t| !t.delivered)
            .map(|t| (t.order, t.to_result()));

        let subs = self
            .sub_agent_calls
            .values()
            .filter(|s| !s.delivered)
            .map(|s| (s.order, s.to_result()));

        let mut ordered: Vec<(u64, ToolResult)> = tools.chain(subs).collect();

        ordered.sort_by_key(|(order, _)| *order);

        ordered
            .into_iter()
            .map(|(_, result)| match overlay {
                // The in-flight result isn't in state yet; substitute it in.
                Some(o) if o.tool_call_id == result.tool_call_id => o.clone(),
                _ => result,
            })
            .collect()
    }

    pub fn has_undelivered_effects(&self) -> bool {
        self.tool_calls.values().any(|t| !t.delivered)
            || self.sub_agent_calls.values().any(|s| !s.delivered)
    }

    fn mark_results_delivered(&mut self, tool_call_ids: &[String]) {
        for id in tool_call_ids {
            if let Some(t) = self.tool_calls.get_mut(id) {
                t.delivered = true;
            }
            for s in self.sub_agent_calls.values_mut() {
                if &s.tool_call_id == id {
                    s.delivered = true;
                }
            }
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

    pub fn has_pending_worker_decision(&self) -> bool {
        self.worker_decisions
            .values()
            .any(|wd| wd.tracking.status == EffectStatus::Pending)
    }

    pub fn next_queued_decision(&self) -> Option<&WorkerDecisionState> {
        self.worker_decisions
            .values()
            .filter(|wd| wd.tracking.status == EffectStatus::Queued)
            .min_by_key(|wd| wd.source_event_sequence)
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
        if !self.has_pending_worker_decision() && self.drainable_batch().is_some() {
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

    pub fn derived_state(&self) -> DerivedState {
        DerivedState {
            status: self.status.clone(),
            wake_at: self.wake_at(),
            owner: self.owner.clone(),
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
