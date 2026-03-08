use std::cmp::min;
use std::collections::{BTreeMap, HashMap};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::decision::{DecisionTrigger, WorkerDecisionRequested};
use super::dispatch::{ToolCallDispatch, WorkerDispatch};
use super::types::{Artifact, CompletionDelivery, ToolHandler};
use crate::runtime::aggregate::{AggregateState, AggregateStatus, Emit};
use crate::runtime::budget::{self, BudgetContext, BudgetError};
use crate::runtime::config::{ClientIdentity, ExhaustionStrategy, RetryPolicy};
use crate::runtime::event::EventPayload;
use crate::runtime::llm::{LlmCallRequested, LlmRequest, LlmResponse};
use crate::runtime::llm::StreamDelta;
use crate::runtime::message::{Message, Role};
use super::system::SessionSystem;
use crate::runtime::span::SpanContext;
use async_trait::async_trait;

use super::command::{CommandPayload, SessionError};

// ---------------------------------------------------------------------------
// SessionContext — transient state passed through handle_command/on_event
// ---------------------------------------------------------------------------

/// Transient context for command handling and event reactions — not persisted.
pub struct SessionContext {
    pub session_id: Uuid,
    pub auth: ClientIdentity,
    pub stream: bool,
    /// Shared infrastructure handle — inter-session messaging, LLM, workers.
    pub system: SessionSystem,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub name: String,
    pub content: String,
    pub is_error: bool,
}

pub(super) fn new_call_id() -> String {
    Uuid::new_v4().to_string()
}

// ---------------------------------------------------------------------------
// Session status
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    /// Work in flight: LLM calls, tool calls, worker decisions.
    Active,
    /// Nothing in flight. Wake scheduler will check wake_at() for retry timing.
    Idle,
    /// Paused for external input (e.g., human approval).
    Interrupted { interrupt_id: String },
    /// Agent loop finished. Waiting for next user input.
    Done,
}

// ---------------------------------------------------------------------------
// Retry tracking (per-call)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetryState {
    pub attempts: u32,
    pub next_at: Option<DateTime<Utc>>,
}

// ---------------------------------------------------------------------------
// LLM call lifecycle
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LlmCallStatus {
    Pending,
    Completed,
    Failed,
    RetryScheduled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCallState {
    pub call_id: String,
    pub status: LlmCallStatus,
    pub retry: RetryState,
    pub deadline: DateTime<Utc>,
    pub retry_policy: RetryPolicy,
    /// Original request, stored for retries and crash recovery.
    pub request: LlmRequest,
    pub stream: bool,
    /// Which LLM provider to use (e.g. "openrouter", "mock").
    #[serde(default)]
    pub llm_client: String,
}

// ---------------------------------------------------------------------------
// Tool call lifecycle
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ToolCallStatus {
    Pending,
    Completed,
    Failed,
    RetryScheduled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallState {
    pub tool_call_id: String,
    pub name: String,
    pub status: ToolCallStatus,
    pub retry: RetryState,
    pub retry_policy: RetryPolicy,
    pub deadline: DateTime<Utc>,
    /// Opaque context from the worker, passed through to transport dispatch.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub context: serde_json::Value,
    #[serde(default)]
    pub handler: ToolHandler,
    /// Original arguments, stored for retries and crash recovery.
    #[serde(default)]
    pub arguments: String,
    /// Result content, stored for enriching AllToolsResolved trigger.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    #[serde(default)]
    pub is_error: bool,
}

// ---------------------------------------------------------------------------
// DerivedState — session-specific query optimization data stamped on events
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedState {
    pub status: SessionStatus,
    pub wake_at: Option<DateTime<Utc>>,
}

// ---------------------------------------------------------------------------
// SessionState — the reducer state for session aggregates
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    pub session_id: Uuid,
    pub status: SessionStatus,
    pub agent_name: Option<String>,
    pub auth: Option<ClientIdentity>,
    pub token_usage: BTreeMap<String, u64>,

    /// Opaque worker state — owned and managed exclusively by the worker.
    /// Session stores but never interprets these bytes.
    #[serde(default)]
    pub worker_state: Vec<u8>,

    /// Sub-agent completion delivery target.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on_done: Option<CompletionDelivery>,

    /// Artifacts produced when the session completes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<Artifact>,

    // Call lifecycle tracking
    pub llm_calls: HashMap<String, LlmCallState>,
    pub tool_calls: HashMap<String, ToolCallState>,
}

impl SessionState {
    pub fn new(session_id: Uuid) -> Self {
        SessionState {
            session_id,
            status: SessionStatus::Done,
            agent_name: None,
            auth: None,
            token_usage: BTreeMap::new(),
            worker_state: Vec::new(),
            on_done: None,
            artifacts: vec![],
            llm_calls: HashMap::new(),
            tool_calls: HashMap::new(),
        }
    }

    /// Apply a single event payload to the session state.
    pub fn apply_core(&mut self, payload: &EventPayload) {
        match payload {
            EventPayload::SessionCreated(payload) => {
                self.status = SessionStatus::Idle;
                self.agent_name = Some(payload.agent_name.clone());
                self.auth = Some(payload.auth.clone());
                self.on_done = payload.on_done.clone();
            }
            EventPayload::MessageUser(_) => {
                self.status = SessionStatus::Idle;
            }
            EventPayload::MessageAssistant(_) => {}
            EventPayload::MessageTool(_) => {}
            EventPayload::LlmCallRequested(payload) => {
                self.status = SessionStatus::Active;
                if let Some(existing) = self.llm_calls.get_mut(&payload.call_id) {
                    // Re-request (retry): preserve retry count, update request
                    existing.status = LlmCallStatus::Pending;
                    existing.deadline = payload.deadline;
                    existing.retry.next_at = None;
                    existing.request = payload.request.clone();
                    existing.stream = payload.stream;
                } else {
                    // New call
                    self.llm_calls.insert(
                        payload.call_id.clone(),
                        LlmCallState {
                            call_id: payload.call_id.clone(),
                            status: LlmCallStatus::Pending,
                            retry: RetryState::default(),
                            deadline: payload.deadline,
                            retry_policy: Self::resolve_llm_retry(
                                payload.timeout_secs,
                                payload.max_retries,
                            ),
                            request: payload.request.clone(),
                            stream: payload.stream,
                            llm_client: payload.llm_client.clone(),
                        },
                    );
                }
            }
            EventPayload::LlmCallCompleted(payload) => {
                self.track_usage(&payload.response);
                if let Some(call) = self.llm_calls.get_mut(&payload.call_id) {
                    call.status = LlmCallStatus::Completed;
                }
                self.status = SessionStatus::Idle;
            }
            EventPayload::LlmCallErrored(payload) => {
                if let Some(call) = self.llm_calls.get_mut(&payload.call_id) {
                    call.retry.attempts += 1;
                    let policy = &call.retry_policy;
                    if payload.retryable && call.retry.attempts < policy.max_retries {
                        call.status = LlmCallStatus::RetryScheduled;
                        let backoff = min(
                            policy.backoff_base_secs.saturating_pow(call.retry.attempts),
                            policy.backoff_max_secs,
                        );
                        call.retry.next_at =
                            Some(Utc::now() + chrono::Duration::seconds(i64::from(backoff)));
                    } else {
                        call.status = LlmCallStatus::Failed;
                        call.retry.next_at = None;
                    }
                }
                self.status = SessionStatus::Idle;
            }
            EventPayload::ToolCallRequested(payload) => {
                if let Some(existing) = self.tool_calls.get_mut(&payload.tool_call_id) {
                    // Re-request (sub-agent retry): reset to pending with fresh deadline
                    existing.status = ToolCallStatus::Pending;
                    existing.deadline = payload.deadline;
                    existing.arguments = payload.arguments.clone();
                } else {
                    self.tool_calls.insert(
                        payload.tool_call_id.clone(),
                        ToolCallState {
                            tool_call_id: payload.tool_call_id.clone(),
                            name: payload.name.clone(),
                            status: ToolCallStatus::Pending,
                            retry: RetryState::default(),
                            retry_policy: Self::resolve_tool_retry(
                                payload.timeout_secs,
                                payload.max_retries,
                            ),
                            deadline: payload.deadline,
                            context: payload.context.clone(),
                            handler: payload.handler.clone(),
                            arguments: payload.arguments.clone(),
                            result: None,
                            is_error: false,
                        },
                    );
                }
                // Active only if there's worker-handled work to do;
                // Idle if all pending calls are client tools.
                let has_worker_work = self.tool_calls.values().any(|tc| {
                    tc.status == ToolCallStatus::Pending && tc.handler != ToolHandler::Client
                });
                self.status = if has_worker_work {
                    SessionStatus::Active
                } else {
                    SessionStatus::Idle
                };
            }
            EventPayload::ToolCallCompleted(payload) => {
                if let Some(tc) = self.tool_calls.get_mut(&payload.tool_call_id) {
                    tc.status = ToolCallStatus::Completed;
                    tc.result = Some(payload.result.clone());
                    tc.is_error = false;
                }
                if !self.has_inflight_tools() {
                    self.status = SessionStatus::Idle;
                }
            }
            EventPayload::ToolCallErrored(payload) => {
                if let Some(tc) = self.tool_calls.get_mut(&payload.tool_call_id) {
                    tc.retry.attempts += 1;
                    let policy = &tc.retry_policy;
                    if payload.retryable && tc.retry.attempts < policy.max_retries {
                        tc.status = ToolCallStatus::RetryScheduled;
                        let backoff = min(
                            policy.backoff_base_secs.saturating_pow(tc.retry.attempts),
                            policy.backoff_max_secs,
                        );
                        tc.retry.next_at =
                            Some(Utc::now() + chrono::Duration::seconds(i64::from(backoff)));
                    } else {
                        tc.status = ToolCallStatus::Failed;
                        tc.retry.next_at = None;
                        tc.result = Some(payload.error.clone());
                        tc.is_error = true;
                    }
                }
                if !self.has_inflight_tools() {
                    self.status = SessionStatus::Idle;
                }
            }
            EventPayload::SessionInterrupted(payload) => {
                self.status = SessionStatus::Interrupted {
                    interrupt_id: payload.interrupt_id.clone(),
                };
                // Cancel pending LLM calls so they don't block future RequestLlmCall
                for call in self.llm_calls.values_mut() {
                    if call.status == LlmCallStatus::Pending {
                        call.status = LlmCallStatus::Failed;
                    }
                }
            }
            EventPayload::InterruptResumed(_) => {
                self.status = SessionStatus::Active;
            }
            EventPayload::WorkerDecisionRequested(_) => {
                // Keep Active while decision is pending
                self.status = SessionStatus::Active;
            }
            EventPayload::WorkerDecisionCompleted(p) => {
                self.worker_state = p.state.clone();
            }
            EventPayload::WorkerStateUpdated(p) => {
                self.worker_state = p.state.clone();
            }
            EventPayload::SessionCancelled => {
                self.status = SessionStatus::Done;
            }
            EventPayload::SessionDone(payload) => {
                self.artifacts = payload.artifacts.clone();
                if self.on_done.is_some() {
                    self.status = SessionStatus::Done;
                } else {
                    self.status = SessionStatus::Idle;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Query methods
    // -----------------------------------------------------------------------

    pub fn active_interrupt(&self) -> Option<&str> {
        match &self.status {
            SessionStatus::Interrupted { interrupt_id } => Some(interrupt_id),
            _ => None,
        }
    }

    /// True when any tool call is still pending or retrying.
    pub fn has_inflight_tools(&self) -> bool {
        self.tool_calls.values().any(|tc| {
            tc.status == ToolCallStatus::Pending || tc.status == ToolCallStatus::RetryScheduled
        })
    }

    /// True when all tool calls are in a terminal state (Completed or Failed).
    pub fn all_tools_resolved(&self) -> bool {
        !self.tool_calls.is_empty()
            && self.tool_calls.values().all(|tc| {
                tc.status == ToolCallStatus::Completed || tc.status == ToolCallStatus::Failed
            })
    }

    /// True when any LLM call is pending.
    pub fn has_pending_llm(&self) -> bool {
        self.llm_calls
            .values()
            .any(|c| c.status == LlmCallStatus::Pending)
    }

    // -----------------------------------------------------------------------
    // Wake scheduling
    // -----------------------------------------------------------------------

    /// Compute the earliest time this session needs attention.
    /// Returns `None` if the session is done/interrupted or has nothing pending.
    pub fn wake_at(&self) -> Option<DateTime<Utc>> {
        match self.status {
            SessionStatus::Done | SessionStatus::Interrupted { .. } => return None,
            _ => {}
        }
        // Earliest of: pending call deadlines, retry next_at for LLM and tool calls
        let pending_llm = self
            .llm_calls
            .values()
            .filter(|c| c.status == LlmCallStatus::Pending)
            .map(|c| c.deadline);
        let pending_tool = self
            .tool_calls
            .values()
            .filter(|c| c.status == ToolCallStatus::Pending && c.handler != ToolHandler::Client)
            .map(|c| c.deadline);
        let llm_retry_at = self
            .llm_calls
            .values()
            .filter(|c| c.status == LlmCallStatus::RetryScheduled)
            .filter_map(|c| c.retry.next_at);
        let tool_retry_at = self
            .tool_calls
            .values()
            .filter(|c| c.status == ToolCallStatus::RetryScheduled)
            .filter_map(|c| c.retry.next_at);
        pending_llm
            .chain(pending_tool)
            .chain(llm_retry_at)
            .chain(tool_retry_at)
            .min()
    }

    /// Compute the derived state envelope for the current session state.
    pub fn derived_state(&self) -> DerivedState {
        DerivedState {
            status: self.status.clone(),
            wake_at: self.wake_at(),
        }
    }

    fn track_usage(&mut self, response: &LlmResponse) {
        let raw = match response.usage() {
            Some(v) => v,
            None => return,
        };
        let breakdown = budget::flatten_usage(raw);
        for (k, v) in &breakdown {
            *self.token_usage.entry(k.clone()).or_insert(0) += v;
        }
    }
}

impl SessionState {
    pub fn aggregate_status(&self) -> AggregateStatus {
        match self.status {
            SessionStatus::Active => AggregateStatus::Active,
            SessionStatus::Idle | SessionStatus::Interrupted { .. } => AggregateStatus::Idle,
            SessionStatus::Done => AggregateStatus::Done,
        }
    }

    /// Resolve LLM retry policy from per-request hints, fallback to defaults.
    fn resolve_llm_retry(timeout_secs: Option<u32>, max_retries: Option<u32>) -> RetryPolicy {
        let mut policy = RetryPolicy::LLM_DEFAULTS;
        if let Some(t) = timeout_secs {
            policy.timeout_secs = t;
        }
        if let Some(r) = max_retries {
            policy.max_retries = r;
        }
        policy
    }

    /// Resolve tool retry policy from per-request hints, fallback to defaults.
    fn resolve_tool_retry(timeout_secs: Option<u32>, max_retries: Option<u32>) -> RetryPolicy {
        let mut policy = RetryPolicy::TOOL_DEFAULTS;
        if let Some(t) = timeout_secs {
            policy.timeout_secs = t;
        }
        if let Some(r) = max_retries {
            policy.max_retries = r;
        }
        policy
    }

    /// Compute LLM call deadline using system defaults.
    pub(super) fn llm_deadline(&self) -> DateTime<Utc> {
        Utc::now() + chrono::Duration::seconds(i64::from(RetryPolicy::LLM_DEFAULTS.timeout_secs))
    }

    /// Compute tool call deadline using system defaults.
    pub(super) fn tool_deadline(&self) -> DateTime<Utc> {
        Utc::now() + chrono::Duration::seconds(i64::from(RetryPolicy::TOOL_DEFAULTS.timeout_secs))
    }


    pub fn label(&self) -> Option<String> {
        self.agent_name.clone()
    }

    /// Build a WorkerDispatch from session state.
    /// Returns None if no agent is configured.
    pub(super) fn worker_dispatch(
        &self,
        ctx: &SessionContext,
        req: &WorkerDecisionRequested,
        span: &SpanContext,
    ) -> Option<WorkerDispatch> {
        let name = self.agent_name.as_ref()?;
        Some(WorkerDispatch {
            session_id: self.session_id,
            decision_id: req.decision_id.clone(),
            trigger: req.trigger.clone(),
            worker_state: self.worker_state.clone(),
            stream: ctx.stream,
            agent_name: name.clone(),
            auth: ctx.auth.clone(),
            token_usage: self.token_usage.clone(),
            tool_call_statuses: self
                .tool_calls
                .iter()
                .map(|(id, tc)| (id.clone(), tc.status.clone()))
                .collect(),
            llm_call_statuses: self
                .llm_calls
                .iter()
                .map(|(id, c)| (id.clone(), c.status.clone()))
                .collect(),
            span: span.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// on_event helpers — I/O dispatch and worker decisions
// ---------------------------------------------------------------------------

impl SessionState {
    /// Reserve budget for an LLM call. Returns the failing command if a policy rejects.
    async fn reserve_budget(
        &self,
        call: &LlmCallState,
        ctx: &SessionContext,
        span: &SpanContext,
    ) -> Result<(), CommandPayload> {
        let auth = self.auth.as_ref().unwrap_or(&ctx.auth);
        let context = BudgetContext::for_llm_call(ctx.session_id, auth, &call.llm_client, &call.request.model);

        let mut breakdown = BTreeMap::new();
        if let Some(max_completion_tokens) = call.request.max_completion_tokens {
            breakdown.insert("completion_tokens".into(), max_completion_tokens);
        }

        match ctx.system
            .reserve_budget(&auth.tenant_id, ctx.session_id, &call.call_id, context, breakdown, span)
            .await
        {
            Ok(()) => Ok(()),
            Err(BudgetError::Denied(ref denial)) => match denial.status.strategy {
                ExhaustionStrategy::Reject => Err(CommandPayload::FailLlmCall {
                    call_id: call.call_id.clone(),
                    error: denial.to_string(),
                    retryable: false,
                    source: Some(serde_json::to_value(denial).unwrap()),
                }),
                ExhaustionStrategy::Interrupt => Err(CommandPayload::Interrupt {
                    interrupt_id: Uuid::new_v4().to_string(),
                    reason: denial.to_string(),
                    payload: serde_json::to_value(denial).unwrap(),
                }),
            },
        }
    }

    /// Handle an LLM call: resolve client, call API (streaming or not), return command.
    async fn handle_llm_call(
        &self,
        p: &LlmCallRequested,
        ctx: &SessionContext,
        span: &SpanContext,
    ) -> CommandPayload {
        let provider = ctx.system.llm_provider();

        let call = match self.llm_calls.get(&p.call_id) {
            Some(c) => c,
            None => {
                return CommandPayload::FailLlmCall {
                    call_id: p.call_id.clone(),
                    error: "LLM call not found in state".into(),
                    retryable: false,
                    source: None,
                };
            }
        };

        let client = match provider.resolve(&call.llm_client, &ctx.auth).await {
            Ok(c) => c,
            Err(e) => {
                return CommandPayload::FailLlmCall {
                    call_id: p.call_id.clone(),
                    error: e,
                    retryable: true,
                    source: None,
                };
            }
        };

        if let Err(cmd) = self.reserve_budget(call, ctx, span).await {
            return cmd;
        }

        let request = p.request.clone();

        let result = if p.stream {
            let (chunk_tx, mut chunk_rx) = tokio::sync::mpsc::unbounded_channel::<StreamDelta>();

            let call_id = p.call_id.clone();
            let session_id = ctx.session_id;
            let chunk_span = span.child("llm.stream");
            let mut chunk_index: u32 = 0;

            let (result, _) = tokio::join!(client.call_streaming(&request, chunk_tx), async {
                while let Some(delta) = chunk_rx.recv().await {
                    if let Some(text) = delta.text {
                        super::routing::notify_llm_chunk(
                            session_id,
                            call_id.clone(),
                            chunk_index,
                            text,
                            chunk_span.clone(),
                        );
                        chunk_index += 1;
                    }
                }
            });
            result
        } else {
            client.call(&request).await
        };

        match result {
            Ok(response) => CommandPayload::CompleteLlmCall {
                call_id: p.call_id.clone(),
                response,
            },
            Err(e) => CommandPayload::FailLlmCall {
                call_id: p.call_id.clone(),
                error: e.message,
                retryable: e.retryable,
                source: e.source,
            },
        }
    }

    /// Build a ToolResult from a resolved ToolCallState.
    fn tool_result(tc: &ToolCallState) -> ToolResult {
        ToolResult {
            tool_call_id: tc.tool_call_id.clone(),
            name: tc.name.clone(),
            content: tc.result.clone().unwrap_or_default(),
            is_error: tc.is_error,
        }
    }

    /// Detect whether an event should trigger a worker decision.
    fn detect_trigger(&self, event: &EventPayload) -> Option<DecisionTrigger> {
        match event {
            EventPayload::MessageUser(p) => Some(DecisionTrigger::UserMessage {
                stream: p.stream,
                message: p.message.clone(),
            }),
            EventPayload::LlmCallCompleted(p) => {
                let truncated = p.response.finish_reason() == Some("length");
                let message = Message {
                    role: Role::Assistant,
                    content: p.response.content(),
                    tool_calls: p.response.tool_calls(),
                    tool_call_id: None,
                    call_id: Some(p.call_id.clone()),
                    usage: p.response.usage().cloned(),
                };
                Some(DecisionTrigger::LlmCompleted {
                    call_id: p.call_id.clone(),
                    message,
                    truncated,
                })
            }
            EventPayload::LlmCallErrored(p) => {
                // Only consult worker when retries are exhausted.
                // Transient failures are retried by the runtime via wake scheduling.
                let exhausted = self
                    .llm_calls
                    .get(&p.call_id)
                    .is_some_and(|c| c.status == LlmCallStatus::Failed);
                if exhausted {
                    Some(DecisionTrigger::LlmFailed {
                        call_id: p.call_id.clone(),
                        error: p.error.clone(),
                    })
                } else {
                    None
                }
            }
            EventPayload::ToolCallErrored(p) => {
                // Only consult worker when retries are exhausted.
                let exhausted = self
                    .tool_calls
                    .get(&p.tool_call_id)
                    .is_some_and(|tc| tc.status == ToolCallStatus::Failed);
                if exhausted {
                    let tc = self.tool_calls.get(&p.tool_call_id)?;
                    Some(DecisionTrigger::ToolResolved {
                        result: Self::tool_result(tc),
                    })
                } else {
                    None
                }
            }
            EventPayload::ToolCallCompleted(p) => {
                let tc = self.tool_calls.get(&p.tool_call_id)?;
                Some(DecisionTrigger::ToolResolved {
                    result: Self::tool_result(tc),
                })
            }
            EventPayload::InterruptResumed(p) => Some(DecisionTrigger::InterruptResumed {
                interrupt_id: p.interrupt_id.clone(),
            }),
            _ => None,
        }
    }

    /// Dispatch a worker decision via the executor (fire-and-forget).
    fn dispatch_worker_decision(
        &self,
        req: &WorkerDecisionRequested,
        ctx: &SessionContext,
        span: &SpanContext,
    ) {
        if let Some(dispatch) = self.worker_dispatch(ctx, req, span) {
            ctx.system.worker_executor().dispatch_decision(dispatch);
        }
    }
}

// ---------------------------------------------------------------------------
// AggregateState impl — makes SessionState the aggregate directly
// ---------------------------------------------------------------------------

#[async_trait]
impl AggregateState for SessionState {
    type Event = EventPayload;
    type Command = CommandPayload;
    type Error = SessionError;
    type Context = SessionContext;
    type Derived = DerivedState;

    fn aggregate_type() -> &'static str {
        "session"
    }

    fn apply(&mut self, event: &Self::Event) {
        self.apply_core(event);
    }

    fn handle_command(
        &self,
        cmd: Self::Command,
        ctx: &Self::Context,
    ) -> Result<Vec<Emit<Self::Event>>, Self::Error> {
        self.handle(cmd, ctx)
    }

    async fn on_event(
        &self,
        event: &Self::Event,
        ctx: &Self::Context,
        span: &SpanContext,
    ) -> Option<Self::Command> {
        match event {
            EventPayload::LlmCallRequested(p) => {
                if self
                    .llm_calls
                    .get(&p.call_id)
                    .is_some_and(|c| c.status == LlmCallStatus::Pending)
                {
                    return Some(self.handle_llm_call(p, ctx, span).await);
                }
                None
            }
            EventPayload::ToolCallRequested(p) => {
                let tc = self.tool_calls.get(&p.tool_call_id);
                if let Some(tc) = tc.filter(|tc| tc.status == ToolCallStatus::Pending) {
                    match tc.handler {
                        ToolHandler::Worker => {
                            ctx.system.worker_executor().dispatch_tool_call(ToolCallDispatch {
                                session_id: self.session_id,
                                tool_call_id: p.tool_call_id.clone(),
                                name: p.name.clone(),
                                arguments: p.arguments.clone(),
                                context: p.context.clone(),
                                agent_name: self.agent_name.clone().unwrap_or_default(),
                                auth: ctx.auth.clone(),
                                span: span.clone(),
                            });
                        }
                        ToolHandler::SubAgent => {
                            let args: serde_json::Value =
                                serde_json::from_str(&p.arguments).unwrap_or_default();
                            let message = args
                                .get("message")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let child_session_id =
                                Uuid::new_v5(&self.session_id, p.tool_call_id.as_bytes());
                            ctx.system.spawn_sub_agent(
                                crate::runtime::types::SubAgentRequest {
                                    session_id: child_session_id,
                                    agent_name: p.name.clone(),
                                    message,
                                    auth: ctx.auth.clone(),
                                    delivery: CompletionDelivery {
                                        parent_session_id: self.session_id,
                                        tool_call_id: p.tool_call_id.clone(),
                                        tool_name: p.name.clone(),
                                        span: span.child("sub_agent.delivery"),
                                    },
                                    span: span.child("sub_agent.spawn"),
                                    stream: ctx.stream,
                                },
                            ).await;
                        }
                        ToolHandler::Client => {}
                    }
                }
                None
            }
            EventPayload::SessionDone(_) => {
                if let Some(ref delivery) = self.on_done {
                    let result = serde_json::to_string(&self.artifacts).unwrap_or_default();
                    ctx.system.deliver(
                        delivery.parent_session_id,
                        CommandPayload::CompleteToolCall {
                            tool_call_id: delivery.tool_call_id.clone(),
                            name: delivery.tool_name.clone(),
                            result,
                            worker_state: None,
                        },
                        span.child("session.done.deliver"),
                    ).await;
                }
                None
            }

            // --- Worker dispatch (fire-and-forget via transport) ---
            EventPayload::WorkerDecisionRequested(req) => {
                self.dispatch_worker_decision(req, ctx, span);
                None
            }

            // --- Everything else: detect triggers ---
            _ => self
                .detect_trigger(event)
                .map(|trigger| CommandPayload::TriggerWorkerDecision { trigger }),
        }
    }

    fn derived_state(&self) -> Self::Derived {
        self.derived_state()
    }

    fn wake_at(&self) -> Option<DateTime<Utc>> {
        self.wake_at()
    }

    fn status(&self) -> AggregateStatus {
        self.aggregate_status()
    }

    fn label(&self) -> Option<String> {
        self.label()
    }
}
