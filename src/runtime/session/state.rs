use std::cmp::min;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::strategy::{DecisionTrigger, Strategy, StrategyCtx};
use super::types::{Artifact, CompletionDelivery, ToolCallMeta, ToolCallRequested, ToolHandler};
use crate::runtime::aggregate::{AggregateState, AggregateStatus, Emit};
use crate::runtime::budget::{self, BudgetContext, BudgetError};
use crate::runtime::config::{AgentConfig, ClientIdentity, ExhaustionStrategy, RetryPolicy};
use crate::runtime::event::EventPayload;
use crate::runtime::llm::{LlmCallRequested, LlmRequest, LlmResponse, LlmTool};
use crate::runtime::mcp::{Content, McpClient};
use crate::runtime::message::{Message, Role};
use crate::runtime::span::SpanContext;
use async_trait::async_trait;

use super::command::{truncate_tool_result, CommandPayload, SessionError};
use crate::runtime::defaults;

// ---------------------------------------------------------------------------
// SessionContext — transient state passed through handle_command/on_event
// ---------------------------------------------------------------------------

/// MCP server info associated with a tool name (transient, populated by runtime).
#[derive(Debug, Clone)]
pub struct McpToolEntry {
    pub server_name: String,
    pub server_version: String,
}

/// Callback for streaming LLM chunks to observers.
pub type NotifyChunkFn = Arc<dyn Fn(Uuid, String, u32, String, SpanContext) + Send + Sync>;
/// Callback for sending a command to a session (fire-and-forget).
pub type SendToSessionFn = Arc<dyn Fn(Uuid, CommandPayload, SpanContext) + Send + Sync>;
/// Callback for spawning a sub-agent.
pub type SpawnSubAgentFn = Arc<dyn Fn(SubAgentParams) + Send + Sync>;

/// Parameters for spawning a sub-agent.
pub struct SubAgentParams {
    pub session_id: Uuid,
    pub agent_name: String,
    pub message: String,
    pub auth: ClientIdentity,
    pub delivery: CompletionDelivery,
    pub span: SpanContext,
    pub stream: bool,
}

/// Transient context for command handling and event reactions — not persisted.
pub struct SessionContext {
    pub mcp_tools: HashMap<String, McpToolEntry>,
    /// All tools (MCP + sub-agents + client), injected into LLM requests.
    pub all_tools: Option<Vec<LlmTool>>,
    pub session_id: Uuid,
    pub auth: ClientIdentity,
    pub stream: bool,
    // Runtime resources for I/O in on_event
    pub llm_provider: Option<Arc<dyn LlmProviderTrait>>,
    pub mcp_clients: Vec<Arc<dyn McpClient>>,
    pub agents: HashMap<String, AgentConfig>,
    pub client_tools: Vec<LlmTool>,
    pub budget_actor: Option<BudgetActorRef>,
    // Callbacks for side-effects
    pub notify_chunk: Option<NotifyChunkFn>,
    pub send_to_session: Option<SendToSessionFn>,
    pub spawn_sub_agent: Option<SpawnSubAgentFn>,
    /// System-wide max tool result bytes (from SystemConfig).
    pub tool_result_max_bytes: Option<usize>,
    /// Resolved strategy implementation.
    pub strategy: Option<Arc<dyn Strategy>>,
}

impl Default for SessionContext {
    fn default() -> Self {
        Self {
            mcp_tools: HashMap::new(),
            all_tools: None,
            session_id: Uuid::nil(),
            auth: ClientIdentity {
                tenant_id: String::new(),
                sub: None,
                attrs: Default::default(),
            },
            stream: false,
            llm_provider: None,
            mcp_clients: Vec::new(),
            agents: HashMap::new(),
            client_tools: Vec::new(),
            budget_actor: None,
            notify_chunk: None,
            send_to_session: None,
            spawn_sub_agent: None,
            tool_result_max_bytes: None,
            strategy: None,
        }
    }
}

use crate::runtime::llm::{LlmProviderTrait, StreamDelta};

/// Opaque handle to the budget actor — avoids leaking ractor types into the domain.
/// Reserve method is implemented in `budget/mod.rs` where the concrete types are known.
pub struct BudgetActorRef {
    pub(crate) inner: Box<dyn std::any::Any + Send + Sync>,
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
    /// Work in flight: LLM calls, tool calls, strategy decisions.
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<ToolCallMeta>,
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

impl ToolCallState {
    pub fn child_session_id(&self) -> Option<Uuid> {
        match &self.meta {
            Some(ToolCallMeta::SubAgent {
                child_session_id, ..
            }) => Some(*child_session_id),
            _ => None,
        }
    }
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
    pub agent: Option<AgentConfig>,
    pub auth: Option<ClientIdentity>,
    pub token_usage: BTreeMap<String, u64>,

    /// Opaque strategy state — owned and managed exclusively by the strategy.
    #[serde(default)]
    pub strategy_state: serde_json::Value,

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
            agent: None,
            auth: None,
            token_usage: BTreeMap::new(),
            strategy_state: serde_json::Value::Null,
            on_done: None,
            artifacts: vec![],
            llm_calls: HashMap::new(),
            tool_calls: HashMap::new(),
        }
    }

    /// Apply a single event payload to the session state.
    pub fn apply_core(&mut self, payload: &EventPayload, _ctx: &SessionContext) {
        match payload {
            EventPayload::SessionCreated(payload) => {
                self.status = SessionStatus::Idle;
                self.agent = Some(payload.agent.clone());
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
                            retry_policy: self.resolve_llm_retry(),
                            request: payload.request.clone(),
                            stream: payload.stream,
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
                            retry_policy: self.resolve_tool_retry(payload.meta.as_ref()),
                            deadline: payload.deadline,
                            meta: payload.meta.clone(),
                            handler: payload.handler.clone(),
                            arguments: payload.arguments.clone(),
                            result: None,
                            is_error: false,
                        },
                    );
                }
                // Active only if there's runtime-handled work to do;
                // Idle if all pending calls are client tools.
                let has_runtime_work = self.tool_calls.values().any(|tc| {
                    tc.status == ToolCallStatus::Pending && tc.handler == ToolHandler::Runtime
                });
                self.status = if has_runtime_work {
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
            EventPayload::StrategyStateChanged(_) => {
                // Legacy: state updates now flow through StrategyDecisionCompleted.
            }
            EventPayload::StrategyDecisionRequested(_) => {
                // Keep Active while decision is pending
                self.status = SessionStatus::Active;
            }
            EventPayload::StrategyDecisionCompleted(p) => {
                // Persist the strategy's updated opaque state.
                self.strategy_state = p.state.clone();
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

    /// Extract conversation messages from strategy state.
    pub fn messages(&self) -> Vec<Message> {
        self.strategy_state
            .get("messages")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| serde_json::from_value::<Message>(v.clone()).ok())
                    .collect()
            })
            .unwrap_or_default()
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

    /// Compute tool call metadata based on the tool name.
    pub(super) fn tool_call_meta(
        &self,
        name: &str,
        tool_call_id: &str,
        mcp_tools: &HashMap<String, McpToolEntry>,
    ) -> Option<ToolCallMeta> {
        // Check sub-agents
        if let Some(agent_name) = self
            .agent
            .as_ref()
            .and_then(|a| a.sub_agents.iter().find(|s| s.as_str() == name))
        {
            return Some(ToolCallMeta::SubAgent {
                child_session_id: Uuid::new_v5(&self.session_id, tool_call_id.as_bytes()),
                agent_name: agent_name.clone(),
            });
        }
        // Check MCP tools
        if let Some(entry) = mcp_tools.get(name) {
            return Some(ToolCallMeta::Mcp {
                server_name: entry.server_name.clone(),
                server_version: entry.server_version.clone(),
            });
        }
        None
    }

    /// Resolve LLM retry policy: agent.llm.retry → defaults.
    pub(super) fn resolve_llm_retry(&self) -> RetryPolicy {
        self.agent
            .as_ref()
            .map(|a| a.llm.retry.resolve(&RetryPolicy::LLM_DEFAULTS))
            .unwrap_or(RetryPolicy::LLM_DEFAULTS)
    }

    /// Resolve tool retry policy: MCP server → defaults.
    pub(super) fn resolve_tool_retry(&self, meta: Option<&ToolCallMeta>) -> RetryPolicy {
        let server_name = match meta {
            Some(ToolCallMeta::Mcp { server_name, .. }) => Some(server_name.as_str()),
            _ => None,
        };
        let mcp_retry = server_name.and_then(|sn| {
            self.agent
                .as_ref()
                .and_then(|a| a.mcp_servers.iter().find(|s| s.name == sn))
                .map(|s| &s.retry)
        });
        mcp_retry
            .map(|r| r.resolve(&RetryPolicy::TOOL_DEFAULTS))
            .unwrap_or(RetryPolicy::TOOL_DEFAULTS)
    }

    /// Compute LLM call deadline from agent config.
    pub(super) fn llm_deadline(&self) -> DateTime<Utc> {
        let policy = self.resolve_llm_retry();
        Utc::now() + chrono::Duration::seconds(i64::from(policy.timeout_secs))
    }

    /// Compute tool call deadline, resolved per MCP server.
    pub(super) fn tool_deadline(&self, meta: Option<&ToolCallMeta>) -> DateTime<Utc> {
        let policy = self.resolve_tool_retry(meta);
        Utc::now() + chrono::Duration::seconds(i64::from(policy.timeout_secs))
    }

    /// Resolve the max tool result size for a given tool.
    ///
    /// Resolution: MCP server → agent → system → hardcoded default.
    /// `Some(0)` at any level means "no limit" (returns `None`).
    pub(super) fn tool_result_max_bytes(
        &self,
        tool_name: &str,
        ctx: &SessionContext,
    ) -> Option<usize> {
        // 1. Per-MCP server
        if let Some(entry) = ctx.mcp_tools.get(tool_name) {
            if let Some(limit) = self
                .agent
                .as_ref()
                .and_then(|a| a.mcp_servers.iter().find(|s| s.name == entry.server_name))
                .and_then(|s| s.tool_result_max_bytes)
            {
                return if limit == 0 { None } else { Some(limit) };
            }
        }
        // 2. Per-agent
        if let Some(limit) = self.agent.as_ref().and_then(|a| a.tool_result_max_bytes) {
            return if limit == 0 { None } else { Some(limit) };
        }
        // 3. System-wide
        if let Some(limit) = ctx.tool_result_max_bytes {
            return if limit == 0 { None } else { Some(limit) };
        }
        // 4. Hardcoded default
        Some(defaults::TOOL_RESULT_MAX_BYTES)
    }

    pub fn label(&self) -> Option<String> {
        self.agent.as_ref().map(|a| a.name.clone())
    }

    /// Build a StrategyCtx from the current state and context.
    pub(super) fn strategy_ctx<'a>(&'a self, ctx: &'a SessionContext) -> Option<StrategyCtx<'a>> {
        let agent = self.agent.as_ref()?;
        Some(StrategyCtx {
            session_id: self.session_id,
            stream: ctx.stream,
            agent,
            all_tools: &ctx.all_tools,
            token_usage: &self.token_usage,
            has_inflight_tools: self.has_inflight_tools(),
            has_pending_llm: self.has_pending_llm(),
            failed_llm_calls: self
                .llm_calls
                .values()
                .filter(|c| c.status == LlmCallStatus::Failed)
                .map(|c| c.call_id.as_str())
                .collect(),
        })
    }
}

// ---------------------------------------------------------------------------
// on_event helpers — I/O dispatch and strategy decisions
// ---------------------------------------------------------------------------

impl SessionState {
    /// Reserve budget for an LLM call. Returns the failing command if a policy rejects.
    async fn reserve_budget(
        &self,
        call_id: &str,
        ctx: &SessionContext,
        span: &SpanContext,
    ) -> Result<(), CommandPayload> {
        let Some(ref budget) = ctx.budget_actor else {
            return Ok(());
        };

        let agent = match self.agent.as_ref() {
            Some(a) => a,
            None => return Ok(()),
        };

        let auth = self.auth.as_ref().unwrap_or(&ctx.auth);
        let client_id = &agent.llm.client;
        let model = &agent.llm.model;

        let context = BudgetContext::for_llm_call(ctx.session_id, auth, client_id, model);

        let mut breakdown = BTreeMap::new();
        if let Some(max_completion_tokens) = agent.llm.max_completion_tokens {
            breakdown.insert("completion_tokens".into(), max_completion_tokens);
        }

        match budget
            .reserve(ctx.session_id, call_id, context, breakdown, span)
            .await
        {
            Ok(()) => Ok(()),
            Err(BudgetError::Denied(ref denial)) => match denial.status.strategy {
                ExhaustionStrategy::Reject => Err(CommandPayload::FailLlmCall {
                    call_id: call_id.to_string(),
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
        let provider = match &ctx.llm_provider {
            Some(p) => p,
            None => {
                return CommandPayload::FailLlmCall {
                    call_id: p.call_id.clone(),
                    error: "no LLM provider configured".into(),
                    retryable: false,
                    source: None,
                };
            }
        };

        let client_id = self
            .agent
            .as_ref()
            .map(|a| a.llm.client.clone())
            .unwrap_or_default();

        let client = match provider.resolve(&client_id, &ctx.auth).await {
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

        if let Err(cmd) = self.reserve_budget(&p.call_id, ctx, span).await {
            return cmd;
        }

        // Inject tools into the request
        let request = p.request.clone().with_tools(ctx.all_tools.clone());

        let result = if p.stream {
            let (chunk_tx, mut chunk_rx) = tokio::sync::mpsc::unbounded_channel::<StreamDelta>();

            let call_id = p.call_id.clone();
            let session_id = ctx.session_id;
            let notify = ctx.notify_chunk.clone();
            let chunk_span = span.child("llm.stream");
            let mut chunk_index: u32 = 0;

            let (result, _) = tokio::join!(client.call_streaming(&request, chunk_tx), async {
                while let Some(delta) = chunk_rx.recv().await {
                    if let Some(text) = delta.text {
                        if let Some(ref notify) = notify {
                            notify(
                                session_id,
                                call_id.clone(),
                                chunk_index,
                                text,
                                chunk_span.clone(),
                            );
                            chunk_index += 1;
                        }
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

    /// Handle a tool call: sub-agent spawn, MCP call, or client tool (no-op).
    async fn handle_tool_call(
        &self,
        p: &ToolCallRequested,
        ctx: &SessionContext,
        span: &SpanContext,
    ) -> Option<CommandPayload> {
        // Sub-agent tool call
        if let Some(child_session_id) = self
            .tool_calls
            .get(&p.tool_call_id)
            .and_then(|tc| tc.child_session_id())
        {
            let args: serde_json::Value = serde_json::from_str(&p.arguments).unwrap_or_default();
            let message = args
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            if let Some(ref spawn) = ctx.spawn_sub_agent {
                spawn(SubAgentParams {
                    session_id: child_session_id,
                    agent_name: p.name.clone(),
                    message,
                    auth: ctx.auth.clone(),
                    delivery: CompletionDelivery {
                        parent_session_id: ctx.session_id,
                        tool_call_id: p.tool_call_id.clone(),
                        tool_name: p.name.clone(),
                        span: span.child("sub_agent.delivery"),
                    },
                    span: span.child("sub_agent.spawn"),
                    stream: ctx.stream,
                });
            }
            return None; // Sub-agent runs async, result arrives via CompleteToolCall command
        }

        // MCP tool call
        let mcp = ctx
            .mcp_clients
            .iter()
            .find(|c| c.tools().iter().any(|t| t.name == p.name))
            .cloned();

        if let Some(mcp) = mcp {
            let args: serde_json::Value = serde_json::from_str(&p.arguments).unwrap_or_default();

            match mcp.call_tool(&p.name, args).await {
                Ok(result) => {
                    let text = result
                        .content
                        .iter()
                        .filter_map(|c| match c {
                            Content::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    let max = self.tool_result_max_bytes(&p.name, ctx);
                    let text = truncate_tool_result(text, max);
                    if result.is_error {
                        return Some(CommandPayload::FailToolCall {
                            tool_call_id: p.tool_call_id.clone(),
                            name: p.name.clone(),
                            error: text,
                        });
                    } else {
                        return Some(CommandPayload::CompleteToolCall {
                            tool_call_id: p.tool_call_id.clone(),
                            name: p.name.clone(),
                            result: text,
                        });
                    }
                }
                Err(e) => {
                    return Some(CommandPayload::FailToolCall {
                        tool_call_id: p.tool_call_id.clone(),
                        name: p.name.clone(),
                        error: e.to_string(),
                    });
                }
            }
        }

        // Unknown tool — fail immediately so the LLM gets feedback
        Some(CommandPayload::FailToolCall {
            tool_call_id: p.tool_call_id.clone(),
            name: p.name.clone(),
            error: format!("unknown tool: {}", p.name),
        })
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

    /// Detect whether an event should trigger a strategy decision.
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
                // Only consult strategy when retries are exhausted.
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
                // Only consult strategy when retries are exhausted.
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

    /// Execute the strategy and return a SubmitStrategyDecision command.
    fn execute_strategy_decision(
        &self,
        req: &super::strategy::StrategyDecisionRequested,
        ctx: &SessionContext,
    ) -> Option<CommandPayload> {
        let strategy = ctx.strategy.as_ref()?;
        let strategy_ctx = self.strategy_ctx(ctx)?;
        let decision = strategy.decide(&req.trigger, &self.strategy_state, &strategy_ctx);
        Some(CommandPayload::SubmitStrategyDecision {
            decision_id: req.decision_id.clone(),
            actions: decision.actions,
            state: decision.state,
        })
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

    fn apply(&mut self, event: &Self::Event, ctx: &Self::Context) {
        self.apply_core(event, ctx);
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
                if self.tool_calls.get(&p.tool_call_id).is_some_and(|tc| {
                    tc.status == ToolCallStatus::Pending && tc.handler == ToolHandler::Runtime
                }) {
                    return self.handle_tool_call(p, ctx, span).await;
                }
                None
            }
            EventPayload::SessionDone(_) => {
                if let Some(ref delivery) = self.on_done {
                    let result = serde_json::to_string(&self.artifacts).unwrap_or_default();
                    if let Some(ref send) = ctx.send_to_session {
                        send(
                            delivery.parent_session_id,
                            CommandPayload::CompleteToolCall {
                                tool_call_id: delivery.tool_call_id.clone(),
                                name: delivery.tool_name.clone(),
                                result,
                            },
                            span.child("session.done.deliver"),
                        );
                    }
                }
                None
            }

            // --- Strategy executor (returns early) ---
            EventPayload::StrategyDecisionRequested(req) => {
                self.execute_strategy_decision(req, ctx)
            }

            // --- Everything else: detect triggers ---
            _ => self
                .detect_trigger(event)
                .map(|trigger| CommandPayload::TriggerStrategyDecision { trigger }),
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
