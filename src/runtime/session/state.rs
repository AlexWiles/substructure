use std::cmp::min;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use super::types::{Artifact, CompletionDelivery, ToolCallMeta, ToolCallRequested, ToolHandler};
use crate::runtime::aggregate::{AggregateState, AggregateStatus, Emit};
use crate::runtime::budget::{self, BudgetContext, BudgetError};
use crate::runtime::config::{
    AgentConfig, ClientIdentity, ExhaustionStrategy, LlmRequestParams, RetryPolicy,
};
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
    pub messages: Vec<Message>,
    pub token_usage: BTreeMap<String, u64>,
    pub strategy_state: Value,

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
            messages: Vec::new(),
            token_usage: BTreeMap::new(),
            strategy_state: Value::Null,
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
                self.agent = Some(payload.agent.clone());
                self.auth = Some(payload.auth.clone());
                self.on_done = payload.on_done.clone();
            }
            EventPayload::MessageUser(payload) => {
                self.messages.push(payload.message.clone());
                self.status = SessionStatus::Idle;
            }
            EventPayload::MessageAssistant(payload) => {
                self.messages.push(payload.message.clone());
            }
            EventPayload::MessageTool(payload) => {
                self.messages.push(payload.message.clone());
            }
            EventPayload::LlmCallRequested(payload) => {
                self.status = SessionStatus::Active;
                if let Some(existing) = self.llm_calls.get_mut(&payload.call_id) {
                    // Re-request (retry): preserve retry count, reset call state
                    existing.status = LlmCallStatus::Pending;
                    existing.deadline = payload.deadline;
                    existing.retry.next_at = None;
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
                }
                if self.pending_tool_results() == 0 {
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
                    }
                }
                // Stay Active while any tool call is pending or retrying
                let has_inflight = self.tool_calls.values().any(|tc| {
                    tc.status == ToolCallStatus::Pending
                        || tc.status == ToolCallStatus::RetryScheduled
                });
                if !has_inflight && self.pending_tool_results() == 0 {
                    self.status = SessionStatus::Idle;
                }
            }
            EventPayload::SessionInterrupted(payload) => {
                self.status = SessionStatus::Interrupted {
                    interrupt_id: payload.interrupt_id.clone(),
                };
            }
            EventPayload::InterruptResumed(_) => {
                self.status = SessionStatus::Active;
            }
            EventPayload::StrategyStateChanged(payload) => {
                self.strategy_state = payload.state.clone();
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

    /// Derive pending tool result count from the message history.
    pub fn pending_tool_results(&self) -> usize {
        // Walk backwards: count Tool messages, then find the Assistant message.
        let mut tool_msgs = 0;
        for msg in self.messages.iter().rev() {
            match msg.role {
                Role::Tool => tool_msgs += 1,
                Role::Assistant => {
                    return msg.tool_calls.len().saturating_sub(tool_msgs);
                }
                _ => return 0,
            }
        }
        0
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

    // -----------------------------------------------------------------------
    // LLM request building
    // -----------------------------------------------------------------------

    pub fn build_llm_request(
        &self,
        tools: Option<Vec<LlmTool>>,
        overrides: Option<&LlmRequestParams>,
    ) -> Option<LlmRequest> {
        let agent = self.agent.as_ref()?;

        let mut messages = vec![Message {
            role: Role::System,
            content: Some(agent.system_prompt.clone()),
            tool_calls: vec![],
            tool_call_id: None,
            call_id: None,
            usage: None,
        }];
        messages.extend(self.messages.iter().cloned());

        Some(self.make_llm_request(messages, tools, overrides))
    }

    /// Build an LLM request using a custom context (e.g. compacted history).
    /// The context messages are used as-is (no system prompt prepended).
    pub fn build_llm_request_with_context(
        &self,
        context: &[Message],
        tools: Option<Vec<LlmTool>>,
        overrides: Option<&LlmRequestParams>,
    ) -> Option<LlmRequest> {
        self.agent.as_ref()?;
        let messages: Vec<Message> = context.to_vec();
        Some(self.make_llm_request(messages, tools, overrides))
    }

    fn make_llm_request(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<LlmTool>>,
        overrides: Option<&LlmRequestParams>,
    ) -> LlmRequest {
        let agent = self.agent.as_ref().expect("agent must be set");
        let model = agent.llm.model.clone();
        let temperature = overrides
            .and_then(|o| o.temperature)
            .or(agent.llm.temperature);
        let max_completion_tokens = overrides
            .and_then(|o| o.max_completion_tokens)
            .or(agent.llm.max_completion_tokens);

        LlmRequest {
            model,
            messages,
            tools,
            temperature,
            max_completion_tokens,
        }
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
            Err(BudgetError::Denied {
                strategy: ExhaustionStrategy::Reject,
                ref policy_name,
                current,
                limit,
            }) => Err(CommandPayload::FailLlmCall {
                call_id: call_id.to_string(),
                error: format!("budget denied by {policy_name}: {current}/{limit}"),
                retryable: false,
                source: None,
            }),
            Err(BudgetError::Denied {
                strategy: ExhaustionStrategy::Interrupt,
                ..
            }) => {
                // Interrupt strategy: allow the call to proceed. The session
                // will be interrupted after it completes.
                Ok(())
            }
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

        // Client tool — no-op, session is Idle and waits for external result
        None
    }

    /// Inlined strategy decisions (replaces DefaultStrategy).
    fn strategy_decision(
        &self,
        event: &EventPayload,
        ctx: &SessionContext,
    ) -> Option<CommandPayload> {
        match event {
            EventPayload::MessageUser(p) => {
                let stream = p.stream;
                let request = self.build_llm_request(ctx.all_tools.clone(), None)?;
                Some(CommandPayload::RequestLlmCall {
                    call_id: new_call_id(),
                    request,
                    stream,
                    deadline: self.llm_deadline(),
                })
            }
            EventPayload::LlmCallCompleted(p) => {
                let (content, tool_calls, _usage) = p.response.as_parts();
                if tool_calls.is_empty() {
                    // No tool calls → done
                    let artifacts = match content {
                        Some(ref text) if !text.is_empty() => vec![Artifact {
                            name: None,
                            description: None,
                            parts: vec![super::types::Part::Text { text: text.clone() }],
                        }],
                        _ => vec![],
                    };
                    Some(CommandPayload::MarkDone { artifacts })
                } else {
                    // Has tool calls → execute them (they're already emitted by command_handler)
                    None
                }
            }
            EventPayload::LlmCallErrored(p) => {
                // Only act when retries are exhausted (status == Failed)
                let call = self.llm_calls.get(&p.call_id)?;
                if call.status != LlmCallStatus::Failed {
                    return None;
                }
                Some(CommandPayload::MarkDone {
                    artifacts: vec![Artifact {
                        name: None,
                        description: None,
                        parts: vec![super::types::Part::Text {
                            text: format!("Error: {}", p.error),
                        }],
                    }],
                })
            }
            EventPayload::MessageTool(_) => {
                // Wait until all tool calls are done (including retries in flight)
                if self.tool_calls.values().any(|tc| {
                    tc.status == ToolCallStatus::Pending
                        || tc.status == ToolCallStatus::RetryScheduled
                }) {
                    return None;
                }
                let request = self.build_llm_request(ctx.all_tools.clone(), None)?;
                Some(CommandPayload::RequestLlmCall {
                    call_id: new_call_id(),
                    request,
                    stream: ctx.stream,
                    deadline: self.llm_deadline(),
                })
            }
            EventPayload::InterruptResumed(_) => {
                let request = self.build_llm_request(ctx.all_tools.clone(), None)?;
                Some(CommandPayload::RequestLlmCall {
                    call_id: new_call_id(),
                    request,
                    stream: ctx.stream,
                    deadline: self.llm_deadline(),
                })
            }
            _ => None,
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
        // --- Mechanical I/O dispatch ---
        match event {
            EventPayload::LlmCallRequested(p) => {
                if self
                    .llm_calls
                    .get(&p.call_id)
                    .is_some_and(|c| c.status == LlmCallStatus::Pending)
                {
                    return Some(self.handle_llm_call(p, ctx, span).await);
                }
            }
            EventPayload::ToolCallRequested(p) => {
                if self.tool_calls.get(&p.tool_call_id).is_some_and(|tc| {
                    tc.status == ToolCallStatus::Pending && tc.handler == ToolHandler::Runtime
                }) {
                    return self.handle_tool_call(p, ctx, span).await;
                }
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
                return None;
            }
            _ => {}
        }

        // --- Inlined strategy decisions ---
        self.strategy_decision(event, ctx)
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
