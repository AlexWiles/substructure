use std::cmp::min;
use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use async_trait::async_trait;
use crate::domain::aggregate::{AggregateState, AggregateStatus};
use crate::domain::event::{
    AgentConfig, Artifact, ClientIdentity, CompletionDelivery, EventPayload, LlmCallRequested,
    LlmRequest, LlmResponse, Message, Role, ToolCallMeta, ToolCallRequested, ToolHandler,
};
use crate::domain::openai;

use super::command_handler::{CommandPayload, SessionError};

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
pub type NotifyChunkFn = Arc<dyn Fn(Uuid, String, u32, String, crate::domain::span::SpanContext) + Send + Sync>;
/// Callback for sending a command to a session (fire-and-forget).
pub type SendToSessionFn = Arc<dyn Fn(Uuid, CommandPayload, crate::domain::event::SpanContext) + Send + Sync>;
/// Callback for spawning a sub-agent.
pub type SpawnSubAgentFn = Arc<
    dyn Fn(SubAgentParams) + Send + Sync,
>;

/// Parameters for spawning a sub-agent.
pub struct SubAgentParams {
    pub session_id: Uuid,
    pub agent_name: String,
    pub message: String,
    pub auth: ClientIdentity,
    pub delivery: CompletionDelivery,
    pub span: crate::domain::event::SpanContext,
    pub token_budget: Option<u64>,
    pub stream: bool,
}

/// Transient context for command handling and event reactions — not persisted.
pub struct SessionContext {
    pub mcp_tools: HashMap<String, McpToolEntry>,
    /// All tools (MCP + sub-agents + client), injected into LLM requests.
    pub all_tools: Option<Vec<openai::Tool>>,
    pub session_id: Uuid,
    pub auth: ClientIdentity,
    pub stream: bool,
    // Runtime resources for I/O in on_event
    pub llm_provider: Option<Arc<dyn LlmClientTrait>>,
    pub mcp_clients: Vec<Arc<dyn McpClientTrait>>,
    pub agents: HashMap<String, AgentConfig>,
    pub client_tools: Vec<openai::Tool>,
    pub budget_actor: Option<BudgetActorRef>,
    // Callbacks for side-effects
    pub notify_chunk: Option<NotifyChunkFn>,
    pub send_to_session: Option<SendToSessionFn>,
    pub spawn_sub_agent: Option<SpawnSubAgentFn>,
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
        }
    }
}

// Trait aliases for runtime types used by SessionContext.
// These avoid pulling runtime crate types directly into the domain.
// The runtime module provides the concrete implementations.

/// Trait for LLM client providers (resolved by the runtime).
pub trait LlmClientTrait: Send + Sync {
    fn resolve<'a>(
        &'a self,
        client_id: &'a str,
        auth: &'a ClientIdentity,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Arc<dyn LlmCallable>, String>> + Send + 'a>>;
}

/// Trait for calling an LLM (single call or streaming).
pub trait LlmCallable: Send + Sync {
    fn call<'a>(
        &'a self,
        request: &'a LlmRequest,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<LlmResponse, LlmCallError>> + Send + 'a>>;

    fn call_streaming<'a>(
        &'a self,
        request: &'a LlmRequest,
        tx: tokio::sync::mpsc::UnboundedSender<StreamDelta>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<LlmResponse, LlmCallError>> + Send + 'a>>;
}

#[derive(Debug, Clone)]
pub struct LlmCallError {
    pub message: String,
    pub retryable: bool,
    pub source: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct StreamDelta {
    pub text: Option<String>,
}

/// Trait for MCP tool clients.
pub trait McpClientTrait: Send + Sync {
    fn server_info(&self) -> McpServerInfo;
    fn tools(&self) -> Vec<McpToolDefinition>;
    fn call_tool<'a>(
        &'a self,
        name: &'a str,
        arguments: serde_json::Value,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<McpToolResult, String>> + Send + 'a>>;
}

#[derive(Debug, Clone)]
pub struct McpServerInfo {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone)]
pub struct McpToolDefinition {
    pub name: String,
}

impl McpToolDefinition {
    pub fn to_openai_tool(&self) -> openai::Tool {
        openai::Tool {
            tool_type: "function".to_string(),
            function: openai::ToolFunction {
                name: self.name.clone(),
                description: String::new(),
                parameters: serde_json::json!({}),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct McpToolResult {
    pub content: Vec<McpToolContent>,
    pub is_error: bool,
}

#[derive(Debug, Clone)]
pub enum McpToolContent {
    Text { text: String },
    Other,
}

/// Opaque handle to the budget actor — avoids leaking ractor types into the domain.
/// Opaque handle to the budget actor — avoids leaking ractor types into the domain.
#[allow(dead_code)]
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
// Token tracking
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    pub cached_tokens: u64,
    pub cache_write_tokens: u64,
    pub audio_tokens: u64,
    pub video_tokens: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: u64,
    pub audio_tokens: u64,
    pub accepted_prediction_tokens: u64,
    pub rejected_prediction_tokens: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub prompt_tokens_details: PromptTokensDetails,
    pub completion_tokens_details: CompletionTokensDetails,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenBudget {
    pub limit: Option<u64>,
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
    pub request: LlmRequest,
    pub status: LlmCallStatus,
    pub response: Option<LlmResponse>,
    pub retry: RetryState,
    pub deadline: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// Tool call lifecycle
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ToolCallStatus {
    Pending,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallState {
    pub tool_call_id: String,
    pub name: String,
    pub status: ToolCallStatus,
    pub result: Option<ToolCallResult>,
    pub retry: RetryState,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResult {
    pub content: String,
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
// AgentState — the reducer state for session aggregates
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub session_id: Uuid,
    pub status: SessionStatus,
    pub agent: Option<AgentConfig>,
    pub auth: Option<ClientIdentity>,
    pub messages: Vec<Message>,
    pub token_usage: TokenUsage,
    pub token_budget: TokenBudget,
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

impl AgentState {
    pub fn new(session_id: Uuid) -> Self {
        AgentState {
            session_id,
            status: SessionStatus::Done,
            agent: None,
            auth: None,
            messages: Vec::new(),
            token_usage: TokenUsage::default(),
            token_budget: TokenBudget { limit: None },
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
                self.token_budget.limit = payload.agent.token_budget;
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
                    existing.request = payload.request.clone();
                    existing.response = None;
                    existing.deadline = payload.deadline;
                    existing.retry.next_at = None;
                } else {
                    // New call
                    self.llm_calls.insert(
                        payload.call_id.clone(),
                        LlmCallState {
                            call_id: payload.call_id.clone(),
                            request: payload.request.clone(),
                            status: LlmCallStatus::Pending,
                            response: None,
                            retry: RetryState::default(),
                            deadline: payload.deadline,
                        },
                    );
                }
            }
            EventPayload::LlmCallCompleted(payload) => {
                self.track_usage(&payload.response);
                if let Some(call) = self.llm_calls.get_mut(&payload.call_id) {
                    call.status = LlmCallStatus::Completed;
                    call.response = Some(payload.response.clone());
                }
                self.status = SessionStatus::Idle;
            }
            EventPayload::LlmCallErrored(payload) => {
                if let Some(call) = self.llm_calls.get_mut(&payload.call_id) {
                    call.retry.attempts += 1;
                    let max_retries = self
                        .agent
                        .as_ref()
                        .map(|a| a.retry.max_retries)
                        .unwrap_or(3);
                    if payload.retryable && call.retry.attempts < max_retries {
                        call.status = LlmCallStatus::RetryScheduled;
                        let backoff_max = self
                            .agent
                            .as_ref()
                            .map(|a| a.retry.backoff_max_secs)
                            .unwrap_or(60);
                        let backoff = min(2u64.pow(call.retry.attempts), backoff_max);
                        call.retry.next_at =
                            Some(Utc::now() + chrono::Duration::seconds(backoff as i64));
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
                            result: None,
                            retry: RetryState::default(),
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
                    tc.result = Some(ToolCallResult {
                        content: payload.result.clone(),
                        is_error: false,
                    });
                }
                if self.pending_tool_results() == 0 {
                    self.status = SessionStatus::Idle;
                }
            }
            EventPayload::ToolCallErrored(payload) => {
                if let Some(tc) = self.tool_calls.get_mut(&payload.tool_call_id) {
                    tc.status = ToolCallStatus::Failed;
                    tc.result = Some(ToolCallResult {
                        content: payload.error.clone(),
                        is_error: true,
                    });
                    tc.retry.attempts += 1;
                }
                if self.pending_tool_results() == 0 {
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
            EventPayload::BudgetExceeded => {
                self.status = SessionStatus::Idle;
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

    pub fn is_over_budget(&self) -> bool {
        self.token_budget
            .limit
            .is_some_and(|limit| self.token_usage.total_tokens >= limit)
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

    /// Build the tool result batch from completed/errored tool calls
    /// belonging to the most recent assistant message.
    pub fn collect_tool_results(&self) -> Vec<ToolResult> {
        let tool_call_ids: Vec<&str> = self
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::Assistant && !m.tool_calls.is_empty())
            .map(|m| m.tool_calls.iter().map(|tc| tc.id.as_str()).collect())
            .unwrap_or_default();

        tool_call_ids
            .iter()
            .filter_map(|id| {
                let tc = self.tool_calls.get(*id)?;
                let result = tc.result.as_ref()?;
                Some(ToolResult {
                    tool_call_id: tc.tool_call_id.clone(),
                    name: tc.name.clone(),
                    content: result.content.clone(),
                    is_error: result.is_error,
                })
            })
            .collect()
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
        // Earliest of: pending call deadlines, failed call retry.next_at
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
        let retry_at = self
            .llm_calls
            .values()
            .filter(|c| c.status == LlmCallStatus::RetryScheduled)
            .filter_map(|c| c.retry.next_at);
        pending_llm.chain(pending_tool).chain(retry_at).min()
    }

    // -----------------------------------------------------------------------
    // LLM request building
    // -----------------------------------------------------------------------

    pub fn build_llm_request(&self, tools: Option<Vec<openai::Tool>>) -> Option<LlmRequest> {
        let agent = self.agent.as_ref()?;

        let mut messages = vec![openai::ChatMessage {
            role: openai::Role::System,
            content: Some(agent.system_prompt.clone()),
            tool_calls: None,
            tool_call_id: None,
        }];

        for msg in &self.messages {
            messages.push(to_openai_message(msg));
        }

        Some(self.make_llm_request(messages, tools))
    }

    /// Build an LLM request using a custom context (e.g. compacted history).
    /// The context messages are used as-is (no system prompt prepended).
    pub fn build_llm_request_with_context(
        &self,
        context: &[Message],
        tools: Option<Vec<openai::Tool>>,
    ) -> Option<LlmRequest> {
        self.agent.as_ref()?;
        let messages = context.iter().map(to_openai_message).collect();
        Some(self.make_llm_request(messages, tools))
    }

    fn make_llm_request(
        &self,
        messages: Vec<openai::ChatMessage>,
        tools: Option<Vec<openai::Tool>>,
    ) -> LlmRequest {
        let agent = self.agent.as_ref().expect("agent must be set");
        let model = agent
            .llm
            .params
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let temperature = agent
            .llm
            .params
            .get("temperature")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32);
        let max_tokens = agent
            .llm
            .params
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);

        LlmRequest::OpenAi(openai::ChatCompletionRequest {
            model,
            messages,
            tools,
            tool_choice: None,
            temperature,
            max_tokens,
        })
    }

    /// Compute the derived state envelope for the current session state.
    pub fn derived_state(&self) -> DerivedState {
        DerivedState {
            status: self.status.clone(),
            wake_at: self.wake_at(),
        }
    }

    fn track_usage(&mut self, response: &LlmResponse) {
        let usage = match response {
            LlmResponse::OpenAi(resp) => match &resp.usage {
                Some(u) => u,
                None => return,
            },
        };
        self.token_usage.prompt_tokens += usage.prompt_tokens as u64;
        self.token_usage.completion_tokens += usage.completion_tokens as u64;
        self.token_usage.total_tokens += usage.total_tokens as u64;
        if let Some(details) = &usage.prompt_tokens_details {
            self.token_usage.prompt_tokens_details.cached_tokens += details.cached_tokens as u64;
            self.token_usage.prompt_tokens_details.cache_write_tokens +=
                details.cache_write_tokens as u64;
            self.token_usage.prompt_tokens_details.audio_tokens += details.audio_tokens as u64;
            self.token_usage.prompt_tokens_details.video_tokens += details.video_tokens as u64;
        }
        if let Some(details) = &usage.completion_tokens_details {
            self.token_usage.completion_tokens_details.reasoning_tokens +=
                details.reasoning_tokens.unwrap_or(0) as u64;
            self.token_usage.completion_tokens_details.audio_tokens +=
                details.audio_tokens.unwrap_or(0) as u64;
            self.token_usage
                .completion_tokens_details
                .accepted_prediction_tokens +=
                details.accepted_prediction_tokens.unwrap_or(0) as u64;
            self.token_usage
                .completion_tokens_details
                .rejected_prediction_tokens +=
                details.rejected_prediction_tokens.unwrap_or(0) as u64;
        }
    }
}

impl AgentState {
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

    /// Compute LLM call deadline from agent config.
    pub(super) fn llm_deadline(&self) -> DateTime<Utc> {
        let timeout = self
            .agent
            .as_ref()
            .map(|a| a.retry.llm_timeout_secs)
            .unwrap_or(60);
        Utc::now() + chrono::Duration::seconds(timeout as i64)
    }

    /// Compute tool call deadline from agent config.
    pub(super) fn tool_deadline(&self) -> DateTime<Utc> {
        let timeout = self
            .agent
            .as_ref()
            .map(|a| a.retry.tool_timeout_secs)
            .unwrap_or(120);
        Utc::now() + chrono::Duration::seconds(timeout as i64)
    }

    pub fn label(&self) -> Option<String> {
        self.agent.as_ref().map(|a| a.name.clone())
    }

    pub fn set_token_usage_total(&mut self, total: u64) {
        self.token_usage.total_tokens = total;
    }
}

// ---------------------------------------------------------------------------
// on_event helpers — I/O dispatch and strategy decisions
// ---------------------------------------------------------------------------

impl AgentState {
    /// Handle an LLM call: resolve client, call API (streaming or not), return command.
    async fn handle_llm_call(
        &self,
        p: &LlmCallRequested,
        ctx: &SessionContext,
        span: &crate::domain::span::SpanContext,
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

        // Inject tools into the request
        let LlmRequest::OpenAi(mut oai_req) = p.request.clone();
        oai_req.tools = ctx.all_tools.clone();

        let request = LlmRequest::OpenAi(oai_req);

        let result = if p.stream {
            let (chunk_tx, mut chunk_rx) =
                tokio::sync::mpsc::unbounded_channel::<StreamDelta>();

            let call_id = p.call_id.clone();
            let session_id = ctx.session_id;
            let notify = ctx.notify_chunk.clone();
            let chunk_span = span.child("llm.stream");
            let mut chunk_index: u32 = 0;

            let (result, _) = tokio::join!(
                client.call_streaming(&request, chunk_tx),
                async {
                    while let Some(delta) = chunk_rx.recv().await {
                        if let Some(text) = delta.text {
                            if let Some(ref notify) = notify {
                                notify(session_id, call_id.clone(), chunk_index, text, chunk_span.clone());
                                chunk_index += 1;
                            }
                        }
                    }
                }
            );
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
        span: &crate::domain::span::SpanContext,
    ) -> Option<CommandPayload> {
        // Sub-agent tool call
        if let Some(child_session_id) = self
            .tool_calls
            .get(&p.tool_call_id)
            .and_then(|tc| tc.child_session_id())
        {
            let args: serde_json::Value =
                serde_json::from_str(&p.arguments).unwrap_or_default();
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
                    token_budget: None,
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
            let args: serde_json::Value =
                serde_json::from_str(&p.arguments).unwrap_or_default();

            match mcp.call_tool(&p.name, args).await {
                Ok(result) => {
                    let text = result
                        .content
                        .iter()
                        .filter_map(|c| match c {
                            McpToolContent::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
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
                        error: e,
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
                let request = self.build_llm_request(ctx.all_tools.clone())?;
                Some(CommandPayload::RequestLlmCall {
                    call_id: new_call_id(),
                    request,
                    stream,
                    deadline: self.llm_deadline(),
                })
            }
            EventPayload::LlmCallCompleted(p) => {
                let (content, tool_calls, _token_count) =
                    super::event_handler::extract_assistant_message(&p.response);
                if tool_calls.is_empty() {
                    // No tool calls → done
                    let artifacts = match content {
                        Some(ref text) if !text.is_empty() => vec![Artifact {
                            name: None,
                            description: None,
                            parts: vec![crate::domain::event::Part::Text {
                                text: text.clone(),
                            }],
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
                        parts: vec![crate::domain::event::Part::Text {
                            text: format!("Error: {}", p.error),
                        }],
                    }],
                })
            }
            EventPayload::MessageTool(_) => {
                // Wait until all tool calls are done
                if self.tool_calls.values().any(|tc| tc.status == ToolCallStatus::Pending) {
                    return None;
                }
                let request = self.build_llm_request(ctx.all_tools.clone())?;
                Some(CommandPayload::RequestLlmCall {
                    call_id: new_call_id(),
                    request,
                    stream: ctx.stream,
                    deadline: self.llm_deadline(),
                })
            }
            EventPayload::InterruptResumed(_) => {
                let request = self.build_llm_request(ctx.all_tools.clone())?;
                Some(CommandPayload::RequestLlmCall {
                    call_id: new_call_id(),
                    request,
                    stream: ctx.stream,
                    deadline: self.llm_deadline(),
                })
            }
            EventPayload::BudgetExceeded => {
                Some(CommandPayload::Interrupt {
                    interrupt_id: Uuid::new_v4().to_string(),
                    reason: "token_budget_exceeded".to_string(),
                    payload: serde_json::json!({
                        "total_tokens": self.token_usage.total_tokens,
                        "limit": self.token_budget.limit,
                    }),
                })
            }
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// AggregateState impl — makes AgentState the aggregate directly
// ---------------------------------------------------------------------------

#[async_trait]
impl AggregateState for AgentState {
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
    ) -> Result<Vec<Self::Event>, Self::Error> {
        self.handle(cmd, ctx)
    }

    async fn on_event(&self, event: &Self::Event, ctx: &Self::Context, span: &crate::domain::span::SpanContext) -> Option<Self::Command> {
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
                if self
                    .tool_calls
                    .get(&p.tool_call_id)
                    .is_some_and(|tc| tc.status == ToolCallStatus::Pending && tc.handler == ToolHandler::Runtime)
                {
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

fn to_openai_message(msg: &Message) -> openai::ChatMessage {
    openai::ChatMessage {
        role: match msg.role {
            Role::System => openai::Role::System,
            Role::User => openai::Role::User,
            Role::Assistant => openai::Role::Assistant,
            Role::Tool => openai::Role::Tool,
        },
        content: msg.content.clone(),
        tool_calls: if msg.tool_calls.is_empty() {
            None
        } else {
            Some(
                msg.tool_calls
                    .iter()
                    .map(|tc| openai::ToolCall {
                        id: tc.id.clone(),
                        call_type: "function".to_string(),
                        function: openai::FunctionCall {
                            name: tc.name.clone(),
                            arguments: tc.arguments.clone(),
                        },
                    })
                    .collect(),
            )
        },
        tool_call_id: msg.tool_call_id.clone(),
    }
}
