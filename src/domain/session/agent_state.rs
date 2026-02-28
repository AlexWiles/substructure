use std::cmp::min;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::domain::aggregate::AggregateStatus;
use crate::domain::event::{
    AgentConfig, Artifact, ClientIdentity, CompletionDelivery, EventPayload, LlmRequest,
    LlmResponse, Message, Role, ToolCallMeta, ToolHandler,
};
use crate::domain::openai;

use super::strategy::{Strategy, ToolResult};

// ---------------------------------------------------------------------------
// StrategySlot — Arc<dyn Strategy> wrapper with Debug + Clone + Default
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
pub struct StrategySlot(pub Option<Arc<dyn Strategy>>);

impl fmt::Debug for StrategySlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Some(_) => f.write_str("Some(<strategy>)"),
            None => f.write_str("None"),
        }
    }
}

impl std::ops::Deref for StrategySlot {
    type Target = Option<Arc<dyn Strategy>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
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
    pub last_reacted: Option<u64>,
    pub strategy_state: Value,

    /// The strategy driving this session. Skipped during serialization;
    /// must be re-attached after deserialization (e.g. via `from_snapshot`).
    #[serde(skip)]
    pub strategy: StrategySlot,

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
            last_reacted: None,
            strategy_state: Value::Null,
            strategy: StrategySlot(None),
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
