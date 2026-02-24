use std::cmp::min;
use std::collections::HashMap;

use chrono::{DateTime, Utc};
use uuid::Uuid;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::domain::event::{
    AgentConfig, Event, EventPayload, LlmRequest, LlmResponse, Message, Role, SessionAuth,
};
use crate::domain::openai;

use super::strategy::ToolResult;

// ---------------------------------------------------------------------------
// Token tracking
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenBudget {
    pub used: u64,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCallState {
    pub call_id: String,
    pub request: LlmRequest,
    pub status: LlmCallStatus,
    pub response: Option<LlmResponse>,
    pub response_processed: bool,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResult {
    pub content: String,
    pub is_error: bool,
}

// ---------------------------------------------------------------------------
// AgentState — the complete aggregate projection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub session_id: Uuid,
    pub status: SessionStatus,
    pub agent: Option<AgentConfig>,
    pub auth: Option<SessionAuth>,
    pub messages: Vec<Message>,
    pub tokens: TokenBudget,
    pub stream_version: u64,
    pub last_applied: Option<u64>,
    pub last_reacted: Option<u64>,
    pub strategy_state: Value,

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
            tokens: TokenBudget {
                used: 0,
                limit: None,
            },
            stream_version: 0,
            last_applied: None,
            last_reacted: None,
            strategy_state: Value::Null,
            llm_calls: HashMap::new(),
            tool_calls: HashMap::new(),
        }
    }

    /// Apply an event to the aggregate. Returns `true` if applied (not a duplicate).
    pub fn apply_core(&mut self, event: &Event) -> bool {
        if self.last_applied.is_some_and(|seq| event.sequence <= seq) {
            return false;
        }
        self.last_applied = Some(event.sequence);
        self.stream_version += 1;
        match &event.payload {
            EventPayload::SessionCreated(payload) => {
                self.session_id = event.session_id;
                self.agent = Some(payload.agent.clone());
                self.auth = Some(payload.auth.clone());
            }
            EventPayload::MessageUser(payload) => {
                self.track_tokens(&payload.message);
                self.messages.push(payload.message.clone());
            }
            EventPayload::MessageAssistant(payload) => {
                self.track_tokens(&payload.message);
                self.messages.push(payload.message.clone());
                if let Some(call) = self.llm_calls.get_mut(&payload.call_id) {
                    call.response_processed = true;
                }
            }
            EventPayload::MessageTool(payload) => {
                self.track_tokens(&payload.message);
                self.messages.push(payload.message.clone());
            }
            EventPayload::LlmCallRequested(payload) => {
                self.status = SessionStatus::Active;
                self.llm_calls.insert(
                    payload.call_id.clone(),
                    LlmCallState {
                        call_id: payload.call_id.clone(),
                        request: payload.request.clone(),
                        status: LlmCallStatus::Pending,
                        response: None,
                        response_processed: false,
                        retry: RetryState::default(),
                        deadline: payload.deadline,
                    },
                );
            }
            EventPayload::LlmCallCompleted(payload) => {
                if let Some(call) = self.llm_calls.get_mut(&payload.call_id) {
                    call.status = LlmCallStatus::Completed;
                    call.response = Some(payload.response.clone());
                }
            }
            EventPayload::LlmCallErrored(payload) => {
                if let Some(call) = self.llm_calls.get_mut(&payload.call_id) {
                    call.status = LlmCallStatus::Failed;
                    call.retry.attempts += 1;
                    let backoff_max = self
                        .agent
                        .as_ref()
                        .map(|a| a.retry.backoff_max_secs)
                        .unwrap_or(60);
                    let backoff = min(2u64.pow(call.retry.attempts), backoff_max);
                    call.retry.next_at =
                        Some(event.occurred_at + chrono::Duration::seconds(backoff as i64));
                }
            }
            EventPayload::ToolCallRequested(payload) => {
                self.status = SessionStatus::Active;
                self.tool_calls.insert(
                    payload.tool_call_id.clone(),
                    ToolCallState {
                        tool_call_id: payload.tool_call_id.clone(),
                        name: payload.name.clone(),
                        status: ToolCallStatus::Pending,
                        result: None,
                        retry: RetryState::default(),
                        deadline: payload.deadline,
                    },
                );
            }
            EventPayload::ToolCallCompleted(payload) => {
                if let Some(tc) = self.tool_calls.get_mut(&payload.tool_call_id) {
                    tc.status = ToolCallStatus::Completed;
                    tc.result = Some(ToolCallResult {
                        content: payload.result.clone(),
                        is_error: false,
                    });
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
            _ => {}
        }
        true
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
            .filter(|c| c.status == ToolCallStatus::Pending)
            .map(|c| c.deadline);
        let retry_at = self
            .llm_calls
            .values()
            .filter(|c| c.status == LlmCallStatus::Failed)
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

        Some(LlmRequest::OpenAi(openai::ChatCompletionRequest {
            model,
            messages,
            tools,
            tool_choice: None,
            temperature,
            max_tokens,
        }))
    }

    fn track_tokens(&mut self, message: &Message) {
        if let Some(count) = message.token_count {
            self.tokens.used += count as u64;
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
