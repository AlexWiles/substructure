use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub use super::agent::{AgentConfig, LlmConfig};
use super::openai;
use super::session::SessionStatus;
pub use super::span::{SpanContext, SpanId, TraceId};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedState {
    pub status: SessionStatus,
    pub wake_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: Uuid,
    pub tenant_id: String,
    pub session_id: Uuid,
    pub sequence: u64,
    pub span: SpanContext,
    pub occurred_at: DateTime<Utc>,
    pub payload: EventPayload,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derived: Option<DerivedState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum EventPayload {
    #[serde(rename = "session.created")]
    SessionCreated(SessionCreated),
    #[serde(rename = "message.user")]
    MessageUser(MessageUser),
    #[serde(rename = "message.assistant")]
    MessageAssistant(MessageAssistant),
    #[serde(rename = "llm.call.requested")]
    LlmCallRequested(LlmCallRequested),
    #[serde(rename = "llm.stream.chunk")]
    LlmStreamChunk(LlmStreamChunk),
    #[serde(rename = "llm.call.completed")]
    LlmCallCompleted(LlmCallCompleted),
    #[serde(rename = "llm.call.errored")]
    LlmCallErrored(LlmCallErrored),
    #[serde(rename = "message.tool")]
    MessageTool(MessageTool),
    #[serde(rename = "tool.call.requested")]
    ToolCallRequested(ToolCallRequested),
    #[serde(rename = "tool.call.completed")]
    ToolCallCompleted(ToolCallCompleted),
    #[serde(rename = "tool.call.errored")]
    ToolCallErrored(ToolCallErrored),
    #[serde(rename = "session.interrupted")]
    SessionInterrupted(SessionInterrupted),
    #[serde(rename = "session.interrupt_resumed")]
    InterruptResumed(InterruptResumed),
    #[serde(rename = "strategy.state_changed")]
    StrategyStateChanged(StrategyStateChanged),
}

// --- Session ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    pub name: String,
    pub transport: McpTransportConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum McpTransportConfig {
    #[serde(rename = "stdio")]
    Stdio { command: String, args: Vec<String> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionAuth {
    pub tenant_id: String,
    pub client_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sub: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCreated {
    pub agent: AgentConfig,
    pub auth: SessionAuth,
}

// --- Messages (internal, provider-agnostic) ---

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_count: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageUser {
    pub message: Message,
    pub stream: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageAssistant {
    pub call_id: String,
    pub message: Message,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageTool {
    pub message: Message,
}

// --- Tool Calls ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRequested {
    pub tool_call_id: String,
    pub name: String,
    pub arguments: String,
    pub deadline: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallCompleted {
    pub tool_call_id: String,
    pub name: String,
    pub result: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallErrored {
    pub tool_call_id: String,
    pub name: String,
    pub error: String,
}

// --- Interrupts ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInterrupted {
    pub interrupt_id: String,
    pub reason: String,
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterruptResumed {
    pub interrupt_id: String,
    pub payload: serde_json::Value,
}

// --- Strategy ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyStateChanged {
    pub state: serde_json::Value,
}

// --- LLM ---

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "provider")]
pub enum LlmRequest {
    #[serde(rename = "openai")]
    OpenAi(openai::ChatCompletionRequest),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "provider")]
pub enum LlmResponse {
    #[serde(rename = "openai")]
    OpenAi(openai::ChatCompletionResponse),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmStreamChunk {
    pub call_id: String,
    pub chunk_index: u32,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCallRequested {
    pub call_id: String,
    pub request: LlmRequest,
    pub stream: bool,
    pub deadline: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCallCompleted {
    pub call_id: String,
    pub response: LlmResponse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCallErrored {
    pub call_id: String,
    pub error: String,
}
