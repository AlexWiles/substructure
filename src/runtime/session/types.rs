use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::runtime::config::ClientIdentity;
use crate::runtime::message::Message;
use crate::runtime::span::SpanContext;

// --- Message event wrappers ---

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

// --- Completion delivery ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionDelivery {
    /// Session ID of the parent session to deliver to
    pub parent_session_id: Uuid,
    /// Tool call ID that this result satisfies on the parent session
    pub tool_call_id: String,
    /// Tool name for the completion
    pub tool_name: String,
    /// Span context for tracing
    pub span: SpanContext,
}

// --- Session lifecycle ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCreated {
    pub agent_name: String,
    pub auth: ClientIdentity,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on_done: Option<CompletionDelivery>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionDone {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<Artifact>,
}

// --- Artifacts ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parts: Vec<Part>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Part {
    Text { text: String },
    Data { data: serde_json::Value },
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

// --- Tool calls ---

/// Who is responsible for executing a tool call.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolHandler {
    /// Dispatched to the work queue for the worker to execute.
    #[default]
    Worker,
    /// Executed by the client. Session goes Idle while waiting.
    Client,
    /// Sub-agent: runtime spawns a child session.
    SubAgent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRequested {
    pub tool_call_id: String,
    pub name: String,
    pub arguments: String,
    pub deadline: DateTime<Utc>,
    #[serde(default)]
    pub handler: ToolHandler,
    /// Opaque context from the worker, passed through to transport dispatch.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub context: serde_json::Value,
    /// Per-request retry hints from the worker.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_secs: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<u32>,
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
    #[serde(default)]
    pub retryable: bool,
}
