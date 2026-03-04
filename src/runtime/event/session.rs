use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::runtime::config::{AgentConfig, RetryConfig};
use crate::runtime::span::SpanContext;

// --- CompletionDelivery ---

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

// --- MCP server config ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    pub name: String,
    pub transport: McpTransportConfig,
    /// Maximum tool result size in bytes. `None` = inherit, `Some(0)` = no limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_result_max_bytes: Option<usize>,
    /// Per-server retry/timeout overrides (inherits from agent when `None`).
    #[serde(default, skip_serializing_if = "RetryConfig::is_empty")]
    pub retry: RetryConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum McpTransportConfig {
    #[serde(rename = "stdio")]
    Stdio { command: String, args: Vec<String> },
}

// --- Identity ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientIdentity {
    pub tenant_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sub: Option<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub attrs: HashMap<String, String>,
}

// --- Session lifecycle ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCreated {
    pub agent: AgentConfig,
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

// --- Strategy ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyStateChanged {
    pub state: serde_json::Value,
}
