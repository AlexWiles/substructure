use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Who is responsible for executing a tool call.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolHandler {
    /// Executed by the runtime (MCP or sub-agent).
    #[default]
    Runtime,
    /// Executed by the client. Session goes Idle while waiting.
    Client,
}

/// Metadata describing the type of tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolCallMeta {
    /// Tool is a sub-agent invocation.
    SubAgent {
        child_session_id: Uuid,
        agent_name: String,
    },
    /// Tool is served by an MCP server.
    Mcp {
        server_name: String,
        server_version: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRequested {
    pub tool_call_id: String,
    pub name: String,
    pub arguments: String,
    pub deadline: DateTime<Utc>,
    #[serde(default)]
    pub handler: ToolHandler,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<ToolCallMeta>,
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
