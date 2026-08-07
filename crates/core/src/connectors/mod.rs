//! Connections the engine holds and executes tools against, on the agent's
//! behalf. The agent config names a connection by id and never sees a URL or a
//! credential; the protocol below that id is an implementation detail, so a
//! non-MCP source is additive.

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub mod credential;
pub mod filter;
pub mod mcp;
pub mod oauth;
pub mod registry;

/// A tool as a connection offers it, before filtering or prefixing. The neutral
/// shape every protocol lowers to.
///
/// Stored verbatim on `connector.sync.completed`, unfiltered: filtering is pure,
/// so recording the offer lets a filter change re-derive the model's tool list
/// without another round trip, and lets the log answer what was filtered *out*.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RemoteTool {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,
    #[serde(default)]
    pub annotations: ToolAnnotations,
}

/// Behavioural hints a connection attaches to a tool. Every field is optional
/// because a server need not annotate anything, and absent is not false — a
/// capability filter treats an unannotated tool as failing, so a bare server
/// yields nothing under `read_only` instead of passing everything through.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolAnnotations {
    pub read_only: Option<bool>,
    pub destructive: Option<bool>,
    pub idempotent: Option<bool>,
    pub open_world: Option<bool>,
}

/// The outcome of a tool call on a connection. `is_error` is the connection's
/// own signal that the tool failed, distinct from a transport failure.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolOutcome {
    pub content: String,
    pub structured: Option<Value>,
    pub is_error: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConnectorError {
    pub message: String,
    /// Whether another attempt could plausibly succeed. Transport faults and 5xx
    /// are retryable; a rejected credential or an unknown tool is not.
    pub retryable: bool,
    /// Set when the connection rejected our credential, so the caller can raise
    /// a re-auth interrupt rather than settling the call as a plain failure.
    pub needs_reauth: bool,
}

impl ConnectorError {
    pub fn permanent(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: false,
            needs_reauth: false,
        }
    }

    pub fn retryable(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: true,
            needs_reauth: false,
        }
    }

    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: false,
            needs_reauth: true,
        }
    }
}

impl std::fmt::Display for ConnectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ConnectorError {}
