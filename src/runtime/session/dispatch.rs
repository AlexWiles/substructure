use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::decision::DecisionTrigger;
use super::state::{LlmCallStatus, ToolCallStatus};
use crate::runtime::config::ClientIdentity;
use crate::runtime::span::SpanContext;

// ---------------------------------------------------------------------------
// Worker dispatch — the wire message for decisions
// ---------------------------------------------------------------------------

/// Serializable request sent to a worker (local or remote).
///
/// Carries session-specific data. The executor combines this with its own
/// tool/sub-agent knowledge to build the full `WorkerCtx` for `decide()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerDispatch {
    pub session_id: Uuid,
    pub decision_id: String,
    pub trigger: DecisionTrigger,
    pub worker_state: Vec<u8>,
    // Session snapshot
    pub stream: bool,
    pub agent_name: String,
    pub auth: ClientIdentity,
    pub token_usage: BTreeMap<String, u64>,
    #[serde(default)]
    pub tool_call_statuses: HashMap<String, ToolCallStatus>,
    #[serde(default)]
    pub llm_call_statuses: HashMap<String, LlmCallStatus>,
    pub span: SpanContext,
}

// ---------------------------------------------------------------------------
// Tool call dispatch — the wire message for tool execution
// ---------------------------------------------------------------------------

/// Serializable request sent to a tool executor (local or remote).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDispatch {
    pub session_id: Uuid,
    pub tool_call_id: String,
    pub name: String,
    pub arguments: String,
    /// Opaque context from the worker (e.g. sub-agent AgentConfig).
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub context: serde_json::Value,
    pub agent_name: String,
    pub auth: ClientIdentity,
    pub span: SpanContext,
}

// ---------------------------------------------------------------------------
// WorkerExecutor — abstracts local vs remote worker execution
// ---------------------------------------------------------------------------

/// Transport abstraction for worker execution.
///
/// A message channel to a worker "somewhere" (local, remote, different process).
/// The session aggregate calls dispatch methods and returns — fire-and-forget.
/// Results come back asynchronously via commands delivered to the session.
pub trait WorkerExecutor: Send + Sync {
    /// Dispatch a decision request to the worker.
    fn dispatch_decision(&self, request: WorkerDispatch);
    /// Dispatch a tool call to the worker for execution.
    fn dispatch_tool_call(&self, request: ToolCallDispatch);
}
