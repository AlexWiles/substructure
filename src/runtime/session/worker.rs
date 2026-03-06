use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::fmt;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::runtime::config::AgentConfig;
use crate::runtime::llm::{LlmRequest, LlmTool};
use crate::runtime::message::{Message, ToolCall};
use crate::runtime::session::state::{LlmCallStatus, ToolCallStatus, ToolResult};
use crate::runtime::session::types::Artifact;

// ---------------------------------------------------------------------------
// Worker trait
// ---------------------------------------------------------------------------

/// A worker is a pure decision-maker for a session.
///
/// The runtime handles execution mechanics (I/O, retries, timeouts) and stores
/// the worker's opaque state. The worker handles business logic: given a
/// trigger and its current state, produce actions and an updated state.
///
/// In the future, `decide` becomes an RPC call to a remote worker.
pub trait Worker: Send + Sync + fmt::Debug {
    /// Make a decision given a trigger, current opaque state, and runtime context.
    /// Returns actions for the runtime to execute and the updated opaque state.
    fn decide(
        &self,
        trigger: &DecisionTrigger,
        state: &serde_json::Value,
        ctx: &WorkerCtx,
    ) -> WorkerDecision;
}

/// The result of a worker decision: actions to execute and updated state.
#[derive(Debug, Clone)]
pub struct WorkerDecision {
    pub actions: Vec<WorkerAction>,
    pub state: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Decision triggers — what caused the worker to be consulted
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionTrigger {
    UserMessage {
        stream: bool,
        message: Message,
    },
    LlmCompleted {
        call_id: String,
        message: Message,
        /// True when finish_reason was "length" (output truncated).
        truncated: bool,
    },
    LlmFailed {
        call_id: String,
        error: String,
    },
    ToolResolved {
        result: ToolResult,
    },
    InterruptResumed {
        interrupt_id: String,
    },
    Stall,
}

// ---------------------------------------------------------------------------
// Worker actions — what the worker wants the runtime to do
// ---------------------------------------------------------------------------

/// A tool call annotated with opaque worker context.
///
/// The worker attaches context (e.g. a sub-agent's `AgentConfig`) that flows
/// through the runtime untouched and arrives at the transport for execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallAction {
    pub tool_call: ToolCall,
    /// Opaque context from the worker, passed through to transport dispatch.
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub context: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkerAction {
    RequestLlm { request: LlmRequest, stream: bool },
    RequestToolCalls { tool_calls: Vec<ToolCallAction> },
    Done { artifacts: Vec<Artifact> },
    UpdateState { state: Option<String> },
}

// ---------------------------------------------------------------------------
// Worker context — curated view of runtime state for decisions
// ---------------------------------------------------------------------------

/// Serializable snapshot of runtime state provided to the worker.
/// Fully owned — can cross process/language boundaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerCtx {
    pub session_id: Uuid,
    pub stream: bool,
    pub agent: AgentConfig,
    pub all_tools: Option<Vec<LlmTool>>,
    /// Sub-agent configs available for spawning.
    #[serde(default)]
    pub sub_agents: HashMap<String, AgentConfig>,
    pub token_usage: BTreeMap<String, u64>,
    /// Status of each tool call, keyed by tool_call_id.
    #[serde(default)]
    pub tool_call_statuses: HashMap<String, ToolCallStatus>,
    /// Status of each LLM call, keyed by call_id.
    #[serde(default)]
    pub llm_call_statuses: HashMap<String, LlmCallStatus>,
}

// ---------------------------------------------------------------------------
// Worker decision events
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerDecisionRequested {
    pub decision_id: String,
    pub trigger: DecisionTrigger,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerDecisionCompleted {
    pub decision_id: String,
    pub state: serde_json::Value,
}
