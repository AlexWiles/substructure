use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::runtime::config::AgentConfig;
use crate::runtime::llm::{LlmRequest, LlmTool};
use crate::runtime::message::{Message, ToolCall};
use crate::runtime::session::state::ToolResult;
use crate::runtime::session::types::Artifact;

// ---------------------------------------------------------------------------
// Strategy trait
// ---------------------------------------------------------------------------

/// A strategy is a pure decision-maker for a session.
///
/// The runtime handles execution mechanics (I/O, retries, timeouts) and stores
/// the strategy's opaque state. The strategy handles business logic: given a
/// trigger and its current state, produce actions and an updated state.
///
/// In the future, `decide` becomes an RPC call to a remote client.
pub trait Strategy: Send + Sync + fmt::Debug {
    /// Make a decision given a trigger, current opaque state, and runtime context.
    /// Returns actions for the runtime to execute and the updated opaque state.
    fn decide(
        &self,
        trigger: &DecisionTrigger,
        state: &serde_json::Value,
        ctx: &StrategyCtx,
    ) -> StrategyDecision;

    /// Extract conversation messages from opaque strategy state.
    fn messages(&self, state: &serde_json::Value) -> Vec<Message>;
}

/// The result of a strategy decision: actions to execute and updated state.
#[derive(Debug, Clone)]
pub struct StrategyDecision {
    pub actions: Vec<StrategyAction>,
    pub state: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Decision triggers — what caused the strategy to be consulted
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
// Strategy actions — what the strategy wants the runtime to do
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StrategyAction {
    RequestLlm { request: LlmRequest, stream: bool },
    RequestToolCalls { tool_calls: Vec<ToolCall> },
    Done { artifacts: Vec<Artifact> },
    UpdateState { state: Option<String> },
}

// ---------------------------------------------------------------------------
// Strategy context — curated view of runtime state for decisions
// ---------------------------------------------------------------------------

/// Serializable snapshot of runtime state provided to the strategy.
/// Fully owned — can cross process/language boundaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyCtx {
    pub session_id: Uuid,
    pub stream: bool,
    pub agent: AgentConfig,
    pub all_tools: Option<Vec<LlmTool>>,
    pub token_usage: BTreeMap<String, u64>,
    pub has_inflight_tools: bool,
    pub has_pending_llm: bool,
    pub failed_llm_calls: Vec<String>,
}

// ---------------------------------------------------------------------------
// Strategy decision events
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyDecisionRequested {
    pub decision_id: String,
    pub trigger: DecisionTrigger,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyDecisionCompleted {
    pub decision_id: String,
    pub state: serde_json::Value,
}

