use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::agent_state::AgentState;
use crate::domain::event::{Message, ToolCall};
use crate::domain::openai;

// ---------------------------------------------------------------------------
// Strategy trait — decisions only
// ---------------------------------------------------------------------------

pub trait Strategy: Send + Sync {
    /// Initial strategy state for a new or cold-replayed session.
    fn default_state(&self) -> Value {
        Value::Null
    }

    /// A new user message arrived.
    fn on_user_message(
        &self,
        state: &Value,
        core: &AgentState,
        tools: Option<&[openai::Tool]>,
    ) -> Turn;

    /// The LLM produced a response.
    fn on_llm_response(
        &self,
        state: &Value,
        core: &AgentState,
        tools: Option<&[openai::Tool]>,
        response: &LlmResponseSummary,
    ) -> Turn;

    /// A batch of tool calls completed.
    fn on_tool_results(
        &self,
        state: &Value,
        core: &AgentState,
        tools: Option<&[openai::Tool]>,
        results: &[ToolResult],
    ) -> Turn;

    /// An interrupt was resumed by the client.
    fn on_interrupt_resume(
        &self,
        state: &Value,
        core: &AgentState,
        tools: Option<&[openai::Tool]>,
        _interrupt_id: &str,
        _response: &Value,
    ) -> Turn {
        self.on_user_message(state, core, tools)
    }
}

// ---------------------------------------------------------------------------
// Turn — strategy decision plus updated state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Turn {
    pub action: Action,
    pub state: Value,
}

// ---------------------------------------------------------------------------
// Action — what the strategy tells the runtime to do next
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Action {
    /// Call the LLM.
    CallLlm(LlmTurnParams),
    /// Execute tool calls, then consult strategy again.
    ExecuteTools(ToolExecutionPlan),
    /// Agent loop is done — idle until next external input.
    Done,
    /// Pause the session and ask the client to resume with a response.
    Interrupt(InterruptRequest),
}

#[derive(Debug, Clone)]
pub struct InterruptRequest {
    pub id: String,
    pub reason: String,
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct LlmTurnParams {
    /// None = default context (system prompt + full message history).
    pub context: Option<Vec<Message>>,
    /// None = use the runtime's default (from the user message stream preference).
    pub stream: Option<bool>,
}

// ---------------------------------------------------------------------------
// LlmResponseSummary — what the strategy sees from an LLM response
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LlmResponseSummary {
    pub call_id: String,
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub token_count: Option<u32>,
}

// ---------------------------------------------------------------------------
// Tool execution types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub name: String,
    pub content: String,
    pub is_error: bool,
}

#[derive(Debug, Clone)]
pub struct ToolExecutionPlan {
    pub calls: Vec<ToolCall>,
    pub mode: ToolExecutionMode,
}

#[derive(Debug, Clone, Default)]
pub enum ToolExecutionMode {
    /// Fire all concurrently, call on_tool_results when all complete.
    #[default]
    Concurrent,
    /// Execute one at a time in order, call on_tool_results when all complete.
    Sequential,
}

// ---------------------------------------------------------------------------
// StrategyKind — serializable enum for agent config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum StrategyKind {
    #[default]
    React,
}

// ---------------------------------------------------------------------------
// ReactStrategy — concurrent ReAct with full history
// ---------------------------------------------------------------------------

pub struct ReactStrategy;

impl ReactStrategy {
    pub fn new() -> Self {
        ReactStrategy
    }
}

impl Strategy for ReactStrategy {
    fn on_user_message(
        &self,
        state: &Value,
        _core: &AgentState,
        _tools: Option<&[openai::Tool]>,
    ) -> Turn {
        Turn {
            action: Action::CallLlm(LlmTurnParams {
                context: None,
                stream: None,
            }),
            state: state.clone(),
        }
    }

    fn on_llm_response(
        &self,
        state: &Value,
        _core: &AgentState,
        _tools: Option<&[openai::Tool]>,
        response: &LlmResponseSummary,
    ) -> Turn {
        let action = if response.tool_calls.is_empty() {
            Action::Done
        } else {
            Action::ExecuteTools(ToolExecutionPlan {
                calls: response.tool_calls.clone(),
                mode: ToolExecutionMode::Concurrent,
            })
        };
        Turn {
            action,
            state: state.clone(),
        }
    }

    fn on_tool_results(
        &self,
        state: &Value,
        _core: &AgentState,
        _tools: Option<&[openai::Tool]>,
        _results: &[ToolResult],
    ) -> Turn {
        Turn {
            action: Action::CallLlm(LlmTurnParams {
                context: None,
                stream: Some(true),
            }),
            state: state.clone(),
        }
    }
}
