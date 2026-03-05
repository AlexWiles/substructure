use std::collections::BTreeMap;
use std::fmt;

use ractor::ActorRef;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::runtime::config::{AgentConfig, StrategyConfig};
use crate::runtime::llm::{LlmRequest, LlmTool};
use crate::runtime::message::{Message, Role, ToolCall};
use crate::runtime::session::command::CommandPayload;
use crate::runtime::session::state::ToolResult;
use crate::runtime::session::types::{Artifact, Part};
use crate::runtime::span::SpanContext;
use crate::runtime::types::RuntimeMessage;

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

// ---------------------------------------------------------------------------
// Default strategy — reproduces current inline behavior
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct DefaultStrategy;

impl DefaultStrategy {
    /// Ensure the opaque state is an object with a messages array.
    fn ensure_state(state: &mut serde_json::Value) {
        if !state.is_object() {
            *state = serde_json::json!({"messages": []});
        }
        let obj = state.as_object_mut().unwrap();
        obj.entry("messages")
            .or_insert_with(|| serde_json::json!([]));
    }

    /// Push a message into the opaque state's messages array.
    fn push_message(state: &mut serde_json::Value, message: &Message) {
        Self::ensure_state(state);
        if let Ok(v) = serde_json::to_value(message) {
            state["messages"].as_array_mut().unwrap().push(v);
        }
    }

    /// Build an LLM request from the opaque strategy state.
    fn build_llm_request(state: &serde_json::Value, ctx: &StrategyCtx) -> Option<LlmRequest> {
        let agent = &ctx.agent;

        // System prompt as first message
        let mut messages = vec![Message {
            role: Role::System,
            content: Some(agent.system_prompt.clone()),
            tool_calls: vec![],
            tool_call_id: None,
            call_id: None,
            usage: None,
        }];

        // Deserialize conversation messages from opaque state
        if let Some(stored) = state.get("messages").and_then(|v| v.as_array()) {
            for msg_val in stored {
                if let Ok(msg) = serde_json::from_value::<Message>(msg_val.clone()) {
                    messages.push(msg);
                }
            }
        }

        let temperature = agent.llm.temperature;
        let max_completion_tokens = agent.llm.max_completion_tokens;

        Some(LlmRequest {
            model: agent.llm.model.clone(),
            messages,
            tools: None, // Runtime injects tools via with_tools()
            temperature,
            max_completion_tokens,
        })
    }

    /// Build an LLM request action, returning empty actions if request cannot be built.
    fn request_llm(state: &serde_json::Value, ctx: &StrategyCtx, stream: bool) -> Vec<StrategyAction> {
        match Self::build_llm_request(state, ctx) {
            Some(request) => vec![StrategyAction::RequestLlm { request, stream }],
            None => vec![],
        }
    }
}

impl Strategy for DefaultStrategy {
    fn decide(
        &self,
        trigger: &DecisionTrigger,
        state: &serde_json::Value,
        ctx: &StrategyCtx,
    ) -> StrategyDecision {
        let mut state = state.clone();

        let actions = match trigger {
            DecisionTrigger::UserMessage { stream, message } => {
                Self::push_message(&mut state, message);
                Self::request_llm(&state, ctx, *stream)
            }
            DecisionTrigger::LlmCompleted {
                message,
                truncated,
                ..
            } => {
                Self::push_message(&mut state, message);
                if *truncated {
                    Self::request_llm(&state, ctx, ctx.stream)
                } else if message.tool_calls.is_empty() {
                    let artifacts = match &message.content {
                        Some(ref text) if !text.is_empty() => vec![Artifact {
                            name: None,
                            description: None,
                            parts: vec![Part::Text { text: text.clone() }],
                        }],
                        _ => vec![],
                    };
                    vec![StrategyAction::Done { artifacts }]
                } else {
                    vec![StrategyAction::RequestToolCalls {
                        tool_calls: message.tool_calls.clone(),
                    }]
                }
            }
            DecisionTrigger::LlmFailed { error, .. } => {
                vec![StrategyAction::Done {
                    artifacts: vec![Artifact {
                        name: None,
                        description: None,
                        parts: vec![Part::Text {
                            text: format!("Error: {error}"),
                        }],
                    }],
                }]
            }
            DecisionTrigger::ToolResolved { result } => {
                // Add tool result message to state
                let msg = Message {
                    role: Role::Tool,
                    content: Some(result.content.clone()),
                    tool_calls: vec![],
                    tool_call_id: Some(result.tool_call_id.clone()),
                    call_id: None,
                    usage: None,
                };
                Self::push_message(&mut state, &msg);
                // Only request next LLM call when all tools are done
                if !ctx.has_inflight_tools {
                    Self::request_llm(&state, ctx, ctx.stream)
                } else {
                    vec![]
                }
            }
            DecisionTrigger::InterruptResumed { .. } => {
                Self::request_llm(&state, ctx, ctx.stream)
            }
            DecisionTrigger::Stall => Self::request_llm(&state, ctx, ctx.stream),
        };

        StrategyDecision { actions, state }
    }

    fn messages(&self, state: &serde_json::Value) -> Vec<Message> {
        state
            .get("messages")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| serde_json::from_value::<Message>(v.clone()).ok())
                    .collect()
            })
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// Strategy transport — abstracts local vs remote strategy execution
// ---------------------------------------------------------------------------

/// Serializable request sent to a strategy executor (local or remote).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyDispatch {
    pub session_id: Uuid,
    pub decision_id: String,
    pub trigger: DecisionTrigger,
    pub strategy_state: serde_json::Value,
    pub ctx: StrategyCtx,
    pub span: SpanContext,
}

/// Transport abstraction for strategy execution.
///
/// The session aggregate calls `dispatch()` and returns — fire-and-forget.
/// The transport handles execution and delivers the result back via
/// `RuntimeMessage::DeliverToSession`.
pub trait StrategyTransport: Send + Sync {
    fn dispatch(&self, request: StrategyDispatch);
}

/// Executes strategies in-process and delivers results back via the runtime actor.
pub struct LocalStrategyTransport {
    pub runtime: ActorRef<RuntimeMessage>,
}

impl StrategyTransport for LocalStrategyTransport {
    fn dispatch(&self, request: StrategyDispatch) {
        let strategy = resolve_strategy(&request.ctx.agent.strategy);
        let decision = strategy.decide(
            &request.trigger,
            &request.strategy_state,
            &request.ctx,
        );
        let _ = self.runtime.send_message(RuntimeMessage::DeliverToSession {
            session_id: request.session_id,
            payload: CommandPayload::SubmitStrategyDecision {
                decision_id: request.decision_id,
                actions: decision.actions,
                state: decision.state,
            },
            span: request.span,
        });
    }
}

// ---------------------------------------------------------------------------
// Strategy resolution
// ---------------------------------------------------------------------------

/// Resolve a strategy implementation from agent config.
pub fn resolve_strategy(_config: &StrategyConfig) -> Box<dyn Strategy> {
    // V1: only the default strategy. Future: match on config.kind.
    Box::new(DefaultStrategy)
}
