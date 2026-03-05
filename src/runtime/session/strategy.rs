use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::runtime::config::{AgentConfig, StrategyConfig};
use crate::runtime::event::EventPayload;
use crate::runtime::llm::{LlmRequest, LlmTool};
use crate::runtime::message::{Message, Role, ToolCall};
use crate::runtime::session::types::{Artifact, Part};

// ---------------------------------------------------------------------------
// Strategy trait
// ---------------------------------------------------------------------------

/// A strategy is a deterministic decision-maker for a session.
///
/// The runtime handles execution mechanics (I/O, retries, timeouts).
/// The strategy handles business logic: what to do after each event.
///
/// Two methods:
/// - `apply`: maintain opaque state from events
/// - `decide`: given a trigger and current state, produce actions
pub trait Strategy: Send + Sync + fmt::Debug {
    /// Update opaque strategy state in response to an event.
    fn apply(&self, event: &EventPayload, state: &mut serde_json::Value);

    /// Make a decision given a trigger, current state, and runtime context.
    fn decide(
        &self,
        trigger: &DecisionTrigger,
        state: &serde_json::Value,
        ctx: &StrategyCtx,
    ) -> Vec<StrategyAction>;

    /// Extract conversation messages from opaque strategy state.
    fn messages(&self, state: &serde_json::Value) -> Vec<Message>;
}

// ---------------------------------------------------------------------------
// Decision triggers — what caused the strategy to be consulted
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionTrigger {
    UserMessage {
        stream: bool,
    },
    LlmCompleted {
        call_id: String,
        content: Option<String>,
        tool_calls: Vec<ToolCall>,
    },
    LlmFailed {
        call_id: String,
        error: String,
    },
    ToolFailed {
        tool_call_id: String,
        name: String,
        error: String,
    },
    AllToolsResolved,
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

pub struct StrategyCtx<'a> {
    pub session_id: Uuid,
    pub stream: bool,
    pub agent: &'a AgentConfig,
    pub all_tools: &'a Option<Vec<LlmTool>>,
    pub token_usage: &'a BTreeMap<String, u64>,
    pub has_inflight_tools: bool,
    pub has_pending_llm: bool,
    pub failed_llm_calls: Vec<&'a str>,
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
}

// ---------------------------------------------------------------------------
// Default strategy — reproduces current inline behavior
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct DefaultStrategy;

impl Strategy for DefaultStrategy {
    fn apply(&self, event: &EventPayload, state: &mut serde_json::Value) {
        // Ensure state is an object with a messages array
        if !state.is_object() {
            *state = serde_json::json!({"messages": []});
        }
        let obj = state.as_object_mut().unwrap();
        obj.entry("messages")
            .or_insert_with(|| serde_json::json!([]));
        let messages = obj
            .get_mut("messages")
            .unwrap()
            .as_array_mut()
            .unwrap();

        match event {
            EventPayload::MessageUser(p) => {
                if let Ok(v) = serde_json::to_value(&p.message) {
                    messages.push(v);
                }
            }
            EventPayload::MessageAssistant(p) => {
                if let Ok(v) = serde_json::to_value(&p.message) {
                    messages.push(v);
                }
            }
            EventPayload::MessageTool(p) => {
                if let Ok(v) = serde_json::to_value(&p.message) {
                    messages.push(v);
                }
            }
            EventPayload::StrategyStateChanged(p) => {
                let obj = state.as_object_mut().unwrap();
                match &p.state {
                    Some(s) => {
                        obj.insert(
                            "user_state".to_string(),
                            serde_json::Value::String(s.clone()),
                        );
                    }
                    None => {
                        obj.remove("user_state");
                    }
                }
            }
            _ => {}
        }
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

    fn decide(
        &self,
        trigger: &DecisionTrigger,
        state: &serde_json::Value,
        ctx: &StrategyCtx,
    ) -> Vec<StrategyAction> {
        match trigger {
            DecisionTrigger::UserMessage { stream } => {
                match self.build_llm_request(state, ctx, None) {
                    Some(request) => vec![StrategyAction::RequestLlm {
                        request,
                        stream: *stream,
                    }],
                    None => vec![],
                }
            }
            DecisionTrigger::LlmCompleted {
                content,
                tool_calls,
                ..
            } => {
                if tool_calls.is_empty() {
                    // No tool calls → done
                    let artifacts = match content {
                        Some(ref text) if !text.is_empty() => vec![Artifact {
                            name: None,
                            description: None,
                            parts: vec![Part::Text { text: text.clone() }],
                        }],
                        _ => vec![],
                    };
                    vec![StrategyAction::Done { artifacts }]
                } else {
                    // Has tool calls → request them
                    vec![StrategyAction::RequestToolCalls {
                        tool_calls: tool_calls.clone(),
                    }]
                }
            }
            DecisionTrigger::LlmFailed { error, .. } => {
                // Retries exhausted — give up
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
            DecisionTrigger::ToolFailed { .. } => {
                // Retries exhausted — wait for all tools to resolve, then retry LLM
                vec![]
            }
            DecisionTrigger::AllToolsResolved => {
                match self.build_llm_request(state, ctx, None) {
                    Some(request) => vec![StrategyAction::RequestLlm {
                        request,
                        stream: ctx.stream,
                    }],
                    None => vec![],
                }
            }
            DecisionTrigger::InterruptResumed { .. } => {
                match self.build_llm_request(state, ctx, None) {
                    Some(request) => vec![StrategyAction::RequestLlm {
                        request,
                        stream: ctx.stream,
                    }],
                    None => vec![],
                }
            }
            DecisionTrigger::Stall => match self.build_llm_request(state, ctx, None) {
                Some(request) => vec![StrategyAction::RequestLlm {
                    request,
                    stream: ctx.stream,
                }],
                None => vec![],
            },
        }
    }
}

impl DefaultStrategy {
    /// Build an LLM request from the opaque strategy state.
    fn build_llm_request(
        &self,
        state: &serde_json::Value,
        ctx: &StrategyCtx,
        _overrides: Option<&crate::runtime::config::LlmRequestParams>,
    ) -> Option<LlmRequest> {
        let agent = ctx.agent;

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
}

// ---------------------------------------------------------------------------
// Strategy resolution
// ---------------------------------------------------------------------------

/// Resolve a strategy implementation from agent config.
pub fn resolve_strategy(_config: &StrategyConfig) -> Box<dyn Strategy> {
    // V1: only the default strategy. Future: match on config.kind.
    Box::new(DefaultStrategy)
}
