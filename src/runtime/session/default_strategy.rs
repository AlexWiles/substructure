use crate::runtime::llm::LlmRequest;
use crate::runtime::message::{Message, Role};
use crate::runtime::session::types::{Artifact, Part};

use super::strategy::{
    DecisionTrigger, Strategy, StrategyAction, StrategyCtx, StrategyDecision,
};

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
}
