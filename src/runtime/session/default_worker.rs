use crate::runtime::llm::LlmRequest;
use crate::runtime::message::{Message, Role, ToolCall};
use crate::runtime::session::state::ToolCallStatus;
use crate::runtime::session::types::{Artifact, Part};

use super::worker::{
    DecisionTrigger, ToolCallAction, Worker, WorkerAction, WorkerCtx, WorkerDecision,
};

// ---------------------------------------------------------------------------
// Default worker — reproduces current inline behavior
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct DefaultWorker;

impl DefaultWorker {
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

    /// Build an LLM request from the opaque worker state.
    fn build_llm_request(state: &serde_json::Value, ctx: &WorkerCtx) -> Option<LlmRequest> {
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
            tools: ctx.all_tools.clone(),
            temperature,
            max_completion_tokens,
        })
    }

    /// Build annotated tool call actions from LLM tool calls.
    ///
    /// Sub-agent tools (matched via `ctx.sub_agents`) get the full `AgentConfig`
    /// as context. Regular tools get null context.
    fn build_tool_call_actions(tool_calls: &[ToolCall], ctx: &WorkerCtx) -> Vec<ToolCallAction> {
        tool_calls
            .iter()
            .map(|tc| {
                let context = ctx
                    .sub_agents
                    .get(&tc.name)
                    .and_then(|agent| serde_json::to_value(agent).ok())
                    .unwrap_or(serde_json::Value::Null);
                ToolCallAction {
                    tool_call: tc.clone(),
                    context,
                }
            })
            .collect()
    }

    /// Build an LLM request action, returning empty actions if request cannot be built.
    fn request_llm(state: &serde_json::Value, ctx: &WorkerCtx, stream: bool) -> Vec<WorkerAction> {
        match Self::build_llm_request(state, ctx) {
            Some(request) => vec![WorkerAction::RequestLlm { request, stream }],
            None => vec![],
        }
    }
}

impl Worker for DefaultWorker {
    fn decide(
        &self,
        trigger: &DecisionTrigger,
        state: &serde_json::Value,
        ctx: &WorkerCtx,
    ) -> WorkerDecision {
        let mut state = state.clone();

        let actions = match trigger {
            DecisionTrigger::UserMessage { stream, message } => {
                Self::push_message(&mut state, message);
                Self::request_llm(&state, ctx, *stream)
            }
            DecisionTrigger::LlmCompleted {
                message, truncated, ..
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
                    vec![WorkerAction::Done { artifacts }]
                } else {
                    vec![WorkerAction::RequestToolCalls {
                        tool_calls: Self::build_tool_call_actions(&message.tool_calls, ctx),
                    }]
                }
            }
            DecisionTrigger::LlmFailed { error, .. } => {
                vec![WorkerAction::Done {
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
                let has_inflight = ctx.tool_call_statuses.values().any(|s| {
                    *s == ToolCallStatus::Pending || *s == ToolCallStatus::RetryScheduled
                });
                if !has_inflight {
                    Self::request_llm(&state, ctx, ctx.stream)
                } else {
                    vec![]
                }
            }
            DecisionTrigger::InterruptResumed { .. } => Self::request_llm(&state, ctx, ctx.stream),
            DecisionTrigger::Stall => Self::request_llm(&state, ctx, ctx.stream),
        };

        WorkerDecision { actions, state }
    }
}
