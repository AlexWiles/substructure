//! Default worker — simple LLM agent loop using proto types.

use super::{
    decision_trigger, part, worker_action, Artifact, CallStatus, DecisionTrigger, Done,
    LlmRequest, Message, Part, RequestLlm, RequestToolCalls, Role, ToolCallAction, Worker,
    WorkerAction, WorkerCtx, WorkerDecision,
};

#[derive(Debug)]
pub struct DefaultWorker;

impl DefaultWorker {
    /// Deserialize opaque state bytes into a JSON value for manipulation.
    fn parse_state(state: &[u8]) -> serde_json::Value {
        if state.is_empty() {
            serde_json::json!({"messages": []})
        } else {
            serde_json::from_slice(state).unwrap_or_else(|_| serde_json::json!({"messages": []}))
        }
    }

    /// Serialize JSON state back to opaque bytes.
    fn encode_state(state: &serde_json::Value) -> Vec<u8> {
        serde_json::to_vec(state).unwrap_or_default()
    }

    /// Ensure the opaque state is an object with a messages array.
    fn ensure_state(state: &mut serde_json::Value) {
        if !state.is_object() {
            *state = serde_json::json!({"messages": []});
        }
        let obj = state.as_object_mut().unwrap();
        obj.entry("messages")
            .or_insert_with(|| serde_json::json!([]));
    }

    /// Push a proto message into the opaque state's messages array.
    fn push_message(state: &mut serde_json::Value, message: &Message) {
        Self::ensure_state(state);
        if let Ok(v) = serde_json::to_value(message) {
            state["messages"].as_array_mut().unwrap().push(v);
        }
    }

    /// Build an LLM request from the opaque worker state.
    fn build_llm_request(state: &serde_json::Value, ctx: &WorkerCtx) -> Option<LlmRequest> {
        let agent = ctx.agent.as_ref()?;

        // System prompt as first message
        let mut messages = vec![Message {
            role: Role::System as i32,
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

        Some(LlmRequest {
            model: agent.model.clone(),
            messages,
            tools: ctx.tools.clone(),
            temperature: agent.temperature,
            max_completion_tokens: agent.max_completion_tokens,
        })
    }

    /// Build annotated tool call actions from LLM tool calls.
    fn build_tool_call_actions(
        tool_calls: &[super::ToolCall],
        _ctx: &WorkerCtx,
    ) -> Vec<ToolCallAction> {
        tool_calls
            .iter()
            .map(|tc| ToolCallAction {
                tool_call: Some(tc.clone()),
                context: None,
            })
            .collect()
    }

    /// Build an LLM request action.
    fn request_llm(
        state: &serde_json::Value,
        ctx: &WorkerCtx,
        stream: bool,
    ) -> Vec<WorkerAction> {
        match Self::build_llm_request(state, ctx) {
            Some(request) => vec![WorkerAction {
                action: Some(worker_action::Action::RequestLlm(RequestLlm {
                    request: Some(request),
                    stream,
                })),
            }],
            None => vec![],
        }
    }
}

impl Worker for DefaultWorker {
    fn decide(
        &self,
        trigger: &DecisionTrigger,
        state: &[u8],
        ctx: &WorkerCtx,
    ) -> WorkerDecision {
        let mut state = Self::parse_state(state);

        let actions = match trigger.trigger.as_ref() {
            Some(decision_trigger::Trigger::UserMessage(um)) => {
                if let Some(ref message) = um.message {
                    Self::push_message(&mut state, message);
                }
                Self::request_llm(&state, ctx, um.stream)
            }
            Some(decision_trigger::Trigger::LlmCompleted(lc)) => {
                if let Some(ref message) = lc.message {
                    Self::push_message(&mut state, message);
                    if lc.truncated {
                        Self::request_llm(&state, ctx, ctx.stream)
                    } else if message.tool_calls.is_empty() {
                        let artifacts = match &message.content {
                            Some(text) if !text.is_empty() => vec![Artifact {
                                name: None,
                                description: None,
                                parts: vec![Part {
                                    kind: Some(part::Kind::Text(text.clone())),
                                }],
                            }],
                            _ => vec![],
                        };
                        vec![WorkerAction {
                            action: Some(worker_action::Action::Done(Done { artifacts })),
                        }]
                    } else {
                        vec![WorkerAction {
                            action: Some(worker_action::Action::RequestToolCalls(
                                RequestToolCalls {
                                    tool_calls: Self::build_tool_call_actions(
                                        &message.tool_calls,
                                        ctx,
                                    ),
                                },
                            )),
                        }]
                    }
                } else {
                    vec![]
                }
            }
            Some(decision_trigger::Trigger::LlmFailed(lf)) => {
                vec![WorkerAction {
                    action: Some(worker_action::Action::Done(Done {
                        artifacts: vec![Artifact {
                            name: None,
                            description: None,
                            parts: vec![Part {
                                kind: Some(part::Kind::Text(format!("Error: {}", lf.error))),
                            }],
                        }],
                    })),
                }]
            }
            Some(decision_trigger::Trigger::ToolResolved(tr)) => {
                if let Some(ref result) = tr.result {
                    let msg = Message {
                        role: Role::Tool as i32,
                        content: Some(result.content.clone()),
                        tool_calls: vec![],
                        tool_call_id: Some(result.tool_call_id.clone()),
                        call_id: None,
                        usage: None,
                    };
                    Self::push_message(&mut state, &msg);
                    // Only request next LLM call when all tools are done
                    let has_inflight = ctx.tool_call_statuses.values().any(|s| {
                        *s == CallStatus::Pending as i32
                            || *s == CallStatus::RetryScheduled as i32
                    });
                    if !has_inflight {
                        Self::request_llm(&state, ctx, ctx.stream)
                    } else {
                        vec![]
                    }
                } else {
                    vec![]
                }
            }
            Some(decision_trigger::Trigger::InterruptResumed(_)) => {
                Self::request_llm(&state, ctx, ctx.stream)
            }
            Some(decision_trigger::Trigger::Stall(_)) => {
                Self::request_llm(&state, ctx, ctx.stream)
            }
            None => vec![],
        };

        WorkerDecision {
            actions,
            state: Self::encode_state(&state),
        }
    }
}
