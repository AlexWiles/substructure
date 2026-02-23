use uuid::Uuid;

use crate::domain::event::*;
use super::state::SessionState;
use super::command::CommandPayload;
use super::effect::Effect;

fn new_call_id() -> String {
    Uuid::new_v4().to_string()
}

pub fn react(
    session_id: Uuid,
    state: &mut SessionState,
    tools: Option<Vec<crate::domain::openai::Tool>>,
    event: &Event,
) -> Vec<Effect> {
    match &event.payload {
        EventPayload::SessionCreated(payload) => {
            if payload.agent.mcp_servers.is_empty() {
                vec![]
            } else {
                vec![Effect::StartMcpServers(payload.agent.mcp_servers.clone())]
            }
        }

        EventPayload::MessageUser(_) => {
            if state.active_llm_call().is_some() {
                println!(
                    "[session:{}] MessageUser -> LLM call already pending, dirty",
                    session_id
                );
                vec![]
            } else {
                match state.build_llm_request(tools) {
                    Some(request) => {
                        println!(
                            "[session:{}] MessageUser -> requesting LLM call",
                            session_id
                        );
                        vec![Effect::Command(CommandPayload::RequestLlmCall {
                            call_id: new_call_id(),
                            request,
                            stream: state.stream,
                        })]
                    }
                    None => vec![],
                }
            }
        }

        EventPayload::LlmCallRequested(payload) => {
            println!(
                "[session:{}] LlmCallRequested [{}] -> calling LLM client",
                session_id, payload.call_id,
            );
            vec![Effect::CallLlm {
                call_id: payload.call_id.clone(),
                request: payload.request.clone(),
                stream: payload.stream,
            }]
        }

        EventPayload::LlmCallCompleted(payload) => {
            if state.dirty {
                println!(
                    "[session:{}] LlmCallCompleted [{}] -> stale (dirty), re-triggering",
                    session_id, payload.call_id,
                );
                state.dirty = false;
                match state.build_llm_request(tools) {
                    Some(request) => {
                        vec![Effect::Command(CommandPayload::RequestLlmCall {
                            call_id: new_call_id(),
                            request,
                            stream: state.stream,
                        })]
                    }
                    None => vec![],
                }
            } else {
                let (content, tool_calls, token_count) =
                    extract_assistant_message(&payload.response);

                println!(
                    "[session:{}] LlmCallCompleted [{}] -> sending assistant message",
                    session_id, payload.call_id,
                );

                let mut effects =
                    vec![Effect::Command(CommandPayload::SendAssistantMessage {
                        call_id: payload.call_id.clone(),
                        content,
                        tool_calls: tool_calls.clone(),
                        token_count,
                    })];

                for tc in &tool_calls {
                    effects.push(Effect::Command(CommandPayload::RequestToolCall {
                        tool_call_id: tc.id.clone(),
                        name: tc.name.clone(),
                        arguments: tc.arguments.clone(),
                    }));
                }

                effects
            }
        }

        EventPayload::LlmCallErrored(payload) => {
            if state.dirty {
                println!(
                    "[session:{}] LlmCallErrored [{}] -> dirty, re-triggering",
                    session_id, payload.call_id,
                );
                state.dirty = false;
                match state.build_llm_request(tools) {
                    Some(request) => {
                        vec![Effect::Command(CommandPayload::RequestLlmCall {
                            call_id: new_call_id(),
                            request,
                            stream: state.stream,
                        })]
                    }
                    None => vec![],
                }
            } else {
                println!(
                    "[session:{}] LlmCallErrored [{}] -> no action",
                    session_id, payload.call_id,
                );
                vec![]
            }
        }

        EventPayload::ToolCallRequested(payload) => {
            let args: serde_json::Value =
                serde_json::from_str(&payload.arguments).unwrap_or_default();
            println!(
                "[session:{}] ToolCallRequested [{}] -> dispatching MCP tool '{}'",
                session_id, payload.tool_call_id, payload.name,
            );
            vec![Effect::CallMcpTool {
                tool_call_id: payload.tool_call_id.clone(),
                name: payload.name.clone(),
                arguments: args,
            }]
        }

        EventPayload::ToolCallCompleted(payload) => {
            println!(
                "[session:{}] ToolCallCompleted [{}] -> sending tool message",
                session_id, payload.tool_call_id,
            );
            vec![Effect::Command(CommandPayload::SendToolMessage {
                tool_call_id: payload.tool_call_id.clone(),
                content: payload.result.clone(),
                token_count: None,
            })]
        }

        EventPayload::ToolCallErrored(payload) => {
            println!(
                "[session:{}] ToolCallErrored [{}] -> sending tool error message",
                session_id, payload.tool_call_id,
            );
            vec![Effect::Command(CommandPayload::SendToolMessage {
                tool_call_id: payload.tool_call_id.clone(),
                content: format!("Error: {}", payload.error),
                token_count: None,
            })]
        }

        EventPayload::MessageTool(_) => {
            if state.pending_tool_results == 0 {
                println!(
                    "[session:{}] MessageTool -> all tool results in, re-triggering LLM",
                    session_id,
                );
                match state.build_llm_request(tools) {
                    Some(request) => vec![Effect::Command(CommandPayload::RequestLlmCall {
                        call_id: new_call_id(),
                        stream: true,
                        request,
                    })],
                    None => vec![],
                }
            } else {
                println!(
                    "[session:{}] MessageTool -> {} tool results still pending",
                    session_id, state.pending_tool_results,
                );
                vec![]
            }
        }

        _ => vec![],
    }
}

pub fn extract_assistant_message(
    response: &LlmResponse,
) -> (Option<String>, Vec<ToolCall>, Option<u32>) {
    match response {
        LlmResponse::OpenAi(resp) => {
            let choice = &resp.choices[0];
            let content = choice.message.content.clone();
            let tool_calls = choice
                .message
                .tool_calls
                .as_ref()
                .map(|tcs| {
                    tcs.iter()
                        .map(|tc| ToolCall {
                            id: tc.id.clone(),
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.clone(),
                        })
                        .collect()
                })
                .unwrap_or_default();
            let token_count = resp.usage.as_ref().map(|u| u.total_tokens);
            (content, tool_calls, token_count)
        }
    }
}
