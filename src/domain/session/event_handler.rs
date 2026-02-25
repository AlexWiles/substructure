use crate::domain::event::*;
use crate::domain::openai;

use super::agent_session::{new_call_id, AgentSession};
use super::command_handler::CommandPayload;
use super::strategy::{extract_response_summary, Action, Turn};

// ---------------------------------------------------------------------------
// Effect types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Effect {
    Command(CommandPayload),
    CallLlm {
        call_id: String,
        request: LlmRequest,
        stream: bool,
    },
    CallMcpTool {
        tool_call_id: String,
        name: String,
        arguments: serde_json::Value,
    },
    StartMcpServers(Vec<McpServerConfig>),
}

// ---------------------------------------------------------------------------
// Response extraction helpers (re-export for external use)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// React — runtime reactor
// ---------------------------------------------------------------------------

impl AgentSession {
    pub async fn react(&self, tools: Option<Vec<openai::Tool>>, event: &Event) -> Vec<Effect> {
        let mut effects = Vec::new();

        // --- Mechanical work: infrastructure I/O and bookkeeping ---
        match &event.payload {
            EventPayload::SessionCreated(payload) => {
                if !payload.agent.mcp_servers.is_empty() {
                    effects.push(Effect::StartMcpServers(payload.agent.mcp_servers.clone()));
                }
            }

            EventPayload::LlmCallRequested(payload) => {
                let LlmRequest::OpenAi(mut oai_req) = payload.request.clone();
                oai_req.tools = tools.clone();
                effects.push(Effect::CallLlm {
                    call_id: payload.call_id.clone(),
                    request: LlmRequest::OpenAi(oai_req),
                    stream: payload.stream,
                });
            }

            EventPayload::ToolCallRequested(payload) => {
                let args: serde_json::Value =
                    serde_json::from_str(&payload.arguments).unwrap_or_default();
                effects.push(Effect::CallMcpTool {
                    tool_call_id: payload.tool_call_id.clone(),
                    name: payload.name.clone(),
                    arguments: args,
                });
            }

            EventPayload::ToolCallCompleted(payload) => {
                effects.push(Effect::Command(CommandPayload::SendToolMessage {
                    tool_call_id: payload.tool_call_id.clone(),
                    content: payload.result.clone(),
                    token_count: None,
                }));
            }

            EventPayload::ToolCallErrored(payload) => {
                effects.push(Effect::Command(CommandPayload::SendToolMessage {
                    tool_call_id: payload.tool_call_id.clone(),
                    content: format!("Error: {}", payload.error),
                    token_count: None,
                }));
            }

            EventPayload::LlmCallCompleted(payload) => {
                let summary = extract_response_summary(payload);
                effects.push(Effect::Command(CommandPayload::SendAssistantMessage {
                    call_id: summary.call_id,
                    content: summary.content,
                    tool_calls: summary.tool_calls,
                    token_count: summary.token_count,
                }));
            }

            _ => {}
        }

        // --- Consult strategy ---
        effects.extend(self.consult_strategy(tools, &event.payload).await);

        effects
    }

    /// Consult the strategy and translate its decision into effects.
    async fn consult_strategy(
        &self,
        tools: Option<Vec<openai::Tool>>,
        event: &EventPayload,
    ) -> Vec<Effect> {
        let strategy = match self.agent_state.strategy.as_ref() {
            Some(s) => s,
            None => return vec![],
        };
        match strategy.on_event(&self.agent_state, event).await {
            Some(turn) => self.apply_turn(turn, tools),
            None => vec![],
        }
    }

    /// Translate strategy Turn into Effects.
    fn apply_turn(&self, turn: Turn, tools: Option<Vec<openai::Tool>>) -> Vec<Effect> {
        let mut effects = Vec::new();
        if turn.state != self.agent_state.strategy_state {
            effects.push(Effect::Command(CommandPayload::UpdateStrategyState {
                state: turn.state,
            }));
        }
        if let Some(action) = turn.action {
            effects.extend(self.execute_action(action, tools));
        }
        effects
    }

    /// Translate a strategy Action into Effects.
    fn execute_action(&self, action: Action, tools: Option<Vec<openai::Tool>>) -> Vec<Effect> {
        match action {
            Action::CallLlm(params) => {
                let stream = params.stream.unwrap_or(true);
                let request = if let Some(ref context) = params.context {
                    self.agent_state
                        .build_llm_request_with_context(context, tools)
                } else {
                    self.agent_state.build_llm_request(tools)
                };
                match request {
                    Some(request) => vec![Effect::Command(CommandPayload::RequestLlmCall {
                        call_id: new_call_id(),
                        request,
                        stream,
                        deadline: self.llm_deadline(),
                    })],
                    None => vec![],
                }
            }
            Action::ExecuteTools(plan) => plan
                .calls
                .iter()
                .map(|tc| {
                    Effect::Command(CommandPayload::RequestToolCall {
                        tool_call_id: tc.id.clone(),
                        name: tc.name.clone(),
                        arguments: tc.arguments.clone(),
                        deadline: self.tool_deadline(),
                    })
                })
                .collect(),
            Action::Done => {
                vec![Effect::Command(CommandPayload::MarkDone)]
            }
            Action::Interrupt(req) => vec![Effect::Command(CommandPayload::Interrupt {
                interrupt_id: req.id,
                reason: req.reason,
                payload: req.payload,
            })],
        }
    }
}
