//! Default worker — simple LLM agent loop using proto types.
//!
//! Executes tool calls via MCP clients. Sub-agent tool calls are detected
//! by name and returned as `RequestSubAgent` actions for the runtime to handle.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::runtime::config::{AgentConfig, SecretProviderConfig};
use crate::runtime::mcp::{Content, McpClient, ToolDefinition};

use super::{
    decision_trigger, part, tool_call_result, worker_action, Artifact, CallStatus, DecisionTrigger,
    Done, LlmRequest, Message, Part, RequestLlm, RequestSubAgent, RequestToolCalls, Role,
    ToolCallAction, ToolCallDispatch, ToolCallResult, Worker, WorkerAction, WorkerCtx,
    WorkerDecision,
};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DefaultWorkerConfig {
    #[serde(default)]
    pub secret_providers: HashMap<String, SecretProviderConfig>,
    #[serde(default)]
    pub agents: HashMap<String, AgentConfig>,
}

pub struct DefaultWorker {
    agents: HashMap<String, AgentConfig>,
    mcp_clients: Vec<Arc<dyn McpClient>>,
}

impl std::fmt::Debug for DefaultWorker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DefaultWorker")
            .field("agents", &self.agents.keys().collect::<Vec<_>>())
            .field("mcp_clients", &self.mcp_clients.len())
            .finish()
    }
}

impl DefaultWorker {
    pub fn new(config: DefaultWorkerConfig, mcp_clients: Vec<Arc<dyn McpClient>>) -> Self {
        Self {
            agents: config.agents,
            mcp_clients,
        }
    }

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

    /// Build the tool list from MCP clients and agent's sub-agent names.
    fn build_tools(&self, agent: &AgentConfig) -> Vec<super::LlmTool> {
        // MCP tools
        let mut tools: Vec<super::LlmTool> = self
            .mcp_clients
            .iter()
            .flat_map(|c| {
                c.tools().iter().map(|t| {
                    let lt = t.to_tool();
                    super::LlmTool::from(&lt)
                })
            })
            .collect();

        // Sub-agent tools
        for name in &agent.sub_agents {
            let sanitized = ToolDefinition::sanitized_name(name);
            tools.push(super::LlmTool {
                tool_type: "function".to_string(),
                function: Some(super::LlmToolFunction {
                    name: sanitized.clone(),
                    description: name.clone(),
                    parameters: serde_json::from_value(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "The message to send to the sub-agent"
                            }
                        },
                        "required": ["message"]
                    }))
                    .ok(),
                }),
            });
        }

        tools
    }

    /// Build an LLM request from the opaque worker state.
    fn build_llm_request(
        &self,
        state: &serde_json::Value,
        agent: &AgentConfig,
    ) -> Option<LlmRequest> {
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
            model: agent.llm.model.clone(),
            messages,
            tools: self.build_tools(agent),
            temperature: agent.llm.temperature,
            max_completion_tokens: agent.llm.max_completion_tokens,
        })
    }

    /// Build worker actions from LLM tool calls, splitting sub-agents from regular tools.
    fn build_tool_actions(
        tool_calls: &[super::ToolCall],
        agent: &AgentConfig,
    ) -> Vec<WorkerAction> {
        let sub_agent_names: Vec<String> = agent
            .sub_agents
            .iter()
            .map(|n| ToolDefinition::sanitized_name(n))
            .collect();

        let mut regular = Vec::new();
        let mut actions = Vec::new();

        for tc in tool_calls {
            if sub_agent_names.contains(&tc.name) {
                // Sub-agent: extract message from arguments, return RequestSubAgent
                let args: serde_json::Value =
                    serde_json::from_str(&tc.arguments).unwrap_or_default();
                let message = args
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                actions.push(WorkerAction {
                    action: Some(worker_action::Action::RequestSubAgent(RequestSubAgent {
                        agent_name: tc.name.clone(),
                        message,
                    })),
                });
            } else {
                regular.push(ToolCallAction {
                    tool_call: Some(tc.clone()),
                    context: None,
                    timeout_secs: None,
                    max_retries: None,
                });
            }
        }

        if !regular.is_empty() {
            actions.push(WorkerAction {
                action: Some(worker_action::Action::RequestToolCalls(RequestToolCalls {
                    tool_calls: regular,
                })),
            });
        }

        actions
    }

    /// Build an LLM request action.
    fn request_llm(
        &self,
        state: &serde_json::Value,
        agent: &AgentConfig,
        stream: bool,
    ) -> Vec<WorkerAction> {
        match self.build_llm_request(state, agent) {
            Some(request) => vec![WorkerAction {
                action: Some(worker_action::Action::RequestLlm(RequestLlm {
                    request: Some(request),
                    stream,
                    llm_client: agent.llm.client.clone(),
                    timeout_secs: agent.llm.retry.timeout_secs,
                    max_retries: agent.llm.retry.max_retries,
                })),
            }],
            None => vec![],
        }
    }

    /// Look up internal agent config by name.
    fn agent(&self, ctx: &WorkerCtx) -> Option<&AgentConfig> {
        self.agents.get(&ctx.agent_name)
    }

    /// Find the MCP client that owns a tool by name.
    fn find_mcp_client(&self, tool_name: &str) -> Option<&Arc<dyn McpClient>> {
        self.mcp_clients
            .iter()
            .find(|c| c.tools().iter().any(|t| t.name == tool_name))
    }
}

#[async_trait::async_trait]
impl Worker for DefaultWorker {
    fn agent_names(&self) -> Vec<String> {
        self.agents.keys().cloned().collect()
    }

    fn decide(&self, trigger: &DecisionTrigger, state: &[u8], ctx: &WorkerCtx) -> WorkerDecision {
        let mut state = Self::parse_state(state);

        let agent = match self.agent(ctx) {
            Some(a) => a,
            None => {
                return WorkerDecision {
                    actions: vec![WorkerAction {
                        action: Some(worker_action::Action::Done(Done {
                            artifacts: vec![Artifact {
                                name: None,
                                description: None,
                                parts: vec![Part {
                                    kind: Some(part::Kind::Text(format!(
                                        "Error: unknown agent '{}'",
                                        ctx.agent_name
                                    ))),
                                }],
                            }],
                        })),
                    }],
                    state: Self::encode_state(&state),
                };
            }
        };

        let actions = match trigger.trigger.as_ref() {
            Some(decision_trigger::Trigger::UserMessage(um)) => {
                if let Some(ref message) = um.message {
                    Self::push_message(&mut state, message);
                }
                self.request_llm(&state, agent, um.stream)
            }
            Some(decision_trigger::Trigger::LlmCompleted(lc)) => {
                if let Some(ref message) = lc.message {
                    Self::push_message(&mut state, message);
                    if lc.truncated {
                        self.request_llm(&state, agent, ctx.stream)
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
                        Self::build_tool_actions(&message.tool_calls, agent)
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
                        *s == CallStatus::Pending as i32 || *s == CallStatus::RetryScheduled as i32
                    });
                    if !has_inflight {
                        self.request_llm(&state, agent, ctx.stream)
                    } else {
                        vec![]
                    }
                } else {
                    vec![]
                }
            }
            Some(decision_trigger::Trigger::InterruptResumed(_)) => {
                self.request_llm(&state, agent, ctx.stream)
            }
            Some(decision_trigger::Trigger::Stall(_)) => {
                self.request_llm(&state, agent, ctx.stream)
            }
            None => vec![],
        };

        WorkerDecision {
            actions,
            state: Self::encode_state(&state),
        }
    }

    async fn execute_tool_call(&self, dispatch: &ToolCallDispatch) -> ToolCallResult {
        let Some(mcp) = self.find_mcp_client(&dispatch.name) else {
            return ToolCallResult {
                tool_call_id: dispatch.tool_call_id.clone(),
                name: dispatch.name.clone(),
                outcome: Some(tool_call_result::Outcome::Error(format!(
                    "unknown tool: {}",
                    dispatch.name
                ))),
                worker_state: None,
            };
        };

        let args: serde_json::Value = serde_json::from_str(&dispatch.arguments).unwrap_or_default();

        match mcp.call_tool(&dispatch.name, args).await {
            Ok(result) => {
                let text = result
                    .content
                    .iter()
                    .filter_map(|c| match c {
                        Content::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                if result.is_error {
                    ToolCallResult {
                        tool_call_id: dispatch.tool_call_id.clone(),
                        name: dispatch.name.clone(),
                        outcome: Some(tool_call_result::Outcome::Error(text)),
                        worker_state: None,
                    }
                } else {
                    ToolCallResult {
                        tool_call_id: dispatch.tool_call_id.clone(),
                        name: dispatch.name.clone(),
                        outcome: Some(tool_call_result::Outcome::Result(text)),
                        worker_state: None,
                    }
                }
            }
            Err(e) => ToolCallResult {
                tool_call_id: dispatch.tool_call_id.clone(),
                name: dispatch.name.clone(),
                outcome: Some(tool_call_result::Outcome::Error(e.to_string())),
                worker_state: None,
            },
        }
    }
}
