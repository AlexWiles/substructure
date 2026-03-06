//! Worker client — dispatches decisions and tool calls.
//!
//! Decisions are always enqueued into the shared DecisionQueue.
//! Tool execution is always local (MCP + sub-agents).

use std::collections::HashMap;
use std::sync::Arc;

use ractor::ActorRef;
use uuid::Uuid;

use crate::runtime::config::{AgentConfig, ClientIdentity};
use crate::runtime::decision_queue::DecisionQueue;
use crate::runtime::llm::{LlmTool, LlmToolFunction};
use crate::runtime::mcp::{Content, McpClient, ToolDefinition};
use crate::runtime::session::types::CompletionDelivery;
use crate::runtime::session::{
    truncate_tool_result, CommandPayload, ToolCallDispatch, WorkerDispatch, WorkerExecutor,
};
use crate::runtime::types::{RuntimeMessage, SubAgentRequest};
use crate::worker as proto;

// ---------------------------------------------------------------------------
// WorkerClient
// ---------------------------------------------------------------------------

/// Dispatches worker decisions and tool calls, delivering results back
/// to sessions via the runtime actor.
pub struct WorkerClient {
    queue: Arc<DecisionQueue>,
    runtime: ActorRef<RuntimeMessage>,
    mcp_clients: Vec<Arc<dyn McpClient>>,
    /// Pre-computed tool definitions (MCP + sub-agent tools).
    tools: Vec<LlmTool>,
    /// Sub-agent configs keyed by sanitized tool name.
    sub_agent_configs: HashMap<String, AgentConfig>,
    auth: ClientIdentity,
    stream: bool,
}

impl WorkerClient {
    /// Build a new client, pre-computing tool definitions from MCP clients
    /// and sub-agent configs from the agents map.
    pub fn new(
        queue: Arc<DecisionQueue>,
        runtime: ActorRef<RuntimeMessage>,
        mcp_clients: Vec<Arc<dyn McpClient>>,
        agent: &AgentConfig,
        agents: &HashMap<String, AgentConfig>,
        auth: ClientIdentity,
        stream: bool,
    ) -> Self {
        // MCP tools
        let mut tools: Vec<LlmTool> = mcp_clients
            .iter()
            .flat_map(|c| c.tools().iter().map(|t| t.to_tool()))
            .collect();

        // Sub-agent tools + configs
        let mut sub_agent_configs = HashMap::new();
        for name in &agent.sub_agents {
            if let Some(sub) = agents.get(name) {
                let tool_name = ToolDefinition::sanitized_name(name);
                tools.push(LlmTool {
                    tool_type: "function".to_string(),
                    function: LlmToolFunction {
                        name: tool_name.clone(),
                        description: sub.description.clone().unwrap_or_else(|| sub.name.clone()),
                        parameters: serde_json::json!({
                            "type": "object",
                            "properties": {
                                "message": {
                                    "type": "string",
                                    "description": "The message to send to the sub-agent"
                                }
                            },
                            "required": ["message"]
                        }),
                    },
                });
                sub_agent_configs.insert(tool_name, sub.clone());
            }
        }

        Self {
            queue,
            runtime,
            mcp_clients,
            tools,
            sub_agent_configs,
            auth,
            stream,
        }
    }

    /// Build a proto WorkerDispatch from an internal dispatch, including tools.
    fn build_proto_dispatch(&self, request: &WorkerDispatch) -> proto::WorkerDispatch {
        proto::WorkerDispatch {
            session_id: request.session_id.to_string(),
            decision_id: request.decision_id.clone(),
            trigger: Some((&request.trigger).into()),
            worker_state: request.worker_state.clone(),
            stream: request.stream,
            agent: Some((&request.agent).into()),
            token_usage: request
                .token_usage
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
            tool_call_statuses: request
                .tool_call_statuses
                .iter()
                .map(|(id, s)| {
                    let cs: proto::CallStatus = s.into();
                    (id.clone(), cs as i32)
                })
                .collect(),
            llm_call_statuses: request
                .llm_call_statuses
                .iter()
                .map(|(id, s)| {
                    let cs: proto::CallStatus = s.into();
                    (id.clone(), cs as i32)
                })
                .collect(),
            span_json: serde_json::to_string(&request.span).unwrap_or_default(),
            tools: self.tools.iter().map(Into::into).collect(),
            sub_agent_names: self.sub_agent_configs.keys().cloned().collect(),
        }
    }
}

impl WorkerExecutor for WorkerClient {
    fn dispatch_decision(&self, request: WorkerDispatch) {
        let proto_dispatch = self.build_proto_dispatch(&request);
        let queue = self.queue.clone();
        tokio::spawn(async move {
            queue.enqueue(proto_dispatch).await;
        });
    }

    fn dispatch_tool_call(&self, request: ToolCallDispatch) {
        // Sub-agent tool call — executor owns the configs
        if let Some(agent) = self.sub_agent_configs.get(&request.name) {
            let args: serde_json::Value =
                serde_json::from_str(&request.arguments).unwrap_or_default();
            let message = args
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let child_session_id =
                Uuid::new_v5(&request.session_id, request.tool_call_id.as_bytes());
            if let Err(e) = self
                .runtime
                .send_message(RuntimeMessage::RunSubAgent(SubAgentRequest {
                    session_id: child_session_id,
                    agent_name: agent.name.clone(),
                    message,
                    auth: self.auth.clone(),
                    delivery: CompletionDelivery {
                        parent_session_id: request.session_id,
                        tool_call_id: request.tool_call_id.clone(),
                        tool_name: request.name.clone(),
                        span: request.span.child("sub_agent.delivery"),
                    },
                    span: request.span.child("sub_agent.spawn"),
                    stream: self.stream,
                })) {
                tracing::warn!(error = %e, "failed to dispatch sub-agent request");
            }
            return;
        }

        // MCP tool call — async, so spawn a task
        let mcp = self
            .mcp_clients
            .iter()
            .find(|c| c.tools().iter().any(|t| t.name == request.name))
            .cloned();

        if let Some(mcp) = mcp {
            let runtime = self.runtime.clone();
            tokio::spawn(async move {
                let args: serde_json::Value =
                    serde_json::from_str(&request.arguments).unwrap_or_default();
                let payload = match mcp.call_tool(&request.name, args).await {
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
                        let text = truncate_tool_result(text, request.max_result_bytes);
                        if result.is_error {
                            CommandPayload::FailToolCall {
                                tool_call_id: request.tool_call_id,
                                name: request.name,
                                error: text,
                                worker_state: None,
                            }
                        } else {
                            CommandPayload::CompleteToolCall {
                                tool_call_id: request.tool_call_id,
                                name: request.name,
                                result: text,
                                worker_state: None,
                            }
                        }
                    }
                    Err(e) => CommandPayload::FailToolCall {
                        tool_call_id: request.tool_call_id,
                        name: request.name,
                        error: e.to_string(),
                        worker_state: None,
                    },
                };
                if let Err(e) = runtime.send_message(RuntimeMessage::DeliverToSession {
                    session_id: request.session_id,
                    payload,
                    span: request.span,
                }) {
                    tracing::warn!(error = %e, "failed to deliver tool call result to session");
                }
            });
            return;
        }

        // Unknown tool — fail immediately
        if let Err(e) = self.runtime.send_message(RuntimeMessage::DeliverToSession {
            session_id: request.session_id,
            payload: CommandPayload::FailToolCall {
                tool_call_id: request.tool_call_id.clone(),
                name: request.name.clone(),
                error: format!("unknown tool: {}", request.name),
                worker_state: None,
            },
            span: request.span,
        }) {
            tracing::warn!(error = %e, "failed to deliver unknown tool error to session");
        }
    }
}
