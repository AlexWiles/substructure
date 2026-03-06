use std::collections::HashMap;
use std::sync::Arc;

use ractor::ActorRef;
use uuid::Uuid;

use crate::runtime::config::{AgentConfig, ClientIdentity, WorkerConfig};
use crate::runtime::llm::{LlmTool, LlmToolFunction};
use crate::runtime::mcp::{Content, McpClient, ToolDefinition};
use crate::runtime::session::{truncate_tool_result, CommandPayload};
use crate::runtime::session::default_worker::DefaultWorker;
use crate::runtime::session::transport::{ToolCallDispatch, WorkerDispatch, WorkerExecutor};
use crate::runtime::session::types::CompletionDelivery;
use crate::runtime::session::worker::{Worker, WorkerCtx};
use crate::runtime::types::{RuntimeMessage, SubAgentRequest};

// ---------------------------------------------------------------------------
// Local worker transport — in-process executor
// ---------------------------------------------------------------------------

/// Executes worker decisions and tool calls in-process, delivering results
/// back via the runtime actor.
pub struct LocalWorkerExecutor {
    pub runtime: ActorRef<RuntimeMessage>,
    pub mcp_clients: Vec<Arc<dyn McpClient>>,
    /// Pre-computed tool definitions (MCP + sub-agent tools).
    pub tools: Vec<LlmTool>,
    /// Sub-agent configs keyed by sanitized tool name.
    pub sub_agent_configs: HashMap<String, AgentConfig>,
    pub auth: ClientIdentity,
    pub stream: bool,
}

impl LocalWorkerExecutor {
    /// Build a new transport, pre-computing tool definitions from MCP clients
    /// and sub-agent configs from the agents map.
    pub fn new(
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
            runtime,
            mcp_clients,
            tools,
            sub_agent_configs,
            auth,
            stream,
        }
    }
}

impl WorkerExecutor for LocalWorkerExecutor {
    fn dispatch_decision(&self, request: WorkerDispatch) {
        // Build full WorkerCtx from session data (dispatch) + executor's tools
        let ctx = WorkerCtx {
            session_id: request.session_id,
            stream: request.stream,
            agent: request.agent.clone(),
            all_tools: if self.tools.is_empty() {
                None
            } else {
                Some(self.tools.clone())
            },
            sub_agents: self.sub_agent_configs.clone(),
            token_usage: request.token_usage.clone(),
            tool_call_statuses: request.tool_call_statuses.clone(),
            llm_call_statuses: request.llm_call_statuses.clone(),
        };

        let worker = resolve_worker(&request.agent.worker);
        let decision = worker.decide(&request.trigger, &request.worker_state, &ctx);
        let _ = self.runtime.send_message(RuntimeMessage::DeliverToSession {
            session_id: request.session_id,
            payload: CommandPayload::SubmitWorkerDecision {
                decision_id: request.decision_id,
                actions: decision.actions,
                state: decision.state,
            },
            span: request.span,
        });
    }

    fn dispatch_tool_call(&self, request: ToolCallDispatch) {
        // Sub-agent tool call — context carries the full AgentConfig
        if let Ok(agent) = serde_json::from_value::<AgentConfig>(request.context.clone()) {
            let args: serde_json::Value =
                serde_json::from_str(&request.arguments).unwrap_or_default();
            let message = args
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let child_session_id =
                Uuid::new_v5(&request.session_id, request.tool_call_id.as_bytes());
            let _ = self
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
                }));
            return; // Sub-agent runs async, result arrives via CompleteToolCall
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
                let _ = runtime.send_message(RuntimeMessage::DeliverToSession {
                    session_id: request.session_id,
                    payload,
                    span: request.span,
                });
            });
            return;
        }

        // Unknown tool — fail immediately
        let _ = self.runtime.send_message(RuntimeMessage::DeliverToSession {
            session_id: request.session_id,
            payload: CommandPayload::FailToolCall {
                tool_call_id: request.tool_call_id.clone(),
                name: request.name.clone(),
                error: format!("unknown tool: {}", request.name),
                worker_state: None,
            },
            span: request.span,
        });
    }
}

// ---------------------------------------------------------------------------
// Worker resolution
// ---------------------------------------------------------------------------

/// Resolve a worker implementation from agent config.
pub fn resolve_worker(_config: &WorkerConfig) -> Box<dyn Worker> {
    // V1: only the default worker. Future: match on config.kind.
    Box::new(DefaultWorker)
}
