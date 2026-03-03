use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use ractor::ActorRef;
use uuid::Uuid;

use crate::domain::agent::AgentConfig;
use crate::domain::event::{ClientIdentity, LlmRequest, LlmResponse};
use crate::domain::session::{
    BudgetActorRef, LlmCallError, McpToolEntry, SessionContext, StreamDelta,
};

use super::budget;
use super::llm::{LlmClient, LlmClientProvider, StreamDelta as LlmStreamDelta};
use super::mcp::{Content, McpClient, ToolDefinition};
use super::routing::notify_observers;
use super::session_client::Notification;

// ---------------------------------------------------------------------------
// Build SessionContext — wires runtime resources into the domain context
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(super) fn build_session_context(
    session_id: Uuid,
    auth: &ClientIdentity,
    mcp_clients: &[Arc<dyn McpClient>],
    llm_provider: &Arc<dyn LlmClientProvider>,
    agents: &HashMap<String, AgentConfig>,
    agent: Option<&AgentConfig>,
    budget_actor: Option<ActorRef<budget::BudgetMessage>>,
    stream: bool,
) -> SessionContext {
    let mcp_tools: HashMap<String, McpToolEntry> = mcp_clients
        .iter()
        .flat_map(|c| {
            let info = c.server_info();
            let server_name = info.name.clone();
            let server_version = info.version.clone();
            c.tools().iter().map(move |t| {
                (
                    t.name.clone(),
                    McpToolEntry {
                        server_name: server_name.clone(),
                        server_version: server_version.clone(),
                    },
                )
            })
        })
        .collect();

    // Build all_tools from MCP + sub-agents + client tools (client tools added later via UpdateContext)
    let mut tools: Vec<crate::domain::openai::Tool> = mcp_clients
        .iter()
        .flat_map(|c| c.tools().iter().map(|t| t.to_openai_tool()))
        .collect();

    // Add sub-agent tools
    if let Some(agent) = agent {
        for name in &agent.sub_agents {
            if let Some(sub) = agents.get(name) {
                let tool_name = ToolDefinition::sanitized_name(name);
                tools.push(crate::domain::openai::Tool {
                    tool_type: "function".to_string(),
                    function: crate::domain::openai::ToolFunction {
                        name: tool_name,
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
            }
        }
    }

    let all_tools = if tools.is_empty() { None } else { Some(tools) };

    // Wrap the LlmClientProvider in our domain-level trait
    let llm_adapter: Arc<dyn crate::domain::session::LlmClientTrait> =
        Arc::new(LlmProviderAdapter(llm_provider.clone()));

    // Wrap MCP clients
    let mcp_adapters: Vec<Arc<dyn crate::domain::session::McpClientTrait>> = mcp_clients
        .iter()
        .map(|c| Arc::new(McpClientAdapter(c.clone())) as Arc<dyn crate::domain::session::McpClientTrait>)
        .collect();

    // Wrap budget actor
    let budget_ref = budget_actor.map(|a| BudgetActorRef {
        inner: Box::new(a),
    });

    // Build callbacks
    let notify_chunk: crate::domain::session::NotifyChunkFn = Arc::new(
        |session_id, call_id, chunk_index, text, span| {
            notify_observers(
                session_id,
                Arc::new(Notification::LlmStreamChunk {
                    call_id,
                    chunk_index,
                    text,
                    span,
                }),
            );
        },
    );

    // send_to_session is set below after we have the runtime ref

    SessionContext {
        mcp_tools,
        all_tools,
        session_id,
        auth: auth.clone(),
        stream,
        llm_provider: Some(llm_adapter),
        mcp_clients: mcp_adapters,
        agents: agents.clone(),
        client_tools: Vec::new(),
        budget_actor: budget_ref,
        notify_chunk: Some(notify_chunk),
        send_to_session: None, // set below after we have runtime ref
        spawn_sub_agent: None, // set below after we have runtime ref
    }
}

// ---------------------------------------------------------------------------
// Adapter: LlmClientProvider → domain LlmClientTrait
// ---------------------------------------------------------------------------

struct LlmProviderAdapter(Arc<dyn LlmClientProvider>);

#[async_trait]
impl crate::domain::session::LlmClientTrait for LlmProviderAdapter {
    async fn resolve(&self, client_id: &str, auth: &ClientIdentity) -> Result<Arc<dyn crate::domain::session::LlmCallable>, String> {
        let client = self
            .0
            .resolve(client_id, auth)
            .await
            .map_err(|e| e.to_string())?;
        Ok(Arc::new(LlmClientAdapter(client)) as Arc<dyn crate::domain::session::LlmCallable>)
    }
}

struct LlmClientAdapter(Arc<dyn LlmClient>);

#[async_trait]
impl crate::domain::session::LlmCallable for LlmClientAdapter {
    async fn call(&self, request: &LlmRequest) -> Result<LlmResponse, LlmCallError> {
        self.0.call(request).await.map_err(|e| {
            LlmCallError {
                message: e.message,
                retryable: e.retryable,
                source: serde_json::to_value(&e.source).ok(),
            }
        })
    }

    async fn call_streaming(&self, request: &LlmRequest, tx: tokio::sync::mpsc::UnboundedSender<StreamDelta>) -> Result<LlmResponse, LlmCallError> {
        let (bridge_tx, mut bridge_rx) =
            tokio::sync::mpsc::unbounded_channel::<LlmStreamDelta>();

        let forward = tokio::spawn(async move {
            while let Some(delta) = bridge_rx.recv().await {
                let _ = tx.send(StreamDelta {
                    text: delta.text,
                });
            }
        });

        let result = self.0.call_streaming(request, bridge_tx).await.map_err(|e| {
            LlmCallError {
                message: e.message,
                retryable: e.retryable,
                source: serde_json::to_value(&e.source).ok(),
            }
        });

        forward.abort();
        result
    }
}

// ---------------------------------------------------------------------------
// Adapter: McpClient → domain McpClientTrait
// ---------------------------------------------------------------------------

struct McpClientAdapter(Arc<dyn McpClient>);

#[async_trait]
impl crate::domain::session::McpClientTrait for McpClientAdapter {
    fn server_info(&self) -> crate::domain::session::McpServerInfo {
        let info = self.0.server_info();
        crate::domain::session::McpServerInfo {
            name: info.name.clone(),
            version: info.version.clone(),
        }
    }

    fn tools(&self) -> Vec<crate::domain::session::McpToolDefinition> {
        self.0
            .tools()
            .iter()
            .map(|t| crate::domain::session::McpToolDefinition {
                name: t.name.clone(),
            })
            .collect()
    }

    async fn call_tool(&self, name: &str, arguments: serde_json::Value) -> Result<crate::domain::session::McpToolResult, String> {
        let result = self.0.call_tool(name, arguments).await.map_err(|e| e.to_string())?;
        Ok(crate::domain::session::McpToolResult {
            content: result
                .content
                .iter()
                .map(|c| match c {
                    Content::Text { text } => {
                        crate::domain::session::McpToolContent::Text { text: text.clone() }
                    }
                    _ => crate::domain::session::McpToolContent::Other,
                })
                .collect(),
            is_error: result.is_error,
        })
    }
}
