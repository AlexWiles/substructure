use async_trait::async_trait;
use ractor::{call_t, Actor, ActorCell, ActorProcessingErr, ActorRef, RpcReplyPort, SpawnErr};

use crate::domain::event::{McpServerConfig, McpTransportConfig};

use super::client::{
    CallToolResult, McpClient, McpError, ServerCapabilities, ServerInfo, ToolDefinition,
};
use super::stdio::StdioMcpClient;

// ---------------------------------------------------------------------------
// Actor naming
// ---------------------------------------------------------------------------

pub fn mcp_actor_name(agent_name: &str, server_name: &str) -> String {
    format!("mcp-{agent_name}-{server_name}")
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

pub enum McpMessage {
    CallTool(
        String,
        serde_json::Value,
        RpcReplyPort<Result<CallToolResult, McpError>>,
    ),
    GetMetadata(RpcReplyPort<McpMetadata>),
}

#[derive(Clone)]
pub struct McpMetadata {
    pub tools: Vec<ToolDefinition>,
    pub server_info: ServerInfo,
    pub capabilities: ServerCapabilities,
    pub instructions: Option<String>,
}

// ---------------------------------------------------------------------------
// McpActor — owns a StdioMcpClient, serializes access via actor mailbox
// ---------------------------------------------------------------------------

pub struct McpActor;

pub struct McpActorArgs {
    pub config: McpServerConfig,
}

pub struct McpActorState {
    client: StdioMcpClient,
}

impl Actor for McpActor {
    type Msg = McpMessage;
    type State = McpActorState;
    type Arguments = McpActorArgs;

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        let config = &args.config;
        match &config.transport {
            McpTransportConfig::Stdio { command, args } => {
                let client = StdioMcpClient::new(command, args)
                    .await
                    .map_err(|e| format!("MCP '{}' failed to start: {e}", config.name))?;
                Ok(McpActorState { client })
            }
        }
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match message {
            McpMessage::CallTool(name, arguments, reply) => {
                let result = state.client.call_tool(&name, arguments).await;
                let _ = reply.send(result);
            }
            McpMessage::GetMetadata(reply) => {
                let metadata = McpMetadata {
                    tools: state.client.tools().to_vec(),
                    server_info: state.client.server_info().clone(),
                    capabilities: state.client.capabilities().clone(),
                    instructions: state.client.instructions().map(String::from),
                };
                let _ = reply.send(metadata);
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// McpActorClient — implements McpClient by delegating to an McpActor
// ---------------------------------------------------------------------------

pub struct McpActorClient {
    actor: ActorRef<McpMessage>,
    tools: Vec<ToolDefinition>,
    server_info: ServerInfo,
    capabilities: ServerCapabilities,
    instructions: Option<String>,
}

impl McpActorClient {
    /// Create a client wrapping an existing actor, fetching metadata from it.
    pub async fn from_actor(actor: ActorRef<McpMessage>) -> Result<Self, McpError> {
        let metadata = call_t!(actor, McpMessage::GetMetadata, 10_000)
            .map_err(|e| McpError::Transport(format!("failed to get MCP metadata: {e}")))?;
        Ok(Self {
            actor,
            tools: metadata.tools,
            server_info: metadata.server_info,
            capabilities: metadata.capabilities,
            instructions: metadata.instructions,
        })
    }
}

#[async_trait]
impl McpClient for McpActorClient {
    fn tools(&self) -> &[ToolDefinition] {
        &self.tools
    }

    fn capabilities(&self) -> &ServerCapabilities {
        &self.capabilities
    }

    fn server_info(&self) -> &ServerInfo {
        &self.server_info
    }

    fn instructions(&self) -> Option<&str> {
        self.instructions.as_deref()
    }

    async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<CallToolResult, McpError> {
        call_t!(
            self.actor,
            McpMessage::CallTool,
            120_000,
            name.to_string(),
            arguments
        )
        .map_err(|e| McpError::Transport(format!("actor call failed: {e}")))?
    }

    async fn refresh_tools(&mut self) -> Result<&[ToolDefinition], McpError> {
        let metadata = call_t!(self.actor, McpMessage::GetMetadata, 10_000)
            .map_err(|e| McpError::Transport(format!("actor call failed: {e}")))?;
        self.tools = metadata.tools;
        Ok(&self.tools)
    }

    async fn shutdown(&mut self) -> Result<(), McpError> {
        self.actor.stop(None);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Spawning
// ---------------------------------------------------------------------------

pub async fn spawn_mcp_actor(
    agent_name: &str,
    config: McpServerConfig,
    supervisor: ActorCell,
) -> Result<ActorRef<McpMessage>, SpawnErr> {
    let actor_name = mcp_actor_name(agent_name, &config.name);
    let (actor_ref, _) = Actor::spawn_linked(
        Some(actor_name),
        McpActor,
        McpActorArgs { config },
        supervisor,
    )
    .await?;
    Ok(actor_ref)
}
