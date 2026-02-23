use crate::domain::event::{LlmRequest, McpServerConfig};
use super::command::SessionCommand;

#[derive(Debug, Clone)]
pub enum Effect {
    Command(SessionCommand),
    CallLlm {
        call_id: String,
        request: LlmRequest,
    },
    CallMcpTool {
        tool_call_id: String,
        name: String,
        arguments: serde_json::Value,
    },
    StartMcpServers(Vec<McpServerConfig>),
}
