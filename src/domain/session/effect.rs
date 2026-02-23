use chrono::{DateTime, Utc};

use crate::domain::event::{LlmRequest, McpServerConfig};
use super::command::CommandPayload;

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
    ScheduleWake {
        wake_at: DateTime<Utc>,
    },
}
