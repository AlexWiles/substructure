use serde::{Deserialize, Serialize};

use super::message::Message;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub name: String,
    pub content: String,
    pub is_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionTrigger {
    UserMessage {
        stream: bool,
        message: Message,
    },
    LlmCompleted {
        call_id: String,
        message: Message,
        /// True when finish_reason was "length" (output truncated).
        truncated: bool,
    },
    LlmFailed {
        call_id: String,
        error: String,
    },
    ToolResolved {
        result: ToolResult,
    },
    InterruptResumed {
        interrupt_id: String,
    },
    Stall,
}
