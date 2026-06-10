use axum::response::sse::Event as SseEvent;
use serde::Serialize;
use serde_json::Value;

use crate::session::message::ToolCall;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum AgUiEvent {
    #[serde(rename = "RUN_STARTED", rename_all = "camelCase")]
    RunStarted { thread_id: String, run_id: String },

    #[serde(rename = "RUN_FINISHED", rename_all = "camelCase")]
    RunFinished {
        thread_id: String,
        run_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        result: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        outcome: Option<RunOutcome>,
    },

    #[serde(rename = "RUN_ERROR")]
    RunError { message: String },

    #[serde(rename = "MESSAGES_SNAPSHOT", rename_all = "camelCase")]
    MessagesSnapshot { messages: Vec<SnapshotMessage> },

    #[serde(rename = "TEXT_MESSAGE_START", rename_all = "camelCase")]
    TextMessageStart {
        message_id: String,
        role: &'static str,
    },

    #[serde(rename = "TEXT_MESSAGE_CONTENT", rename_all = "camelCase")]
    TextMessageContent { message_id: String, delta: String },

    #[serde(rename = "TEXT_MESSAGE_END", rename_all = "camelCase")]
    TextMessageEnd { message_id: String },

    #[serde(rename = "TOOL_CALL_START", rename_all = "camelCase")]
    ToolCallStart {
        tool_call_id: String,
        tool_call_name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_message_id: Option<String>,
    },

    #[serde(rename = "TOOL_CALL_ARGS", rename_all = "camelCase")]
    ToolCallArgs { tool_call_id: String, delta: String },

    #[serde(rename = "TOOL_CALL_END", rename_all = "camelCase")]
    ToolCallEnd { tool_call_id: String },

    #[serde(rename = "TOOL_CALL_RESULT", rename_all = "camelCase")]
    ToolCallResult {
        message_id: String,
        tool_call_id: String,
        content: String,
        role: &'static str,
    },

    #[serde(rename = "REASONING_START", rename_all = "camelCase")]
    ReasoningStart { message_id: String },

    #[serde(rename = "REASONING_MESSAGE_START", rename_all = "camelCase")]
    ReasoningMessageStart {
        message_id: String,
        /// Must be the literal `"reasoning"` — the client's Zod schema rejects
        /// the event otherwise.
        role: &'static str,
    },

    #[serde(rename = "REASONING_MESSAGE_CONTENT", rename_all = "camelCase")]
    ReasoningMessageContent { message_id: String, delta: String },

    #[serde(rename = "REASONING_MESSAGE_END", rename_all = "camelCase")]
    ReasoningMessageEnd { message_id: String },

    #[serde(rename = "REASONING_END", rename_all = "camelCase")]
    ReasoningEnd { message_id: String },
}

/// `RUN_FINISHED.outcome` per the AG-UI interrupt-aware run lifecycle.
/// Omitted entirely for legacy normal completion.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum RunOutcome {
    Success,
    Interrupt { interrupts: Vec<AgUiInterrupt> },
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AgUiInterrupt {
    pub id: String,
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

impl AgUiInterrupt {
    /// Lift the AG-UI interrupt fields out of a session interrupt. The
    /// interrupt payload's `message`, `toolCallId`, `responseSchema`,
    /// `expiresAt` and `metadata` keys map onto the spec fields.
    pub fn from_session(p: &crate::session::events::SessionInterrupted) -> Self {
        let obj = p.payload.as_object();
        let take = |key: &str| obj.and_then(|o| o.get(key)).cloned();
        let take_str = |key: &str| {
            obj.and_then(|o| o.get(key))
                .and_then(|v| v.as_str())
                .map(str::to_string)
        };
        Self {
            id: p.interrupt_id.clone(),
            reason: p.reason.clone(),
            message: take_str("message"),
            tool_call_id: take_str("toolCallId"),
            response_schema: take("responseSchema"),
            expires_at: take_str("expiresAt"),
            metadata: take("metadata"),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum SnapshotMessage {
    System {
        id: String,
        content: String,
    },
    User {
        id: String,
        content: String,
    },
    #[serde(rename_all = "camelCase")]
    Assistant {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ToolCall>,
    },
    #[serde(rename_all = "camelCase")]
    Tool {
        id: String,
        tool_call_id: String,
        content: String,
    },
}

impl AgUiEvent {
    pub fn type_name(&self) -> &'static str {
        match self {
            AgUiEvent::RunStarted { .. } => "RUN_STARTED",
            AgUiEvent::RunFinished { .. } => "RUN_FINISHED",
            AgUiEvent::RunError { .. } => "RUN_ERROR",
            AgUiEvent::MessagesSnapshot { .. } => "MESSAGES_SNAPSHOT",
            AgUiEvent::TextMessageStart { .. } => "TEXT_MESSAGE_START",
            AgUiEvent::TextMessageContent { .. } => "TEXT_MESSAGE_CONTENT",
            AgUiEvent::TextMessageEnd { .. } => "TEXT_MESSAGE_END",
            AgUiEvent::ToolCallStart { .. } => "TOOL_CALL_START",
            AgUiEvent::ToolCallArgs { .. } => "TOOL_CALL_ARGS",
            AgUiEvent::ToolCallEnd { .. } => "TOOL_CALL_END",
            AgUiEvent::ToolCallResult { .. } => "TOOL_CALL_RESULT",
            AgUiEvent::ReasoningStart { .. } => "REASONING_START",
            AgUiEvent::ReasoningMessageStart { .. } => "REASONING_MESSAGE_START",
            AgUiEvent::ReasoningMessageContent { .. } => "REASONING_MESSAGE_CONTENT",
            AgUiEvent::ReasoningMessageEnd { .. } => "REASONING_MESSAGE_END",
            AgUiEvent::ReasoningEnd { .. } => "REASONING_END",
        }
    }

    pub fn to_sse(&self) -> SseEvent {
        let data = serde_json::to_string(self).unwrap_or_default();
        SseEvent::default().event(self.type_name()).data(data)
    }
}
