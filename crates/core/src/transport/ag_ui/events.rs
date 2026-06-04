use axum::response::sse::Event as SseEvent;
use serde::Serialize;
use serde_json::Value;

use crate::session::message::ToolCall;

/// A single AG-UI output event. Serializes to the wire shape: a
/// SCREAMING_SNAKE_CASE `type` discriminator plus camelCase fields, with absent
/// optionals omitted.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum AgUiEvent {
    #[serde(rename = "RUN_STARTED", rename_all = "camelCase")]
    RunStarted { thread_id: String, run_id: String },

    #[serde(rename = "RUN_FINISHED", rename_all = "camelCase")]
    RunFinished {
        thread_id: String,
        run_id: String,
        /// The turn result, when the engine produced one.
        #[serde(skip_serializing_if = "Option::is_none")]
        result: Option<Value>,
    },

    #[serde(rename = "RUN_ERROR")]
    RunError { message: String },

    /// Full conversation history, applied wholesale to the client's message list
    /// — used to hydrate a thread on (re)connect.
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
        /// The producing llm call, giving the client a stable assistant message
        /// id; omitted when unknown.
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

    // Reasoning blocks nest as REASONING_START → REASONING_MESSAGE_START →
    // REASONING_MESSAGE_CONTENT* → REASONING_MESSAGE_END → REASONING_END.
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

/// One message in a [`AgUiEvent::MessagesSnapshot`], matching AG-UI's role-tagged
/// message union. Reconstructed from the session's persisted `message.new` events.
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
        /// Serializes as `[{ id, type: "function", function: { name, arguments } }]`.
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
    /// The wire `type` discriminator, reused as the SSE `event:` name. Kept in
    /// sync with the serde `rename` on each variant.
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

    /// Serialize to an SSE frame: the JSON body as `data:`, the type name as the
    /// (debug-friendly) `event:` name.
    pub fn to_sse(&self) -> SseEvent {
        let data = serde_json::to_string(self).unwrap_or_default();
        SseEvent::default().event(self.type_name()).data(data)
    }
}
