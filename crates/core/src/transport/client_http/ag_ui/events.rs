use serde::Serialize;
use serde_json::Value;

/// A single AG-UI protocol output event.
///
/// Serializes to the AG-UI wire shape — an object with a SCREAMING_SNAKE_CASE
/// `type` discriminator and camelCase fields — so the variant set *is* the
/// protocol surface this endpoint emits. Optional fields are omitted when
/// absent. The driver loop turns each value into one SSE frame via
/// [`AgUiEvent::type_name`] (the `event:` name) plus the serialized JSON (the
/// `data:` payload).
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
        /// The assistant message this tool call belongs to (the producing llm
        /// call). Gives the client a stable server message id from the moment
        /// the call appears; omitted when unknown.
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

    // Reasoning / thinking. Transient: streamed live for reasoning models, never
    // persisted, so it does not replay. A block nests as REASONING_START →
    // REASONING_MESSAGE_START → REASONING_MESSAGE_CONTENT* → REASONING_MESSAGE_END
    // → REASONING_END, mirroring the THINKING_* family.
    #[serde(rename = "REASONING_START", rename_all = "camelCase")]
    ReasoningStart { message_id: String },

    #[serde(rename = "REASONING_MESSAGE_START", rename_all = "camelCase")]
    ReasoningMessageStart {
        message_id: String,
        /// AG-UI requires this literal on the reasoning message — the client's
        /// Zod schema rejects the event without `role: "reasoning"`.
        role: &'static str,
    },

    #[serde(rename = "REASONING_MESSAGE_CONTENT", rename_all = "camelCase")]
    ReasoningMessageContent { message_id: String, delta: String },

    #[serde(rename = "REASONING_MESSAGE_END", rename_all = "camelCase")]
    ReasoningMessageEnd { message_id: String },

    #[serde(rename = "REASONING_END", rename_all = "camelCase")]
    ReasoningEnd { message_id: String },
}

impl AgUiEvent {
    /// The wire `type` discriminator, reused as the SSE `event:` name. Kept in
    /// sync with the serde `rename` on each variant.
    pub fn type_name(&self) -> &'static str {
        match self {
            AgUiEvent::RunStarted { .. } => "RUN_STARTED",
            AgUiEvent::RunFinished { .. } => "RUN_FINISHED",
            AgUiEvent::RunError { .. } => "RUN_ERROR",
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
}
