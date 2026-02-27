use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// RunAgentInput — the input payload to start a new agent run
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunAgentInput {
    /// Thread identifier — maps to session_id in the runtime.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
    /// Unique identifier for this run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    /// Conversation messages (context for the agent).
    #[serde(default)]
    pub messages: Vec<Message>,
    /// Tools the agent can use.
    #[serde(default)]
    pub tools: Vec<Tool>,
    /// Additional context items.
    #[serde(default)]
    pub context: Vec<Context>,
    /// Arbitrary state to pass through.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state: Option<serde_json::Value>,
    /// Forwarded properties from the caller.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub forwarded_props: serde_json::Map<String, serde_json::Value>,
    /// Resume a previously interrupted run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resume: Option<ResumeInfo>,
}

// ---------------------------------------------------------------------------
// Interrupt types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InterruptInfo {
    pub id: String,
    pub reason: String,
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResumeInfo {
    pub interrupt_id: String,
    pub payload: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "camelCase")]
pub enum Message {
    #[serde(rename = "user")]
    User {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        content: MessageContent,
    },
    #[serde(rename = "assistant")]
    Assistant {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty", rename = "toolCalls")]
        tool_calls: Vec<ToolCallInfo>,
    },
    #[serde(rename = "system")]
    System {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        content: String,
    },
    #[serde(rename = "tool")]
    Tool {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    #[serde(rename = "developer")]
    Developer {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        content: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<InputContent>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum InputContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageUrl {
    pub url: String,
}

// ---------------------------------------------------------------------------
// Tools & tool calls
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallInfo {
    pub id: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Context {
    pub description: String,
    pub value: serde_json::Value,
}

// ---------------------------------------------------------------------------
// AgUiEvent — the full AG-UI event protocol
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AgUiEvent {
    // -- Lifecycle --
    RunStarted {
        #[serde(rename = "threadId")]
        thread_id: String,
        #[serde(rename = "runId")]
        run_id: String,
    },
    RunFinished {
        #[serde(rename = "threadId")]
        thread_id: String,
        #[serde(rename = "runId")]
        run_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        outcome: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        interrupt: Option<InterruptInfo>,
    },
    RunError {
        message: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        code: Option<String>,
    },

    // -- Steps --
    StepStarted {
        #[serde(rename = "stepId")]
        step_id: String,
        #[serde(rename = "stepName")]
        step_name: String,
    },
    StepFinished {
        #[serde(rename = "stepId")]
        step_id: String,
        #[serde(rename = "stepName")]
        step_name: String,
    },

    // -- Text message streaming --
    TextMessageStart {
        #[serde(rename = "messageId")]
        message_id: String,
    },
    TextMessageContent {
        #[serde(rename = "messageId")]
        message_id: String,
        delta: String,
    },
    TextMessageEnd {
        #[serde(rename = "messageId")]
        message_id: String,
    },

    // -- Tool call streaming --
    ToolCallStart {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        #[serde(rename = "toolCallName")]
        tool_call_name: String,
        #[serde(skip_serializing_if = "Option::is_none", rename = "parentMessageId")]
        parent_message_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none", rename = "childSessionId")]
        child_session_id: Option<String>,
    },
    ToolCallArgs {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        delta: String,
    },
    ToolCallEnd {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
    },
    ToolCallResult {
        #[serde(rename = "messageId")]
        message_id: String,
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        role: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },

    // -- State management --
    StateSnapshot {
        snapshot: serde_json::Value,
    },
    StateDelta {
        delta: Vec<serde_json::Value>,
    },
    MessagesSnapshot {
        messages: Vec<Message>,
    },

    // -- Special --
    Raw {
        event: serde_json::Value,
    },
    Custom {
        name: String,
        value: serde_json::Value,
    },
}
