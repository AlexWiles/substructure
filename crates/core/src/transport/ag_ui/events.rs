use axum::response::sse::Event as SseEvent;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::protocol::ToolCall;

/// A `role` is a constant the client schema needs, not data. Deserialization
/// skips it and uses these.
fn assistant() -> &'static str {
    "assistant"
}

fn tool() -> &'static str {
    "tool"
}

fn reasoning() -> &'static str {
    "reasoning"
}

fn is_false(flag: &bool) -> bool {
    !*flag
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AgUiEvent {
    #[serde(rename = "RUN_STARTED", rename_all = "camelCase")]
    RunStarted { thread_id: String, run_id: String },

    #[serde(rename = "RUN_FINISHED", rename_all = "camelCase")]
    RunFinished {
        thread_id: String,
        run_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        result: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        outcome: Option<RunOutcome>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<Value>,
    },

    #[serde(rename = "RUN_ERROR")]
    RunError { message: String },

    #[serde(rename = "MESSAGES_SNAPSHOT", rename_all = "camelCase")]
    MessagesSnapshot { messages: Vec<SnapshotMessage> },

    #[serde(rename = "TEXT_MESSAGE_START", rename_all = "camelCase")]
    TextMessageStart {
        message_id: String,
        #[serde(skip_deserializing, default = "assistant")]
        role: &'static str,
    },

    #[serde(rename = "TEXT_MESSAGE_CONTENT", rename_all = "camelCase")]
    TextMessageContent { message_id: String, delta: String },

    #[serde(rename = "TEXT_MESSAGE_END", rename_all = "camelCase")]
    TextMessageEnd {
        message_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<Value>,
    },

    #[serde(rename = "TOOL_CALL_START", rename_all = "camelCase")]
    ToolCallStart {
        tool_call_id: String,
        tool_call_name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_message_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<Value>,
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
        #[serde(default, skip_serializing_if = "is_false")]
        is_error: bool,
        #[serde(default, skip_serializing_if = "is_false")]
        retryable: bool,
        #[serde(skip_deserializing, default = "tool")]
        role: &'static str,
    },

    #[serde(rename = "REASONING_START", rename_all = "camelCase")]
    ReasoningStart { message_id: String },

    #[serde(rename = "REASONING_MESSAGE_START", rename_all = "camelCase")]
    ReasoningMessageStart {
        message_id: String,
        /// Must be the literal `"reasoning"` — the client's Zod schema rejects
        /// the event otherwise.
        #[serde(skip_deserializing, default = "reasoning")]
        role: &'static str,
    },

    #[serde(rename = "REASONING_MESSAGE_CONTENT", rename_all = "camelCase")]
    ReasoningMessageContent { message_id: String, delta: String },

    #[serde(rename = "REASONING_MESSAGE_END", rename_all = "camelCase")]
    ReasoningMessageEnd { message_id: String },

    #[serde(rename = "REASONING_END", rename_all = "camelCase")]
    ReasoningEnd { message_id: String },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgUiUsage {
    pub input: u64,
    pub output: u64,
}

impl AgUiUsage {
    pub fn metadata(&self) -> Value {
        serde_json::json!({ "usage": self })
    }

    pub fn of(metadata: Option<&Value>) -> Option<Self> {
        serde_json::from_value(metadata?.get("usage")?.clone()).ok()
    }
}

/// `RUN_FINISHED.outcome` per the AG-UI interrupt-aware run lifecycle.
/// Omitted entirely for legacy normal completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum RunOutcome {
    Success,
    Interrupt { interrupts: Vec<AgUiInterrupt> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgUiInterrupt {
    pub id: String,
    pub reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

impl AgUiInterrupt {
    /// Lift the AG-UI spec fields out of an interrupt payload — the payload
    /// convention is the spec shape, so this is a plain field read.
    fn lift(interrupt_id: &str, reason: &str, payload: &Value) -> Self {
        let obj = payload.as_object();
        let take = |key: &str| obj.and_then(|o| o.get(key)).cloned();
        let take_str = |key: &str| take(key)?.as_str().map(str::to_string);
        Self {
            id: interrupt_id.to_string(),
            reason: reason.to_string(),
            message: take_str("message"),
            tool_call_id: take_str("toolCallId").or_else(|| take_str("tool_call_id")),
            response_schema: take("responseSchema"),
            expires_at: take_str("expiresAt"),
            metadata: take("metadata"),
        }
    }

    pub fn from_session(p: &crate::session::events::SessionInterrupted) -> Self {
        Self::lift(&p.interrupt_id, &p.reason, &p.payload)
    }

    pub fn from_open(p: &crate::session::state::OpenInterrupt) -> Self {
        Self::lift(&p.interrupt_id, &p.reason, &p.payload)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The CLI reads these events, so each variant must survive the trip.
    #[test]
    fn every_event_survives_the_round_trip() {
        let events = vec![
            AgUiEvent::RunStarted {
                thread_id: "s".into(),
                run_id: "r".into(),
            },
            AgUiEvent::RunFinished {
                thread_id: "s".into(),
                run_id: "r".into(),
                result: None,
                outcome: None,
                metadata: None,
            },
            AgUiEvent::RunError {
                message: "no".into(),
            },
            AgUiEvent::TextMessageStart {
                message_id: "m".into(),
                role: "assistant",
            },
            AgUiEvent::TextMessageContent {
                message_id: "m".into(),
                delta: "hi".into(),
            },
            AgUiEvent::TextMessageEnd {
                message_id: "m".into(),
                metadata: None,
            },
            AgUiEvent::ToolCallStart {
                tool_call_id: "t".into(),
                tool_call_name: "search".into(),
                parent_message_id: None,
                metadata: None,
            },
            AgUiEvent::ToolCallArgs {
                tool_call_id: "t".into(),
                delta: "{}".into(),
            },
            AgUiEvent::ToolCallEnd {
                tool_call_id: "t".into(),
            },
            AgUiEvent::ToolCallResult {
                message_id: "m".into(),
                tool_call_id: "t".into(),
                content: "ok".into(),
                is_error: false,
                retryable: false,
                role: "tool",
            },
            AgUiEvent::ReasoningMessageStart {
                message_id: "m".into(),
                role: "reasoning",
            },
        ];

        for event in events {
            let json = serde_json::to_string(&event).unwrap();
            let back: AgUiEvent = serde_json::from_str(&json)
                .unwrap_or_else(|e| panic!("{} did not survive: {e}\n{json}", event.type_name()));
            assert_eq!(
                back.type_name(),
                event.type_name(),
                "came back as a different event: {json}"
            );
            assert_eq!(
                serde_json::to_string(&back).unwrap(),
                json,
                "re-serialized differently"
            );
        }
    }

    #[test]
    fn a_skipped_role_comes_back_as_its_constant() {
        let json = r#"{"type":"TEXT_MESSAGE_START","messageId":"m","role":"assistant"}"#;
        match serde_json::from_str::<AgUiEvent>(json).unwrap() {
            AgUiEvent::TextMessageStart { role, .. } => assert_eq!(role, "assistant"),
            other => panic!("expected TEXT_MESSAGE_START, got {}", other.type_name()),
        }

        // Also when absent: deserialization does not read it.
        let json = r#"{"type":"REASONING_MESSAGE_START","messageId":"m"}"#;
        match serde_json::from_str::<AgUiEvent>(json).unwrap() {
            AgUiEvent::ReasoningMessageStart { role, .. } => assert_eq!(role, "reasoning"),
            other => panic!(
                "expected REASONING_MESSAGE_START, got {}",
                other.type_name()
            ),
        }
    }
}
