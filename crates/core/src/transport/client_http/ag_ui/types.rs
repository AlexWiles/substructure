use serde::Deserialize;

use crate::session::message::{Content, Message, Role};

/// AG-UI [`RunAgentInput`](https://docs.ag-ui.com). The client posts the full
/// conversation; fields beyond `threadId`/`runId`/`messages` are accepted for
/// protocol compatibility but unused here (`state`, `tools`, `context`, and
/// `forwardedProps` back deferred shared-state and dynamic frontend-tool work —
/// the basic frontend-tool flow declares the tool on the worker instead).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunAgentInput {
    pub thread_id: String,
    pub run_id: String,
    #[serde(default)]
    pub messages: Vec<AgUiMessage>,
    // Accepted for protocol compatibility; not yet read.
    #[serde(default)]
    #[allow(dead_code)]
    pub state: Option<serde_json::Value>,
    #[serde(default)]
    #[allow(dead_code)]
    pub tools: Vec<serde_json::Value>,
    #[serde(default)]
    #[allow(dead_code)]
    pub context: Vec<serde_json::Value>,
    /// Arbitrary client→server passthrough. We read `aguiClient` from it to
    /// pick the client dialect (see `ClientToolSignal`).
    #[serde(default)]
    pub forwarded_props: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgUiMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    pub id: Option<String>,
    /// Set on `role: "tool"` messages — correlates a frontend tool result back
    /// to the suspended tool call.
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

/// What a `RunAgentInput` asks the engine to do, derived from the trailing
/// message. A new `user` turn starts a turn; a trailing `tool` message is a
/// frontend tool result that resumes the turn suspended on that call.
pub enum AgUiInput {
    /// Start a new turn from this user message.
    UserTurn(Message),
    /// Resume the suspended turn by completing this client tool call.
    ToolResult {
        tool_call_id: String,
        content: String,
    },
}

impl RunAgentInput {
    /// Classify the run: a frontend tool result (trailing `tool` message) takes
    /// precedence over a new user turn. The worker owns history server-side, so
    /// we never replay the whole `messages` array — only the trailing action.
    pub fn classify(&self) -> Option<AgUiInput> {
        if let Some(last) = self.messages.last() {
            if last.role.eq_ignore_ascii_case("tool") {
                if let Some(tool_call_id) = last.tool_call_id.clone() {
                    return Some(AgUiInput::ToolResult {
                        tool_call_id,
                        content: last.content.clone().unwrap_or_default(),
                    });
                }
            }
        }
        self.last_user_message().map(AgUiInput::UserTurn)
    }

    /// The last user-authored message, mapped to an engine [`Message`].
    pub fn last_user_message(&self) -> Option<Message> {
        self.messages
            .iter()
            .rev()
            .find(|m| m.role.eq_ignore_ascii_case("user"))
            .map(|m| Message {
                role: Role::User,
                content: m.content.clone().map(Content::Text),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn deserializes_minimal_input() {
        let input: RunAgentInput = serde_json::from_value(json!({
            "threadId": "t1",
            "runId": "r1",
            "messages": [{"role": "user", "content": "hello"}],
        }))
        .unwrap();
        assert_eq!(input.thread_id, "t1");
        assert_eq!(input.run_id, "r1");
    }

    #[test]
    fn picks_last_user_message() {
        let input: RunAgentInput = serde_json::from_value(json!({
            "threadId": "t1",
            "runId": "r1",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "second"},
            ],
        }))
        .unwrap();
        let msg = input.last_user_message().unwrap();
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content.unwrap().text_owned(), "second");
    }

    #[test]
    fn none_when_no_user_message() {
        let input: RunAgentInput = serde_json::from_value(json!({
            "threadId": "t1",
            "runId": "r1",
            "messages": [{"role": "assistant", "content": "hi"}],
        }))
        .unwrap();
        assert!(input.last_user_message().is_none());
    }

    #[test]
    fn classify_user_turn() {
        let input: RunAgentInput = serde_json::from_value(json!({
            "threadId": "t1", "runId": "r1",
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .unwrap();
        match input.classify() {
            Some(AgUiInput::UserTurn(m)) => assert_eq!(m.role, Role::User),
            _ => panic!("expected UserTurn"),
        }
    }

    #[test]
    fn classify_tool_result_takes_precedence() {
        // A continuation run: full history ending in a frontend tool result.
        let input: RunAgentInput = serde_json::from_value(json!({
            "threadId": "t1", "runId": "r2",
            "messages": [
                {"role": "user", "content": "what's my timezone?"},
                {"role": "assistant", "content": ""},
                {"role": "tool", "toolCallId": "call-1", "content": "America/Los_Angeles"},
            ],
        }))
        .unwrap();
        match input.classify() {
            Some(AgUiInput::ToolResult {
                tool_call_id,
                content,
            }) => {
                assert_eq!(tool_call_id, "call-1");
                assert_eq!(content, "America/Los_Angeles");
            }
            _ => panic!("expected ToolResult"),
        }
    }
}
