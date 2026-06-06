use serde::Deserialize;

use crate::session::message::{Content, Message, Role};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunAgentInput {
    pub thread_id: String,
    pub run_id: String,
    #[serde(default)]
    pub messages: Vec<AgUiMessage>,
    #[serde(default)]
    #[allow(dead_code)]
    pub state: Option<serde_json::Value>,
    #[serde(default)]
    #[allow(dead_code)]
    pub tools: Vec<serde_json::Value>,
    #[serde(default)]
    #[allow(dead_code)]
    pub context: Vec<serde_json::Value>,
    #[serde(default)]
    #[allow(dead_code)]
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
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

pub enum AgUiInput {
    UserTurn(Message),
    ToolResults(Vec<ToolResultItem>),
}

pub struct ToolResultItem {
    pub tool_call_id: String,
    pub content: String,
}

impl RunAgentInput {
    /// Trailing `reasoning` messages are skipped — CopilotKit echoes the
    /// transient reasoning back after the tool result, which would otherwise
    /// mask the tool block and look like a fresh user turn.
    pub fn classify(&self) -> Option<AgUiInput> {
        let mut results: Vec<ToolResultItem> = Vec::new();
        for m in self.messages.iter().rev() {
            if m.role.eq_ignore_ascii_case("reasoning") {
                continue;
            }
            if !m.role.eq_ignore_ascii_case("tool") {
                break;
            }
            if let Some(tool_call_id) = m.tool_call_id.clone() {
                results.push(ToolResultItem {
                    tool_call_id,
                    content: m.content.clone().unwrap_or_default(),
                });
            }
        }
        if !results.is_empty() {
            results.reverse();
            return Some(AgUiInput::ToolResults(results));
        }
        self.last_user_message().map(AgUiInput::UserTurn)
    }

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
    fn classify_tool_result_with_trailing_reasoning() {
        let input: RunAgentInput = serde_json::from_value(json!({
            "threadId": "t1", "runId": "r2",
            "messages": [
                {"role": "user", "content": "what is the color"},
                {"role": "assistant", "content": "",
                 "toolCalls": [{"id": "call-1", "type": "function",
                     "function": {"name": "get_color", "arguments": "{}"}}]},
                {"role": "tool", "toolCallId": "call-1", "content": "{\"hex\":\"#6366f1\"}"},
                {"role": "reasoning", "content": "Thinking Process: ..."},
            ],
        }))
        .unwrap();
        match input.classify() {
            Some(AgUiInput::ToolResults(results)) => {
                assert_eq!(results.len(), 1);
                assert_eq!(results[0].tool_call_id, "call-1");
                assert_eq!(results[0].content, "{\"hex\":\"#6366f1\"}");
            }
            _ => panic!("expected ToolResults"),
        }
    }

    #[test]
    fn classify_mixed_parallel_batch_collects_all_trailing_results() {
        let input: RunAgentInput = serde_json::from_value(json!({
            "threadId": "t1", "runId": "r2",
            "messages": [
                {"role": "user", "content": "weather in bangkok and the color"},
                {"role": "assistant", "content": "", "toolCalls": [
                    {"id": "call-weather", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{}"}},
                    {"id": "call-color", "type": "function",
                     "function": {"name": "get_color", "arguments": "{}"}}]},
                {"role": "tool", "toolCallId": "call-weather", "content": "{\"temp\":62}"},
                {"role": "tool", "toolCallId": "call-color", "content": "{\"hex\":\"#6366f1\"}"},
                {"role": "reasoning", "content": "..."},
            ],
        }))
        .unwrap();
        match input.classify() {
            Some(AgUiInput::ToolResults(results)) => {
                let ids: Vec<&str> = results.iter().map(|r| r.tool_call_id.as_str()).collect();
                assert_eq!(ids, ["call-weather", "call-color"]);
            }
            _ => panic!("expected ToolResults"),
        }
    }

    #[test]
    fn classify_tool_result_takes_precedence() {
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
            Some(AgUiInput::ToolResults(results)) => {
                assert_eq!(results.len(), 1);
                assert_eq!(results[0].tool_call_id, "call-1");
                assert_eq!(results[0].content, "America/Los_Angeles");
            }
            _ => panic!("expected ToolResults"),
        }
    }

    #[test]
    fn classify_collects_all_trailing_tool_results() {
        let input: RunAgentInput = serde_json::from_value(json!({
            "threadId": "t1", "runId": "r2",
            "messages": [
                {"role": "user", "content": "set it to teal and tell me the hex"},
                {"role": "assistant", "content": ""},
                {"role": "tool", "toolCallId": "call-a", "content": "{\"hex\":\"#008080\"}"},
                {"role": "tool", "toolCallId": "call-b", "content": "ok"},
            ],
        }))
        .unwrap();
        match input.classify() {
            Some(AgUiInput::ToolResults(results)) => {
                let ids: Vec<&str> = results.iter().map(|r| r.tool_call_id.as_str()).collect();
                assert_eq!(ids, ["call-a", "call-b"]);
            }
            _ => panic!("expected ToolResults"),
        }
    }

    #[test]
    fn classify_user_turn_when_trailing_message_not_tool() {
        let input: RunAgentInput = serde_json::from_value(json!({
            "threadId": "t1", "runId": "r3",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": ""},
                {"role": "tool", "toolCallId": "old", "content": "done"},
                {"role": "assistant", "content": "all set"},
                {"role": "user", "content": "again"},
            ],
        }))
        .unwrap();
        assert!(matches!(input.classify(), Some(AgUiInput::UserTurn(_))));
    }
}
