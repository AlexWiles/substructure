use serde::Deserialize;

use crate::protocol::{Content, DraftMessage, Role};

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
    #[serde(default)]
    pub resume: Vec<ResumeEntry>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResumeEntry {
    pub interrupt_id: String,
    pub status: ResumeStatus,
    #[serde(default)]
    pub payload: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ResumeStatus {
    Resolved,
    Cancelled,
}

impl ResumeStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ResumeStatus::Resolved => "resolved",
            ResumeStatus::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgUiMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

impl RunAgentInput {
    /// Unknown roles (e.g. `reasoning`) are dropped.
    pub fn to_messages(&self) -> Vec<DraftMessage> {
        self.messages
            .iter()
            .filter_map(|m| {
                let role = match m.role.to_ascii_lowercase().as_str() {
                    "system" => Role::System,
                    "user" => Role::User,
                    "assistant" => Role::Assistant,
                    "tool" => Role::Tool,
                    _ => return None,
                };
                Some(DraftMessage {
                    id: m.id.clone(),
                    role,
                    content: m.content.clone().map(Content::Text),
                    tool_calls: None,
                    tool_call_id: m.tool_call_id.clone(),
                    name: None,
                })
            })
            .collect()
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
    fn to_messages_maps_known_roles_keeps_ids_drops_unknown() {
        let input: RunAgentInput = serde_json::from_value(json!({
            "threadId": "t1", "runId": "r1",
            "messages": [
                {"role": "system", "content": "S", "id": "s1"},
                {"role": "user", "content": "U", "id": "u1"},
                {"role": "assistant", "content": "A", "id": "a1"},
                {"role": "tool", "toolCallId": "call-1", "content": "R", "id": "t1"},
                {"role": "reasoning", "content": "ignored"},
            ],
        }))
        .unwrap();
        let msgs = input.to_messages();
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(msgs[1].role, Role::User);
        assert_eq!(msgs[2].role, Role::Assistant);
        assert_eq!(msgs[3].role, Role::Tool);
        assert_eq!(msgs[3].tool_call_id.as_deref(), Some("call-1"));
        let ids: Vec<&str> = msgs.iter().map(|m| m.id.as_deref().unwrap()).collect();
        assert_eq!(ids, ["s1", "u1", "a1", "t1"]);
    }

    #[test]
    fn parses_resume_entries() {
        let input: RunAgentInput = serde_json::from_value(json!({
            "threadId": "t1", "runId": "r2",
            "resume": [
                {"interruptId": "int-1", "status": "resolved", "payload": {"approved": true}},
                {"interruptId": "int-2", "status": "cancelled"},
            ],
        }))
        .unwrap();
        assert_eq!(input.resume.len(), 2);
        assert_eq!(input.resume[0].interrupt_id, "int-1");
        assert_eq!(input.resume[0].status, ResumeStatus::Resolved);
        assert_eq!(input.resume[0].payload, Some(json!({"approved": true})));
        assert_eq!(input.resume[1].status, ResumeStatus::Cancelled);
        assert!(input.resume[1].payload.is_none());
    }
}
