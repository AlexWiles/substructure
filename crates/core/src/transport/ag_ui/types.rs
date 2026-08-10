use serde::Deserialize;

use crate::protocol::{
    AgentTool, ClientContext, Content, DraftMessage, Handler, ResumeStatus, Role, ToolCall,
};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunAgentInput {
    pub thread_id: String,
    pub run_id: String,
    #[serde(default)]
    pub messages: Vec<AgUiMessage>,
    #[serde(default)]
    pub state: Option<serde_json::Value>,
    #[serde(default)]
    pub tools: Vec<AgUiTool>,
    #[serde(default)]
    pub context: Vec<serde_json::Value>,
    #[serde(default)]
    pub forwarded_props: Option<serde_json::Value>,
    #[serde(default)]
    pub resume: Vec<ResumeEntry>,
}

/// The connect body: only the thread is addressed; `runId` labels the snapshot
/// frames when present.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConnectInput {
    pub thread_id: String,
    #[serde(default)]
    pub run_id: Option<String>,
}

/// A frontend tool a client declares on its run: `{name, description, parameters}`
/// (AG-UI shape). `parameters` is the JSON Schema for its arguments.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgUiTool {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResumeEntry {
    pub interrupt_id: String,
    pub status: ResumeStatus,
    #[serde(default)]
    pub payload: Option<serde_json::Value>,
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
    /// Assistant tool calls echoed back in the client view. Dropping these
    /// orphans the matching `tool` result, which providers reject.
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
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
                    tool_calls: m.tool_calls.clone(),
                    tool_call_id: m.tool_call_id.clone(),
                    name: None,
                })
            })
            .collect()
    }

    /// The client-declared run inputs the worker sees on `client.messages`. Each
    /// frontend tool becomes a client-handled [`AgentTool`] (`parameters` → `input`);
    /// context/state/forwardedProps pass through verbatim.
    pub fn client_context(&self) -> ClientContext {
        ClientContext {
            tools: self
                .tools
                .iter()
                .map(|t| AgentTool {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    input: t.parameters.clone(),
                    output: None,
                    handler: Some(Handler::Client),
                    defer: None,
                })
                .collect(),
            context: self.context.clone(),
            state: self.state.clone(),
            forwarded_props: self.forwarded_props.clone(),
        }
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
    fn connect_input_needs_only_thread_id() {
        let input: ConnectInput = serde_json::from_value(json!({"threadId": "t1"})).unwrap();
        assert_eq!(input.thread_id, "t1");
        assert!(input.run_id.is_none());
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
    fn to_messages_keeps_assistant_tool_calls() {
        // A follow-up turn resends the assistant tool call + its result in the
        // client view. Dropping `toolCalls` orphans the result and providers 400.
        let input: RunAgentInput = serde_json::from_value(json!({
            "threadId": "t1", "runId": "r1",
            "messages": [
                {"role": "user", "content": "what time is it?", "id": "u1"},
                {"role": "assistant", "id": "a1", "toolCalls": [
                    {"id": "call-1", "type": "function",
                     "function": {"name": "get_current_time", "arguments": "{}"}},
                ]},
                {"role": "tool", "toolCallId": "call-1", "content": "2026", "id": "t1"},
                {"role": "assistant", "content": "It's 2026.", "id": "a2"},
            ],
        }))
        .unwrap();
        let msgs = input.to_messages();
        let calls = msgs[1].tool_calls.as_ref().expect("tool_calls preserved");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call-1");
        assert_eq!(calls[0].function.name, "get_current_time");
        assert_eq!(msgs[2].tool_call_id.as_deref(), Some("call-1"));
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

    #[test]
    fn parses_resume_and_messages_together() {
        // steerAway: one run input resumes the interrupt and branches away.
        let input: RunAgentInput = serde_json::from_value(json!({
            "threadId": "t1", "runId": "r2",
            "messages": [
                {"role": "user", "content": "hi", "id": "u1"},
                {"role": "user", "content": "actually, do this instead"},
            ],
            "resume": [
                {"interruptId": "int-1", "status": "cancelled"},
            ],
        }))
        .unwrap();
        assert_eq!(input.resume.len(), 1);
        assert_eq!(input.to_messages().len(), 2);
    }

    #[test]
    fn client_context_normalizes_frontend_tools_and_passes_context_through() {
        let input: RunAgentInput = serde_json::from_value(json!({
            "threadId": "t1", "runId": "r1",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "name": "get_timezone",
                "description": "the browser's time zone",
                "parameters": {"type": "object", "properties": {}},
            }],
            "context": [{"description": "system", "value": "be nice"}],
            "state": {"count": 1},
            "forwardedProps": {"a": true},
        }))
        .unwrap();
        let cc = input.client_context();
        assert_eq!(cc.tools.len(), 1);
        let tool = &cc.tools[0];
        assert_eq!(tool.name, "get_timezone");
        assert_eq!(
            tool.handler,
            Some(Handler::Client),
            "frontend ⇒ client-handled"
        );
        assert_eq!(
            tool.input,
            Some(json!({"type": "object", "properties": {}})),
            "AG-UI `parameters` ⇒ AgentTool `input`"
        );
        assert_eq!(
            cc.context,
            vec![json!({"description": "system", "value": "be nice"})]
        );
        assert_eq!(cc.state, Some(json!({"count": 1})));
        assert_eq!(cc.forwarded_props, Some(json!({"a": true})));
    }
}
