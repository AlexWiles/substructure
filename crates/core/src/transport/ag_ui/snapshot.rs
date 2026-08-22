use crate::protocol::{Content, Message, MessageTree, Role, StoredContent};
use crate::session::state::SessionState;

use super::events::{AgUiEvent, AgUiInterrupt, RunOutcome, SnapshotMessage};

/// The active conversation: the tree's head-to-root path, not abandoned branches.
pub fn session_messages(tree: &MessageTree) -> Vec<SnapshotMessage> {
    match &tree.head_id {
        Some(head) => tree.path_to(head).into_iter().map(to_snapshot).collect(),
        None => Vec::new(),
    }
}

fn to_snapshot(message: Message) -> SnapshotMessage {
    let id = message.id;
    let content = content_text(message.content);
    match message.role {
        Role::System => SnapshotMessage::System {
            id,
            content: content.unwrap_or_default(),
        },
        Role::User => SnapshotMessage::User {
            id,
            content: content.unwrap_or_default(),
        },
        Role::Assistant => SnapshotMessage::Assistant {
            id,
            content,
            tool_calls: message.tool_calls,
        },
        Role::Tool => SnapshotMessage::Tool {
            id,
            tool_call_id: message.tool_call_id.unwrap_or_default(),
            content: content.unwrap_or_default(),
        },
    }
}

pub fn open_interrupts(session: &SessionState) -> Vec<AgUiInterrupt> {
    session
        .interrupts_for(session.head_id.as_deref())
        .into_iter()
        .map(AgUiInterrupt::from_open)
        .collect()
}

/// Rehydration frames: the active branch plus every open interrupt on the
/// head path. `None` = session not created yet.
pub fn snapshot_events(
    thread_id: String,
    run_id: String,
    session: Option<&SessionState>,
) -> Vec<AgUiEvent> {
    let tree = session.map(SessionState::message_tree).unwrap_or_default();
    let outcome = session.and_then(|s| {
        let interrupts = open_interrupts(s);
        (!interrupts.is_empty()).then_some(RunOutcome::Interrupt { interrupts })
    });
    vec![
        AgUiEvent::RunStarted {
            thread_id: thread_id.clone(),
            run_id: run_id.clone(),
        },
        AgUiEvent::MessagesSnapshot {
            messages: session_messages(&tree),
        },
        AgUiEvent::RunFinished {
            thread_id,
            run_id,
            result: None,
            outcome,
            metadata: None,
        },
    ]
}

fn content_text(content: Option<Content>) -> Option<String> {
    match content {
        Some(Content::Text(t)) => Some(t),
        Some(Content::Parts(parts)) => {
            let text: String = parts
                .iter()
                .filter_map(|p| match p {
                    StoredContent::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            (!text.is_empty()).then_some(text)
        }
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{InterruptOrigin, NewMessage, ToolCall, ToolCallFunction};
    use crate::session::state::OpenInterrupt;
    use serde_json::{json, Value};
    use uuid::Uuid;

    fn node(id: u128, parent: Option<u128>, mut message: Message) -> NewMessage {
        message.id = Uuid::from_u128(id).to_string();
        NewMessage {
            message,
            parent_id: parent.map(|p| Uuid::from_u128(p).to_string()),
        }
    }

    fn tree(nodes: Vec<NewMessage>, head: u128) -> MessageTree {
        MessageTree {
            nodes,
            head_id: Some(Uuid::from_u128(head).to_string()),
        }
    }

    fn linear(messages: Vec<Message>) -> MessageTree {
        let nodes: Vec<NewMessage> = messages
            .into_iter()
            .enumerate()
            .map(|(i, m)| node((i + 1) as u128, (i > 0).then_some(i as u128), m))
            .collect();
        let head = nodes.len() as u128;
        tree(nodes, head)
    }

    fn user(text: &str) -> Message {
        Message {
            id: String::new(),
            role: Role::User,
            content: Some(Content::Text(text.into())),
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
            reasoning: None,
        }
    }

    fn assistant_with_tool(call_id: &str, name: &str, args: &str) -> Message {
        Message {
            id: String::new(),
            role: Role::Assistant,
            content: None,
            tool_calls: vec![ToolCall {
                id: call_id.into(),
                call_type: "function".into(),
                function: ToolCallFunction {
                    name: name.into(),
                    arguments: args.into(),
                },
            }],
            tool_call_id: None,
            name: None,
            reasoning: None,
        }
    }

    fn tool_result(call_id: &str, result: &str) -> Message {
        Message {
            id: String::new(),
            role: Role::Tool,
            content: Some(Content::Text(result.into())),
            tool_calls: vec![],
            tool_call_id: Some(call_id.into()),
            name: Some("get_weather".into()),
            reasoning: None,
        }
    }

    fn assistant_text(text: &str) -> Message {
        Message {
            id: String::new(),
            role: Role::Assistant,
            content: Some(Content::Text(text.into())),
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
            reasoning: None,
        }
    }

    fn wire(messages: Vec<SnapshotMessage>) -> Vec<Value> {
        messages
            .iter()
            .map(|m| serde_json::to_value(m).unwrap())
            .collect()
    }

    #[test]
    fn snapshot_returns_the_active_branch_only() {
        let t = tree(
            vec![
                node(1, None, user("first")),
                node(2, Some(1), user("second")),
                node(3, Some(2), assistant_text("reply to second")),
                node(4, Some(1), user("second, edited")),
            ],
            4,
        );
        let w = wire(session_messages(&t));
        assert_eq!(w.len(), 2);
        assert_eq!(w[0]["id"], Uuid::from_u128(1).to_string());
        assert_eq!(w[1]["content"], "second, edited");
    }

    #[test]
    fn folds_full_tool_conversation_in_order() {
        let t = linear(vec![
            user("weather in SF?"),
            assistant_with_tool("call-1", "get_weather", r#"{"city":"SF"}"#),
            tool_result("call-1", r#"{"temp":62}"#),
            assistant_text("It's 62°F in SF."),
        ]);
        let w = wire(session_messages(&t));

        assert_eq!(w.len(), 4);

        assert_eq!(w[0]["role"], "user");
        assert_eq!(w[0]["content"], "weather in SF?");
        assert_eq!(w[0]["id"], Uuid::from_u128(1).to_string());

        assert_eq!(w[1]["role"], "assistant");
        assert!(w[1].get("content").is_none());
        assert_eq!(w[1]["toolCalls"][0]["id"], "call-1");
        assert_eq!(w[1]["toolCalls"][0]["type"], "function");
        assert_eq!(w[1]["toolCalls"][0]["function"]["name"], "get_weather");
        assert_eq!(
            w[1]["toolCalls"][0]["function"]["arguments"],
            r#"{"city":"SF"}"#
        );

        assert_eq!(w[2]["role"], "tool");
        assert_eq!(w[2]["toolCallId"], "call-1");
        assert_eq!(w[2]["content"], r#"{"temp":62}"#);
        assert!(w[2].get("toolCalls").is_none());

        assert_eq!(w[3]["role"], "assistant");
        assert_eq!(w[3]["content"], "It's 62°F in SF.");
        assert!(w[3].get("toolCalls").is_none());
    }

    #[test]
    fn snapshot_events_wrap_the_snapshot_in_a_run() {
        let s = session_of(linear(vec![user("hi")]));
        let out = snapshot_events("thread-1".into(), "snap-1".into(), Some(&s));
        let kinds: Vec<&str> = out.iter().map(|e| e.type_name()).collect();
        assert_eq!(kinds, ["RUN_STARTED", "MESSAGES_SNAPSHOT", "RUN_FINISHED"]);

        let snapshot = serde_json::to_value(&out[1]).unwrap();
        assert_eq!(snapshot["type"], "MESSAGES_SNAPSHOT");
        assert_eq!(snapshot["messages"][0]["role"], "user");
        assert_eq!(snapshot["messages"][0]["content"], "hi");
    }

    #[test]
    fn empty_session_yields_empty_message_list() {
        assert!(session_messages(&MessageTree::default()).is_empty());
    }

    fn session_of(tree: MessageTree) -> SessionState {
        use crate::runtime::session::state::Logged;
        let mut s = SessionState::new("s".to_string());
        s.head_id = tree.head_id.clone();
        s.nodes = tree
            .nodes
            .into_iter()
            .map(|entry| Logged { seq: 0, entry })
            .collect();
        s
    }

    fn open(id: &str, anchor: Option<&str>, payload: Value) -> OpenInterrupt {
        OpenInterrupt {
            interrupt_id: id.to_string(),
            origin: InterruptOrigin::Frontend,
            reason: "confirmation".to_string(),
            payload,
            anchor: anchor.map(str::to_string),
        }
    }

    #[test]
    fn parked_head_reports_every_open_interrupt_on_its_path() {
        let mut s = session_of(linear(vec![user("send the email")]));
        let head = s.head_id.clone();
        s.open_interrupts.push(open(
            "int-1",
            head.as_deref(),
            json!({"message": "Send the email?"}),
        ));
        s.open_interrupts
            .push(open("int-2", head.as_deref(), Value::Null));

        let out = snapshot_events("thread-1".into(), "snap-1".into(), Some(&s));
        let finished = serde_json::to_value(&out[2]).unwrap();
        assert_eq!(finished["type"], "RUN_FINISHED");
        assert_eq!(finished["outcome"]["type"], "interrupt");
        let interrupts = finished["outcome"]["interrupts"].as_array().unwrap();
        assert_eq!(interrupts.len(), 2, "the client owes both a response");
        assert_eq!(interrupts[0]["id"], "int-1");
        assert_eq!(interrupts[0]["reason"], "confirmation");
        assert_eq!(interrupts[0]["message"], "Send the email?");
        assert_eq!(interrupts[1]["id"], "int-2");
    }

    #[test]
    fn escaped_head_reports_no_outcome() {
        // The interrupt is parked on an abandoned branch, off the head path.
        let mut s = session_of(linear(vec![user("hi"), assistant_text("hello")]));
        s.open_interrupts
            .push(open("int-1", Some("m-elsewhere"), Value::Null));

        let out = snapshot_events("thread-1".into(), "snap-1".into(), Some(&s));
        let finished = serde_json::to_value(&out[2]).unwrap();
        assert_eq!(finished["type"], "RUN_FINISHED");
        assert!(finished.get("outcome").is_none());
    }

    #[test]
    fn missing_session_snapshots_empty() {
        let out = snapshot_events("thread-1".into(), "snap-1".into(), None);
        let snapshot = serde_json::to_value(&out[1]).unwrap();
        assert_eq!(snapshot["messages"], json!([]));
        let finished = serde_json::to_value(&out[2]).unwrap();
        assert!(finished.get("outcome").is_none());
    }
}
