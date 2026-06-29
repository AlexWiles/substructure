use crate::event_store::Event;
use crate::session::events::{EventPayload, MessageTree, SessionInterrupted};
use crate::session::message::{Content, ContentPart, Message, Role};

use super::events::{AgUiEvent, AgUiInterrupt, RunOutcome, SnapshotMessage};

/// The active conversation — the materialized tree's head-to-root path — so a
/// branched session snapshots its live thread, not the branches it abandoned.
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
            tool_calls: message.tool_calls.unwrap_or_default(),
        },
        Role::Tool => SnapshotMessage::Tool {
            id,
            tool_call_id: message.tool_call_id.unwrap_or_default(),
            content: content.unwrap_or_default(),
        },
    }
}

fn active_interrupt(events: &[Event]) -> Option<SessionInterrupted> {
    let mut active: Option<SessionInterrupted> = None;
    for event in events {
        match serde_json::from_value::<EventPayload>(event.payload.clone()) {
            Ok(EventPayload::SessionInterrupted(p)) => active = Some(p),
            Ok(EventPayload::InterruptResumed(_)) => active = None,
            _ => {}
        }
    }
    active
}

pub fn snapshot_events(
    thread_id: String,
    run_id: String,
    tree: &MessageTree,
    events: &[Event],
) -> Vec<AgUiEvent> {
    let outcome = active_interrupt(events).map(|p| RunOutcome::Interrupt {
        interrupts: vec![AgUiInterrupt::from_session(&p)],
    });
    vec![
        AgUiEvent::RunStarted {
            thread_id: thread_id.clone(),
            run_id: run_id.clone(),
        },
        AgUiEvent::MessagesSnapshot {
            messages: session_messages(tree),
        },
        AgUiEvent::RunFinished {
            thread_id,
            run_id,
            result: None,
            outcome,
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
                    ContentPart::Text { text } => Some(text.as_str()),
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
    use crate::session::events::{NewMessage, Node};
    use crate::session::message::{Content, Message, Role, ToolCall, ToolCallFunction};
    use crate::span::SpanContext;
    use chrono::{DateTime, Utc};
    use serde_json::{json, Value};
    use std::collections::HashMap;
    use uuid::Uuid;

    fn node(id: u128, parent: Option<u128>, mut message: Message) -> Node {
        message.id = Uuid::from_u128(id).to_string();
        Node::Message(NewMessage {
            message,
            parent_id: parent.map(|p| Uuid::from_u128(p).to_string()),
        })
    }

    fn tree(nodes: Vec<Node>, head: u128) -> MessageTree {
        MessageTree {
            nodes,
            head_id: Some(Uuid::from_u128(head).to_string()),
        }
    }

    /// A linear thread: each message parents off the previous, head at the last.
    fn linear(messages: Vec<Message>) -> MessageTree {
        let nodes: Vec<Node> = messages
            .into_iter()
            .enumerate()
            .map(|(i, m)| node((i + 1) as u128, (i > 0).then(|| i as u128), m))
            .collect();
        let head = nodes.len() as u128;
        tree(nodes, head)
    }

    fn user(text: &str) -> Message {
        Message {
            id: String::new(),
            role: Role::User,
            content: Some(Content::Text(text.into())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    fn assistant_with_tool(call_id: &str, name: &str, args: &str) -> Message {
        Message {
            id: String::new(),
            role: Role::Assistant,
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: call_id.into(),
                call_type: "function".into(),
                function: ToolCallFunction {
                    name: name.into(),
                    arguments: args.into(),
                },
            }]),
            tool_call_id: None,
            name: None,
        }
    }

    fn tool_result(call_id: &str, result: &str) -> Message {
        Message {
            id: String::new(),
            role: Role::Tool,
            content: Some(Content::Text(result.into())),
            tool_calls: None,
            tool_call_id: Some(call_id.into()),
            name: Some("get_weather".into()),
        }
    }

    fn assistant_text(text: &str) -> Message {
        Message {
            id: String::new(),
            role: Role::Assistant,
            content: Some(Content::Text(text.into())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
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
        // 1 ← 2 ← 3, then an edit forks node 4 off 1 and head moves to it.
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
        let t = linear(vec![user("hi")]);
        let out = snapshot_events("thread-1".into(), "snap-1".into(), &t, &[]);
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

    fn lifecycle_event(id: u128, payload: Value) -> Event {
        let ts = DateTime::<Utc>::from_timestamp(0, 0).unwrap();
        Event {
            position: 0,
            id: Uuid::from_u128(id),
            tenant_id: "t".into(),
            aggregate_type: "session".into(),
            aggregate_id: "s".into(),
            sequence: id as u64,
            span: SpanContext::root(),
            occurred_at: ts,
            payload,
            derived: None,
            metadata: HashMap::new(),
            start_time: ts,
            end_time: ts,
        }
    }

    #[test]
    fn snapshot_reports_pending_interrupt() {
        let t = linear(vec![user("send the email")]);
        let events = vec![lifecycle_event(
            2,
            json!({
                "type": "session.interrupted",
                "interrupt_id": "int-1",
                "origin": "frontend",
                "reason": "confirmation",
                "payload": {"message": "Send the email?"},
            }),
        )];
        let out = snapshot_events("thread-1".into(), "snap-1".into(), &t, &events);
        let finished = serde_json::to_value(&out[2]).unwrap();
        assert_eq!(finished["type"], "RUN_FINISHED");
        assert_eq!(finished["outcome"]["type"], "interrupt");
        let interrupt = &finished["outcome"]["interrupts"][0];
        assert_eq!(interrupt["id"], "int-1");
        assert_eq!(interrupt["reason"], "confirmation");
        assert_eq!(interrupt["message"], "Send the email?");
    }

    #[test]
    fn snapshot_omits_outcome_after_resume() {
        let events = vec![
            lifecycle_event(
                1,
                json!({
                    "type": "session.interrupted",
                    "interrupt_id": "int-1",
                    "origin": "frontend",
                    "reason": "confirmation",
                    "payload": null,
                }),
            ),
            lifecycle_event(
                2,
                json!({
                    "type": "session.interrupt_resumed",
                    "interrupt_id": "int-1",
                    "payload": null,
                }),
            ),
        ];
        let out = snapshot_events(
            "thread-1".into(),
            "snap-1".into(),
            &MessageTree::default(),
            &events,
        );
        let finished = serde_json::to_value(&out[2]).unwrap();
        assert_eq!(finished["type"], "RUN_FINISHED");
        assert!(finished.get("outcome").is_none());
    }
}
