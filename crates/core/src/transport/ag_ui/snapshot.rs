use crate::event_store::Event;
use crate::session::events::EventPayload;
use crate::session::message::{Content, ContentPart, Role};

use super::events::{AgUiEvent, SnapshotMessage};

pub fn session_messages(events: &[Event]) -> Vec<SnapshotMessage> {
    let mut out = Vec::new();
    for event in events {
        let Ok(EventPayload::NewMessage(new_message)) =
            serde_json::from_value::<EventPayload>(event.payload.clone())
        else {
            continue;
        };
        let id = event.id.to_string();
        let message = new_message.message;
        let content = content_text(message.content);
        out.push(match message.role {
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
        });
    }
    out
}

pub fn snapshot_events(thread_id: String, run_id: String, events: &[Event]) -> Vec<AgUiEvent> {
    vec![
        AgUiEvent::RunStarted {
            thread_id: thread_id.clone(),
            run_id: run_id.clone(),
        },
        AgUiEvent::MessagesSnapshot {
            messages: session_messages(events),
        },
        AgUiEvent::RunFinished {
            thread_id,
            run_id,
            result: None,
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
    use crate::session::events::NewMessage;
    use crate::session::message::{Content, Message, Role, ToolCall, ToolCallFunction};
    use crate::span::SpanContext;
    use chrono::{DateTime, Utc};
    use serde_json::{json, Value};
    use std::collections::HashMap;
    use uuid::Uuid;

    fn message_event(id: u128, message: Message) -> Event {
        let ts = DateTime::<Utc>::from_timestamp(0, 0).unwrap();
        let payload =
            serde_json::to_value(EventPayload::NewMessage(NewMessage { message })).unwrap();
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

    fn user(text: &str) -> Message {
        Message {
            role: Role::User,
            content: Some(Content::Text(text.into())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    fn assistant_with_tool(call_id: &str, name: &str, args: &str) -> Message {
        Message {
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
            role: Role::Tool,
            content: Some(Content::Text(result.into())),
            tool_calls: None,
            tool_call_id: Some(call_id.into()),
            name: Some("get_weather".into()),
        }
    }

    fn assistant_text(text: &str) -> Message {
        Message {
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
    fn folds_full_tool_conversation_in_order() {
        let events = vec![
            message_event(1, user("weather in SF?")),
            message_event(
                2,
                assistant_with_tool("call-1", "get_weather", r#"{"city":"SF"}"#),
            ),
            message_event(3, tool_result("call-1", r#"{"temp":62}"#)),
            message_event(4, assistant_text("It's 62°F in SF.")),
        ];
        let w = wire(session_messages(&events));

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
    fn ignores_non_message_events() {
        let ts = DateTime::<Utc>::from_timestamp(0, 0).unwrap();
        let lifecycle = Event {
            position: 0,
            id: Uuid::from_u128(9),
            tenant_id: "t".into(),
            aggregate_type: "session".into(),
            aggregate_id: "s".into(),
            sequence: 9,
            span: SpanContext::root(),
            occurred_at: ts,
            payload: json!({"type": "turn.completed", "turn_id": "r1"}),
            derived: None,
            metadata: HashMap::new(),
            start_time: ts,
            end_time: ts,
        };
        let events = vec![message_event(1, user("hi")), lifecycle];
        let messages = session_messages(&events);
        assert_eq!(messages.len(), 1);
    }

    #[test]
    fn snapshot_events_wrap_the_snapshot_in_a_run() {
        let events = vec![message_event(1, user("hi"))];
        let out = snapshot_events("thread-1".into(), "snap-1".into(), &events);
        let kinds: Vec<&str> = out.iter().map(|e| e.type_name()).collect();
        assert_eq!(kinds, ["RUN_STARTED", "MESSAGES_SNAPSHOT", "RUN_FINISHED"]);

        let snapshot = serde_json::to_value(&out[1]).unwrap();
        assert_eq!(snapshot["type"], "MESSAGES_SNAPSHOT");
        assert_eq!(snapshot["messages"][0]["role"], "user");
        assert_eq!(snapshot["messages"][0]["content"], "hi");
    }

    #[test]
    fn empty_session_yields_empty_message_list() {
        assert!(session_messages(&[]).is_empty());
    }
}
