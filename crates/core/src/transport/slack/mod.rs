mod activity;
mod bot;
mod socket;
mod state;
mod webhook;

pub use bot::{Routing, SlackBot, Workspace, WorkspaceResolver};
pub use socket::{MissingEnv, SlackChannel};
pub use state::StreamStore;
pub use webhook::webhook_router;

use serde_json::Value;

use crate::protocol::{Content, DraftMessage, InterruptOption, InterruptPayload, Role};

/// A message the bot must answer: a mention or a DM.
#[derive(Debug, PartialEq)]
struct Inbound {
    channel: String,
    /// The session anchor. A top-level message starts its thread at its own ts.
    thread_ts: String,
    /// Keys the turn.
    ts: String,
    user: String,
    /// The asker's workspace (their own for a Slack Connect guest).
    team: Option<String>,
    text: String,
}

/// A usable `app_mention`. A DM mention also arrives as `message.im`;
/// that path owns it.
fn app_mention(payload: &Value) -> Option<Inbound> {
    let event = &payload["event"];
    if event["type"].as_str() != Some("app_mention") || event["bot_id"].is_string() {
        return None;
    }
    let channel = event["channel"].as_str()?;
    if channel.starts_with('D') {
        return None;
    }
    let ts = event["ts"].as_str()?;
    Some(Inbound {
        channel: channel.to_string(),
        thread_ts: event["thread_ts"].as_str().unwrap_or(ts).to_string(),
        ts: ts.to_string(),
        user: event["user"].as_str()?.to_string(),
        team: asker_team(payload),
        text: event["text"].as_str()?.to_string(),
    })
}

/// `user_team` for a Slack Connect guest, else the delivered workspace.
fn asker_team(payload: &Value) -> Option<String> {
    ["user_team", "team"]
        .iter()
        .find_map(|k| payload["event"][k].as_str())
        .or_else(|| payload["team_id"].as_str())
        .map(str::to_string)
}

/// A user's DM. Bot echoes and subtyped messages are `None`.
fn dm_message(payload: &Value) -> Option<Inbound> {
    let event = &payload["event"];
    if event["type"].as_str() != Some("message")
        || event["channel_type"].as_str() != Some("im")
        || event["subtype"].is_string()
        || event["bot_id"].is_string()
    {
        return None;
    }
    let ts = event["ts"].as_str()?;
    Some(Inbound {
        channel: event["channel"].as_str()?.to_string(),
        thread_ts: event["thread_ts"].as_str().unwrap_or(ts).to_string(),
        ts: ts.to_string(),
        user: event["user"].as_str()?.to_string(),
        team: asker_team(payload),
        text: event["text"].as_str()?.to_string(),
    })
}

/// The `(channel, thread_ts)` behind a `slack:{channel}:{thread_ts}` session.
fn slack_session(session_id: &str) -> Option<(&str, &str)> {
    session_id.strip_prefix("slack:")?.split_once(':')
}

const REPLY_EVENT_TYPE: &str = "substructure_reply";

/// Engine ids stamped on a posted reply.
#[derive(Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
struct ReplyMeta {
    #[serde(skip_serializing_if = "Option::is_none")]
    turn_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    message_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    interrupt_id: Option<String>,
}

#[derive(Debug, PartialEq)]
struct SlackMsg {
    ts: String,
    /// The user id, or a bot's `bot_id`.
    author: Option<String>,
    meta: Option<ReplyMeta>,
    text: String,
    blocks: Vec<Value>,
}

/// The last message of ours in `resp` with no reply stamp: an open stream's
/// message, since a stamp only lands when a message settles.
fn unstamped_ours(resp: &Value, ours: &[String]) -> Option<String> {
    let messages = resp["messages"].as_array()?;
    messages
        .iter()
        .filter(|m| {
            m["type"].as_str() == Some("message")
                && m["metadata"]["event_type"].as_str() != Some(REPLY_EVENT_TYPE)
                && ["user", "bot_id"]
                    .iter()
                    .filter_map(|k| m[k].as_str())
                    .any(|id| ours.iter().any(|o| o == id))
        })
        .filter_map(|m| m["ts"].as_str())
        .next_back()
        .map(str::to_string)
}

/// Thread messages. Drops subtyped messages and our own unstamped posts.
fn parse_replies(resp: &Value, mine: &[String]) -> Vec<SlackMsg> {
    let Some(messages) = resp["messages"].as_array() else {
        return Vec::new();
    };
    messages
        .iter()
        .filter_map(|m| {
            if m["type"].as_str() != Some("message") || m["subtype"].is_string() {
                return None;
            }
            let meta: Option<ReplyMeta> = (m["metadata"]["event_type"].as_str()
                == Some(REPLY_EVENT_TYPE))
            .then(|| serde_json::from_value(m["metadata"]["event_payload"].clone()).ok())
            .flatten();
            let ours = ["user", "bot_id"]
                .iter()
                .filter_map(|k| m[k].as_str())
                .any(|id| mine.iter().any(|m| m == id));
            if ours && meta.is_none() {
                return None;
            }
            Some(SlackMsg {
                ts: m["ts"].as_str()?.to_string(),
                author: m["user"]
                    .as_str()
                    .or_else(|| m["bot_id"].as_str())
                    .map(str::to_string),
                meta,
                text: m["text"].as_str().unwrap_or_default().to_string(),
                blocks: m["blocks"].as_array().cloned().unwrap_or_default(),
            })
        })
        .collect()
}

/// The blocks with the buttons removed and the outcome added.
fn settled_blocks(blocks: &[Value], text: &str, resolution: &str) -> Vec<Value> {
    let mut settled: Vec<Value> = blocks
        .iter()
        .filter(|b| b["type"] != "actions")
        .cloned()
        .collect();
    if settled.is_empty() {
        settled.push(section_block(text));
    }
    settled.push(serde_json::json!({
        "type": "context",
        "elements": [{ "type": "mrkdwn", "text": resolution }],
    }));
    settled
}

/// The unseen thread delta as drafts. Our stamped replies map to their
/// recorded assistant nodes.
fn build_batch(
    path: &[crate::protocol::Message],
    thread: &[SlackMsg],
    inbound: &Inbound,
) -> Vec<DraftMessage> {
    let mut batch = Vec::new();
    let mut seen: std::collections::HashSet<String> = path.iter().map(|m| m.id.clone()).collect();
    let mut thread: Vec<&SlackMsg> = thread.iter().collect();
    thread.sort_by(|a, b| a.ts.cmp(&b.ts));
    for msg in thread {
        let (id, role, text) = match &msg.meta {
            // A stamp with no message id is skipped.
            Some(meta) => match &meta.message_id {
                Some(id) => (id.clone(), Role::Assistant, msg.text.clone()),
                None => continue,
            },
            None => {
                let text = match &msg.author {
                    Some(author) => format!("<@{author}>: {}", msg.text),
                    None => msg.text.clone(),
                };
                (format!("slack:{}", msg.ts), Role::User, text)
            }
        };
        if seen.insert(id.clone()) {
            batch.push(draft(&id, role, text));
        }
    }
    let inbound_id = format!("slack:{}", inbound.ts);
    if seen.insert(inbound_id.clone()) {
        batch.push(draft(
            &inbound_id,
            Role::User,
            format!("<@{}>: {}", inbound.user, inbound.text),
        ));
    }
    batch
}

fn draft(id: &str, role: Role, content: String) -> DraftMessage {
    DraftMessage {
        id: Some(id.to_string()),
        role,
        content: Some(Content::Text(content)),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }
}

/// The renderable part of an AG-UI interrupt payload.
struct Display {
    message: String,
    options: Vec<InterruptOption>,
    expires_at: Option<String>,
}

fn display_of(payload: &Value) -> Option<Display> {
    let p: InterruptPayload = serde_json::from_value(payload.clone()).ok()?;
    let options = p
        .metadata
        .as_ref()
        .and_then(|m| m.get("options"))
        .and_then(|o| serde_json::from_value(o.clone()).ok())
        .unwrap_or_default();
    Some(Display {
        message: p.message?,
        options,
        expires_at: p.expires_at,
    })
}

/// A button carries only the interrupt id and the option index. The value
/// is read from the recorded interrupt at click time.
fn prompt_blocks(display: &Display, interrupt_id: &str) -> Vec<Value> {
    let mut blocks = vec![section_block(&display.message)];
    if !display.options.is_empty() {
        let buttons: Vec<Value> = display
            .options
            .iter()
            .enumerate()
            .map(|(idx, option)| {
                let mut button = serde_json::json!({
                    "type": "button",
                    "action_id": format!("prompt_option_{idx}"),
                    "text": { "type": "plain_text", "text": option.label },
                    "value": serde_json::json!({
                        "interrupt_id": interrupt_id,
                        "option": idx,
                    })
                    .to_string(),
                });
                if let Some(style @ ("primary" | "danger")) = option.style.as_deref() {
                    button["style"] = style.into();
                }
                button
            })
            .collect();
        blocks.push(serde_json::json!({ "type": "actions", "elements": buttons }));
    }
    if let Some((raw, ts)) = display
        .expires_at
        .as_deref()
        .and_then(|e| Some((e, chrono::DateTime::parse_from_rfc3339(e).ok()?)))
    {
        blocks.push(serde_json::json!({ "type": "context", "elements": [{
            "type": "mrkdwn",
            "text": format!("Expires <!date^{}^{{date_short_pretty}} {{time}}|{raw}>", ts.timestamp()),
        }]}));
    }
    blocks
}

fn section_block(text: &str) -> Value {
    serde_json::json!({
        "type": "section",
        "text": { "type": "mrkdwn", "text": clip(text, MAX_SECTION) },
    })
}

/// A button click from a `block_actions` payload.
#[derive(Debug, PartialEq)]
struct Click {
    interrupt_id: String,
    option: usize,
    user: String,
    channel: String,
    message_ts: String,
    thread_ts: String,
    message_text: String,
    message_blocks: Vec<Value>,
}

fn block_action(payload: &Value) -> Option<Click> {
    if payload["type"].as_str() != Some("block_actions") {
        return None;
    }
    let action = payload["actions"].as_array()?.first()?;
    let value: Value = serde_json::from_str(action["value"].as_str()?).ok()?;
    let message_ts = payload["message"]["ts"].as_str()?;
    Some(Click {
        interrupt_id: value["interrupt_id"].as_str()?.to_string(),
        option: usize::try_from(value["option"].as_u64()?).ok()?,
        user: payload["user"]["id"].as_str()?.to_string(),
        channel: payload["channel"]["id"].as_str()?.to_string(),
        message_ts: message_ts.to_string(),
        thread_ts: payload["message"]["thread_ts"]
            .as_str()
            .unwrap_or(message_ts)
            .to_string(),
        message_text: payload["message"]["text"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
        message_blocks: payload["message"]["blocks"]
            .as_array()
            .cloned()
            .unwrap_or_default(),
    })
}

fn resolution_text(payload: &Value) -> String {
    if payload["expired"].as_bool() == Some(true) {
        return "⏱ Expired".to_string();
    }
    let (mark, word) = if payload["status"].as_str() == Some("cancelled") {
        ("✖", "Cancelled")
    } else {
        ("✅", "Resolved")
    };
    let responder = &payload["responder"];
    match (responder["label"].as_str(), responder["user"].as_str()) {
        (Some(label), Some(user)) => format!("{mark} {label} — <@{user}>"),
        (None, Some(user)) => format!("{mark} {word} by <@{user}>"),
        _ => format!("{mark} {word}"),
    }
}

fn with_footer(text: &str, footer: Option<&str>) -> String {
    match footer {
        Some(footer) => format!("{text}\n\n_{footer}_"),
        None => text.to_string(),
    }
}

/// Over the cap, `chat.postMessage` fails with `msg_too_long`.
const MAX_FALLBACK: usize = 4_000;
const MAX_MARKDOWN: usize = 12_000;
const MAX_SECTION: usize = 3_000;

fn clip(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let cut: String = text.chars().take(max.saturating_sub(1)).collect();
    format!("{cut}…")
}

fn turn_result_text(t: &crate::session::events::TurnCompleted) -> String {
    if let Some(err) = &t.error {
        return format!("Error: {err}");
    }
    match &t.data {
        Value::Null => "(no result)".to_string(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope_payload(event: Value) -> Value {
        serde_json::json!({ "type": "event_callback", "event": event })
    }

    #[test]
    fn mention_maps_thread_to_reply_target() {
        let payload = envelope_payload(serde_json::json!({
            "type": "app_mention",
            "user": "U1",
            "text": "<@UBOT> what is up",
            "ts": "2.0",
            "thread_ts": "1.0",
            "channel": "C1",
        }));
        assert_eq!(
            app_mention(&payload),
            Some(Inbound {
                channel: "C1".into(),
                thread_ts: "1.0".into(),
                ts: "2.0".into(),
                user: "U1".into(),
                team: None,
                text: "<@UBOT> what is up".into(),
            })
        );
    }

    #[test]
    fn unthreaded_mention_starts_the_thread_at_its_own_ts() {
        let payload = envelope_payload(serde_json::json!({
            "type": "app_mention",
            "user": "U1",
            "text": "<@UBOT> hi",
            "ts": "3.0",
            "channel": "C1",
        }));
        assert_eq!(app_mention(&payload).unwrap().thread_ts, "3.0");
    }

    #[test]
    fn top_level_dm_starts_a_thread_at_its_own_ts() {
        let payload = envelope_payload(serde_json::json!({
            "type": "message",
            "channel_type": "im",
            "user": "U1",
            "text": "hi there",
            "ts": "5.0",
            "channel": "D1",
        }));
        assert_eq!(
            dm_message(&payload),
            Some(Inbound {
                channel: "D1".into(),
                thread_ts: "5.0".into(),
                ts: "5.0".into(),
                user: "U1".into(),
                team: None,
                text: "hi there".into(),
            })
        );
    }

    #[test]
    fn threaded_dm_keeps_its_thread_as_the_session() {
        let payload = envelope_payload(serde_json::json!({
            "type": "message",
            "channel_type": "im",
            "user": "U1",
            "text": "in thread",
            "ts": "6.0",
            "thread_ts": "5.0",
            "channel": "D1",
        }));
        assert_eq!(dm_message(&payload).unwrap().thread_ts, "5.0");
    }

    #[test]
    fn dm_echoes_subtypes_and_channel_messages_are_not_dms() {
        let echo = envelope_payload(serde_json::json!({
            "type": "message",
            "channel_type": "im",
            "bot_id": "B1",
            "text": "my own reply",
            "ts": "7.0",
            "channel": "D1",
        }));
        assert_eq!(dm_message(&echo), None);
        let edit = envelope_payload(serde_json::json!({
            "type": "message",
            "channel_type": "im",
            "subtype": "message_changed",
            "user": "U1",
            "text": "edited",
            "ts": "8.0",
            "channel": "D1",
        }));
        assert_eq!(dm_message(&edit), None);
        let channel_msg = envelope_payload(serde_json::json!({
            "type": "message",
            "channel_type": "channel",
            "user": "U1",
            "text": "x",
            "ts": "9.0",
            "channel": "C1",
        }));
        assert_eq!(dm_message(&channel_msg), None);
    }

    #[test]
    fn dm_mention_defers_to_its_message_event() {
        // A DM mention fires both events; only the message path claims it.
        let payload = envelope_payload(serde_json::json!({
            "type": "app_mention",
            "user": "U1",
            "text": "<@UBOT> hi",
            "ts": "5.0",
            "channel": "D1",
        }));
        assert_eq!(app_mention(&payload), None);
    }

    #[test]
    fn bot_and_non_mention_events_are_ignored() {
        let bot = envelope_payload(serde_json::json!({
            "type": "app_mention",
            "bot_id": "B1",
            "user": "U1",
            "text": "x",
            "ts": "1.0",
            "channel": "C1",
        }));
        assert_eq!(app_mention(&bot), None);
        let message = envelope_payload(serde_json::json!({
            "type": "message",
            "user": "U1",
            "text": "x",
            "ts": "1.0",
            "channel": "C1",
        }));
        assert_eq!(app_mention(&message), None);
    }

    #[test]
    fn session_id_round_trips_channel_and_thread() {
        assert_eq!(slack_session("slack:C1:1.0"), Some(("C1", "1.0")));
        assert_eq!(slack_session("other:C1:1.0"), None);
        assert_eq!(slack_session("slack:C1"), None);
    }

    fn text_of(d: &DraftMessage) -> &str {
        match d.content.as_ref().unwrap() {
            Content::Text(t) => t,
            _ => panic!("expected text"),
        }
    }

    fn message(id: &str, role: Role, content: &str) -> crate::protocol::Message {
        crate::protocol::Message {
            id: id.into(),
            role,
            content: Some(Content::Text(content.into())),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    fn slack_msg(ts: &str, author: &str, text: &str) -> SlackMsg {
        SlackMsg {
            ts: ts.into(),
            author: Some(author.into()),
            meta: None,
            text: text.into(),
            blocks: Vec::new(),
        }
    }

    fn ours_msg(ts: &str, message_id: &str, text: &str) -> SlackMsg {
        SlackMsg {
            ts: ts.into(),
            author: Some("UBOT".into()),
            meta: Some(ReplyMeta {
                turn_id: Some("t".into()),
                message_id: Some(message_id.into()),
                session_id: Some("slack:C1:1.0".into()),
                ..Default::default()
            }),
            text: text.into(),
            blocks: Vec::new(),
        }
    }

    fn mention_at(ts: &str) -> Inbound {
        Inbound {
            channel: "C1".into(),
            thread_ts: "1.0".into(),
            ts: ts.into(),
            user: "U2".into(),
            team: None,
            text: "<@UBOT> go".into(),
        }
    }

    #[test]
    fn replies_keep_plain_messages_and_parse_our_stamp() {
        let resp = serde_json::json!({ "ok": true, "messages": [
            { "type": "message", "user": "U1", "text": "parent", "ts": "1.0" },
            { "type": "message", "subtype": "channel_join", "user": "U2", "text": "joined", "ts": "2.0" },
            { "type": "message", "bot_id": "B1", "text": "reply", "ts": "3.0",
              "metadata": { "event_type": "substructure_reply",
                            "event_payload": { "turn_id": "t", "message_id": "uuid-a1", "session_id": "slack:C1:1.0" } } },
            { "type": "message", "bot_id": "B9", "text": "other bot", "ts": "4.0" },
            { "type": "file", "ts": "5.0" },
        ]});
        let msgs = parse_replies(&resp, &["B1".into()]);
        assert_eq!(msgs.len(), 3);
        assert!(msgs[0].meta.is_none());
        let meta = msgs[1].meta.as_ref().unwrap();
        assert_eq!(meta.turn_id.as_deref(), Some("t"));
        assert_eq!(meta.message_id.as_deref(), Some("uuid-a1"));
        assert!(msgs[2].meta.is_none());
        assert_eq!(msgs[2].author.as_deref(), Some("B9"));
    }

    #[test]
    fn the_orphaned_stream_is_our_last_unstamped_message() {
        let resp = serde_json::json!({ "ok": true, "messages": [
            { "type": "message", "user": "U1", "text": "asker", "ts": "1.0" },
            { "type": "message", "bot_id": "B1", "text": "old reply", "ts": "2.0",
              "metadata": { "event_type": "substructure_reply", "event_payload": { "turn_id": "t0" } } },
            { "type": "message", "bot_id": "B1", "text": "streaming…", "ts": "3.0" },
            { "type": "message", "bot_id": "B9", "text": "other bot", "ts": "4.0" },
        ]});
        // The stamped reply and the foreign bot are not candidates.
        assert_eq!(
            unstamped_ours(&resp, &["B1".into()]),
            Some("3.0".to_string())
        );
        // Without an identity nothing can be claimed.
        assert_eq!(unstamped_ours(&resp, &[]), None);
    }

    #[test]
    fn our_own_unstamped_posts_are_not_conversation() {
        // An in-flight stream has no stamp; it is not conversation.
        let resp = serde_json::json!({ "ok": true, "messages": [
            { "type": "message", "user": "U1", "text": "parent", "ts": "1.0" },
            { "type": "message", "user": "UBOT", "bot_id": "B1", "text": "🔄 working", "ts": "2.0" },
        ]});
        let msgs = parse_replies(&resp, &["UBOT".into(), "B1".into()]);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].ts, "1.0");
        // No identity: no filter.
        assert_eq!(parse_replies(&resp, &[]).len(), 2);
    }

    #[test]
    fn batch_is_the_unseen_attributed_delta() {
        let path = vec![
            message("slack:1.0", Role::User, "<@U1>: we need a slogan"),
            message("uuid-a1", Role::Assistant, "how about: subs"),
        ];
        let thread = vec![
            slack_msg("3.0", "U2", "<@UBOT> go"),
            ours_msg("2.5", "uuid-a1", "how about: subs"),
            slack_msg("1.0", "U1", "we need a slogan"),
        ];
        let batch = build_batch(&path, &thread, &mention_at("3.0"));
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].id.as_deref(), Some("slack:3.0"));
        assert_eq!(text_of(&batch[0]), "<@U2>: <@UBOT> go");
    }

    #[test]
    fn lost_session_rebuilds_from_the_thread() {
        let thread = vec![
            slack_msg("1.0", "U1", "we need a slogan"),
            ours_msg("2.5", "uuid-a1", "how about: subs"),
            slack_msg("3.0", "U2", "<@UBOT> go"),
        ];
        let batch = build_batch(&[], &thread, &mention_at("3.0"));
        assert_eq!(
            batch
                .iter()
                .map(|d| d.id.clone().unwrap())
                .collect::<Vec<_>>(),
            ["slack:1.0", "uuid-a1", "slack:3.0"]
        );
        assert!(matches!(batch[1].role, Role::Assistant));
        assert_eq!(text_of(&batch[1]), "how about: subs");
    }

    #[test]
    fn unstamped_and_duplicate_stamped_replies_do_not_duplicate() {
        // A stamp with no message id is skipped.
        let unmapped = SlackMsg {
            ts: "2.5".into(),
            author: Some("UBOT".into()),
            meta: Some(ReplyMeta::default()),
            text: "reply".into(),
            blocks: Vec::new(),
        };
        // A double post with the same id becomes one node.
        let thread = vec![
            unmapped,
            ours_msg("2.6", "uuid-a1", "how about: subs"),
            ours_msg("2.7", "uuid-a1", "how about: subs"),
        ];
        let batch = build_batch(&[], &thread, &mention_at("3.0"));
        assert_eq!(
            batch
                .iter()
                .map(|d| d.id.clone().unwrap())
                .collect::<Vec<_>>(),
            ["uuid-a1", "slack:3.0"]
        );
    }

    #[test]
    fn mention_missing_from_fetch_is_appended_once() {
        let batch = build_batch(&[], &[], &mention_at("9.0"));
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].id.as_deref(), Some("slack:9.0"));
        assert_eq!(text_of(&batch[0]), "<@U2>: <@UBOT> go");

        let path = vec![message("slack:9.0", Role::User, "<@U2>: <@UBOT> go")];
        let batch = build_batch(&path, &[], &mention_at("9.0"));
        assert!(batch.is_empty());
    }

    #[test]
    fn display_reads_ag_ui_shape_with_options_in_metadata() {
        let payload = serde_json::json!({
            "message": "Run `send_email`?",
            "toolCallId": "tc1",
            "metadata": {
                "options": [
                    { "label": "Approve", "value": { "decision": "approve" }, "style": "primary" },
                    { "label": "Deny", "value": { "decision": "deny" }, "style": "danger" },
                ],
                "pending": { "tool_call": { "id": "tc1" } },
            },
        });
        let display = display_of(&payload).unwrap();
        assert_eq!(display.message, "Run `send_email`?");
        assert_eq!(display.options.len(), 2);
        assert_eq!(display.options[0].label, "Approve");
        assert_eq!(
            display.options[1].value,
            serde_json::json!({ "decision": "deny" })
        );

        // No message: nothing to render.
        assert!(display_of(&serde_json::json!({ "custom": "x" })).is_none());
        assert!(display_of(&serde_json::json!(null)).is_none());
        // Bad options: the message still renders.
        assert!(display_of(&serde_json::json!({ "message": "hold" }))
            .unwrap()
            .options
            .is_empty());
        assert!(display_of(
            &serde_json::json!({ "message": "hold", "metadata": { "options": "bad" } })
        )
        .unwrap()
        .options
        .is_empty());
    }

    #[test]
    fn prompt_blocks_carry_coordinates_not_values() {
        let display = display_of(&serde_json::json!({
            "message": "Run it?",
            "expiresAt": "2026-07-24T18:00:00Z",
            "metadata": { "options": [
                { "label": "Approve", "value": { "decision": "approve" }, "style": "primary" },
                { "label": "Maybe", "value": {}, "style": "fancy" },
            ]},
        }))
        .unwrap();
        let blocks = prompt_blocks(&display, "int-1");
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0]["text"]["text"], "Run it?");
        let buttons = blocks[1]["elements"].as_array().unwrap();
        let value: Value = serde_json::from_str(buttons[0]["value"].as_str().unwrap()).unwrap();
        assert_eq!(
            value,
            serde_json::json!({ "interrupt_id": "int-1", "option": 0 })
        );
        assert_eq!(buttons[0]["style"], "primary");
        // Unknown styles are dropped.
        assert!(buttons[1].get("style").is_none());
        assert!(blocks[2]["elements"][0]["text"]
            .as_str()
            .unwrap()
            .contains("<!date^1784916000^"));
    }

    #[test]
    fn message_only_prompt_renders_without_buttons() {
        let display = display_of(&serde_json::json!({ "message": "hold" })).unwrap();
        assert_eq!(prompt_blocks(&display, "int-1").len(), 1);
    }

    #[test]
    fn block_action_maps_a_click_to_its_coordinates() {
        let payload = serde_json::json!({
            "type": "block_actions",
            "user": { "id": "U9" },
            "channel": { "id": "D1" },
            "message": { "ts": "8.0", "thread_ts": "5.0", "text": "Run it?" },
            "actions": [{ "value": "{\"interrupt_id\":\"int-1\",\"option\":1}" }],
        });
        assert_eq!(
            block_action(&payload),
            Some(Click {
                interrupt_id: "int-1".into(),
                option: 1,
                user: "U9".into(),
                channel: "D1".into(),
                message_ts: "8.0".into(),
                thread_ts: "5.0".into(),
                message_text: "Run it?".into(),
                message_blocks: Vec::new(),
            })
        );
        // A prompt on a top-level message anchors its own thread.
        let mut top_level = payload.clone();
        top_level["message"]
            .as_object_mut()
            .unwrap()
            .remove("thread_ts");
        assert_eq!(block_action(&top_level).unwrap().thread_ts, "8.0");

        let mut wrong_type = payload.clone();
        wrong_type["type"] = "view_submission".into();
        assert_eq!(block_action(&wrong_type), None);
        // A foreign button value is not a click.
        let mut foreign = payload;
        foreign["actions"][0]["value"] = "not json".into();
        assert_eq!(block_action(&foreign), None);
    }

    #[test]
    fn settling_spends_the_buttons_and_keeps_the_turn() {
        // Keep the task cards when the buttons go.
        let blocks = vec![
            serde_json::json!({ "type": "task_card", "task_id": "tc1", "title": "search_web" }),
            section_block("Run `send_email`?"),
            serde_json::json!({ "type": "actions", "elements": [{ "type": "button" }] }),
        ];
        let settled = settled_blocks(&blocks, "Run `send_email`?", "✅ Approve — <@U9>");
        assert_eq!(settled.len(), 3);
        assert_eq!(settled[0]["type"], "task_card");
        assert_eq!(settled[1]["type"], "section");
        assert_eq!(settled[2]["elements"][0]["text"], "✅ Approve — <@U9>");
        assert!(settled.iter().all(|b| b["type"] != "actions"));

        // A message with no blocks still settles.
        let bare = settled_blocks(&[], "Run it?", "✖ Cancelled");
        assert_eq!(bare[0]["text"]["text"], "Run it?");
        assert_eq!(bare[1]["elements"][0]["text"], "✖ Cancelled");
    }

    #[test]
    fn resolution_renders_status_responder_and_expiry() {
        assert_eq!(
            resolution_text(&serde_json::json!({
                "status": "resolved",
                "payload": { "decision": "approve" },
                "responder": { "channel": "slack", "user": "U9", "label": "Approve" },
            })),
            "✅ Approve — <@U9>"
        );
        assert_eq!(
            resolution_text(&serde_json::json!({ "responder": { "user": "U9" } })),
            "✅ Resolved by <@U9>"
        );
        assert_eq!(resolution_text(&serde_json::json!({})), "✅ Resolved");
        assert_eq!(
            resolution_text(&serde_json::json!({ "status": "cancelled" })),
            "✖ Cancelled"
        );
        assert_eq!(
            resolution_text(&serde_json::json!({ "expired": true })),
            "⏱ Expired"
        );
    }

    #[test]
    fn prompt_posts_are_stamped_but_never_join_the_batch() {
        // A prompt stamp has no message id; batches skip it.
        let prompt_post = SlackMsg {
            ts: "6.0".into(),
            author: Some("UBOT".into()),
            meta: Some(ReplyMeta {
                interrupt_id: Some("int-1".into()),
                session_id: Some("slack:C1:1.0".into()),
                ..Default::default()
            }),
            text: "Run it?".into(),
            blocks: Vec::new(),
        };
        let batch = build_batch(&[], &[prompt_post], &mention_at("9.0"));
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].id.as_deref(), Some("slack:9.0"));
    }

    #[test]
    fn text_is_clipped_to_each_slack_limit() {
        assert_eq!(clip("short", MAX_FALLBACK), "short");
        for max in [MAX_FALLBACK, MAX_MARKDOWN, MAX_SECTION] {
            let out = clip(&"é".repeat(max + 500), max);
            assert_eq!(out.chars().count(), max);
            assert!(out.ends_with('…'));
        }
        // A section block clips at its own limit.
        let long = "é".repeat(MAX_SECTION + 500);
        assert_eq!(
            section_block(&long)["text"]["text"]
                .as_str()
                .unwrap()
                .chars()
                .count(),
            MAX_SECTION
        );
    }

    #[test]
    fn turn_result_renders_string_json_and_error() {
        use crate::session::events::TurnCompleted;
        let mut t = TurnCompleted {
            turn_id: "t".into(),
            data: Value::String("done".into()),
            turn_cost: Default::default(),
            turn_token_usage: Default::default(),
            error: None,
        };
        assert_eq!(turn_result_text(&t), "done");
        t.data = serde_json::json!({"a": 1});
        assert_eq!(turn_result_text(&t), r#"{"a":1}"#);
        t.error = Some("boom".into());
        assert_eq!(turn_result_text(&t), "Error: boom");
    }
}
