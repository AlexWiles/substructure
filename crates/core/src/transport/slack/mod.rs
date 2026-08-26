mod activity;
mod bot;
mod proposer;
mod render;
mod socket;
mod state;

pub use bot::{SlackBot, Workspace, WorkspaceResolver};
pub use proposer::SlackProposer;
pub use render::{context_block, section_block, PromptOption, PromptView, Rendered};
pub use socket::{env_var, MissingEnv, SlackChannel, SlackTokens};
pub use state::StreamStore;

use std::collections::HashMap;

use serde_json::Value;

use crate::protocol::{
    Content, DraftMessage, InterruptOption, InterruptPayload, Role, StoredContent,
};

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
    files: Vec<SlackFile>,
}

/// An attachment on a Slack message.
#[derive(Debug, Clone, PartialEq)]
struct SlackFile {
    id: String,
    name: Option<String>,
    mimetype: String,
    url_private: String,
    size: u64,
}

fn files_of(event: &Value) -> Vec<SlackFile> {
    let Some(files) = event["files"].as_array() else {
        return Vec::new();
    };
    files
        .iter()
        .filter_map(|f| {
            Some(SlackFile {
                id: f["id"].as_str()?.to_string(),
                name: f["name"].as_str().map(str::to_string),
                mimetype: f["mimetype"].as_str()?.to_string(),
                url_private: f["url_private"].as_str()?.to_string(),
                size: f["size"].as_u64().unwrap_or(0),
            })
        })
        .collect()
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
        files: files_of(event),
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

/// An upload arrives as `file_share`; every other subtype is not conversation.
fn subtype_ok(m: &Value) -> bool {
    match m["subtype"].as_str() {
        None => true,
        Some(subtype) => subtype == "file_share",
    }
}

/// A user's DM. Bot echoes and subtyped messages (uploads aside) are `None`.
fn dm_message(payload: &Value) -> Option<Inbound> {
    let event = &payload["event"];
    if event["type"].as_str() != Some("message")
        || event["channel_type"].as_str() != Some("im")
        || !subtype_ok(event)
        || event["bot_id"].is_string()
    {
        return None;
    }
    let ts = event["ts"].as_str()?;
    let files = files_of(event);
    let text = event["text"].as_str().unwrap_or_default();
    if text.is_empty() && files.is_empty() {
        return None;
    }
    Some(Inbound {
        channel: event["channel"].as_str()?.to_string(),
        thread_ts: event["thread_ts"].as_str().unwrap_or(ts).to_string(),
        ts: ts.to_string(),
        user: event["user"].as_str()?.to_string(),
        team: asker_team(payload),
        text: text.to_string(),
        files,
    })
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
    files: Vec<SlackFile>,
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

/// Thread messages. Drops subtyped messages (uploads aside) and our own
/// unstamped posts.
fn parse_replies(resp: &Value, mine: &[String]) -> Vec<SlackMsg> {
    let Some(messages) = resp["messages"].as_array() else {
        return Vec::new();
    };
    messages
        .iter()
        .filter_map(|m| {
            if m["type"].as_str() != Some("message") || !subtype_ok(m) {
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
                files: files_of(m),
            })
        })
        .collect()
}

/// The blocks with the buttons removed and the outcome added.
/// The unseen thread delta as drafts. Our stamped replies map to their
/// recorded assistant nodes. `uploads` maps a file id to its stored blob uri.
fn build_batch(
    path: &[crate::protocol::Message],
    thread: &[SlackMsg],
    inbound: &Inbound,
    uploads: &HashMap<String, StoredContent>,
) -> Vec<DraftMessage> {
    let mut batch = Vec::new();
    let mut seen: std::collections::HashSet<String> = path.iter().map(|m| m.id.clone()).collect();
    let mut thread: Vec<&SlackMsg> = thread.iter().collect();
    thread.sort_by(|a, b| a.ts.cmp(&b.ts));
    for msg in thread {
        let (id, role, content) = match &msg.meta {
            // A stamp with no message id is skipped.
            Some(meta) => match &meta.message_id {
                Some(id) => (id.clone(), Role::Assistant, Content::Text(msg.text.clone())),
                None => continue,
            },
            None => {
                let text = match &msg.author {
                    Some(author) => format!("<@{author}>: {}", msg.text),
                    None => msg.text.clone(),
                };
                let content = with_attachments(text, &msg.files, uploads);
                (format!("slack:{}", msg.ts), Role::User, content)
            }
        };
        if seen.insert(id.clone()) {
            batch.push(draft(&id, role, content));
        }
    }
    let inbound_id = format!("slack:{}", inbound.ts);
    if seen.insert(inbound_id.clone()) {
        batch.push(draft(
            &inbound_id,
            Role::User,
            with_attachments(
                format!("<@{}>: {}", inbound.user, inbound.text),
                &inbound.files,
                uploads,
            ),
        ));
    }
    batch
}

/// The text plus the message's stored attachments. A file with no stored
/// part (unreadable type, too big, or the store failed) becomes a note, so
/// the model knows the attachment exists.
fn with_attachments(
    text: String,
    files: &[SlackFile],
    uploads: &HashMap<String, StoredContent>,
) -> Content {
    let mut attached = Vec::new();
    let mut notes = Vec::new();
    for f in files {
        match uploads.get(&f.id) {
            Some(part) => attached.push(part.clone()),
            None => notes.push(format!(
                "{} ({})",
                f.name.as_deref().unwrap_or("attachment"),
                f.mimetype
            )),
        }
    }
    let mut text = text;
    if !notes.is_empty() {
        text.push_str(&format!("\n[unreadable attachments: {}]", notes.join(", ")));
    }
    if attached.is_empty() {
        return Content::Text(text);
    }
    let mut parts = vec![StoredContent::Text { text }];
    parts.extend(attached);
    Content::Parts(parts)
}

fn draft(id: &str, role: Role, content: Content) -> DraftMessage {
    DraftMessage {
        id: Some(id.to_string()),
        role,
        content: Some(content),
        tool_calls: None,
        tool_call_id: None,
        name: None,
        reasoning: None,
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

/// What the bot puts in the `value` of a button it makes. A value that does
/// not deserialize as one of these belongs to the worker.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
enum ButtonValue {
    /// One of a prompt's answers. The button holds only its position; the
    /// open interrupt holds the value.
    #[serde(rename = "interrupt.option")]
    InterruptOption { interrupt_id: String, option: usize },
}

fn prompt_options(display: &Display, interrupt_id: &str) -> Vec<PromptOption> {
    display
        .options
        .iter()
        .enumerate()
        .map(|(idx, option)| PromptOption {
            label: option.label.clone(),
            style: option.style.clone(),
            action_id: format!("prompt_option_{idx}"),
            value: serde_json::to_string(&ButtonValue::InterruptOption {
                interrupt_id: interrupt_id.to_string(),
                option: idx,
            })
            .unwrap_or_default(),
            url: None,
        })
        .collect()
}

/// The button that opens a way in. It answers nothing, so it holds no value
/// and the session never sees its click.
pub(super) const AUTHORIZE_ACTION: &str = "authorize";

fn authorize_option(url: String, label: String) -> PromptOption {
    PromptOption {
        label,
        style: Some("primary".to_string()),
        action_id: AUTHORIZE_ACTION.to_string(),
        value: String::new(),
        url: Some(url),
    }
}

/// A button click, read out of a `block_actions` payload.
#[derive(Debug, PartialEq)]
struct Click {
    action_id: String,
    value: String,
    user: String,
    channel: String,
    message_ts: String,
    thread_ts: String,
    message_text: String,
    message_blocks: Vec<Value>,
    /// The session stamped on the clicked message, if it has one.
    session: Option<String>,
}

impl Click {
    /// The click as the action's `args`.
    fn args(&self) -> Value {
        serde_json::json!({
            "action_id": self.action_id,
            "value": self.value,
            "user": self.user,
            "channel": self.channel,
            "message_ts": self.message_ts,
            "thread_ts": self.thread_ts,
            "message_text": self.message_text,
            "message_blocks": self.message_blocks,
        })
    }
}

fn block_action(payload: &Value) -> Option<Click> {
    if payload["type"].as_str() != Some("block_actions") {
        return None;
    }
    let action = payload["actions"].as_array()?.first()?;
    let message_ts = payload["message"]["ts"].as_str()?;
    let session = (payload["message"]["metadata"]["event_type"].as_str() == Some(REPLY_EVENT_TYPE))
        .then(|| {
            payload["message"]["metadata"]["event_payload"]["session_id"]
                .as_str()
                .map(str::to_string)
        })
        .flatten();
    Some(Click {
        session,
        action_id: action["action_id"].as_str().unwrap_or_default().to_string(),
        value: action["value"].as_str().unwrap_or_default().to_string(),
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
    let responder = &payload["responder"];
    // A `danger` pick stops something. It never gets a green tick.
    let (mark, word) = match responder["style"].as_str() {
        _ if payload["status"].as_str() == Some("cancelled") => ("✖", "Cancelled"),
        Some("danger") => ("❌", "Declined"),
        _ => ("✅", "Resolved"),
    };
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
                files: Vec::new(),
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
                files: Vec::new(),
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

    fn file_json(id: &str, mimetype: &str) -> Value {
        serde_json::json!({
            "id": id,
            "name": "shot.png",
            "mimetype": mimetype,
            "url_private": format!("https://files.slack.com/{id}"),
            "size": 1234,
        })
    }

    #[test]
    fn dm_file_share_keeps_its_files() {
        let payload = envelope_payload(serde_json::json!({
            "type": "message",
            "channel_type": "im",
            "subtype": "file_share",
            "user": "U1",
            "text": "look at this",
            "ts": "5.0",
            "channel": "D1",
            "files": [file_json("F1", "image/png")],
        }));
        let inbound = dm_message(&payload).unwrap();
        assert_eq!(
            inbound.files,
            vec![SlackFile {
                id: "F1".into(),
                name: Some("shot.png".into()),
                mimetype: "image/png".into(),
                url_private: "https://files.slack.com/F1".into(),
                size: 1234,
            }]
        );
    }

    #[test]
    fn image_only_dm_is_still_a_message() {
        let payload = envelope_payload(serde_json::json!({
            "type": "message",
            "channel_type": "im",
            "subtype": "file_share",
            "user": "U1",
            "text": "",
            "ts": "5.0",
            "channel": "D1",
            "files": [file_json("F1", "image/png")],
        }));
        assert!(dm_message(&payload).is_some());
        // No text and no files is still nothing to answer.
        let empty = envelope_payload(serde_json::json!({
            "type": "message",
            "channel_type": "im",
            "user": "U1",
            "text": "",
            "ts": "6.0",
            "channel": "D1",
        }));
        assert_eq!(dm_message(&empty), None);
    }

    #[test]
    fn mention_keeps_its_files() {
        let payload = envelope_payload(serde_json::json!({
            "type": "app_mention",
            "user": "U1",
            "text": "<@UBOT> see attached",
            "ts": "2.0",
            "channel": "C1",
            "files": [file_json("F2", "image/jpeg")],
        }));
        assert_eq!(app_mention(&payload).unwrap().files.len(), 1);
    }

    #[test]
    fn replies_keep_file_share_messages_and_their_files() {
        let resp = serde_json::json!({ "ok": true, "messages": [
            { "type": "message", "subtype": "file_share", "user": "U1", "text": "here",
              "ts": "1.0", "files": [file_json("F1", "image/png")] },
            { "type": "message", "subtype": "channel_join", "user": "U2", "text": "joined", "ts": "2.0" },
        ]});
        let msgs = parse_replies(&resp, &[]);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].files[0].id, "F1");
    }

    #[test]
    fn batch_attaches_stored_images_and_skips_unstored_files() {
        let mut msg = slack_msg("1.0", "U1", "here");
        msg.files = vec![
            SlackFile {
                id: "F1".into(),
                name: None,
                mimetype: "image/png".into(),
                url_private: "https://files.slack.com/F1".into(),
                size: 10,
            },
            SlackFile {
                id: "F2".into(),
                name: None,
                mimetype: "application/pdf".into(),
                url_private: "https://files.slack.com/F2".into(),
                size: 10,
            },
        ];
        let uploads = HashMap::from_iter([(
            "F1".to_string(),
            StoredContent::Blob {
                uri: "blob://t/ab?mime=image%2Fpng&size=10".to_string(),
            },
        )]);
        let batch = build_batch(&[], &[msg], &mention_at("3.0"), &uploads);
        match batch[0].content.as_ref().unwrap() {
            Content::Parts(parts) => {
                assert_eq!(parts.len(), 2);
                // The unstored pdf becomes a note the model can act on.
                assert!(matches!(
                    &parts[0],
                    StoredContent::Text { text }
                        if text == "<@U1>: here\n[unreadable attachments: attachment (application/pdf)]"
                ));
                assert!(matches!(
                    &parts[1],
                    StoredContent::Blob { uri } if uri.starts_with("blob://")
                ));
            }
            _ => panic!("expected parts"),
        }
        // No files at all: plain text, as before.
        let batch = build_batch(
            &[],
            &[slack_msg("1.0", "U1", "here")],
            &mention_at("3.0"),
            &uploads,
        );
        assert!(matches!(
            batch[0].content.as_ref().unwrap(),
            Content::Text(_)
        ));
    }

    #[test]
    fn a_stored_pdf_rides_as_a_file_part() {
        let mut msg = slack_msg("1.0", "U1", "read this");
        msg.files = vec![SlackFile {
            id: "F9".into(),
            name: Some("q3.pdf".into()),
            mimetype: "application/pdf".into(),
            url_private: "https://files.slack.com/F9".into(),
            size: 10,
        }];
        let uploads = HashMap::from_iter([(
            "F9".to_string(),
            StoredContent::Blob {
                uri: "blob://t/ab?mime=application%2Fpdf&size=10&name=q3.pdf".into(),
            },
        )]);
        let batch = build_batch(&[], &[msg], &mention_at("3.0"), &uploads);
        match batch[0].content.as_ref().unwrap() {
            Content::Parts(parts) => {
                assert!(matches!(
                    &parts[1],
                    StoredContent::Blob { uri } if uri.contains("q3.pdf")
                ));
            }
            _ => panic!("expected parts"),
        }
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
            reasoning: None,
        }
    }

    fn slack_msg(ts: &str, author: &str, text: &str) -> SlackMsg {
        SlackMsg {
            ts: ts.into(),
            author: Some(author.into()),
            meta: None,
            text: text.into(),
            blocks: Vec::new(),
            files: Vec::new(),
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
            files: Vec::new(),
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
            files: Vec::new(),
        }
    }

    fn no_uploads() -> HashMap<String, StoredContent> {
        HashMap::new()
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
        let batch = build_batch(&path, &thread, &mention_at("3.0"), &no_uploads());
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
        let batch = build_batch(&[], &thread, &mention_at("3.0"), &no_uploads());
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
            files: Vec::new(),
        };
        // A double post with the same id becomes one node.
        let thread = vec![
            unmapped,
            ours_msg("2.6", "uuid-a1", "how about: subs"),
            ours_msg("2.7", "uuid-a1", "how about: subs"),
        ];
        let batch = build_batch(&[], &thread, &mention_at("3.0"), &no_uploads());
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
        let batch = build_batch(&[], &[], &mention_at("9.0"), &no_uploads());
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].id.as_deref(), Some("slack:9.0"));
        assert_eq!(text_of(&batch[0]), "<@U2>: <@UBOT> go");

        let path = vec![message("slack:9.0", Role::User, "<@U2>: <@UBOT> go")];
        let batch = build_batch(&path, &[], &mention_at("9.0"), &no_uploads());
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
        let options = prompt_options(&display, "int-1");
        let blocks = render::prompt_blocks(&render::PromptView {
            message: &display.message,
            options: &options,
            expires_at: display.expires_at.as_deref(),
        });
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0]["text"]["text"], "Run it?");
        let buttons = blocks[1]["elements"].as_array().unwrap();
        let value: Value = serde_json::from_str(buttons[0]["value"].as_str().unwrap()).unwrap();
        assert_eq!(
            value,
            serde_json::json!({ "type": "interrupt.option", "interrupt_id": "int-1", "option": 0 })
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
        let options = prompt_options(&display, "int-1");
        let blocks = render::prompt_blocks(&render::PromptView {
            message: &display.message,
            options: &options,
            expires_at: display.expires_at.as_deref(),
        });
        assert_eq!(blocks.len(), 1);
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
                session: None,
                action_id: String::new(),
                value: "{\"interrupt_id\":\"int-1\",\"option\":1}".into(),
                user: "U9".into(),
                channel: "D1".into(),
                message_ts: "8.0".into(),
                thread_ts: "5.0".into(),
                message_text: "Run it?".into(),
                message_blocks: Vec::new(),
            })
        );
        // A stamped message sends the click to the session it names.
        let mut stamped = payload.clone();
        stamped["message"]["metadata"] = serde_json::json!({
            "event_type": "substructure_reply",
            "event_payload": { "session_id": "my-api-session" },
        });
        assert_eq!(
            block_action(&stamped).unwrap().session.as_deref(),
            Some("my-api-session")
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
        // A worker's own button is still a click; its value passes through.
        let mut foreign = payload;
        foreign["actions"][0]["value"] = "not json".into();
        assert_eq!(block_action(&foreign).unwrap().value, "not json");
    }

    #[test]
    fn settling_spends_the_buttons_and_keeps_the_turn() {
        // Keep the task cards when the buttons go.
        let blocks = vec![
            serde_json::json!({ "type": "task_card", "task_id": "tc1", "title": "search_web" }),
            section_block("Run `send_email`?"),
            serde_json::json!({ "type": "actions", "elements": [{ "type": "button" }] }),
        ];
        let settled =
            render::settled_prompt_blocks(&blocks, "Run `send_email`?", "✅ Approve — <@U9>");
        assert_eq!(settled.len(), 3);
        assert_eq!(settled[0]["type"], "task_card");
        assert_eq!(settled[1]["type"], "section");
        assert_eq!(settled[2]["elements"][0]["text"], "✅ Approve — <@U9>");
        assert!(settled.iter().all(|b| b["type"] != "actions"));

        // A message with no blocks still settles.
        let bare = render::settled_prompt_blocks(&[], "Run it?", "✖ Cancelled");
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
            resolution_text(&serde_json::json!({
                "status": "resolved",
                "payload": { "approved": false },
                "responder": { "channel": "slack", "user": "U9", "label": "Decline", "style": "danger" },
            })),
            "❌ Decline — <@U9>"
        );
        assert_eq!(
            resolution_text(&serde_json::json!({
                "responder": { "user": "U9", "style": "danger" }
            })),
            "❌ Declined by <@U9>",
            "a `danger` pick with no label still says which way it went"
        );
        assert_eq!(
            resolution_text(&serde_json::json!({
                "responder": { "user": "U9", "label": "Run it", "style": "primary" }
            })),
            "✅ Run it — <@U9>"
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
            files: Vec::new(),
        };
        let batch = build_batch(&[], &[prompt_post], &mention_at("9.0"), &no_uploads());
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
}
