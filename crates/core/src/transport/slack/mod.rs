use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use tokio_tungstenite::tungstenite::Message as WsMessage;

use crate::processor::{EventProcessor, EventProcessorRunnerConfig, ProcessorError};
use crate::protocol::{
    ClientInput, Content, DraftMessage, InterruptOption, InterruptPayload, InterruptResolution,
    InterruptResponder, InterruptResumption, ResumeStatus, Role, SessionOwner,
};
use crate::session::command::SessionError;
use crate::session::events::EventPayload;
use crate::session::SessionEvent;
use crate::transport::channel::{Channel, ChannelContext};
use crate::{Caller, HandleClientInput, RuntimeError};

const RECONNECT_DELAY: Duration = Duration::from_secs(3);

/// Slack Socket Mode bot. A channel mention or a DM message submits a turn
/// for the configured agent — the thread is the session, everywhere: a
/// top-level DM message starts a thread and the bot answers in it.
/// Replies are posted by a checkpointed processor watching the event log, so
/// a completion survives a restart; the event is acked only after the submit
/// is durably recorded, so Slack redelivers what a crash swallows (the
/// deterministic turn id dedupes the replay).
/// An interrupt whose payload follows the AG-UI Interrupt shape posts its
/// `message` (with `metadata.options` as buttons); a click resumes the
/// interrupt with the chosen option's value.
#[derive(Clone)]
pub struct SlackChannel {
    agent_id: String,
    app_token: String,
    bot_token: String,
    tenant_id: String,
    api_base: String,
    http: reqwest::Client,
    /// Set by `run`; lets the outbound processor read sessions.
    ctx: Arc<OnceLock<ChannelContext>>,
}

impl SlackChannel {
    pub fn new(
        agent_id: String,
        app_token: String,
        bot_token: String,
        tenant_id: String,
        api_base: String,
    ) -> Self {
        Self {
            agent_id,
            app_token,
            bot_token,
            tenant_id,
            api_base,
            http: reqwest::Client::new(),
            ctx: Arc::new(OnceLock::new()),
        }
    }

    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// SLACK_APP_TOKEN and SLACK_BOT_TOKEN, reported EnvVars-style when
    /// missing. SLACK_API_BASE overrides the Slack API origin (tests).
    pub fn from_env(agent_id: String, tenant_id: String) -> Result<Self, ()> {
        let specs = [
            (
                "SLACK_APP_TOKEN",
                "App-level token with connections:write, for Socket Mode (xapp-…)",
            ),
            (
                "SLACK_BOT_TOKEN",
                "Bot token with app_mentions:read, chat:write, channels:history, im:history (xoxb-…)",
            ),
        ];
        let mut values = Vec::new();
        let mut missing = Vec::new();
        for (name, desc) in specs {
            match std::env::var(name) {
                Ok(v) => values.push(v),
                Err(_) => missing.push((name, desc)),
            }
        }
        if !missing.is_empty() {
            eprintln!("error: missing required environment variable(s):");
            for (name, desc) in &missing {
                eprintln!("  - {name}: {desc}");
            }
            eprintln!("\nSet them and try again, e.g.:");
            for (name, _) in &missing {
                eprintln!("  export {name}=...");
            }
            return Err(());
        }
        let mut it = values.into_iter();
        let api_base =
            std::env::var("SLACK_API_BASE").unwrap_or_else(|_| "https://slack.com/api".to_string());
        Ok(Self::new(
            agent_id,
            it.next().unwrap(),
            it.next().unwrap(),
            tenant_id,
            api_base,
        ))
    }

    async fn connections_open(&self) -> anyhow::Result<String> {
        let resp: Value = self
            .http
            .post(format!("{}/apps.connections.open", self.api_base))
            .bearer_auth(&self.app_token)
            .send()
            .await?
            .json()
            .await?;
        if resp["ok"].as_bool() != Some(true) {
            anyhow::bail!("apps.connections.open failed: {}", resp["error"]);
        }
        resp["url"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| anyhow::anyhow!("apps.connections.open returned no url"))
    }

    async fn connect_and_listen(&self, ctx: &ChannelContext) -> anyhow::Result<()> {
        let url = self.connections_open().await?;
        let (mut ws, _) = tokio_tungstenite::connect_async(&url).await?;
        tracing::info!("slack socket connected");

        while let Some(msg) = ws.next().await {
            match msg? {
                WsMessage::Ping(p) => ws.send(WsMessage::Pong(p)).await?,
                WsMessage::Text(text) => {
                    let envelope: Value = match serde_json::from_str(text.as_str()) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    if envelope["type"].as_str() == Some("events_api") {
                        let payload = &envelope["payload"];
                        if let Some(inbound) = app_mention(payload).or_else(|| dm_message(payload))
                        {
                            self.submit(ctx, inbound).await;
                        }
                    }
                    if envelope["type"].as_str() == Some("interactive") {
                        if let Some(click) = block_action(&envelope["payload"]) {
                            self.resolve_click(ctx, click).await;
                        }
                    }
                    // Ack after the submit: an unacked event is redelivered.
                    if let Some(id) = envelope["envelope_id"].as_str() {
                        let ack = serde_json::json!({ "envelope_id": id }).to_string();
                        ws.send(WsMessage::text(ack)).await?;
                    }
                    // Slack rotates sockets; reconnect via a fresh url.
                    if envelope["type"].as_str() == Some("disconnect") {
                        return Ok(());
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// The thread past `oldest` (exclusive), or the whole thread from the
    /// top. One page; a first fetch past 200 messages loses its tail.
    async fn fetch_thread(
        &self,
        channel: &str,
        thread_ts: &str,
        oldest: Option<&str>,
    ) -> anyhow::Result<Vec<SlackMsg>> {
        let mut url = format!(
            "{}/conversations.replies?channel={channel}&ts={thread_ts}&limit=200&include_all_metadata=true",
            self.api_base
        );
        if let Some(oldest) = oldest {
            url.push_str(&format!("&oldest={oldest}"));
        }
        let resp: Value = self
            .http
            .get(url)
            .bearer_auth(&self.bot_token)
            .send()
            .await?
            .json()
            .await?;
        if resp["ok"].as_bool() != Some(true) {
            anyhow::bail!("conversations.replies failed: {}", resp["error"]);
        }
        Ok(parse_replies(&resp))
    }

    /// The session's active path, or empty when it doesn't resolve.
    async fn session_path(&self, session_id: &str) -> Vec<crate::protocol::Message> {
        let Some(ctx) = self.ctx.get() else {
            return Vec::new();
        };
        match ctx.get_session(&self.tenant_id, session_id).await {
            Ok(session) => {
                let tree = session.state.message_tree();
                match &tree.head_id {
                    Some(head) => tree.path_to(head),
                    None => Vec::new(),
                }
            }
            Err(_) => Vec::new(),
        }
    }

    async fn submit(&self, ctx: &ChannelContext, inbound: Inbound) {
        let session_id = format!("slack:{}:{}", inbound.channel, inbound.thread_ts);
        // Deterministic per Slack message, so a redelivery dedupes.
        let turn_id = Some(format!("slack:{}:{}", inbound.channel, inbound.ts));

        // The recorded path is the cursor: its highest `slack:{ts}` id marks
        // how far into the thread the session has seen, so the fetch is just
        // the delta.
        let path = self.session_path(&session_id).await;
        let cursor = path
            .iter()
            .filter_map(|m| m.id.strip_prefix("slack:"))
            .max();

        // The fetched delta appends at the head — materialized against the
        // path at delivery, so a message queued behind an active turn lands
        // after its reply instead of forking. Without a usable fetch, append
        // the message alone with a note that context may be missing.
        let input = match self
            .fetch_thread(&inbound.channel, &inbound.thread_ts, cursor)
            .await
        {
            Ok(thread) => ClientInput::Append {
                agent_id: self.agent_id.clone(),
                turn_id: turn_id.clone(),
                messages: build_batch(&path, &thread, &inbound),
                stream: false,
                client: Default::default(),
            },
            Err(e) => {
                let hint = if !e.to_string().contains("missing_scope") {
                    ""
                } else if inbound.channel.starts_with('D') {
                    " (bot token lacks im:history?)"
                } else {
                    " (bot token lacks channels:history?)"
                };
                tracing::warn!(error = %e, "slack: history fetch failed{hint}; appending message only");
                ClientInput::Message {
                    agent_id: self.agent_id.clone(),
                    turn_id: turn_id.clone(),
                    message: draft(
                        &format!("slack:{}", inbound.ts),
                        Role::User,
                        format!(
                            "<@{}>: {}\n\n[note: the Slack conversation could not be fetched — \
                             earlier messages may be missing from your context]",
                            inbound.user, inbound.text
                        ),
                    ),
                    stream: false,
                }
            }
        };
        let submitted = ctx
            .handle_client_input(HandleClientInput {
                session_id: session_id.clone(),
                caller: Caller::System {
                    tenant_id: self.tenant_id.clone(),
                },
                owner: SessionOwner {
                    tenant_id: self.tenant_id.clone(),
                    id: Some(format!("slack:{}", inbound.user)),
                    metadata: HashMap::from([
                        ("slack_channel".into(), inbound.channel.clone()),
                        ("slack_thread_ts".into(), inbound.thread_ts.clone()),
                    ]),
                },
                input,
                span: crate::span::SpanContext::root().child("slack_inbound"),
            })
            .await;
        match submitted {
            Ok(_) => {}
            // A redelivered message whose turn already ran (or is running),
            // or a message while a prompt is pending — the thread delta
            // carries it into the next turn after the resume.
            Err(RuntimeError::Session(
                SessionError::TurnAlreadyActive { .. }
                | SessionError::TurnAlreadyCompleted { .. }
                | SessionError::SessionInterrupted,
            )) => {}
            Err(e) => {
                let meta = ReplyMeta {
                    turn_id,
                    session_id: Some(session_id),
                    ..Default::default()
                };
                if let Err(post) = self
                    .post(
                        &inbound.channel,
                        &inbound.thread_ts,
                        &format!("Error: {e}"),
                        &meta,
                    )
                    .await
                {
                    tracing::warn!(error = %post, "slack: failed to post submit error");
                }
            }
        }
    }

    async fn post(
        &self,
        channel: &str,
        thread_ts: &str,
        text: &str,
        meta: &ReplyMeta,
    ) -> Result<(), PostError> {
        self.post_blocks(channel, thread_ts, text, vec![section_block(text)], meta)
            .await
    }

    async fn post_blocks(
        &self,
        channel: &str,
        thread_ts: &str,
        text: &str,
        blocks: Vec<Value>,
        meta: &ReplyMeta,
    ) -> Result<(), PostError> {
        self.api_call(
            "chat.postMessage",
            serde_json::json!({
                "channel": channel,
                "thread_ts": thread_ts,
                // Notification fallback; the blocks carry the rendered reply.
                "text": text,
                "blocks": blocks,
                // Round-trips on fetches: maps the message back to its engine ids.
                "metadata": {
                    "event_type": REPLY_EVENT_TYPE,
                    "event_payload": meta,
                },
            }),
        )
        .await
    }

    async fn update(
        &self,
        channel: &str,
        ts: &str,
        text: &str,
        blocks: Vec<Value>,
        meta: &ReplyMeta,
    ) -> Result<(), PostError> {
        self.api_call(
            "chat.update",
            serde_json::json!({
                "channel": channel,
                "ts": ts,
                "text": text,
                "blocks": blocks,
                "metadata": {
                    "event_type": REPLY_EVENT_TYPE,
                    "event_payload": meta,
                },
            }),
        )
        .await
    }

    async fn api_call(&self, method: &str, body: Value) -> Result<(), PostError> {
        let resp = self
            .http
            .post(format!("{}/{method}", self.api_base))
            .bearer_auth(&self.bot_token)
            .json(&body)
            .send()
            .await
            .map_err(|e| PostError::Retryable(e.to_string()))?;
        let status = resp.status();
        if status.is_server_error() || status.as_u16() == 429 {
            return Err(PostError::Retryable(format!("http {status}")));
        }
        let v: Value = resp
            .json()
            .await
            .map_err(|e| PostError::Retryable(e.to_string()))?;
        if v["ok"].as_bool() == Some(true) {
            return Ok(());
        }
        let error = v["error"].as_str().unwrap_or("unknown_error").to_string();
        match error.as_str() {
            "rate_limited" | "ratelimited" | "internal_error" | "service_unavailable" => {
                Err(PostError::Retryable(error))
            }
            _ => Err(PostError::Terminal(error)),
        }
    }

    /// Resolve a clicked option against the recorded interrupt — the value
    /// comes from the stored payload, not the wire — and resume with it. The
    /// prompt message settles when the resume's event lands; a click on an
    /// already-resolved prompt just strips its stale buttons.
    async fn resolve_click(&self, ctx: &ChannelContext, click: Click) {
        let session_id = format!("slack:{}:{}", click.channel, click.thread_ts);
        let open = match ctx.get_session(&self.tenant_id, &session_id).await {
            Ok(session) => session
                .state
                .open_interrupts
                .iter()
                .find(|i| i.interrupt_id == click.interrupt_id)
                .map(|i| i.payload.clone()),
            Err(e) => {
                tracing::warn!(error = %e, %session_id, "slack: click on unreadable session");
                return;
            }
        };
        let Some(payload) = open else {
            let meta = ReplyMeta {
                interrupt_id: Some(click.interrupt_id),
                session_id: Some(session_id),
                ..Default::default()
            };
            let text = format!("{}\n\n(no longer active)", click.message_text);
            let cleared = self
                .update(
                    &click.channel,
                    &click.message_ts,
                    &text,
                    vec![section_block(&text)],
                    &meta,
                )
                .await;
            if let Err(e) = cleared {
                tracing::warn!(error = %e, "slack: failed to clear stale prompt");
            }
            return;
        };
        let option = display_of(&payload).and_then(|d| d.options.into_iter().nth(click.option));
        let Some(option) = option else {
            tracing::warn!(
                interrupt_id = %click.interrupt_id,
                option = click.option,
                "slack: click has no matching option"
            );
            return;
        };
        let resumed = ctx
            .handle_client_input(HandleClientInput {
                session_id: session_id.clone(),
                caller: Caller::System {
                    tenant_id: self.tenant_id.clone(),
                },
                // Resumes never touch ownership; addressing only.
                owner: SessionOwner {
                    tenant_id: self.tenant_id.clone(),
                    id: None,
                    metadata: HashMap::new(),
                },
                input: ClientInput::InterruptResume {
                    resumption: InterruptResumption {
                        interrupt_id: click.interrupt_id,
                        payload: serde_json::to_value(InterruptResolution {
                            status: ResumeStatus::Resolved,
                            payload: option.value,
                            responder: Some(InterruptResponder {
                                channel: "slack".to_string(),
                                user: Some(click.user),
                                label: Some(option.label),
                            }),
                        })
                        .unwrap_or_default(),
                    },
                },
                span: crate::span::SpanContext::root().child("slack_click"),
            })
            .await;
        if let Err(e) = resumed {
            tracing::warn!(error = %e, %session_id, "slack: interrupt resume failed");
        }
    }
}

#[async_trait::async_trait]
impl Channel for SlackChannel {
    fn kind(&self) -> &'static str {
        "slack"
    }

    async fn run(&self, ctx: ChannelContext) {
        let _ = self.ctx.set(ctx.clone());
        let spawned = ctx
            .spawn_processor(
                Arc::new(self.clone()),
                EventProcessorRunnerConfig {
                    owner_id: Some("slack_outbound".to_string()),
                    ..Default::default()
                },
                // A new deployment must not replay history into Slack.
                true,
            )
            .await;
        if let Err(e) = spawned {
            tracing::error!(error = %e, "slack: failed to start outbound processor");
            return;
        }

        loop {
            tokio::select! {
                _ = ctx.shutdown.cancelled() => return,
                r = self.connect_and_listen(&ctx) => {
                    if let Err(e) = r {
                        tracing::warn!(error = %e, "slack socket error");
                    }
                }
            }
            tokio::select! {
                _ = ctx.shutdown.cancelled() => return,
                _ = tokio::time::sleep(RECONNECT_DELAY) => {}
            }
        }
    }
}

/// Durable outbound side: everything a reply needs is recoverable from the
/// event log — the session id encodes the destination, the event carries the
/// text. At-least-once: a crash between post and checkpoint may repost.
#[async_trait::async_trait]
impl EventProcessor for SlackChannel {
    fn name(&self) -> &'static str {
        "slack_outbound_v1"
    }

    async fn apply(&self, event: SessionEvent) -> Result<(), ProcessorError> {
        if event.tenant_id != self.tenant_id {
            return Ok(());
        }
        let Some((channel_id, thread_ts)) = slack_session(&event.session_id) else {
            return Ok(());
        };
        let result = match &event.payload {
            EventPayload::TurnCompleted(t) => {
                self.post_turn(channel_id, thread_ts, &event.session_id, t)
                    .await
            }
            EventPayload::SessionInterrupted(p) => {
                self.post_interrupt(channel_id, thread_ts, &event.session_id, p)
                    .await
            }
            EventPayload::InterruptResumed(p) => self.settle_prompt(channel_id, thread_ts, p).await,
            _ => return Ok(()),
        };
        match result {
            Ok(()) => Ok(()),
            // Erring would wedge the processor on this event forever.
            Err(PostError::Terminal(e)) => {
                tracing::warn!(session_id = %event.session_id, error = %e, "slack: dropping undeliverable reply");
                Ok(())
            }
            Err(PostError::Retryable(e)) => Err(ProcessorError::Apply(e)),
        }
    }
}

impl SlackChannel {
    async fn post_turn(
        &self,
        channel_id: &str,
        thread_ts: &str,
        session_id: &str,
        t: &crate::session::events::TurnCompleted,
    ) -> Result<(), PostError> {
        // A crash between post and checkpoint redelivers the event.
        // The reply always posts after its trigger message, whose ts
        // the turn id carries — fetch past it and skip if the reply
        // is already there. A failed fetch posts anyway: at-least-once.
        let oldest = slack_session(&t.turn_id).map(|(_, ts)| ts);
        if let Ok(thread) = self.fetch_thread(channel_id, thread_ts, oldest).await {
            if thread.iter().any(|m| {
                m.meta.as_ref().and_then(|r| r.turn_id.as_deref()) == Some(t.turn_id.as_str())
            }) {
                return Ok(());
            }
        }
        // The reply's recorded node — the id fetches map back to.
        let message_id = self
            .session_path(session_id)
            .await
            .last()
            .filter(|m| matches!(m.role, Role::Assistant))
            .map(|m| m.id.clone());
        let meta = ReplyMeta {
            turn_id: Some(t.turn_id.clone()),
            message_id,
            session_id: Some(session_id.to_string()),
            ..Default::default()
        };
        self.post(channel_id, thread_ts, &turn_result_text(t), &meta)
            .await
    }

    /// A prompt-carrying interrupt posts as buttons, anything else as
    /// "Paused: {reason}". Redelivery dedupes on the stamped interrupt id,
    /// like replies dedupe on their turn id.
    async fn post_interrupt(
        &self,
        channel_id: &str,
        thread_ts: &str,
        session_id: &str,
        p: &crate::session::events::SessionInterrupted,
    ) -> Result<(), PostError> {
        if let Ok(thread) = self.fetch_thread(channel_id, thread_ts, None).await {
            if thread.iter().any(|m| {
                m.meta.as_ref().and_then(|r| r.interrupt_id.as_deref())
                    == Some(p.interrupt_id.as_str())
            }) {
                return Ok(());
            }
        }
        let meta = ReplyMeta {
            interrupt_id: Some(p.interrupt_id.clone()),
            session_id: Some(session_id.to_string()),
            ..Default::default()
        };
        match display_of(&p.payload) {
            Some(display) => {
                let blocks = prompt_blocks(&display, &p.interrupt_id);
                self.post_blocks(channel_id, thread_ts, &display.message, blocks, &meta)
                    .await
            }
            // The categorical `reason` is the last resort.
            None => {
                self.post(
                    channel_id,
                    thread_ts,
                    &format!("Paused: {}", p.reason),
                    &meta,
                )
                .await
            }
        }
    }

    /// Close the loop on the posted prompt, whoever resumed it — a click,
    /// the API, a timeout: strip its buttons and stamp the outcome.
    /// Best-effort: an unfindable prompt message is not worth wedging on.
    async fn settle_prompt(
        &self,
        channel_id: &str,
        thread_ts: &str,
        p: &crate::session::events::InterruptResumed,
    ) -> Result<(), PostError> {
        let Ok(thread) = self.fetch_thread(channel_id, thread_ts, None).await else {
            tracing::warn!(interrupt_id = %p.interrupt_id, "slack: fetch failed; prompt not settled");
            return Ok(());
        };
        let Some(msg) = thread.iter().find(|m| {
            m.meta.as_ref().and_then(|r| r.interrupt_id.as_deref()) == Some(p.interrupt_id.as_str())
        }) else {
            return Ok(());
        };
        let text = format!("{}\n\n{}", msg.text, resolution_text(&p.payload));
        let meta = ReplyMeta {
            interrupt_id: Some(p.interrupt_id.clone()),
            session_id: Some(format!("slack:{channel_id}:{thread_ts}")),
            ..Default::default()
        };
        self.update(
            channel_id,
            &msg.ts,
            &text,
            vec![section_block(&text)],
            &meta,
        )
        .await
    }
}

enum PostError {
    Retryable(String),
    Terminal(String),
}

impl std::fmt::Display for PostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PostError::Retryable(e) | PostError::Terminal(e) => write!(f, "{e}"),
        }
    }
}

/// A message the bot should answer: a channel mention or a DM.
#[derive(Debug, PartialEq)]
struct Inbound {
    channel: String,
    /// The parent ts: the session anchor and the reply target. A top-level
    /// message (mention or DM) starts its thread at its own ts.
    thread_ts: String,
    /// The message's own ts; unique per message, keys the turn.
    ts: String,
    user: String,
    text: String,
}

/// A usable `app_mention` from an `events_api` payload; bot echoes,
/// non-mention events, and DM mentions (which arrive as `message.im` too —
/// that path owns them) are `None`.
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
        text: event["text"].as_str()?.to_string(),
    })
}

/// A user's DM (`message` in an `im` channel); bot echoes and subtyped
/// messages (edits, joins) are `None`. Threads always, even in a DM: a
/// top-level message starts one at its own ts, so context stays per-thread.
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
        text: event["text"].as_str()?.to_string(),
    })
}

/// The `(channel, thread_ts)` behind a `slack:{channel}:{thread_ts}` session.
fn slack_session(session_id: &str) -> Option<(&str, &str)> {
    session_id.strip_prefix("slack:")?.split_once(':')
}

const REPLY_EVENT_TYPE: &str = "substructure_reply";

/// The engine ids stamped on a posted reply and read back on fetches — the
/// message becomes addressable in both worlds.
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
    /// The author's user id, or a bot's `bot_id` when no user rides along.
    author: Option<String>,
    /// This bot's reply stamp, when present.
    meta: Option<ReplyMeta>,
    text: String,
}

/// Plain thread messages from a `conversations.replies` response; joins and
/// other subtyped messages are dropped.
fn parse_replies(resp: &Value) -> Vec<SlackMsg> {
    let Some(messages) = resp["messages"].as_array() else {
        return Vec::new();
    };
    messages
        .iter()
        .filter_map(|m| {
            if m["type"].as_str() != Some("message") || m["subtype"].is_string() {
                return None;
            }
            let meta = (m["metadata"]["event_type"].as_str() == Some(REPLY_EVENT_TYPE))
                .then(|| serde_json::from_value(m["metadata"]["event_payload"].clone()).ok())
                .flatten();
            Some(SlackMsg {
                ts: m["ts"].as_str()?.to_string(),
                author: m["user"]
                    .as_str()
                    .or_else(|| m["bot_id"].as_str())
                    .map(str::to_string),
                meta,
                text: m["text"].as_str().unwrap_or_default().to_string(),
            })
        })
        .collect()
}

/// The append batch for an inbound message: fetched messages not yet on the
/// path, as attributed user messages under `slack:{ts}`. Our own replies map
/// back to their recorded assistant nodes via the stamped message id —
/// skipped when already on the path, rebuilt in place as assistant messages
/// when not (a lost session recovers from its conversation). The live
/// message is appended when the fetch missed it (race, or an empty delta).
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
            // Ours: anchor to the recorded assistant node. An id-less stamp
            // has nothing to anchor — skip it.
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

/// The renderable half of an AG-UI-shaped interrupt payload. `message` is
/// the gate; options ride `metadata.options` (unparseable options render a
/// message without buttons, not a fallback).
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

/// Buttons carry coordinates only (interrupt id + option index); the value is
/// read from the recorded interrupt at click time, so a click can't smuggle
/// a value and Slack's button-value size cap never binds.
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
        "text": { "type": "mrkdwn", "text": block_text(text) },
    })
}

/// A button click from a `block_actions` payload.
#[derive(Debug, PartialEq)]
struct Click {
    interrupt_id: String,
    option: usize,
    user: String,
    channel: String,
    /// The prompt message the buttons live on, and its thread.
    message_ts: String,
    thread_ts: String,
    message_text: String,
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
    })
}

/// How a resolution renders on the settled prompt.
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

/// A section block caps mrkdwn text at 3000 chars; overflow would fail the
/// whole post terminally, so truncate the block (the `text` fallback keeps
/// the full reply).
fn block_text(text: &str) -> String {
    const MAX: usize = 3000;
    if text.chars().count() <= MAX {
        return text.to_string();
    }
    let cut: String = text.chars().take(MAX - 1).collect();
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
        // A mention inside a DM fires both app_mention and message.im; only
        // the message path may claim it or two sessions would race.
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
        }
    }

    fn mention_at(ts: &str) -> Inbound {
        Inbound {
            channel: "C1".into(),
            thread_ts: "1.0".into(),
            ts: ts.into(),
            user: "U2".into(),
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
        let msgs = parse_replies(&resp);
        assert_eq!(msgs.len(), 3);
        assert!(msgs[0].meta.is_none());
        let meta = msgs[1].meta.as_ref().unwrap();
        assert_eq!(meta.turn_id.as_deref(), Some("t"));
        assert_eq!(meta.message_id.as_deref(), Some("uuid-a1"));
        assert!(msgs[2].meta.is_none());
        assert_eq!(msgs[2].author.as_deref(), Some("B9"));
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
        // An old-style stamp with no message id has nothing to anchor: skip.
        let unmapped = SlackMsg {
            ts: "2.5".into(),
            author: Some("UBOT".into()),
            meta: Some(ReplyMeta::default()),
            text: "reply".into(),
        };
        // A double post (pre-dedupe crash) stamps the same id twice: one node.
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

        // No message, nothing to render.
        assert!(display_of(&serde_json::json!({ "custom": "x" })).is_none());
        assert!(display_of(&serde_json::json!(null)).is_none());
        // Options are optional or unparseable: the message still renders.
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
        // Unknown styles are dropped, not forwarded to Slack.
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
        // A foreign button whose value isn't our coordinates is not a click.
        let mut foreign = payload;
        foreign["actions"][0]["value"] = "not json".into();
        assert_eq!(block_action(&foreign), None);
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
        // A prompt stamp has an interrupt id but no message id: nothing to
        // anchor in the transcript, so batches skip it.
        let prompt_post = SlackMsg {
            ts: "6.0".into(),
            author: Some("UBOT".into()),
            meta: Some(ReplyMeta {
                interrupt_id: Some("int-1".into()),
                session_id: Some("slack:C1:1.0".into()),
                ..Default::default()
            }),
            text: "Run it?".into(),
        };
        let batch = build_batch(&[], &[prompt_post], &mention_at("9.0"));
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].id.as_deref(), Some("slack:9.0"));
    }

    #[test]
    fn block_text_truncates_at_the_section_limit() {
        assert_eq!(block_text("short"), "short");
        let long = "é".repeat(3500);
        let out = block_text(&long);
        assert_eq!(out.chars().count(), 3000);
        assert!(out.ends_with('…'));
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
