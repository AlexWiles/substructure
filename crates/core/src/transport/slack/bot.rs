use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::Duration;

use serde_json::Value;
use tokio::sync::Notify;

use super::activity::TurnActivity;
use super::controller::{
    Action, ActionOutcome, Context, DefaultController, PromptView, Rendered, SlackController,
};
use super::state::StreamStore;
use super::{
    app_mention, block_action, build_batch, clip, display_of, dm_message, draft, foreign_action,
    prompt_options, resolution_text, section_block, slack_session, unstamped_ours, Click,
    ForeignClick, Inbound, ReplyMeta, MAX_FALLBACK, MAX_MARKDOWN, REPLY_EVENT_TYPE,
};
use crate::event_store::Seq;
use crate::processor::{EventProcessor, EventProcessorRunnerConfig, ProcessorError};
use crate::protocol::{
    ClientInput, InterruptResolution, InterruptResponder, InterruptResumption, ResumeStatus, Role,
    SessionOwner,
};
use crate::session::command::SessionError;
use crate::session::events::EventPayload;
use crate::session::SessionEvent;
use crate::transport::channel::ChannelContext;
use crate::{Caller, HandleClientInput, RuntimeError};

/// Slack limits the rate of `chat.appendStream`.
const ACTIVITY_INTERVAL: Duration = Duration::from_secs(1);
/// Shown as `<app> is thinking…`.
/// Slack removes a status after two minutes. Set it again on this cadence.
const STATUS_REFRESH: Duration = Duration::from_secs(90);

/// Which agent answers where. Three separate questions, so three settings:
/// who takes DMs, who takes a mention in a channel nobody named, and who takes
/// each channel that is named. Silence is the default for all three — nothing
/// answers anywhere until something says so.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Routing {
    dm: Option<String>,
    mentions: Option<String>,
    /// The agent for each named channel, or `None` where the bot stays out.
    channels: HashMap<String, Option<String>>,
}

impl Routing {
    /// Routes nothing. Build it up with [`Routing::dm`],
    /// [`Routing::mentions`], and [`Routing::channel`].
    pub fn new() -> Self {
        Self::default()
    }

    /// `agent` answers direct messages.
    pub fn dm(mut self, agent: Option<String>) -> Self {
        self.dm = agent;
        self
    }

    /// `agent` answers a mention in any channel that `channel` does not name —
    /// a mention being the only way a channel reaches the bot at all. Absent,
    /// an unnamed channel is not served, which is what makes the channel table
    /// an allowlist.
    pub fn mentions(mut self, agent: Option<String>) -> Self {
        self.mentions = agent;
        self
    }

    /// `agent` answers in `id`; `None` keeps the bot out of it, even when
    /// `mentions` is set.
    pub fn channel(mut self, id: impl Into<String>, agent: Option<String>) -> Self {
        self.channels.insert(id.into(), agent);
        self
    }

    /// The agent that answers in `channel`, or `None` where the bot is silent.
    ///
    /// A DM (`D…`) resolves only against `dm`: `mentions` is about channels,
    /// and a DM reaches the bot without anybody mentioning it.
    pub fn agent_for(&self, channel: &str) -> Option<&str> {
        if channel.starts_with('D') {
            return self.dm.as_deref();
        }
        match self.channels.get(channel) {
            Some(entry) => entry.as_deref(),
            None => self.mentions.as_deref(),
        }
    }

    /// Whether this routes anything at all.
    pub fn is_empty(&self) -> bool {
        self.dm.is_none() && self.mentions.is_none() && self.channels.is_empty()
    }
}

/// The line the server logs at startup, so a misrouted channel is visible
/// without the file.
impl std::fmt::Display for Routing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut parts: Vec<String> = Vec::new();
        if let Some(agent) = &self.dm {
            parts.push(format!("dm→{agent}"));
        }
        if let Some(agent) = &self.mentions {
            parts.push(format!("mentions→{agent}"));
        }
        let mut channels: Vec<_> = self.channels.iter().collect();
        channels.sort_by(|a, b| a.0.cmp(b.0));
        for (id, agent) in channels {
            parts.push(match agent {
                Some(agent) => format!("{id}→{agent}"),
                None => format!("{id} off"),
            });
        }
        match parts.is_empty() {
            true => write!(f, "nothing"),
            false => write!(f, "{}", parts.join(", ")),
        }
    }
}

/// One Slack install.
pub struct Workspace {
    pub bot_token: String,
    /// Must be unique for each install, not for each team: session ids carry
    /// only the channel and thread, so two of our apps in one workspace share
    /// a tenant's sessions and stream rows.
    pub tenant_id: String,
    pub routing: Routing,
    identity: tokio::sync::OnceCell<Identity>,
}

impl Workspace {
    pub fn new(bot_token: String, tenant_id: String, routing: Routing) -> Self {
        Self {
            bot_token,
            tenant_id,
            routing,
            identity: tokio::sync::OnceCell::new(),
        }
    }
}

/// Our ids from `auth.test`.
#[derive(Default)]
struct Identity {
    ours: Vec<String>,
    team: Option<String>,
}

/// Maps an event to its install; `None` for an unknown install.
/// `by_tenant` must return the same workspace that `by_install` supplies.
#[async_trait::async_trait]
pub trait WorkspaceResolver: Send + Sync {
    /// An install is a team and an app: one workspace can hold more than one
    /// of our apps, so the team alone is ambiguous. `app_id` is the event's
    /// `api_app_id`. `channel` is the event's channel, so one install can
    /// serve a different tenant per channel.
    async fn by_install(
        &self,
        team_id: Option<&str>,
        app_id: Option<&str>,
        channel: &str,
    ) -> Option<Arc<Workspace>>;
    async fn by_tenant(&self, tenant_id: &str) -> Option<Arc<Workspace>>;
}

/// One turn's stream. A queued turn takes its slot while the turn before it
/// is still settling, so the turn is part of the key; session ids are unique
/// only in one workspace.
#[derive(Clone, PartialEq, Eq, Hash)]
struct StreamKey {
    tenant_id: String,
    session_id: String,
    turn_id: String,
}

impl StreamKey {
    fn new(tenant_id: &str, session_id: &str, turn_id: &str) -> Self {
        Self {
            tenant_id: tenant_id.to_string(),
            session_id: session_id.to_string(),
            turn_id: turn_id.to_string(),
        }
    }
}

/// Where a reply goes.
#[derive(Clone)]
struct Thread {
    channel: String,
    ts: String,
}

impl Thread {
    fn new(channel: &str, ts: &str) -> Self {
        Self {
            channel: channel.to_string(),
            ts: ts.to_string(),
        }
    }
}

/// The live streaming message for the session's current turn. The durable
/// half lives in [`StreamStore`]; a restarted process rebuilds this slot and
/// keeps appending to the same message.
#[derive(Clone)]
struct Stream {
    tenant_id: String,
    session_id: String,
    turn_id: String,
    start_seq: u64,
    started_at: chrono::DateTime<chrono::Utc>,
    /// Required by `chat.startStream` outside a DM.
    recipient: Option<String>,
    recipient_team: Option<String>,
    ts: Option<String>,
    /// The last chunk for each task. An append sends only the changes.
    sent: HashMap<String, Value>,
    /// Slack rejected the stream. Post the reply instead.
    dead: bool,
    /// The store row's version; the fence against a second writer.
    version: u64,
}

impl Stream {
    fn key(&self) -> StreamKey {
        StreamKey::new(&self.tenant_id, &self.session_id, &self.turn_id)
    }
}

/// A poisoned lock guards only bookkeeping; keep serving.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The open streams, and the sessions whose activity is not yet rendered.
#[derive(Default)]
struct Streams {
    open: Mutex<HashMap<StreamKey, Stream>>,
    dirty: Mutex<HashSet<StreamKey>>,
    notify: Notify,
}

impl Streams {
    fn insert(&self, stream: Stream) {
        lock(&self.open).insert(stream.key(), stream);
    }

    fn remove(&self, key: &StreamKey) {
        lock(&self.open).remove(key);
    }

    /// Every turn of the session: a cancel ends them all.
    fn remove_session(&self, tenant_id: &str, session_id: &str) {
        lock(&self.open).retain(|k, _| k.tenant_id != tenant_id || k.session_id != session_id);
    }

    /// Remove the slot; the worker cannot append after this.
    fn take(&self, key: &StreamKey) -> Option<Stream> {
        lock(&self.open).remove(key).filter(|s| !s.dead)
    }

    fn get(&self, key: &StreamKey) -> Option<Stream> {
        lock(&self.open).get(key).cloned()
    }

    fn kill(&self, key: &StreamKey) {
        if let Some(stream) = lock(&self.open).get_mut(key) {
            stream.dead = true;
        }
    }

    /// Another turn of this session still holding a message.
    fn open_elsewhere(&self, key: &StreamKey) -> bool {
        lock(&self.open).iter().any(|(k, s)| {
            k.tenant_id == key.tenant_id
                && k.session_id == key.session_id
                && k.turn_id != key.turn_id
                && !s.dead
                && s.ts.is_some()
        })
    }

    /// Record what was sent, unless the turn ended during the append.
    fn commit(&self, key: &StreamKey, ts: String, version: u64, sent: Vec<(String, Value)>) {
        let mut open = lock(&self.open);
        let Some(stream) = open.get_mut(key) else {
            return;
        };
        stream.ts = Some(ts);
        stream.version = version;
        stream.sent.extend(sent);
    }

    /// Live turns with no card yet.
    fn waiting(&self) -> Vec<StreamKey> {
        lock(&self.open)
            .values()
            .filter(|s| !s.dead && s.ts.is_none())
            .map(Stream::key)
            .collect()
    }

    fn mark_dirty(&self, key: StreamKey) {
        lock(&self.dirty).insert(key);
        self.notify.notify_one();
    }

    fn take_dirty(&self) -> Option<StreamKey> {
        let mut dirty = lock(&self.dirty);
        let key = dirty.iter().next().cloned()?;
        dirty.remove(&key);
        Some(key)
    }

    async fn notified(&self) {
        self.notify.notified().await;
    }
}

/// The bot behavior, resolved to a workspace for each event. A transport
/// (Socket Mode, webhooks) parses its deliveries and calls
/// [`handle_event`](Self::handle_event) or
/// [`handle_interaction`](Self::handle_interaction). Call
/// [`start`](Self::start) first.
#[derive(Clone)]
pub struct SlackBot {
    resolver: Arc<dyn WorkspaceResolver>,
    api_base: String,
    http: reqwest::Client,
    ctx: Arc<OnceLock<ChannelContext>>,
    streams: Arc<Streams>,
    store: Option<Arc<StreamStore>>,
    /// What a turn says. Delivery stays here; the words are the controller's.
    controller: Arc<dyn SlackController>,
}

impl SlackBot {
    /// `store` holds durable stream state: a restart resumes open streaming
    /// messages in place. Without one a restart orphans them.
    pub fn new(
        resolver: Arc<dyn WorkspaceResolver>,
        api_base: String,
        store: Option<StreamStore>,
    ) -> Self {
        Self {
            resolver,
            api_base,
            http: reqwest::Client::new(),
            ctx: Arc::new(OnceLock::new()),
            streams: Arc::new(Streams::default()),
            store: store.map(Arc::new),
            controller: Arc::new(DefaultController),
        }
    }

    /// Speak with a different voice: billing copy for a `budget_exceeded`, a
    /// link under every message. Delivery is unchanged.
    pub fn with_controller(mut self, controller: Arc<dyn SlackController>) -> Self {
        self.controller = controller;
        self
    }

    /// Start the outbound processor and the activity worker.
    pub async fn start(&self, ctx: &ChannelContext) -> Result<(), RuntimeError> {
        let _ = self.ctx.set(ctx.clone());
        let spawned = ctx
            .spawn_processor(
                Arc::new(self.clone()),
                EventProcessorRunnerConfig {
                    owner_id: Some("slack_outbound".to_string()),
                    ..Default::default()
                },
                // Do not replay history into Slack.
                true,
            )
            .await;
        if let Err(e) = spawned {
            tracing::error!(error = %e, "slack: failed to start outbound processor");
            return Err(e);
        }
        let (this, activity_ctx) = (self.clone(), ctx.clone());
        tokio::spawn(async move { this.stream_activity(&activity_ctx).await });
        Ok(())
    }

    /// Handle an `events_api` payload (an `event_callback` body).
    pub async fn handle_event(&self, payload: &Value) {
        let Some(inbound) = app_mention(payload).or_else(|| dm_message(payload)) else {
            return;
        };
        let Some(ctx) = self.ctx.get().cloned() else {
            tracing::warn!("slack: event before start; dropped");
            return;
        };
        // The delivered workspace, not the asker's team.
        let team = payload["team_id"].as_str();
        let app = payload["api_app_id"].as_str();
        let Some(ws) = self.resolver.by_install(team, app, &inbound.channel).await else {
            tracing::warn!(
                team = %team.unwrap_or(""),
                app = %app.unwrap_or(""),
                channel = %inbound.channel,
                "slack: event for unknown workspace"
            );
            return;
        };
        self.submit(&ctx, &ws, inbound).await;
    }

    /// Handle an `interactive` payload (a `block_actions` body).
    ///
    /// A press on one of the bot's own prompt buttons answers its interrupt.
    /// Anything else is a button the controller invented, and goes back to it.
    pub async fn handle_interaction(&self, payload: &Value) {
        let Some(ctx) = self.ctx.get().cloned() else {
            tracing::warn!("slack: interaction before start; dropped");
            return;
        };
        let channel = match block_action(payload) {
            Some(click) => click.channel.clone(),
            None => match foreign_action(payload) {
                Some(click) => click.channel.clone(),
                None => return,
            },
        };
        let team = payload["team"]["id"]
            .as_str()
            .or_else(|| payload["user"]["team_id"].as_str());
        let app = payload["api_app_id"].as_str();
        let Some(ws) = self.resolver.by_install(team, app, &channel).await else {
            tracing::warn!(
                team = %team.unwrap_or(""),
                app = %app.unwrap_or(""),
                %channel,
                "slack: click for unknown workspace"
            );
            return;
        };
        match block_action(payload) {
            Some(click) => self.resolve_click(&ctx, &ws, click).await,
            None => {
                if let Some(click) = foreign_action(payload) {
                    self.controller_action(&ctx, &ws, click).await;
                }
            }
        }
    }

    /// Hand a controller's own button back to it and perform what it asks for.
    async fn controller_action(&self, ctx: &ChannelContext, ws: &Workspace, click: ForeignClick) {
        let session_id = format!("slack:{}:{}", click.channel, click.thread_ts);
        let outcome = self
            .controller
            .on_action(&Action {
                action_id: &click.action_id,
                value: &click.value,
                tenant_id: &ws.tenant_id,
                session_id: &session_id,
                user: &click.user,
                channel: &click.channel,
                message_ts: &click.message_ts,
                thread_ts: &click.thread_ts,
            })
            .await;
        let thread = Thread::new(&click.channel, &click.thread_ts);
        let result = match outcome {
            ActionOutcome::Ignored => {
                tracing::debug!(action_id = %click.action_id, "slack: click ignored");
                return;
            }
            ActionOutcome::Update(r) => {
                let meta = ReplyMeta {
                    session_id: Some(session_id),
                    ..Default::default()
                };
                self.update(
                    ws,
                    &click.channel,
                    &click.message_ts,
                    &r.text,
                    r.blocks,
                    &meta,
                )
                .await
            }
            ActionOutcome::Reply(r) => {
                let meta = ReplyMeta {
                    session_id: Some(session_id),
                    ..Default::default()
                };
                self.post_blocks(ws, &thread, &r.text, r.blocks, &meta)
                    .await
            }
            // The click becomes a turn, as though the user had typed it.
            ActionOutcome::Submit(text) => {
                self.submit(
                    ctx,
                    ws,
                    Inbound {
                        channel: click.channel.clone(),
                        thread_ts: click.thread_ts.clone(),
                        ts: format!("{}:{}", click.message_ts, click.action_id),
                        user: click.user.clone(),
                        team: None,
                        text,
                    },
                )
                .await;
                Ok(())
            }
        };
        if let Err(e) = result {
            tracing::warn!(error = %e, action_id = %click.action_id, "slack: action outcome failed");
        }
    }

    async fn identity<'a>(&self, ws: &'a Workspace) -> Option<&'a Identity> {
        let loaded = ws
            .identity
            .get_or_try_init(|| async {
                let resp = self
                    .api_call(ws, "auth.test", serde_json::json!({}))
                    .await?;
                Ok::<_, Error>(Identity {
                    ours: ["user_id", "bot_id"]
                        .iter()
                        .filter_map(|k| resp[k].as_str().map(str::to_string))
                        .collect(),
                    team: resp["team_id"].as_str().map(str::to_string),
                })
            })
            .await;
        match loaded {
            Ok(identity) => Some(identity),
            Err(e) => {
                tracing::warn!(error = %e, "slack: auth.test failed; own posts unrecognised");
                None
            }
        }
    }

    /// The thread after `oldest` (exclusive). One page of 200 messages.
    async fn fetch_thread(
        &self,
        ws: &Workspace,
        thread: &Thread,
        oldest: Option<&str>,
    ) -> Result<Vec<super::SlackMsg>, Error> {
        let resp = self.fetch_replies_raw(ws, thread, oldest).await?;
        let ours = self.identity(ws).await.map_or(&[][..], |i| &i.ours);
        Ok(super::parse_replies(&resp, ours))
    }

    /// One raw `conversations.replies` page after `oldest` (exclusive).
    async fn fetch_replies_raw(
        &self,
        ws: &Workspace,
        thread: &Thread,
        oldest: Option<&str>,
    ) -> Result<Value, Error> {
        let (channel, thread_ts) = (&thread.channel, &thread.ts);
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
            .bearer_auth(&ws.bot_token)
            .send()
            .await
            .map_err(Error::retryable)?
            .json()
            .await
            .map_err(Error::retryable)?;
        if resp["ok"].as_bool() != Some(true) {
            return Err(Error::from_response(&resp));
        }
        Ok(resp)
    }

    /// The session's active path, or empty.
    async fn session_path(
        &self,
        ws: &Workspace,
        session_id: &str,
    ) -> Vec<crate::protocol::Message> {
        let Some(ctx) = self.ctx.get() else {
            return Vec::new();
        };
        match ctx.get_session(&ws.tenant_id, session_id).await {
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

    async fn submit(&self, ctx: &ChannelContext, ws: &Workspace, inbound: Inbound) {
        // Before the thread fetch: a channel nobody answers in costs no API
        // calls. A click is deliberately not gated the same way — a prompt
        // already posted has to stay answerable after its channel goes off.
        let Some(agent_id) = ws.routing.agent_for(&inbound.channel) else {
            tracing::debug!(channel = %inbound.channel, "slack: no agent for channel; ignored");
            return;
        };
        let agent_id = agent_id.to_string();
        let thread = Thread::new(&inbound.channel, &inbound.thread_ts);
        let session_id = format!("slack:{}:{}", inbound.channel, inbound.thread_ts);
        // Deterministic for each message: a redelivery dedupes.
        let turn_id = Some(format!("slack:{}:{}", inbound.channel, inbound.ts));

        // The highest recorded `slack:{ts}` id is the fetch cursor.
        let path = self.session_path(ws, &session_id).await;
        let cursor = path
            .iter()
            .filter_map(|m| m.id.strip_prefix("slack:"))
            .max();

        let fetched = self.fetch_thread(ws, &thread, cursor).await;

        // If the fetch fails, append the message alone with a note.
        let input = match fetched {
            Ok(replies) => ClientInput::Append {
                agent_id: agent_id.clone(),
                turn_id: turn_id.clone(),
                messages: build_batch(&path, &replies, &inbound),
                stream: false,
                client: Default::default(),
                // A mention that lands mid-turn is a question, not a mistake:
                // hold it and answer it in the next turn.
                queue: true,
            },
            Err(e) => {
                let hint = if e.code() != "missing_scope" {
                    ""
                } else if inbound.channel.starts_with('D') {
                    " (bot token lacks im:history?)"
                } else {
                    " (bot token lacks channels:history?)"
                };
                tracing::warn!(error = %e, "slack: history fetch failed{hint}; appending message only");
                ClientInput::Message {
                    agent_id: agent_id.clone(),
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
                    queue: true,
                }
            }
        };
        let submitted = ctx
            .handle_client_input(HandleClientInput {
                session_id: session_id.clone(),
                caller: Caller::System {
                    tenant_id: ws.tenant_id.clone(),
                },
                owner: SessionOwner {
                    tenant_id: ws.tenant_id.clone(),
                    id: Some(format!("slack:{}", inbound.user)),
                    metadata: HashMap::from_iter(
                        [
                            Some(("slack_channel".into(), inbound.channel.clone())),
                            Some(("slack_thread_ts".into(), inbound.thread_ts.clone())),
                            inbound.team.clone().map(|team| ("slack_team".into(), team)),
                        ]
                        .into_iter()
                        .flatten(),
                    ),
                },
                input,
                span: crate::span::SpanContext::root().child("slack_inbound"),
            })
            .await;
        match submitted {
            Ok(_) => {}
            // A true redelivery of a message already taken, or a message while
            // a prompt is open. Nothing to say: the first copy is queued or
            // running, and an interrupt owes the user an answer first.
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
                if let Err(post) = self.post(ws, &thread, &format!("Error: {e}"), &meta).await {
                    tracing::warn!(error = %post, "slack: failed to post submit error");
                }
            }
        }
    }

    async fn post(
        &self,
        ws: &Workspace,
        thread: &Thread,
        text: &str,
        meta: &ReplyMeta,
    ) -> Result<(), Error> {
        self.post_blocks(ws, thread, text, vec![section_block(text)], meta)
            .await
    }

    async fn post_blocks(
        &self,
        ws: &Workspace,
        thread: &Thread,
        text: &str,
        blocks: Vec<Value>,
        meta: &ReplyMeta,
    ) -> Result<(), Error> {
        self.api_call(
            ws,
            "chat.postMessage",
            serde_json::json!({
                "channel": thread.channel,
                "thread_ts": thread.ts,
                // Notification fallback; the blocks carry the reply.
                "text": clip(text, MAX_FALLBACK),
                "blocks": blocks,
                // Maps the message back to its engine ids.
                "metadata": {
                    "event_type": REPLY_EVENT_TYPE,
                    "event_payload": meta,
                },
            }),
        )
        .await
        .map(|_| ())
    }

    async fn update(
        &self,
        ws: &Workspace,
        channel: &str,
        ts: &str,
        text: &str,
        blocks: Vec<Value>,
        meta: &ReplyMeta,
    ) -> Result<(), Error> {
        self.api_call(
            ws,
            "chat.update",
            serde_json::json!({
                "channel": channel,
                "ts": ts,
                "text": clip(text, MAX_FALLBACK),
                "blocks": blocks,
                "metadata": {
                    "event_type": REPLY_EVENT_TYPE,
                    "event_payload": meta,
                },
            }),
        )
        .await
        .map(|_| ())
    }

    /// Best-effort thread status. An empty status clears it.
    /// The working indicator, as the controller words it. `None` clears it, and
    /// so does the end of a turn.
    async fn set_working(&self, ws: &Workspace, thread: &Thread, ctx: &Context<'_>) {
        let status = self.controller.status(ctx).unwrap_or_default();
        self.set_status(ws, thread, &status).await;
    }

    async fn set_status(&self, ws: &Workspace, thread: &Thread, status: &str) {
        let body = serde_json::json!({
            "channel_id": thread.channel,
            "thread_ts": thread.ts,
            "status": status,
        });
        if let Err(e) = self.api_call(ws, "assistant.threads.setStatus", body).await {
            tracing::debug!(error = %e, channel = %thread.channel, "slack: status not set");
        }
    }

    async fn api_call(&self, ws: &Workspace, method: &str, body: Value) -> Result<Value, Error> {
        let resp = self
            .http
            .post(format!("{}/{method}", self.api_base))
            .bearer_auth(&ws.bot_token)
            .json(&body)
            .send()
            .await
            .map_err(Error::retryable)?;
        let status = resp.status();
        if status.is_server_error() || status.as_u16() == 429 {
            return Err(Error::Retryable(format!("http {status}")));
        }
        let v: Value = resp.json().await.map_err(Error::retryable)?;
        if v["ok"].as_bool() == Some(true) {
            return Ok(v);
        }
        Err(Error::from_response(&v))
    }

    /// Open the turn's message. `recipient_user_id` is required outside a DM.
    async fn start_stream(
        &self,
        ws: &Workspace,
        thread: &Thread,
        recipient: Option<&str>,
        recipient_team: Option<&str>,
    ) -> Result<String, Error> {
        let mut body = serde_json::json!({
            "channel": thread.channel,
            "thread_ts": thread.ts,
            "task_display_mode": "timeline",
        });
        if let Some(recipient) = recipient {
            body["recipient_user_id"] = recipient.into();
        }
        // A Slack Connect guest keeps their own team.
        let team = match recipient_team {
            Some(team) => Some(team.to_string()),
            None => self.identity(ws).await.and_then(|i| i.team.clone()),
        };
        if let Some(team) = team {
            body["recipient_team_id"] = team.into();
        }
        let resp = self.api_call(ws, "chat.startStream", body).await?;
        resp["ts"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| Error::Terminal("startStream returned no ts".into()))
    }

    async fn append_stream(
        &self,
        ws: &Workspace,
        channel: &str,
        ts: &str,
        chunks: Vec<Value>,
    ) -> Result<(), Error> {
        self.api_call(
            ws,
            "chat.appendStream",
            serde_json::json!({ "channel": channel, "ts": ts, "chunks": chunks }),
        )
        .await
        .map(|_| ())
    }

    /// Close the message with a last chunk. Slack accepts the metadata stamp
    /// only here. A plain-text close fails with `streaming_mode_mismatch`.
    async fn stop_stream(
        &self,
        ws: &Workspace,
        channel: &str,
        ts: &str,
        chunk: Value,
        meta: &ReplyMeta,
    ) -> Result<(), Error> {
        self.api_call(
            ws,
            "chat.stopStream",
            serde_json::json!({
                "channel": channel,
                "ts": ts,
                "chunks": [chunk],
                "metadata": {
                    "event_type": REPLY_EVENT_TYPE,
                    "event_payload": meta,
                },
            }),
        )
        .await
        .map(|_| ())
    }

    /// Resume the interrupt with the recorded option value, not the wire
    /// value. A click on a resolved prompt only removes its stale buttons.
    async fn resolve_click(&self, ctx: &ChannelContext, ws: &Workspace, click: Click) {
        let session_id = format!("slack:{}:{}", click.channel, click.thread_ts);
        let open = match ctx.get_session(&ws.tenant_id, &session_id).await {
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
            let blocks = self.controller.settled_prompt_blocks(
                &click.message_blocks,
                &click.message_text,
                "(no longer active)",
            );
            let cleared = self
                .update(ws, &click.channel, &click.message_ts, &text, blocks, &meta)
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
                    tenant_id: ws.tenant_id.clone(),
                },
                // A resume does not change ownership.
                owner: SessionOwner {
                    tenant_id: ws.tenant_id.clone(),
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

/// Outbound side. At-least-once: a crash between post and checkpoint can
/// post again.
#[async_trait::async_trait]
impl EventProcessor for SlackBot {
    fn name(&self) -> &'static str {
        "slack_outbound_v1"
    }

    async fn apply(&self, event: SessionEvent) -> Result<(), ProcessorError> {
        let Some((channel_id, thread_ts)) = slack_session(&event.session_id) else {
            return Ok(());
        };
        let Some(ws) = self.resolver.by_tenant(&event.tenant_id).await else {
            return Ok(());
        };
        let thread = Thread::new(channel_id, thread_ts);
        // The row lands before the checkpoint commits: a lost write replays.
        if let Err(e) = self.track(&ws, &event).await {
            return Err(ProcessorError::Apply(e.to_string()));
        }
        // The indicator is the turn's, start to end: nothing else lights it, so
        // a message waiting its turn does not claim the thread is working.
        match &event.payload {
            EventPayload::TurnCompleted(_) | EventPayload::SessionInterrupted(_) => {
                self.set_status(&ws, &thread, "").await;
            }
            EventPayload::TurnStarted(t) => {
                let ctx = Context {
                    tenant_id: &event.tenant_id,
                    session_id: &event.session_id,
                    turn_id: &t.turn_id,
                    agent_id: event.meta.agent_id.as_deref(),
                    elapsed: None,
                };
                self.set_working(&ws, &thread, &ctx).await
            }
            _ => {}
        }
        let result = match &event.payload {
            EventPayload::TurnCompleted(t) => {
                self.complete_turn(
                    &ws,
                    &thread,
                    &event.session_id,
                    t,
                    event.occurred_at,
                    event.seq,
                )
                .await
            }
            EventPayload::SessionInterrupted(p) => {
                self.post_interrupt(
                    &ws,
                    &thread,
                    &event.session_id,
                    event.meta.turn_id.as_deref(),
                    p,
                    event.seq,
                )
                .await
            }
            EventPayload::InterruptResumed(p) => self.settle_prompt(&ws, &thread, p).await,
            _ => return Ok(()),
        };
        // The turn is settled (or undeliverable) either way: drop its row.
        if matches!(result, Ok(()) | Err(Error::Terminal(_))) {
            if let EventPayload::TurnCompleted(t) = &event.payload {
                if let Some(store) = self.store.as_deref() {
                    if let Err(e) = store
                        .clear_turn(&event.tenant_id, &event.session_id, &t.turn_id)
                        .await
                    {
                        tracing::warn!(session_id = %event.session_id, error = %e, "slack: stream state not cleared");
                    }
                }
            }
        }
        match result {
            Ok(()) => Ok(()),
            // A terminal error must not block the processor.
            Err(Error::Terminal(e)) => {
                tracing::warn!(session_id = %event.session_id, error = %e, "slack: dropping undeliverable reply");
                Ok(())
            }
            Err(Error::Retryable(e)) => Err(ProcessorError::Apply(e)),
        }
    }
}

impl SlackBot {
    /// Post the answer into the live stream, or as a new message. A lost
    /// slot (restart, failover) is rebuilt from the store first.
    async fn complete_turn(
        &self,
        ws: &Workspace,
        thread: &Thread,
        session_id: &str,
        t: &crate::session::events::TurnCompleted,
        at: chrono::DateTime<chrono::Utc>,
        seq: u64,
    ) -> Result<(), Error> {
        let key = StreamKey::new(&ws.tenant_id, session_id, &t.turn_id);
        let live = match self.streams.take(&key) {
            Some(live) => Some(live),
            None => self.recover(ws, &key, seq).await,
        };
        let elapsed = live
            .as_ref()
            .and_then(|s| (at - s.started_at).to_std().ok());
        let present_ctx = Context {
            tenant_id: &ws.tenant_id,
            session_id,
            turn_id: &t.turn_id,
            agent_id: ws.routing.agent_for(&thread.channel),
            elapsed,
        };
        // The turn's visible work, rendered and trimmed. `None` when there is
        // nothing to rebuild from — then the streamed text stands as posted.
        let activity = self.activity_blocks(ws, live.as_ref()).await;
        let Some(rendered) =
            self.controller
                .message(&present_ctx, t, activity.as_deref().unwrap_or_default())
        else {
            tracing::debug!(%session_id, "slack: the controller posts nothing for this turn");
            return Ok(());
        };
        let Some(ts) = live.as_ref().and_then(|s| s.ts.clone()) else {
            return self.post_turn(ws, thread, session_id, t, &rendered).await;
        };
        if self.already_replied(ws, thread, t).await {
            return Ok(());
        }
        let meta = self.reply_meta(ws, session_id, t).await;
        let text = rendered.text.clone();
        let chunk = serde_json::json!({
            "type": "markdown_text",
            "text": clip(&text, MAX_MARKDOWN),
        });
        match self
            .stop_stream(ws, &thread.channel, &ts, chunk, &meta)
            .await
        {
            Ok(()) => {
                // Rebuild from blocks, which have no 256-character chunk
                // limit, so each card settles with its full detail.
                if activity.is_some() {
                    if let Err(e) = self
                        .update(ws, &thread.channel, &ts, &text, rendered.blocks, &meta)
                        .await
                    {
                        tracing::warn!(error = %e, %session_id, "slack: cards not expanded");
                    }
                }
                Ok(())
            }
            // The stream is gone (expired, or already stopped) but the
            // message remains: finalize it in place, not as an orphan.
            Err(e) if e.code() == "message_not_in_streaming_state" => {
                match self
                    .update(
                        ws,
                        &thread.channel,
                        &ts,
                        &text,
                        rendered.blocks.clone(),
                        &meta,
                    )
                    .await
                {
                    Ok(()) => Ok(()),
                    Err(e) => {
                        tracing::warn!(error = %e, %session_id, "slack: in-place finalize failed; posting reply");
                        self.post_turn(ws, thread, session_id, t, &rendered).await
                    }
                }
            }
            // Do not lose the answer to a bad stream.
            Err(e) => {
                tracing::warn!(error = %e, %session_id, "slack: stream finalize failed; posting reply");
                self.post_turn(ws, thread, session_id, t, &rendered).await
            }
        }
    }

    async fn post_turn(
        &self,
        ws: &Workspace,
        thread: &Thread,
        session_id: &str,
        t: &crate::session::events::TurnCompleted,
        rendered: &Rendered,
    ) -> Result<(), Error> {
        if self.already_replied(ws, thread, t).await {
            return Ok(());
        }
        let meta = self.reply_meta(ws, session_id, t).await;
        self.post_blocks(ws, thread, &rendered.text, rendered.blocks.clone(), &meta)
            .await
    }

    /// True if the thread has the turn's stamped reply. A failed fetch
    /// answers again: at-least-once.
    async fn already_replied(
        &self,
        ws: &Workspace,
        thread: &Thread,
        t: &crate::session::events::TurnCompleted,
    ) -> bool {
        let oldest = slack_session(&t.turn_id).map(|(_, ts)| ts);
        let Ok(replies) = self.fetch_thread(ws, thread, oldest).await else {
            return false;
        };
        replies
            .iter()
            .any(|m| m.meta.as_ref().and_then(|r| r.turn_id.as_deref()) == Some(t.turn_id.as_str()))
    }

    async fn reply_meta(
        &self,
        ws: &Workspace,
        session_id: &str,
        t: &crate::session::events::TurnCompleted,
    ) -> ReplyMeta {
        let message_id = self
            .session_path(ws, session_id)
            .await
            .last()
            .filter(|m| matches!(m.role, Role::Assistant))
            .map(|m| m.id.clone());
        ReplyMeta {
            turn_id: Some(t.turn_id.clone()),
            message_id,
            session_id: Some(session_id.to_string()),
            ..Default::default()
        }
    }

    /// Post the prompt (buttons), or "Paused: {reason}". Dedupes on the
    /// interrupt id.
    async fn post_interrupt(
        &self,
        ws: &Workspace,
        thread: &Thread,
        session_id: &str,
        turn_id: Option<&str>,
        p: &crate::session::events::SessionInterrupted,
        seq: u64,
    ) -> Result<(), Error> {
        if let Ok(replies) = self.fetch_thread(ws, thread, None).await {
            if replies.iter().any(|m| {
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
        let (text, blocks) = match display_of(&p.payload) {
            Some(display) => {
                let options = prompt_options(&display, &p.interrupt_id);
                let blocks = self.controller.prompt_blocks(&PromptView {
                    message: &display.message,
                    options: &options,
                    expires_at: display.expires_at.as_deref(),
                });
                (display.message.clone(), blocks)
            }
            None => {
                let text = format!("Paused: {}", p.reason);
                let block = section_block(&text);
                (text, vec![block])
            }
        };
        // A prompt can wait for a long time; do not hold the stream open.
        let key = turn_id.map(|t| StreamKey::new(&ws.tenant_id, session_id, t));
        let open = match &key {
            Some(key) => match self.streams.take(key) {
                Some(open) => Some(open),
                None => self.recover(ws, key, seq).await,
            },
            None => None,
        };
        let posted = 'post: {
            if let Some(ts) = open.as_ref().and_then(|s| s.ts.as_deref()) {
                let chunk = serde_json::json!({ "type": "blocks", "blocks": blocks.clone() });
                match self
                    .stop_stream(ws, &thread.channel, ts, chunk, &meta)
                    .await
                {
                    Ok(()) => break 'post Ok(()),
                    Err(e) => {
                        tracing::warn!(error = %e, %session_id, "slack: prompt into stream failed; posting it")
                    }
                }
            }
            self.post_blocks(ws, thread, &text, blocks, &meta).await
        };
        // The prompt owns the message now; the stream row is spent.
        if posted.is_ok() {
            if let (Some(store), Some(open)) = (self.store.as_deref(), &open) {
                if let Err(e) = store
                    .clear_turn(&ws.tenant_id, session_id, &open.turn_id)
                    .await
                {
                    tracing::warn!(%session_id, error = %e, "slack: stream state not cleared");
                }
            }
        }
        posted
    }

    /// Remove the prompt's buttons and add the outcome. Best-effort.
    async fn settle_prompt(
        &self,
        ws: &Workspace,
        thread: &Thread,
        p: &crate::session::events::InterruptResumed,
    ) -> Result<(), Error> {
        let session_id = format!("slack:{}:{}", thread.channel, thread.ts);
        let Ok(replies) = self.fetch_thread(ws, thread, None).await else {
            tracing::warn!(interrupt_id = %p.interrupt_id, "slack: fetch failed; prompt not settled");
            return Ok(());
        };
        let Some(msg) = replies.iter().find(|m| {
            m.meta.as_ref().and_then(|r| r.interrupt_id.as_deref()) == Some(p.interrupt_id.as_str())
        }) else {
            return Ok(());
        };
        let resolution = resolution_text(&p.payload);
        let text = format!("{}\n\n{resolution}", msg.text);
        let meta = ReplyMeta {
            interrupt_id: Some(p.interrupt_id.clone()),
            session_id: Some(session_id),
            ..Default::default()
        };
        let blocks = self
            .controller
            .settled_prompt_blocks(&msg.blocks, &msg.text, &resolution);
        self.update(ws, &thread.channel, &msg.ts, &text, blocks, &meta)
            .await
    }
}

/// Activity: events mark a session dirty; one worker renders the cards.
impl SlackBot {
    /// Track the turn's slot; durable when a store is attached. A store
    /// failure is retryable so the row always lands before the checkpoint.
    async fn track(&self, ws: &Workspace, event: &SessionEvent) -> Result<(), Error> {
        // Every event names the turn it belongs to; a start names its own.
        let turn_id = match &event.payload {
            EventPayload::TurnStarted(t) => Some(t.turn_id.clone()),
            _ => event.meta.turn_id.clone(),
        };
        match &event.payload {
            EventPayload::TurnStarted(t) => {
                let owner = event.meta.owner.as_ref();
                let recipient = owner
                    .and_then(|o| o.id.as_deref())
                    .and_then(|id| id.strip_prefix("slack:"))
                    .map(str::to_string);
                let recipient_team = owner.and_then(|o| o.metadata.get("slack_team")).cloned();
                let mut stream = Stream {
                    tenant_id: event.tenant_id.clone(),
                    session_id: event.session_id.clone(),
                    turn_id: t.turn_id.clone(),
                    start_seq: event.seq,
                    started_at: event.occurred_at,
                    recipient,
                    recipient_team,
                    ts: None,
                    sent: HashMap::new(),
                    dead: false,
                    version: 0,
                };
                if let Some(store) = self.store.as_deref() {
                    let slot = store
                        .upsert_turn(
                            &event.tenant_id,
                            &event.session_id,
                            &t.turn_id,
                            event.seq,
                            event.occurred_at,
                        )
                        .await
                        .map_err(|e| Error::Retryable(e.to_string()))?;
                    stream.version = slot.version;
                    // A replayed start after a restart: pick the open
                    // stream back up instead of opening a second one.
                    if slot.resumed {
                        if let Some(recovered) = self.recover(ws, &stream.key(), event.seq).await {
                            stream = recovered;
                        }
                    }
                }
                self.streams.insert(stream);
            }
            EventPayload::ToolCallRequested(_)
            | EventPayload::ToolCallCompleted(_)
            | EventPayload::ToolCallErrored(_)
            | EventPayload::SubAgentRequested(_)
            | EventPayload::SubAgentTurnCompleted(_)
            | EventPayload::SubAgentErrored(_) => {}
            // A cancel ends every turn of the session.
            EventPayload::SessionCancelled => {
                self.streams
                    .remove_session(&event.tenant_id, &event.session_id);
                if let Some(store) = self.store.as_deref() {
                    if let Err(e) = store.clear(&event.tenant_id, &event.session_id).await {
                        tracing::warn!(session_id = %event.session_id, error = %e, "slack: stream state not cleared");
                    }
                }
                return Ok(());
            }
            _ => return Ok(()),
        }
        // An event outside any turn renders nothing.
        if let Some(turn_id) = turn_id {
            self.streams.mark_dirty(StreamKey::new(
                &event.tenant_id,
                &event.session_id,
                &turn_id,
            ));
        }
        Ok(())
    }

    /// One worker; the tick is the global append budget.
    async fn stream_activity(&self, ctx: &ChannelContext) {
        loop {
            let Some(key) = self.streams.take_dirty() else {
                // No events come while a turn thinks; refresh the status on
                // a timer.
                tokio::select! {
                    _ = ctx.shutdown.cancelled() => return,
                    _ = self.streams.notified() => {}
                    _ = tokio::time::sleep(STATUS_REFRESH) => self.refresh_statuses().await,
                }
                continue;
            };
            let Some(ws) = self.resolver.by_tenant(&key.tenant_id).await else {
                continue;
            };
            self.stream_turn(ctx, &ws, &key).await;
            tokio::select! {
                _ = ctx.shutdown.cancelled() => return,
                _ = tokio::time::sleep(ACTIVITY_INTERVAL) => {}
            }
        }
    }

    /// Set the status again for each turn that has no card yet.
    async fn refresh_statuses(&self) {
        for key in self.streams.waiting() {
            let Some(ws) = self.resolver.by_tenant(&key.tenant_id).await else {
                continue;
            };
            if let Some((channel, thread_ts)) = slack_session(&key.session_id) {
                let ctx = Context {
                    tenant_id: &key.tenant_id,
                    session_id: &key.session_id,
                    turn_id: &key.turn_id,
                    agent_id: ws.routing.agent_for(channel),
                    elapsed: None,
                };
                self.set_working(&ws, &Thread::new(channel, thread_ts), &ctx)
                    .await;
            }
        }
    }

    /// Fold the turn's events into cards and append the changes.
    async fn stream_turn(&self, ctx: &ChannelContext, ws: &Workspace, key: &StreamKey) {
        let session_id = &key.session_id;
        let Some((channel, thread_ts)) = slack_session(session_id) else {
            return;
        };
        let thread = Thread::new(channel, thread_ts);
        // A missing slot after a restart rebuilds from the store; a dead
        // one stays dead.
        let open = match self.streams.get(key) {
            Some(open) if open.dead => return,
            Some(open) => open,
            None => match self.recover(ws, key, u64::MAX).await {
                Some(recovered) => {
                    self.streams.insert(recovered.clone());
                    recovered
                }
                None => return,
            },
        };
        let caller = Caller::System {
            tenant_id: ws.tenant_id.clone(),
        };
        let events = ctx
            .read_session_events(&caller, session_id, Some(Seq(open.start_seq)), None)
            .await;
        let Ok(events) = events else {
            return;
        };
        let Some(turn) = TurnActivity::fold(&events, Some(key.turn_id.clone())) else {
            return;
        };
        // A newer turn owns its own message.
        if turn.turn_id != key.turn_id {
            return;
        }
        let changed: Vec<(String, Value)> = turn
            .chunks(self.controller.as_ref())
            .into_iter()
            .filter(|(id, chunk)| open.sent.get(id) != Some(chunk))
            .collect();
        // No changes: only keep the status up.
        if changed.is_empty() {
            if open.ts.is_none() {
                let ctx = Context {
                    tenant_id: &key.tenant_id,
                    session_id: &key.session_id,
                    turn_id: &key.turn_id,
                    agent_id: ws.routing.agent_for(&thread.channel),
                    elapsed: None,
                };
                self.set_working(ws, &thread, &ctx).await;
            }
            return;
        }

        let (ts, version) = match open.ts.clone() {
            Some(ts) => (ts, open.version),
            None => {
                // A thread shows one open stream. A queued turn waits for the
                // turn before it to settle rather than opening a second one
                // beside it; the tick that follows tries again.
                if self.stream_open_elsewhere(key).await {
                    self.streams.mark_dirty(key.clone());
                    return;
                }
                let ts = match self
                    .start_stream(
                        ws,
                        &thread,
                        open.recipient.as_deref(),
                        open.recipient_team.as_deref(),
                    )
                    .await
                {
                    Ok(ts) => ts,
                    Err(e) => return self.kill_stream(key, e),
                };
                match self.persist_ts(&open, &ts).await {
                    TsPersist::Kept(version) => (ts, version),
                    // Another writer opened this turn's stream first.
                    TsPersist::Adopted(theirs, version) => {
                        self.delete_message(ws, channel, &ts).await;
                        (theirs, version)
                    }
                    TsPersist::TurnOver => {
                        self.delete_message(ws, channel, &ts).await;
                        self.streams.remove(key);
                        return;
                    }
                }
            }
        };
        let chunks = changed.iter().map(|(_, c)| c.clone()).collect();
        if let Err(e) = self.append_stream(ws, channel, &ts, chunks).await {
            return self.kill_stream(key, e);
        }
        self.streams.commit(key, ts, version, changed);
    }

    /// True while another turn of this session still holds an open message.
    /// A row whose turn the log has already completed is leftover, not open:
    /// its owner clears it, so it must not hold this turn back.
    async fn stream_open_elsewhere(&self, key: &StreamKey) -> bool {
        if self.streams.open_elsewhere(key) {
            return true;
        }
        let Some(store) = self.store.as_deref() else {
            return false;
        };
        let other = store
            .open_other(&key.tenant_id, &key.session_id, &key.turn_id)
            .await;
        let Ok(Some(row)) = other else {
            return false;
        };
        !self.turn_is_over(key, &row).await
    }

    /// Whether the log has settled the row's turn.
    async fn turn_is_over(&self, key: &StreamKey, row: &super::state::StreamRow) -> bool {
        let Some(ctx) = self.ctx.get() else {
            return false;
        };
        let caller = Caller::System {
            tenant_id: key.tenant_id.clone(),
        };
        let events = ctx
            .read_session_events(&caller, &key.session_id, Some(Seq(row.start_seq)), None)
            .await;
        let Ok(events) = events else {
            return false;
        };
        events.iter().any(|e| match &e.payload {
            EventPayload::TurnCompleted(t) => t.turn_id == row.turn_id,
            EventPayload::SessionCancelled => true,
            _ => false,
        })
    }

    /// The finished turn rendered as blocks, ready to replace the stream.
    /// The turn's visible work as blocks, rendered by the controller and
    /// trimmed to Slack's limits. `None` when there is no stream to read from.
    async fn activity_blocks(&self, ws: &Workspace, stream: Option<&Stream>) -> Option<Vec<Value>> {
        let (ctx, stream) = (self.ctx.get()?, stream?);
        let caller = Caller::System {
            tenant_id: ws.tenant_id.clone(),
        };
        let events = ctx
            .read_session_events(
                &caller,
                &stream.session_id,
                Some(Seq(stream.start_seq)),
                None,
            )
            .await
            .ok()?;
        let turn = TurnActivity::fold(&events, Some(stream.turn_id.clone()))?;
        Some(turn.blocks(self.controller.as_ref()))
    }

    /// Abandon a rejected stream. The answer posts as a message.
    fn kill_stream(&self, key: &StreamKey, error: Error) {
        tracing::warn!(
            error = %error,
            session_id = %key.session_id,
            turn_id = %key.turn_id,
            "slack: activity stream failed"
        );
        self.streams.kill(key);
    }

    /// Rebuild a lost slot from its durable row: the log replays the cards,
    /// the row carries the message. `before` bounds the staleness scan; the
    /// caller's own redelivered event sits at it.
    async fn recover(&self, ws: &Workspace, key: &StreamKey, before: u64) -> Option<Stream> {
        let (store, ctx) = (self.store.as_deref()?, self.ctx.get()?);
        let row = store
            .load(&key.tenant_id, &key.session_id, &key.turn_id)
            .await
            .ok()
            .flatten()?;
        let caller = Caller::System {
            tenant_id: key.tenant_id.clone(),
        };
        let events = ctx
            .read_session_events(&caller, &key.session_id, Some(Seq(row.start_seq)), None)
            .await
            .ok()?;
        for event in events.iter().filter(|e| e.seq < before) {
            match &event.payload {
                // The turn is over; the row is leftover. A later turn starting
                // says nothing: it holds a row of its own.
                EventPayload::SessionCancelled => {
                    let _ = store
                        .clear_turn(&key.tenant_id, &key.session_id, &row.turn_id)
                        .await;
                    return None;
                }
                EventPayload::TurnCompleted(t) if t.turn_id == row.turn_id => {
                    let _ = store
                        .clear_turn(&key.tenant_id, &key.session_id, &row.turn_id)
                        .await;
                    return None;
                }
                // A prompt may own the message now; never write over it.
                EventPayload::SessionInterrupted(_) => return None,
                _ => {}
            }
        }
        // Cards set by id and can repeat; only appended text cannot. Seed
        // it as sent — a chunk lost mid-crash reappears at finalize.
        let sent: HashMap<String, Value> = TurnActivity::fold(&events, Some(row.turn_id.clone()))
            .map(|turn| {
                turn.chunks(self.controller.as_ref())
                    .into_iter()
                    .filter(|(id, _)| id.starts_with("say:"))
                    .collect()
            })
            .unwrap_or_default();
        let owner = events.iter().rev().find_map(|e| e.meta.owner.as_ref());
        let recipient = owner
            .and_then(|o| o.id.as_deref())
            .and_then(|id| id.strip_prefix("slack:"))
            .map(str::to_string);
        let recipient_team = owner.and_then(|o| o.metadata.get("slack_team")).cloned();
        let mut stream = Stream {
            tenant_id: key.tenant_id.clone(),
            session_id: key.session_id.clone(),
            turn_id: row.turn_id,
            start_seq: row.start_seq,
            started_at: row.started_at,
            recipient,
            recipient_team,
            ts: row.ts,
            sent,
            dead: false,
            version: row.version,
        };
        if stream.ts.is_none() {
            self.heal_ts(ws, key, &mut stream).await;
        }
        Some(stream)
    }

    /// A crash can beat the record after `chat.startStream`. The thread
    /// knows: stamps land only when a message settles, so the orphan is our
    /// last unstamped message after the turn opened.
    async fn heal_ts(&self, ws: &Workspace, key: &StreamKey, stream: &mut Stream) {
        let Some((_, trigger)) = slack_session(&stream.turn_id) else {
            return;
        };
        let Some((channel, thread_ts)) = slack_session(&key.session_id) else {
            return;
        };
        let thread = Thread::new(channel, thread_ts);
        // A queued turn's trigger predates the whole turn before it, whose own
        // orphan would then look like ours. Never look back past our start.
        let oldest = std::cmp::max(
            trigger.to_string(),
            format!("{}.000000", stream.started_at.timestamp()),
        );
        let Ok(resp) = self.fetch_replies_raw(ws, &thread, Some(&oldest)).await else {
            return;
        };
        let ours = self
            .identity(ws)
            .await
            .map(|i| i.ours.clone())
            .unwrap_or_default();
        let Some(ts) = unstamped_ours(&resp, &ours) else {
            return;
        };
        if let Some(store) = self.store.as_deref() {
            let recorded = store
                .set_ts(
                    &key.tenant_id,
                    &key.session_id,
                    &stream.turn_id,
                    &ts,
                    stream.version,
                )
                .await;
            if matches!(recorded, Ok(true)) {
                stream.version += 1;
            }
        }
        stream.ts = Some(ts);
    }

    /// Record the new stream's ts. Losing the version fence means another
    /// writer opened one first; theirs comes back.
    async fn persist_ts(&self, open: &Stream, ts: &str) -> TsPersist {
        let Some(store) = self.store.as_deref() else {
            return TsPersist::Kept(open.version);
        };
        let mut expected = open.version;
        for _ in 0..2 {
            match store
                .set_ts(
                    &open.tenant_id,
                    &open.session_id,
                    &open.turn_id,
                    ts,
                    expected,
                )
                .await
            {
                Ok(true) => return TsPersist::Kept(expected + 1),
                Ok(false) => match store
                    .load(&open.tenant_id, &open.session_id, &open.turn_id)
                    .await
                {
                    Ok(Some(row)) => match row.ts {
                        Some(theirs) if theirs != ts => {
                            return TsPersist::Adopted(theirs, row.version)
                        }
                        Some(_) => return TsPersist::Kept(row.version),
                        None => expected = row.version,
                    },
                    // The row is gone: the turn settled under us.
                    Ok(None) => return TsPersist::TurnOver,
                    Err(e) => {
                        tracing::warn!(session_id = %open.session_id, error = %e, "slack: stream ts unrecorded");
                        return TsPersist::Kept(open.version);
                    }
                },
                Err(e) => {
                    tracing::warn!(session_id = %open.session_id, error = %e, "slack: stream ts unrecorded");
                    return TsPersist::Kept(open.version);
                }
            }
        }
        TsPersist::Kept(expected)
    }

    /// Remove a message we should not have kept. Best-effort.
    async fn delete_message(&self, ws: &Workspace, channel: &str, ts: &str) {
        let body = serde_json::json!({ "channel": channel, "ts": ts });
        if let Err(e) = self.api_call(ws, "chat.delete", body).await {
            tracing::warn!(error = %e, "slack: duplicate stream not deleted");
        }
    }
}

/// How recording a fresh stream's ts settled.
enum TsPersist {
    /// Ours, at this row version.
    Kept(u64),
    /// A concurrent writer won; use their message.
    Adopted(String, u64),
    /// The row is gone: the turn ended under us.
    TurnOver,
}

#[derive(Debug, thiserror::Error)]
pub(super) enum Error {
    #[error("{0}")]
    Retryable(String),
    #[error("{0}")]
    Terminal(String),
}

impl Error {
    fn retryable(e: impl std::fmt::Display) -> Self {
        Error::Retryable(e.to_string())
    }

    /// The `error` code of a non-`ok` Slack response.
    fn from_response(resp: &Value) -> Self {
        match resp["error"].as_str().unwrap_or("unknown_error") {
            code @ ("rate_limited" | "ratelimited" | "internal_error" | "service_unavailable") => {
                Error::Retryable(code.to_string())
            }
            code => Error::Terminal(code.to_string()),
        }
    }

    fn code(&self) -> &str {
        match self {
            Error::Retryable(e) | Error::Terminal(e) => e,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{ErrorCode, ErrorInfo};
    use crate::session::events::TurnCompleted;
    use std::sync::Mutex as StdMutex;

    const SESSION: &str = "slack:C1:1.0";

    /// A Slack that records what it was posted. `conversations.replies` answers
    /// empty so the bot's idempotency check passes and it goes on to post.
    async fn fake_slack() -> (String, Arc<StdMutex<Vec<Value>>>) {
        use axum::routing::post;
        use axum::{Json, Router};

        let posted: Arc<StdMutex<Vec<Value>>> = Arc::new(StdMutex::new(Vec::new()));
        let recorder = posted.clone();
        let app = Router::new()
            .route(
                "/chat.postMessage",
                post(move |Json(body): Json<Value>| {
                    let recorder = recorder.clone();
                    async move {
                        recorder.lock().unwrap().push(body);
                        Json(serde_json::json!({"ok": true, "ts": "1.1"}))
                    }
                }),
            )
            .route(
                "/conversations.replies",
                post(|| async { Json(serde_json::json!({"ok": true, "messages": []})) }),
            )
            .route(
                "/auth.test",
                post(|| async {
                    Json(serde_json::json!({"ok": true, "user_id": "U0", "bot_id": "B0"}))
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (format!("http://{addr}"), posted)
    }

    struct OneWorkspace(Arc<Workspace>);

    #[async_trait::async_trait]
    impl WorkspaceResolver for OneWorkspace {
        async fn by_install(
            &self,
            _: Option<&str>,
            _: Option<&str>,
            _: &str,
        ) -> Option<Arc<Workspace>> {
            Some(self.0.clone())
        }
        async fn by_tenant(&self, _: &str) -> Option<Arc<Workspace>> {
            Some(self.0.clone())
        }
    }

    fn failed_turn(error: ErrorInfo) -> TurnCompleted {
        TurnCompleted {
            turn_id: "turn-1".into(),
            data: Value::Null,
            turn_cost: Default::default(),
            turn_token_usage: Default::default(),
            error: Some(error),
        }
    }

    /// A deployment supplies its own voice and keeps every other behaviour of
    /// the bot: this controller rewrites one error code and adds a link under
    /// every message, and inherits the rest.
    struct CloudController;

    impl SlackController for CloudController {
        fn turn(&self, ctx: &Context<'_>, t: &TurnCompleted) -> Rendered {
            match &t.error {
                Some(e) if e.code == ErrorCode::BudgetExceeded => Rendered {
                    text: "You are out of credits.".into(),
                    blocks: vec![section_block("You are out of credits.")],
                },
                _ => DefaultController.turn(ctx, t),
            }
        }
        fn trailing_blocks(&self, ctx: &Context<'_>) -> Vec<Value> {
            vec![serde_json::json!({
                "type": "context",
                "elements": [{
                    "type": "mrkdwn",
                    "text": format!("<https://admin.example/s/{}|View in admin>", ctx.session_id),
                }],
            })]
        }
    }

    async fn post_with(controller: Arc<dyn SlackController>, t: &TurnCompleted) -> Value {
        let (api_base, posted) = fake_slack().await;
        let ws = Arc::new(Workspace::new(
            "xoxb-test".into(),
            "tenant-a".into(),
            Routing::new().dm(Some("a".into())),
        ));
        let bot = SlackBot::new(Arc::new(OneWorkspace(ws.clone())), api_base, None)
            .with_controller(controller);
        let thread = Thread {
            channel: "C1".into(),
            ts: "1.0".into(),
        };
        bot.complete_turn(&ws, &thread, SESSION, t, chrono::Utc::now(), 0)
            .await
            .expect("posts");
        let sent = posted.lock().unwrap();
        sent.first().cloned().expect("one message posted")
    }

    /// The engine's own voice, unchanged: the failure's sentence and nothing
    /// appended.
    #[tokio::test]
    async fn the_default_controller_posts_the_failure_sentence() {
        let t = failed_turn(ErrorInfo::new(ErrorCode::BudgetExceeded, "budget spent"));
        let body = post_with(Arc::new(DefaultController), &t).await;
        assert_eq!(body["text"], "Error: budget spent");
        assert_eq!(body["blocks"].as_array().map(Vec::len), Some(1));
    }

    /// The point of the seam: cloud copy for the code it cares about, and a
    /// link under the message, with delivery untouched.
    #[tokio::test]
    async fn a_controller_rewrites_one_code_and_appends_a_link() {
        let t = failed_turn(ErrorInfo::new(ErrorCode::BudgetExceeded, "budget spent"));
        let body = post_with(Arc::new(CloudController), &t).await;

        assert_eq!(body["text"], "You are out of credits.");
        let blocks = body["blocks"].as_array().expect("blocks");
        assert_eq!(blocks.len(), 2, "the card and the trailing link");
        assert!(blocks[1]["elements"][0]["text"]
            .as_str()
            .is_some_and(|t| t.contains("https://admin.example/s/slack:C1:1.0")));
        // Delivery is the bot's: the reply is still stamped for idempotency.
        assert_eq!(body["metadata"]["event_payload"]["turn_id"], "turn-1");
    }

    /// Every other code falls through to the inherited implementation, so a
    /// deployment overrides one branch rather than forking the renderer.
    #[tokio::test]
    async fn an_uninteresting_code_keeps_the_engines_wording() {
        let t = failed_turn(ErrorInfo::new(ErrorCode::InvalidResponse, "bad decision"));
        let body = post_with(Arc::new(CloudController), &t).await;
        assert_eq!(body["text"], "Error: bad decision");
        let blocks = body["blocks"].as_array().expect("blocks");
        assert_eq!(blocks.len(), 2, "still carries the deployment's link");
    }

    /// A channel with nothing of its own is the default's, so declaring one
    /// channel does not take the bot out of every other.
    #[test]
    fn a_channel_falls_to_the_default() {
        let routing = Routing::new()
            .dm(Some("support".into()))
            .mentions(Some("support".into()))
            .channel("C0ENG", Some("oncall".into()));
        assert_eq!(routing.agent_for("C0ENG"), Some("oncall"));
        assert_eq!(routing.agent_for("C0SALES"), Some("support"));
        // A DM is a channel, and resolves the same way.
        assert_eq!(routing.agent_for("D0USER"), Some("support"));
    }

    /// No default is the allowlist: the bot is in the channels the file names
    /// and nowhere else, without a second setting saying so.
    #[test]
    fn without_a_default_only_the_named_channels_are_served() {
        let routing = Routing::new().channel("C0ENG", Some("oncall".into()));
        assert_eq!(routing.agent_for("C0ENG"), Some("oncall"));
        assert_eq!(routing.agent_for("C0SALES"), None);
        assert_eq!(routing.agent_for("D0USER"), None);
    }

    /// And with a default, `off` is how one channel is carved back out.
    #[test]
    fn an_off_channel_is_silent_under_a_default() {
        let routing = Routing::new()
            .mentions(Some("support".into()))
            .channel("C0RANDOM", None);
        assert_eq!(routing.agent_for("C0RANDOM"), None);
        assert_eq!(routing.agent_for("C0SALES"), Some("support"));
    }

    #[test]
    fn routing_reads_back_for_the_startup_line() {
        let routing = Routing::new()
            .dm(Some("support".into()))
            .mentions(Some("helper".into()))
            .channel("C0RANDOM", None)
            .channel("C0ENG", Some("oncall".into()));
        // The line reads back in the file's own words, so a misrouted channel
        // can be checked against the section that set it.
        assert_eq!(
            routing.to_string(),
            "dm→support, mentions→helper, C0ENG→oncall, C0RANDOM off"
        );

        assert!(Routing::default().is_empty());
        assert_eq!(Routing::default().to_string(), "nothing");
        assert!(!Routing::new().channel("C0ENG", None).is_empty());
    }

    fn stream(turn_id: &str, ts: Option<&str>) -> Stream {
        Stream {
            tenant_id: "t".into(),
            session_id: SESSION.into(),
            turn_id: turn_id.into(),
            start_seq: 1,
            started_at: chrono::Utc::now(),
            recipient: None,
            recipient_team: None,
            ts: ts.map(str::to_string),
            sent: HashMap::new(),
            dead: false,
            version: 0,
        }
    }

    fn key(turn_id: &str) -> StreamKey {
        StreamKey::new("t", SESSION, turn_id)
    }

    /// A queued turn takes its slot while the turn before it is still
    /// settling, so one turn's writer must never reach the other's slot.
    #[test]
    fn two_turns_of_one_session_hold_their_own_slots() {
        let streams = Streams::default();
        streams.insert(stream("turn-1", Some("100.1")));
        streams.insert(stream("turn-2", None));

        streams.kill(&key("turn-1"));
        assert!(streams.get(&key("turn-1")).unwrap().dead);
        assert!(!streams.get(&key("turn-2")).unwrap().dead);

        streams.commit(&key("turn-1"), "9.9".into(), 7, vec![]);
        assert_eq!(streams.get(&key("turn-2")).unwrap().ts, None);

        streams.remove(&key("turn-1"));
        assert!(streams.get(&key("turn-2")).is_some());
    }

    /// The guard: a turn with no message yet must see the open one beside it.
    #[test]
    fn an_open_message_is_visible_to_the_turn_behind_it() {
        let streams = Streams::default();
        streams.insert(stream("turn-2", None));
        // Nothing else open yet.
        assert!(!streams.open_elsewhere(&key("turn-2")));

        streams.insert(stream("turn-1", Some("100.1")));
        assert!(streams.open_elsewhere(&key("turn-2")));
        // Its own message is not another's.
        assert!(!streams.open_elsewhere(&key("turn-1")));
        // A turn holding no message blocks nobody.
        streams.insert(stream("turn-1", None));
        assert!(!streams.open_elsewhere(&key("turn-2")));
        // Nor does an abandoned one.
        streams.insert(stream("turn-1", Some("100.1")));
        streams.kill(&key("turn-1"));
        assert!(!streams.open_elsewhere(&key("turn-2")));

        // A cancel takes the whole session.
        streams.remove_session("t", SESSION);
        assert!(streams.get(&key("turn-1")).is_none());
        assert!(streams.get(&key("turn-2")).is_none());
    }
}
