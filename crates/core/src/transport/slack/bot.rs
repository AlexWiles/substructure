use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::Duration;

use serde_json::Value;
use tokio::sync::Notify;

use super::render::{self, PromptView, Rendered};
use super::state::StreamStore;
use super::{
    app_mention, block_action, build_batch, clip, display_of, dm_message, draft, prompt_options,
    resolution_text, section_block, unstamped_ours, with_attachments, Click, Inbound, ReplyMeta,
    SlackFile, MAX_FALLBACK, MAX_MARKDOWN, REPLY_EVENT_TYPE,
};
use crate::event_store::Seq;
use crate::processor::{EventProcessor, EventProcessorRunnerConfig, ProcessorError};
use crate::protocol::{
    ClientInput, Content, Issuer, Requester, Role, SessionOwner, StoredContent, Subject, Visibility,
};
use crate::runtime::blob::{
    audio_format, text_like, video_playable, BlobError, BlobRef, BlobStore, NewBlob,
};
use crate::runtime::session::interrupts::auth::Authorize;
use crate::session::command::SessionError;
use crate::session::events::EventPayload;
use crate::session::state::SessionStatus;
use crate::session::SessionEvent;
use crate::transport::channel::ChannelContext;
use crate::transport::consent::{CliConsent, Consent, WayIn};
use crate::{Caller, HandleClientInput, RuntimeError};

/// Anthropic's own cap; a larger image fails the call anyway.
const MAX_IMAGE_BYTES: u64 = 5 * 1024 * 1024;
/// Under provider request caps with room for base64 growth.
const MAX_PDF_BYTES: u64 = 10 * 1024 * 1024;
/// Text inlines into the prompt, so a little goes a long way.
const MAX_TEXT_BYTES: u64 = 1024 * 1024;
// Media rides in every later model call on the session, so these stay small.
const MAX_AUDIO_BYTES: u64 = 10 * 1024 * 1024;
const MAX_VIDEO_BYTES: u64 = 20 * 1024 * 1024;
const IMAGE_MIMES: [&str; 4] = ["image/png", "image/jpeg", "image/gif", "image/webp"];
const IMAGE_NOT_ATTACHED: &str = "_an image could not be attached_";

/// Slack limits the rate of `chat.appendStream`.
const ACTIVITY_INTERVAL: Duration = Duration::from_secs(1);
/// Shown as `<app> is typing…`.
/// Slack removes a status after two minutes. Set it again on this cadence.
const STATUS_REFRESH: Duration = Duration::from_secs(90);

/// Only an `im` channel (`D…`) is one person's. A group DM is not: more than
/// one person reads it, and `dm_message` never matches it anyway. Read from
/// the channel id rather than `conversations.info` — the installed scopes do
/// not cover that call, and a lookup that fails must read as shared.
fn audience_of(channel: &str) -> Visibility {
    match channel.starts_with('D') {
        true => Visibility::Private,
        false => Visibility::Shared,
    }
}

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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
    /// Where the session's messages go.
    thread: Thread,
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

/// The thread the session's owner records, if it records one.
fn owner_thread(meta: &crate::session::state::EventMeta) -> Option<Thread> {
    let owner = meta.owner.as_ref()?;
    Some(Thread::new(
        owner.metadata.get("slack_channel")?,
        owner.metadata.get("slack_thread_ts")?,
    ))
}

/// A poisoned lock guards only bookkeeping; keep serving.
/// How large each attachment type may be, or `None` for a type no model
/// reads. Text files inline into the prompt, so their cap is the tightest.
fn attachment_cap(mime: &str) -> Option<u64> {
    if IMAGE_MIMES.contains(&mime) {
        Some(MAX_IMAGE_BYTES)
    } else if mime == "application/pdf" {
        Some(MAX_PDF_BYTES)
    } else if text_like(mime) {
        Some(MAX_TEXT_BYTES)
    } else if audio_format(mime).is_some() {
        Some(MAX_AUDIO_BYTES)
    } else if video_playable(mime) {
        Some(MAX_VIDEO_BYTES)
    } else {
        None
    }
}

/// The blocks with the image blocks replaced by a note, or `None` when there
/// is nothing to strip.
fn without_images(blocks: &[Value]) -> Option<Vec<Value>> {
    let mut kept: Vec<Value> = blocks
        .iter()
        .filter(|b| b["type"] != "image")
        .cloned()
        .collect();
    (kept.len() != blocks.len()).then(|| {
        kept.push(render::context_block(IMAGE_NOT_ATTACHED));
        kept
    })
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// What a turn's message should say, as a decision wrote it.
#[derive(Debug, Clone)]
struct View {
    text: String,
    blocks: Vec<Value>,
}

impl View {
    fn parse(v: &Value) -> Option<Self> {
        Some(Self {
            text: v["text"].as_str().unwrap_or_default().to_string(),
            blocks: v["blocks"].as_array()?.clone(),
        })
    }

    /// The view as stream chunks. A card is keyed by its id, text by its
    /// content, so moving text does not send it twice.
    fn chunks(&self) -> Vec<(String, Value)> {
        self.blocks
            .iter()
            .filter_map(|block| match block["type"].as_str() {
                Some("task_card") => {
                    let id = block["task_id"].as_str()?;
                    let mut chunk = serde_json::json!({
                        "type": "task_update",
                        "id": id,
                        "status": block["status"],
                    });
                    if let Some(title) = block["title"].as_str() {
                        chunk["title"] = clip(title, 60).into();
                    }
                    Some((id.to_string(), chunk))
                }
                _ => {
                    let text = block["text"]["text"].as_str()?;
                    let mut hash = std::collections::hash_map::DefaultHasher::new();
                    std::hash::Hash::hash(text, &mut hash);
                    Some((
                        format!("say:view:{:x}", std::hash::Hasher::finish(&hash)),
                        serde_json::json!({ "type": "markdown_text", "text": text }),
                    ))
                }
            })
            .collect()
    }
}

/// The open streams, and the sessions whose activity is not yet rendered.
#[derive(Default)]
struct Streams {
    open: Mutex<HashMap<StreamKey, Stream>>,
    dirty: Mutex<HashSet<StreamKey>>,
    views: Mutex<HashMap<StreamKey, View>>,
    /// The finished message. It does not stream.
    finals: Mutex<HashMap<StreamKey, View>>,
    prompts: Mutex<HashMap<StreamKey, View>>,
    notify: Notify,
}

impl Streams {
    fn insert(&self, stream: Stream) {
        lock(&self.open).insert(stream.key(), stream);
    }

    /// Track a turn that is not tracked yet. A turn can start twice, and the
    /// slot it already holds knows its message.
    fn track_new(&self, stream: Stream) {
        lock(&self.open).entry(stream.key()).or_insert(stream);
    }

    fn remove(&self, key: &StreamKey) {
        lock(&self.open).remove(key);
    }

    /// Every turn of the session, removed and handed back: a cancel ends them
    /// all, and each one that opened a message has to close it.
    fn take_session(&self, tenant_id: &str, session_id: &str) -> Vec<Stream> {
        let mut open = lock(&self.open);
        let mine: Vec<StreamKey> = open
            .keys()
            .filter(|k| k.tenant_id == tenant_id && k.session_id == session_id)
            .cloned()
            .collect();
        mine.iter().filter_map(|key| open.remove(key)).collect()
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

    /// Record what was sent. False when the turn ended during the append.
    fn commit(
        &self,
        key: &StreamKey,
        ts: String,
        version: u64,
        sent: Vec<(String, Value)>,
    ) -> bool {
        let mut open = lock(&self.open);
        let Some(stream) = open.get_mut(key) else {
            return false;
        };
        stream.ts = Some(ts);
        stream.version = version;
        stream.sent.extend(sent);
        true
    }

    /// A turn of this session other than `turn_id` that is still running.
    fn live_elsewhere(
        &self,
        tenant_id: &str,
        session_id: &str,
        turn_id: Option<&str>,
    ) -> Option<String> {
        lock(&self.open)
            .values()
            .find(|s| {
                s.tenant_id == tenant_id
                    && s.session_id == session_id
                    && Some(s.turn_id.as_str()) != turn_id
                    && !s.dead
            })
            .map(|s| s.turn_id.clone())
    }

    /// Every turn still running, whether or not it has opened a message.
    fn live(&self) -> Vec<Stream> {
        lock(&self.open)
            .values()
            .filter(|s| !s.dead)
            .cloned()
            .collect()
    }

    fn set_view(&self, key: StreamKey, view: View) {
        lock(&self.views).insert(key, view);
    }

    fn view(&self, key: &StreamKey) -> Option<View> {
        lock(&self.views).get(key).cloned()
    }

    fn set_final(&self, key: StreamKey, view: View) {
        lock(&self.finals).insert(key, view);
    }

    /// The finished message, or how far the stream got. Both slots are spent.
    fn take_message(&self, key: &StreamKey) -> Option<View> {
        let streamed = lock(&self.views).remove(key);
        lock(&self.finals).remove(key).or(streamed)
    }

    fn set_prompt(&self, key: StreamKey, prompt: View) {
        lock(&self.prompts).insert(key, prompt);
    }

    fn take_prompt(&self, key: &StreamKey) -> Option<View> {
        lock(&self.prompts).remove(key)
    }

    fn clear_session_views(&self, tenant_id: &str, session_id: &str) {
        let mine = |k: &StreamKey| k.tenant_id == tenant_id && k.session_id == session_id;
        lock(&self.views).retain(|k, _| !mine(k));
        lock(&self.finals).retain(|k, _| !mine(k));
        lock(&self.prompts).retain(|k, _| !mine(k));
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
    blobs: Option<Arc<dyn BlobStore>>,
    consent: Arc<dyn Consent>,
}

impl SlackBot {
    /// `store` holds durable stream state: a restart resumes open streaming
    /// messages in place. Without one a restart orphans them.
    /// `blobs` holds uploaded images; without one attachments are dropped.
    pub fn new(
        resolver: Arc<dyn WorkspaceResolver>,
        api_base: String,
        store: Option<StreamStore>,
        blobs: Option<Arc<dyn BlobStore>>,
    ) -> Self {
        Self {
            resolver,
            api_base,
            http: reqwest::Client::new(),
            ctx: Arc::new(OnceLock::new()),
            streams: Arc::new(Streams::default()),
            store: store.map(Arc::new),
            blobs,
            consent: Arc::new(CliConsent),
        }
    }

    /// A Slack person is named within their workspace: `U1` in two workspaces
    /// is two people. The team comes from the signed payload, or from the
    /// install itself; with neither we name nobody rather than conflate them.
    async fn requester(
        &self,
        ws: &Workspace,
        team: Option<&str>,
        user: &str,
        visibility: Visibility,
    ) -> Requester {
        let team = match team {
            Some(team) => Some(team.to_string()),
            None => self.identity(ws).await.and_then(|i| i.team.clone()),
        };
        match team {
            Some(team) => Requester::new(
                Subject::new(Issuer::slack(), format!("{team}:{user}")),
                visibility,
            ),
            None => Requester::machine(),
        }
    }

    /// Where this deployment sends a person to authorize a connection.
    pub fn with_consent(mut self, consent: Arc<dyn Consent>) -> Self {
        self.consent = consent;
        self
    }

    /// The line telling a person how to authorize, appended when a prompt
    /// carries the facts. A link is minted at delivery, so the event log
    /// holds none, and Slack's markup is applied here rather than by whoever
    /// decided where to send them.
    async fn way_in(&self, tenant_id: &str, payload: &serde_json::Value) -> Option<String> {
        let authorize: Authorize =
            serde_json::from_value(payload.get("authorize")?.clone()).ok()?;
        Some(match self.consent.way_in(tenant_id, &authorize).await? {
            WayIn::Link { url, label } => format!("<{url}|{label}>"),
            WayIn::Command(words) => words,
        })
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
    /// Each click goes into the session as a `client.action` decision. What
    /// it means is decided there.
    pub async fn handle_interaction(&self, payload: &Value) {
        let Some(ctx) = self.ctx.get().cloned() else {
            tracing::warn!("slack: interaction before start; dropped");
            return;
        };
        let Some(click) = block_action(payload) else {
            return;
        };
        let team = payload["team"]["id"]
            .as_str()
            .or_else(|| payload["user"]["team_id"].as_str());
        let app = payload["api_app_id"].as_str();
        let Some(ws) = self.resolver.by_install(team, app, &click.channel).await else {
            tracing::warn!(
                team = %team.unwrap_or(""),
                app = %app.unwrap_or(""),
                channel = %click.channel,
                "slack: click for unknown workspace"
            );
            return;
        };
        self.submit_click(&ctx, &ws, click).await;
    }

    async fn submit_click(&self, ctx: &ChannelContext, ws: &Workspace, click: Click) {
        let Some(agent_id) = ws.routing.agent_for(&click.channel) else {
            tracing::warn!(channel = %click.channel, "slack: click in a channel no agent serves");
            return;
        };
        let session_id = click
            .session
            .clone()
            .unwrap_or_else(|| format!("slack:{}:{}", click.channel, click.thread_ts));
        let clicked = self
            .requester(ws, None, &click.user, audience_of(&click.channel))
            .await;
        let submitted = ctx
            .handle_client_input(HandleClientInput {
                session_id: session_id.clone(),
                caller: Caller::System {
                    tenant_id: ws.tenant_id.clone(),
                },
                // Used only if the click starts the session; an existing
                // session keeps its owner.
                owner: SessionOwner {
                    tenant_id: ws.tenant_id.clone(),
                    requester: clicked,
                    metadata: HashMap::from_iter([
                        ("slack_channel".to_string(), click.channel.clone()),
                        ("slack_thread_ts".to_string(), click.thread_ts.clone()),
                    ]),
                },
                input: ClientInput::Action {
                    agent_id: agent_id.to_string(),
                    turn_id: None,
                    name: click.action_id.clone(),
                    args: Some(click.args()),
                },
                span: crate::span::SpanContext::root().child("slack_click"),
            })
            .await;
        if let Err(e) = submitted {
            tracing::warn!(error = %e, %session_id, "slack: click submit failed");
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

    /// The stored part for each attachment, by file id. A file that cannot be
    /// stored (type, size, or a failure) is absent, and its message carries a
    /// note instead.
    async fn store_attachments(
        &self,
        ws: &Workspace,
        files: impl Iterator<Item = SlackFile>,
    ) -> HashMap<String, StoredContent> {
        let Some(blobs) = &self.blobs else {
            return HashMap::new();
        };
        let mut out = HashMap::new();
        for f in files {
            let Some(cap) = attachment_cap(&f.mimetype) else {
                continue;
            };
            if f.size > cap {
                tracing::warn!(file = %f.id, size = f.size, "slack: attachment over size cap; skipped");
                continue;
            }
            let bytes = match self.download_file(ws, &f.url_private).await {
                Ok(bytes) => bytes,
                Err(e) => {
                    tracing::warn!(error = %e, file = %f.id, "slack: file download failed (bot token lacks files:read?)");
                    continue;
                }
            };
            let stored = blobs
                .put(NewBlob {
                    tenant_id: ws.tenant_id.clone(),
                    mime: f.mimetype.clone(),
                    name: f.name.clone(),
                    bytes,
                })
                .await;
            match stored {
                Ok(r) => {
                    out.insert(f.id, StoredContent::Blob { uri: r.uri() });
                }
                Err(e) => tracing::warn!(error = %e, file = %f.id, "slack: blob put failed"),
            }
        }
        out
    }

    /// `url_private` needs the bot token. Without `files:read` Slack answers
    /// 200 with an HTML login page, so the content type is the check.
    async fn download_file(&self, ws: &Workspace, url: &str) -> Result<Vec<u8>, Error> {
        let resp = self
            .http
            .get(url)
            .bearer_auth(&ws.bot_token)
            .send()
            .await
            .map_err(Error::retryable)?;
        if !resp.status().is_success() {
            return Err(Error::Terminal(format!("http {}", resp.status())));
        }
        let html = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.contains("text/html"));
        if html {
            return Err(Error::Terminal("got html, not the file".to_string()));
        }
        Ok(resp.bytes().await.map_err(Error::retryable)?.to_vec())
    }

    /// Blocks for the images on the turn's reply. Each blob uploads to Slack
    /// once per workspace; the block embeds the Slack file. A file that cannot
    /// be delivered becomes a note instead of sinking the reply.
    async fn turn_image_blocks(&self, ws: &Workspace, session_id: &str) -> Vec<Value> {
        if self.blobs.is_none() {
            return Vec::new();
        }
        let path = self.session_path(ws, session_id).await;
        let Some(message) = path.last().filter(|m| matches!(m.role, Role::Assistant)) else {
            return Vec::new();
        };
        let Some(Content::Parts(parts)) = &message.content else {
            return Vec::new();
        };
        let mut blocks = Vec::new();
        for part in parts {
            let StoredContent::Blob { uri } = part else {
                continue;
            };
            match self.slack_image_block(ws, uri).await {
                Ok(Some(block)) => blocks.push(block),
                Ok(None) => {}
                Err(e) => {
                    tracing::warn!(error = %e, %session_id, "slack: image not attached");
                    blocks.push(render::context_block(IMAGE_NOT_ATTACHED));
                }
            }
        }
        blocks
    }

    /// One image part as a block: an https url embeds directly; a blob ref
    /// uploads (or reuses the workspace's upload) and embeds the Slack file.
    async fn slack_image_block(&self, ws: &Workspace, url: &str) -> Result<Option<Value>, Error> {
        if url.starts_with("https://") {
            return Ok(Some(serde_json::json!({
                "type": "image", "image_url": url, "alt_text": "image",
            })));
        }
        let Some(r) = BlobRef::parse(url) else {
            return Ok(None);
        };
        if r.tenant_id != ws.tenant_id || !r.mime.starts_with("image/") {
            return Ok(None);
        }
        let file_id = self.slack_file_for(ws, &r).await?;
        Ok(Some(serde_json::json!({
            "type": "image",
            "slack_file": { "id": file_id },
            "alt_text": r.name.as_deref().unwrap_or("image"),
        })))
    }

    async fn slack_file_for(&self, ws: &Workspace, r: &BlobRef) -> Result<String, Error> {
        if let Some(store) = self.store.as_deref() {
            if let Ok(Some(id)) = store.slack_file(&ws.tenant_id, &r.id).await {
                return Ok(id);
            }
        }
        let blobs = self
            .blobs
            .as_ref()
            .ok_or_else(|| Error::Terminal("no blob store".to_string()))?;
        let bytes = blobs.get(r).await.map_err(|e| match e {
            BlobError::Io(m) => Error::Retryable(m),
            e => Error::Terminal(e.to_string()),
        })?;
        let name = r
            .name
            .clone()
            .unwrap_or_else(|| format!("image.{}", r.mime.strip_prefix("image/").unwrap_or("bin")));
        let file_id = self.upload_file(ws, &name, bytes).await?;
        if let Some(store) = self.store.as_deref() {
            if let Err(e) = store
                .record_slack_file(&ws.tenant_id, &r.id, &file_id)
                .await
            {
                tracing::warn!(error = %e, "slack: uploaded file not recorded");
            }
        }
        Ok(file_id)
    }

    /// The external upload dance: get a one-time url, put the bytes, complete.
    /// The file stays unshared; the stamped reply embeds it by id.
    async fn upload_file(
        &self,
        ws: &Workspace,
        name: &str,
        bytes: Vec<u8>,
    ) -> Result<String, Error> {
        // This method takes form or query args, not JSON.
        let query =
            serde_urlencoded::to_string([("filename", name), ("length", &bytes.len().to_string())])
                .map_err(Error::retryable)?;
        let resp: Value = self
            .http
            .get(format!(
                "{}/files.getUploadURLExternal?{query}",
                self.api_base
            ))
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
        let (Some(upload_url), Some(file_id)) =
            (resp["upload_url"].as_str(), resp["file_id"].as_str())
        else {
            return Err(Error::Terminal(
                "upload url response incomplete".to_string(),
            ));
        };
        let put = self
            .http
            .post(upload_url)
            .body(bytes)
            .send()
            .await
            .map_err(Error::retryable)?;
        if !put.status().is_success() {
            return Err(Error::Retryable(format!("upload http {}", put.status())));
        }
        let file_id = file_id.to_string();
        self.api_call(
            ws,
            "files.completeUploadExternal",
            serde_json::json!({ "files": [{ "id": file_id, "title": name }] }),
        )
        .await?;
        Ok(file_id)
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

    /// The prompt's own words, so a link in it comes back too.
    fn still_waiting(payload: &Value) -> Option<String> {
        let display = display_of(payload)?;
        Some(format!("Still waiting on this:\n\n{}", display.message))
    }

    /// The buttons stay on the message that has them, because two sets are two
    /// ways to answer one prompt.
    async fn remind_of_prompt(
        &self,
        ws: &Workspace,
        thread: &Thread,
        session_id: &str,
        turn_id: Option<String>,
    ) {
        let Some(ctx) = self.ctx.get() else { return };
        let Ok(session) = ctx.get_session(&ws.tenant_id, session_id).await else {
            return;
        };
        let head = session.state.head_id.clone();
        let Some(open) = session.state.active_interrupt_for(head.as_deref()) else {
            return;
        };
        let Some(text) = Self::still_waiting(&open.payload) else {
            return;
        };
        let meta = ReplyMeta {
            turn_id,
            session_id: Some(session_id.to_string()),
            ..Default::default()
        };
        if let Err(e) = self.post(ws, thread, &text, &meta).await {
            tracing::warn!(error = %e, %session_id, "slack: failed to repeat the open prompt");
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

        // Store the unseen images before drafting, so the drafts hold refs.
        let uploads = {
            let seen: HashSet<&str> = path.iter().map(|m| m.id.as_str()).collect();
            let mut unseen: HashMap<String, SlackFile> = HashMap::new();
            let mut collect = |ts: &str, files: &[SlackFile]| {
                if !seen.contains(format!("slack:{ts}").as_str()) {
                    for f in files {
                        unseen.insert(f.id.clone(), f.clone());
                    }
                }
            };
            if let Ok(replies) = &fetched {
                for msg in replies.iter().filter(|m| m.meta.is_none()) {
                    collect(&msg.ts, &msg.files);
                }
            }
            collect(&inbound.ts, &inbound.files);
            self.store_attachments(ws, unseen.into_values()).await
        };

        // If the fetch fails, append the message alone with a note.
        let input = match fetched {
            Ok(replies) => ClientInput::Append {
                agent_id: agent_id.clone(),
                turn_id: turn_id.clone(),
                messages: build_batch(&path, &replies, &inbound, &uploads),
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
                        with_attachments(
                            format!(
                                "<@{}>: {}\n\n[note: the Slack conversation could not be fetched — \
                                 earlier messages may be missing from your context]",
                                inbound.user, inbound.text
                            ),
                            &inbound.files,
                            &uploads,
                        ),
                    ),
                    stream: false,
                    queue: true,
                }
            }
        };
        let inbound_requester = self
            .requester(
                ws,
                inbound.team.as_deref(),
                &inbound.user,
                audience_of(&inbound.channel),
            )
            .await;
        let submitted = ctx
            .handle_client_input(HandleClientInput {
                session_id: session_id.clone(),
                caller: Caller::System {
                    tenant_id: ws.tenant_id.clone(),
                },
                owner: SessionOwner {
                    tenant_id: ws.tenant_id.clone(),
                    requester: inbound_requester,
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
            // A redelivery. The first copy is queued or running.
            Err(RuntimeError::Session(
                SessionError::TurnAlreadyActive { .. } | SessionError::TurnAlreadyCompleted { .. },
            )) => {}
            // The message was refused, so silence leaves a person waiting.
            Err(RuntimeError::Session(SessionError::SessionInterrupted)) => {
                self.remind_of_prompt(ws, &thread, &session_id, turn_id)
                    .await;
            }
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
        let body = |blocks: &[Value]| {
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
            })
        };
        match self.api_call(ws, "chat.postMessage", body(&blocks)).await {
            // Images degrade rather than sink the reply.
            Err(e) if e.code() == "invalid_blocks" => {
                let Some(stripped) = without_images(&blocks) else {
                    return Err(e);
                };
                self.api_call(ws, "chat.postMessage", body(&stripped))
                    .await
                    .map(|_| ())
            }
            r => r.map(|_| ()),
        }
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
        let body = |blocks: &[Value]| {
            serde_json::json!({
                "channel": channel,
                "ts": ts,
                "text": clip(text, MAX_FALLBACK),
                "blocks": blocks,
                "metadata": {
                    "event_type": REPLY_EVENT_TYPE,
                    "event_payload": meta,
                },
            })
        };
        match self.api_call(ws, "chat.update", body(&blocks)).await {
            Err(e) if e.code() == "invalid_blocks" => {
                let Some(stripped) = without_images(&blocks) else {
                    return Err(e);
                };
                self.api_call(ws, "chat.update", body(&stripped))
                    .await
                    .map(|_| ())
            }
            r => r.map(|_| ()),
        }
    }

    /// Whether the event is a running turn's work. A session names its last
    /// turn after that turn ends, so the event's turn id does not say. The
    /// slot does: it is opened at the turn's start and taken by its reply.
    fn working(&self, event: &SessionEvent) -> bool {
        // The prompt took the slot with it; the resume speaks for itself.
        if matches!(event.payload, EventPayload::InterruptResumed(_)) {
            return true;
        }
        // The events that end a turn, and a prompt waiting on a person.
        let stops = matches!(
            event.payload,
            EventPayload::TurnCompleted(_)
                | EventPayload::SessionInterrupted(_)
                | EventPayload::SessionCancelled
                | EventPayload::SessionDone(_)
        ) || matches!(event.meta.status, SessionStatus::Interrupted { .. });
        let Some(turn_id) = event.meta.turn_id.as_deref() else {
            return false;
        };
        let key = StreamKey::new(&event.tenant_id, &event.session_id, turn_id);
        !stops && self.streams.get(&key).is_some()
    }

    /// The working indicator. An empty status clears it.
    async fn set_working(&self, ws: &Workspace, thread: &Thread) {
        self.set_status(ws, thread, render::WORKING_STATUS).await;
    }

    /// The indicator belongs to the thread, not to one turn: a turn ending
    /// hands it to the turn queued behind it rather than clearing it, so a
    /// thread that is still working never looks idle.
    async fn settle_status(
        &self,
        ws: &Workspace,
        thread: &Thread,
        event: &SessionEvent,
        ended: Option<&str>,
    ) {
        if self
            .streams
            .live_elsewhere(&event.tenant_id, &event.session_id, ended)
            .is_none()
        {
            return self.set_status(ws, thread, "").await;
        }
        self.set_working(ws, thread).await
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

    /// Stop the stream without adding to it, for the paths that have nothing
    /// left to say. A message left streaming shows a spinner until Slack
    /// expires it, so every path that opens one closes it.
    ///
    /// Best effort by nature: a stream already stopped, expired, or on a
    /// deleted message is one that is no longer open, which is the point.
    async fn close_stream(&self, ws: &Workspace, channel: &str, ts: &str) {
        let body = serde_json::json!({ "channel": channel, "ts": ts });
        if let Err(e) = self.api_call(ws, "chat.stopStream", body).await {
            tracing::debug!(error = %e, channel, "slack: stream already closed");
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
        // Slack's sessions are the ones whose owner records a thread.
        let Some(thread) = owner_thread(&event.meta) else {
            return Ok(());
        };
        let Some(ws) = self.resolver.by_tenant(&event.tenant_id).await else {
            return Ok(());
        };
        // The row lands before the checkpoint commits: a lost write replays.
        if let Err(e) = self.track(&ws, &thread, &event).await {
            return Err(ProcessorError::Apply(e.to_string()));
        }
        // Slack clears the indicator on every write to the thread, so each
        // event of a running turn sets it again.
        if self.working(&event) {
            self.set_working(&ws, &thread).await;
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
            EventPayload::ChannelsUpdated(c) => self.apply_channels(&ws, &thread, &event, c).await,
            _ => return Ok(()),
        };
        // The turn's end puts the indicator out, after the reply and not
        // before it: the reply takes the slot, so no card can light it again.
        match &event.payload {
            EventPayload::TurnCompleted(t) => {
                self.settle_status(&ws, &thread, &event, Some(&t.turn_id))
                    .await
            }
            EventPayload::SessionInterrupted(_) => {
                self.settle_status(&ws, &thread, &event, event.meta.turn_id.as_deref())
                    .await
            }
            _ => {}
        }
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
    /// Apply what a settled decision said Slack should show.
    async fn apply_channels(
        &self,
        ws: &Workspace,
        thread: &Thread,
        event: &SessionEvent,
        c: &crate::session::events::ChannelsUpdated,
    ) -> Result<(), Error> {
        let Some(slack) = c.channels.get("slack") else {
            return Ok(());
        };
        if let Some(status) = slack["status"].as_str() {
            self.set_status(ws, thread, status).await;
        }
        if let Some(update) = slack.get("update") {
            let channel = update["channel"].as_str().unwrap_or(&thread.channel);
            let (Some(ts), Some(text)) = (update["ts"].as_str(), update["text"].as_str()) else {
                tracing::warn!(session_id = %event.session_id, "slack: update view missing ts or text");
                return Ok(());
            };
            let blocks = update["blocks"].as_array().cloned().unwrap_or_default();
            let meta = ReplyMeta {
                session_id: Some(event.session_id.clone()),
                ..Default::default()
            };
            self.update(ws, channel, ts, text, blocks, &meta).await?;
        }
        if let Some(turn_id) = event.meta.turn_id.as_deref() {
            let key = StreamKey::new(&event.tenant_id, &event.session_id, turn_id);
            if let Some(view) = slack.get("view").and_then(View::parse) {
                // `complete_turn` posts the finished message.
                if c.finishes_turn {
                    self.streams.set_final(key.clone(), view);
                } else {
                    self.streams.set_view(key.clone(), view);
                    self.streams.mark_dirty(key.clone());
                }
            }
            if let Some(prompt) = slack.get("prompt").and_then(View::parse) {
                self.streams.set_prompt(key, prompt);
            }
        }
        Ok(())
    }

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
        // The turn's message is the final one; an error goes on top of it.
        let rendered = match (self.streams.take_message(&key), &t.error) {
            (Some(view), None) => Rendered {
                text: view.text,
                blocks: view.blocks,
            },
            (Some(view), Some(error)) => {
                let line = format!("Error: {error}");
                let mut blocks = view.blocks;
                blocks.push(section_block(&line));
                Rendered { text: line, blocks }
            }
            (None, _) => render::render_turn(t, elapsed),
        };
        let mut rendered = rendered;
        // A failed turn has no new reply message to take images from.
        if t.error.is_none() {
            rendered
                .blocks
                .extend(self.turn_image_blocks(ws, session_id).await);
        }
        let rendered = rendered;
        let since = live.as_ref().map(|s| s.started_at);
        let Some(ts) = live.as_ref().and_then(|s| s.ts.clone()) else {
            return self
                .post_turn(ws, thread, session_id, t, since, &rendered)
                .await;
        };
        if self.already_replied(ws, thread, t, since).await {
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
                if !rendered.blocks.is_empty() {
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
                        self.post_turn(ws, thread, session_id, t, since, &rendered)
                            .await
                    }
                }
            }
            // Do not lose the answer to a bad stream.
            Err(e) => {
                tracing::warn!(error = %e, %session_id, "slack: stream finalize failed; posting reply");
                let posted = self
                    .post_turn(ws, thread, session_id, t, since, &rendered)
                    .await;
                // The answer is in the thread, and the stream that would not
                // take it is still open above it.
                self.close_stream(ws, &thread.channel, &ts).await;
                posted
            }
        }
    }

    async fn post_turn(
        &self,
        ws: &Workspace,
        thread: &Thread,
        session_id: &str,
        t: &crate::session::events::TurnCompleted,
        since: Option<chrono::DateTime<chrono::Utc>>,
        rendered: &Rendered,
    ) -> Result<(), Error> {
        if self.already_replied(ws, thread, t, since).await {
            return Ok(());
        }
        let meta = self.reply_meta(ws, session_id, t).await;
        self.post_blocks(ws, thread, &rendered.text, rendered.blocks.clone(), &meta)
            .await
    }

    /// True if the thread has the turn's stamped reply. A failed fetch says
    /// no, so the reply goes out again. `since` bounds the scan.
    async fn already_replied(
        &self,
        ws: &Workspace,
        thread: &Thread,
        t: &crate::session::events::TurnCompleted,
        since: Option<chrono::DateTime<chrono::Utc>>,
    ) -> bool {
        let oldest = since.map(|at| format!("{}.000000", at.timestamp()));
        let Ok(replies) = self.fetch_thread(ws, thread, oldest.as_deref()).await else {
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
        // The decision that interrupted can write the prompt itself.
        let key = turn_id.map(|t| StreamKey::new(&ws.tenant_id, session_id, t));
        let authored = key.as_ref().and_then(|k| self.streams.take_prompt(k));
        let (text, blocks) = match authored {
            Some(view) => (view.text, view.blocks),
            None => match display_of(&p.payload) {
                Some(display) => {
                    let options = prompt_options(&display, &p.interrupt_id);
                    let message = match self.way_in(&ws.tenant_id, &p.payload).await {
                        Some(how) => format!("{}\n\n{how}", display.message),
                        None => display.message.clone(),
                    };
                    let blocks = render::prompt_blocks(&PromptView {
                        message: &message,
                        options: &options,
                        expires_at: display.expires_at.as_deref(),
                    });
                    (message, blocks)
                }
                None => {
                    let text = format!("Paused: {}", p.reason);
                    let block = section_block(&text);
                    (text, vec![block])
                }
            },
        };
        // A prompt can wait for a long time; do not hold the stream open.
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
        let blocks = render::settled_prompt_blocks(&msg.blocks, &msg.text, &resolution);
        self.update(ws, &thread.channel, &msg.ts, &text, blocks, &meta)
            .await
    }
}

/// Activity: events mark a session dirty; one worker renders the cards.
impl SlackBot {
    /// Track the turn's slot; durable when a store is attached. A store
    /// failure is retryable so the row always lands before the checkpoint.
    async fn track(
        &self,
        ws: &Workspace,
        thread: &Thread,
        event: &SessionEvent,
    ) -> Result<(), Error> {
        // Every event names the turn it belongs to; a start names its own.
        let turn_id = match &event.payload {
            EventPayload::TurnStarted(t) => Some(t.turn_id.clone()),
            _ => event.meta.turn_id.clone(),
        };
        match &event.payload {
            EventPayload::TurnStarted(t) => {
                let owner = event.meta.owner.as_ref();
                let recipient = owner
                    .and_then(|o| o.requester.subject.as_ref().map(|s| s.id.as_str()))
                    .and_then(|id| id.strip_prefix("slack:"))
                    .map(str::to_string);
                let recipient_team = owner.and_then(|o| o.metadata.get("slack_team")).cloned();
                let mut stream = Stream {
                    tenant_id: event.tenant_id.clone(),
                    session_id: event.session_id.clone(),
                    turn_id: t.turn_id.clone(),
                    thread: thread.clone(),
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
                self.streams.track_new(stream);
            }
            EventPayload::ToolCallRequested(_)
            | EventPayload::ToolCallCompleted(_)
            | EventPayload::ToolCallErrored(_)
            | EventPayload::SubAgentRequested(_)
            | EventPayload::SubAgentTurnCompleted(_)
            | EventPayload::SubAgentErrored(_) => {}
            // A cancel ends every turn of the session.
            EventPayload::SessionCancelled => {
                let open = self
                    .streams
                    .take_session(&event.tenant_id, &event.session_id);
                self.close_turns(ws, &event.session_id, thread, open).await;
                self.streams
                    .clear_session_views(&event.tenant_id, &event.session_id);
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
        let mut refreshed = std::time::Instant::now();
        loop {
            // Slack drops a status after two minutes, so every live turn has
            // it set again on this cadence. Off the clock rather than off the
            // queue: a turn that streams a card every second never lets the
            // idle branch run, and it is the turns that work the longest that
            // most need to look like they are working.
            if refreshed.elapsed() >= STATUS_REFRESH {
                self.refresh_statuses().await;
                refreshed = std::time::Instant::now();
            }
            let Some(key) = self.streams.take_dirty() else {
                // Nothing to render: wait for work, or for the status to be
                // due again.
                tokio::select! {
                    _ = ctx.shutdown.cancelled() => return,
                    _ = self.streams.notified() => {}
                    _ = tokio::time::sleep(STATUS_REFRESH) => {}
                }
                continue;
            };
            let Some(ws) = self.resolver.by_tenant(&key.tenant_id).await else {
                continue;
            };
            self.stream_turn(&ws, &key).await;
            tokio::select! {
                _ = ctx.shutdown.cancelled() => return,
                _ = tokio::time::sleep(ACTIVITY_INTERVAL) => {}
            }
        }
    }

    /// Set the status again for every running turn. Slack drops one after two
    /// minutes, and the longest turns need it most.
    async fn refresh_statuses(&self) {
        for stream in self.streams.live() {
            let Some(ws) = self.resolver.by_tenant(&stream.tenant_id).await else {
                continue;
            };
            self.set_working(&ws, &stream.thread).await;
        }
    }

    /// Append what changed in the turn's view.
    async fn stream_turn(&self, ws: &Workspace, key: &StreamKey) {
        // A missing slot rebuilds from the store; a dead one stays dead.
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
        let thread = open.thread.clone();
        // No view yet: nothing to send but the status.
        let Some(view) = self.streams.view(key) else {
            if open.ts.is_none() {
                self.set_working(ws, &thread).await;
            }
            return;
        };
        let changed: Vec<(String, Value)> = view
            .chunks()
            .into_iter()
            .filter(|(id, chunk)| open.sent.get(id) != Some(chunk))
            .collect();
        if changed.is_empty() {
            if open.ts.is_none() {
                self.set_working(ws, &thread).await;
            }
            return;
        }

        // Set when this tick opens the message: only that one is ours to take
        // back.
        let mut opened = false;
        let (ts, version) = match open.ts.clone() {
            Some(ts) => (ts, open.version),
            // The slot has no message, but the row may name one: this slot
            // is younger than the turn.
            None => match self.recorded_ts(key).await {
                Some(recorded) => recorded,
                None => {
                    // A thread shows one open stream. A queued turn waits for
                    // the turn before it to settle rather than opening a second
                    // one beside it; the tick that follows tries again.
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
                        TsPersist::Kept(version) => {
                            // The reply can take the slot while this opens, and
                            // then it posts a message of its own.
                            if self.streams.get(key).is_none() {
                                self.discard_stream(ws, &thread.channel, &ts).await;
                                return;
                            }
                            opened = true;
                            (ts, version)
                        }
                        // Another writer opened this turn's stream first.
                        TsPersist::Adopted(theirs, version) => {
                            self.discard_stream(ws, &thread.channel, &ts).await;
                            (theirs, version)
                        }
                        TsPersist::TurnOver => {
                            self.discard_stream(ws, &thread.channel, &ts).await;
                            self.streams.remove(key);
                            return;
                        }
                    }
                }
            },
        };
        let chunks = changed.iter().map(|(_, c)| c.clone()).collect();
        if let Err(e) = self.append_stream(ws, &thread.channel, &ts, chunks).await {
            // The message takes nothing more, and the answer posts beside it.
            // Delete one this tick opened; stop one that holds cards.
            match opened {
                true => self.discard_stream(ws, &thread.channel, &ts).await,
                false => self.close_stream(ws, &thread.channel, &ts).await,
            }
            return self.kill_stream(key, e);
        }
        // The card cleared the indicator; the turn goes on, so set it again.
        if self.streams.commit(key, ts.clone(), version, changed) {
            return self.set_working(ws, &thread).await;
        }
        // The turn ended during the append. A message this tick opened is one
        // the reply never saw.
        if opened {
            self.discard_stream(ws, &thread.channel, &ts).await;
        }
    }

    /// The message the row records for the turn, if it records one.
    async fn recorded_ts(&self, key: &StreamKey) -> Option<(String, u64)> {
        let row = self
            .store
            .as_deref()?
            .load(&key.tenant_id, &key.session_id, &key.turn_id)
            .await
            .ok()??;
        Some((row.ts?, row.version))
    }

    /// Take back a message this writer opened: stop the stream, then delete
    /// the message.
    async fn discard_stream(&self, ws: &Workspace, channel: &str, ts: &str) {
        self.close_stream(ws, channel, ts).await;
        self.delete_message(ws, channel, ts).await;
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

    /// Close every message these turns left open. A cancel completes no turn,
    /// so nothing else comes along to finish them.
    async fn close_turns(
        &self,
        ws: &Workspace,
        session_id: &str,
        thread: &Thread,
        open: Vec<Stream>,
    ) {
        for stream in open {
            let Some(ts) = stream.ts.clone() else {
                continue;
            };
            let view = self.streams.take_message(&stream.key());
            let blocks = view.map(|v| v.blocks).unwrap_or_default();
            let rendered = render::render_cancelled(&blocks);
            let meta = ReplyMeta {
                session_id: Some(session_id.to_string()),
                ..Default::default()
            };
            if let Err(e) = self
                .update(
                    ws,
                    &stream.thread.channel,
                    &ts,
                    &rendered.text,
                    rendered.blocks,
                    &meta,
                )
                .await
            {
                tracing::warn!(error = %e, %session_id, "slack: cancelled turn not settled");
            }
            // Whatever the message says, the stream still has to stop.
            self.close_stream(ws, &stream.thread.channel, &ts).await;
        }
        self.set_status(ws, thread, "").await;
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
        // The last streamed view is in the log. Cards can be sent twice, text
        // cannot, so only its chunks seed `sent`.
        let view = events.iter().rev().find_map(|e| match &e.payload {
            EventPayload::ChannelsUpdated(c)
                if e.meta.turn_id.as_deref() == Some(&row.turn_id) && !c.finishes_turn =>
            {
                c.channels.get("slack")?.get("view").and_then(View::parse)
            }
            _ => None,
        });
        let sent: HashMap<String, Value> = view
            .as_ref()
            .map(|v| {
                v.chunks()
                    .into_iter()
                    .filter(|(id, _)| id.starts_with("say:"))
                    .collect()
            })
            .unwrap_or_default();
        if let Some(view) = view {
            self.streams.set_view(key.clone(), view);
        }
        let owner = events.iter().rev().find_map(|e| e.meta.owner.as_ref());
        let thread = Thread::new(
            owner.and_then(|o| o.metadata.get("slack_channel"))?,
            owner.and_then(|o| o.metadata.get("slack_thread_ts"))?,
        );
        let recipient = owner
            .and_then(|o| o.requester.subject.as_ref().map(|s| s.id.as_str()))
            .and_then(|id| id.strip_prefix("slack:"))
            .map(str::to_string);
        let recipient_team = owner.and_then(|o| o.metadata.get("slack_team")).cloned();
        let mut stream = Stream {
            tenant_id: key.tenant_id.clone(),
            session_id: key.session_id.clone(),
            thread,
            turn_id: row.turn_id,
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
        let thread = stream.thread.clone();
        // Never look back past our start: an earlier turn's orphan is not ours.
        let oldest = format!("{}.000000", stream.started_at.timestamp());
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
    /// Only an `im` is one person's. A group DM and a channel are shared, and
    /// so is anything unrecognized — the safe value is the fallback.
    #[test]
    fn only_a_direct_message_is_private() {
        use crate::protocol::Visibility;
        assert_eq!(super::audience_of("D0AAAAAAA"), Visibility::Private);
        assert_eq!(super::audience_of("C0AAAAAAA"), Visibility::Shared);
        assert_eq!(super::audience_of("G0AAAAAAA"), Visibility::Shared, "mpim");
        assert_eq!(super::audience_of(""), Visibility::Shared);
    }

    #[test]
    fn audio_and_video_are_accepted_within_their_caps() {
        assert_eq!(super::attachment_cap("audio/mpeg"), Some(MAX_AUDIO_BYTES));
        assert_eq!(super::attachment_cap("audio/ogg"), Some(MAX_AUDIO_BYTES));
        assert_eq!(super::attachment_cap("video/mp4"), Some(MAX_VIDEO_BYTES));
        assert_eq!(
            super::attachment_cap("video/quicktime"),
            Some(MAX_VIDEO_BYTES)
        );
        assert_eq!(
            super::attachment_cap("video/x-matroska"),
            None,
            "a container no provider names is reported, not downloaded"
        );
        assert_eq!(super::attachment_cap("application/zip"), None);
    }

    use super::*;
    use crate::protocol::{ErrorCode, ErrorInfo};
    use crate::session::events::TurnCompleted;
    use std::sync::Mutex as StdMutex;

    const SESSION: &str = "slack:C1:1.0";

    /// Every call the bot made, in order.
    #[derive(Clone, Default)]
    struct Calls(Arc<StdMutex<Vec<(String, Value)>>>);

    impl Calls {
        fn record(&self, method: &str, body: Value) {
            self.0.lock().unwrap().push((method.to_string(), body));
        }

        /// The methods called, in order.
        fn methods(&self) -> Vec<String> {
            self.0
                .lock()
                .unwrap()
                .iter()
                .map(|(m, _)| m.clone())
                .collect()
        }

        fn to(&self, method: &str) -> Vec<Value> {
            self.0
                .lock()
                .unwrap()
                .iter()
                .filter(|(m, _)| m == method)
                .map(|(_, b)| b.clone())
                .collect()
        }
    }

    /// A Slack that records what it was called with. `conversations.replies`
    /// answers empty so the bot's idempotency check passes and it goes on to
    /// post. `stops` is what `chat.stopStream` answers, for the paths that
    /// only happen when a stream refuses to close, and `appends` what
    /// `chat.appendStream` answers.
    async fn fake_slack_with(stops: Value, appends: Value) -> (String, Calls) {
        use axum::routing::post;
        use axum::{Json, Router};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let calls = Calls::default();
        let route = |app: Router, method: &'static str, answer: Value| {
            let calls = calls.clone();
            app.route(
                &format!("/{method}"),
                post(move |Json(body): Json<Value>| {
                    let (calls, answer) = (calls.clone(), answer.clone());
                    async move {
                        calls.record(method, body);
                        Json(answer)
                    }
                }),
            )
        };
        let ok = serde_json::json!({"ok": true, "ts": "1.1"});
        let app = Router::new();
        let app = route(app, "chat.postMessage", ok.clone());
        let app = route(app, "chat.update", ok.clone());
        let app = route(app, "chat.startStream", ok.clone());
        let app = route(app, "chat.appendStream", appends);
        let app = route(app, "chat.stopStream", stops);
        let app = route(
            app,
            "conversations.replies",
            serde_json::json!({"ok": true, "messages": []}),
        );
        let app = route(
            app,
            "auth.test",
            serde_json::json!({"ok": true, "user_id": "U0", "bot_id": "B0"}),
        );
        let app = route(
            app,
            "assistant.threads.setStatus",
            serde_json::json!({"ok": true}),
        );
        let app = {
            let calls = calls.clone();
            app.route(
                "/files.getUploadURLExternal",
                axum::routing::get(move |axum::extract::RawQuery(q): axum::extract::RawQuery| {
                    let calls = calls.clone();
                    async move {
                        calls.record(
                            "files.getUploadURLExternal",
                            serde_json::json!({"query": q}),
                        );
                        Json(serde_json::json!({
                            "ok": true,
                            "upload_url": format!("http://{addr}/upload"),
                            "file_id": "F123",
                        }))
                    }
                }),
            )
        };
        let app = {
            let calls = calls.clone();
            app.route(
                "/upload",
                post(move |body: axum::body::Bytes| {
                    let calls = calls.clone();
                    async move {
                        calls.record("upload", serde_json::json!({"len": body.len()}));
                        "OK"
                    }
                }),
            )
        };
        let app = route(
            app,
            "files.completeUploadExternal",
            serde_json::json!({"ok": true, "files": [{"id": "F123"}]}),
        );

        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (format!("http://{addr}"), calls)
    }

    async fn fake_slack() -> (String, Calls) {
        fake_slack_with(
            serde_json::json!({"ok": true}),
            serde_json::json!({"ok": true}),
        )
        .await
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

    /// What the bot appends to an auth prompt, per deployment.
    mod way_in {
        use super::super::SlackBot;
        use crate::connectors::Requester;
        use crate::protocol::{ConnectorProtocol, Visibility};
        use crate::protocol::{Issuer, Subject};
        use crate::providers::sqlite::{SqliteAuthFlows, SqliteDb};
        use crate::runtime::session::interrupts::auth;
        use crate::transport::consent::{CliConsent, Consent, DashboardConsent, EngineConsent};
        use crate::transport::mcp_auth::AuthorizeLinks;
        use std::sync::Arc;

        fn payload(requester: Requester) -> serde_json::Value {
            serde_json::json!({
                "message": "*gmail* is not authorized yet.",
                "authorize": auth::Authorize { connection: "gmail".into(), requester },
            })
        }

        fn person(audience: Visibility) -> Requester {
            Requester::new(Subject::new(Issuer::slack(), "T1:U1"), audience)
        }

        fn bot(consent: Arc<dyn Consent>) -> SlackBot {
            SlackBot::new(
                Arc::new(super::OneWorkspace(Arc::new(super::Workspace::new(
                    "xoxb-test".into(),
                    "t".into(),
                    super::Routing::new().dm(Some("a".into())),
                )))),
                "http://127.0.0.1:1".into(),
                None,
                None,
            )
            .with_consent(consent)
        }

        fn engine(path: &std::path::Path) -> Arc<dyn Consent> {
            let db =
                SqliteDb::open(path.to_str().unwrap(), std::time::Duration::from_secs(5)).unwrap();
            let spec = crate::connectors::registry::ConnectionSpec {
                url: "https://mcp.gmail.test/mcp".into(),
                protocol: ConnectorProtocol::Mcp,
                auth: None,
                header: None,
                credential: Some(crate::connectors::registry::CredentialScope::User),
                scopes: Vec::new(),
                client_id_env: None,
                client_secret_env: None,
                prefix_tools: true,
            };
            Arc::new(EngineConsent(Arc::new(AuthorizeLinks::new(
                "https://agent.test",
                Arc::new(SqliteAuthFlows::new(db)),
                [("gmail".to_string(), spec)].into(),
            ))))
        }

        fn temp() -> std::path::PathBuf {
            std::env::temp_dir().join(format!("core-way-in-{}.db", uuid::Uuid::now_v7()))
        }

        fn cleanup(path: &std::path::Path) {
            for suffix in ["", "-wal", "-shm"] {
                let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
            }
        }

        #[tokio::test]
        async fn an_engine_that_hosts_the_flow_mints_a_link() {
            let path = temp();
            let bot = bot(engine(&path));
            let how = bot
                .way_in("t", &payload(person(Visibility::Private)))
                .await
                .expect("a private person gets a link");
            assert!(
                how.starts_with("<https://agent.test/mcp/authorize/"),
                "got {how}"
            );
            cleanup(&path);
        }

        /// The mint applies the same decision the call makes, so whoever
        /// could not read the credential is offered no way to store one.
        #[tokio::test]
        async fn a_requester_the_call_would_refuse_gets_no_link() {
            let path = temp();
            let bot = bot(engine(&path));
            for refused in [Requester::machine(), person(Visibility::Shared)] {
                let how = bot.way_in("t", &payload(refused)).await.unwrap();
                assert_eq!(how, "Run `subs mcp login gmail` to authorize it.");
            }
            cleanup(&path);
        }

        #[tokio::test]
        async fn a_hosted_project_points_at_the_dashboard_and_a_laptop_at_the_command() {
            let dashboard = bot(Arc::new(DashboardConsent(
                "https://app.test/overview".into(),
            )));
            assert_eq!(
                dashboard
                    .way_in("t", &payload(person(Visibility::Private)))
                    .await
                    .unwrap(),
                "<https://app.test/overview|Authorize it in the dashboard>"
            );
            assert_eq!(
                bot(Arc::new(CliConsent))
                    .way_in("t", &payload(person(Visibility::Private)))
                    .await
                    .unwrap(),
                "Run `subs mcp login gmail` to authorize it."
            );
        }

        /// A prompt carrying no facts gets nothing appended. A rejected
        /// static token is one; every other interrupt is another.
        #[tokio::test]
        async fn a_prompt_without_the_facts_is_left_alone() {
            let bot = bot(Arc::new(CliConsent));
            let bare = serde_json::json!({ "message": "*gmail* rejected its token." });
            assert_eq!(bot.way_in("t", &bare).await, None);
        }

        /// Two people on one connection get two links.
        #[tokio::test]
        async fn each_person_gets_their_own_link() {
            let path = temp();
            let bot = bot(engine(&path));

            let first = bot
                .way_in("t", &payload(person(Visibility::Private)))
                .await
                .unwrap();
            let second = bot
                .way_in(
                    "t",
                    &payload(Requester::new(
                        Subject::new(Issuer::slack(), "T1:U2"),
                        Visibility::Private,
                    )),
                )
                .await
                .unwrap();
            assert_ne!(first, second);
            cleanup(&path);
        }
    }

    /// The bot, its workspace, and the Slack it talks to.
    fn bot_for(api_base: String) -> (SlackBot, Arc<Workspace>) {
        let ws = Arc::new(Workspace::new(
            "xoxb-test".into(),
            "t".into(),
            Routing::new().dm(Some("a".into())),
        ));
        let bot = SlackBot::new(Arc::new(OneWorkspace(ws.clone())), api_base, None, None);
        (bot, ws)
    }

    /// The bot with a durable store and a blob store, for the upload paths.
    async fn bot_with_blobs(
        api_base: String,
        dir: &std::path::Path,
    ) -> (SlackBot, Arc<Workspace>, Arc<dyn BlobStore>) {
        let ws = Arc::new(Workspace::new(
            "xoxb-test".into(),
            "t".into(),
            Routing::new().dm(Some("a".into())),
        ));
        let db = crate::providers::sqlite::SqliteDb::open(
            dir.join("test.db").to_str().unwrap(),
            std::time::Duration::from_secs(5),
        )
        .unwrap();
        let store = StreamStore::new(db.clone()).unwrap();
        let blobs: Arc<dyn BlobStore> =
            Arc::new(crate::providers::sqlite::SqliteBlobStore::new(db));
        let bot = SlackBot::new(
            Arc::new(OneWorkspace(ws.clone())),
            api_base,
            Some(store),
            Some(blobs.clone()),
        );
        (bot, ws, blobs)
    }

    #[tokio::test]
    async fn a_stored_image_uploads_once_and_embeds_the_slack_file() {
        let (api_base, calls) = fake_slack().await;
        let dir = tempfile::tempdir().unwrap();
        let (bot, ws, blobs) = bot_with_blobs(api_base, dir.path()).await;
        let r = blobs
            .put(NewBlob {
                tenant_id: "t".into(),
                mime: "image/png".into(),
                name: Some("chart.png".into()),
                bytes: vec![1, 2, 3],
            })
            .await
            .unwrap();

        let block = bot.slack_image_block(&ws, &r.uri()).await.unwrap().unwrap();
        assert_eq!(block["type"], "image");
        assert_eq!(block["slack_file"]["id"], "F123");
        assert_eq!(block["alt_text"], "chart.png");
        assert_eq!(calls.to("upload").len(), 1);
        assert_eq!(calls.to("files.completeUploadExternal").len(), 1);

        // The second embed reuses the recorded upload.
        let again = bot.slack_image_block(&ws, &r.uri()).await.unwrap().unwrap();
        assert_eq!(again["slack_file"]["id"], "F123");
        assert_eq!(calls.to("upload").len(), 1);
    }

    #[tokio::test]
    async fn foreign_and_non_image_refs_do_not_embed() {
        let (api_base, calls) = fake_slack().await;
        let dir = tempfile::tempdir().unwrap();
        let (bot, ws, blobs) = bot_with_blobs(api_base, dir.path()).await;
        let foreign = blobs
            .put(NewBlob {
                tenant_id: "other".into(),
                mime: "image/png".into(),
                name: None,
                bytes: vec![1],
            })
            .await
            .unwrap();
        let pdf = blobs
            .put(NewBlob {
                tenant_id: "t".into(),
                mime: "application/pdf".into(),
                name: None,
                bytes: vec![1],
            })
            .await
            .unwrap();
        assert!(bot
            .slack_image_block(&ws, &foreign.uri())
            .await
            .unwrap()
            .is_none());
        assert!(bot
            .slack_image_block(&ws, &pdf.uri())
            .await
            .unwrap()
            .is_none());
        assert!(calls.to("upload").is_empty());

        // A plain https url embeds without an upload.
        let block = bot
            .slack_image_block(&ws, "https://example.com/a.png")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(block["image_url"], "https://example.com/a.png");
        assert!(calls.to("upload").is_empty());
    }

    #[test]
    fn stripping_images_leaves_a_note_and_only_fires_when_needed() {
        let blocks = vec![
            section_block("answer"),
            serde_json::json!({"type": "image", "slack_file": {"id": "F1"}, "alt_text": "x"}),
        ];
        let stripped = without_images(&blocks).unwrap();
        assert_eq!(stripped.len(), 2);
        assert_eq!(stripped[0]["type"], "section");
        assert_eq!(stripped[1]["type"], "context");
        // Nothing to strip: the error was not the images' fault.
        assert!(without_images(&[section_block("answer")]).is_none());
    }

    fn thread() -> Thread {
        Thread {
            channel: "C1".into(),
            ts: "1.0".into(),
        }
    }

    #[test]
    fn a_refused_message_gets_the_open_prompt_back() {
        let text = SlackBot::still_waiting(&serde_json::json!({
            "message": "*sentry* needs to be authorized again.\n\n<https://app.test/x|Authorize>",
            "metadata": { "options": [{ "label": "Retry", "value": {} }] },
        }))
        .expect("a prompt to repeat");

        assert!(text.starts_with("Still waiting on this:"), "got {text}");
        assert!(text.contains("https://app.test/x"), "got {text}");
    }

    #[test]
    fn an_interrupt_with_nothing_to_say_is_not_repeated() {
        assert!(SlackBot::still_waiting(&serde_json::json!({ "metadata": {} })).is_none());
    }

    #[tokio::test]
    async fn a_failed_turn_posts_its_sentence() {
        let (api_base, calls) = fake_slack().await;
        let (bot, ws) = bot_for(api_base);
        let t = failed_turn(ErrorInfo::new(ErrorCode::BudgetExceeded, "budget spent"));
        bot.complete_turn(&ws, &thread(), SESSION, &t, chrono::Utc::now(), 0)
            .await
            .expect("posts");
        let body = calls
            .to("chat.postMessage")
            .first()
            .cloned()
            .expect("one message posted");
        assert_eq!(body["text"], "Error: budget spent");
        assert_eq!(body["blocks"].as_array().map(Vec::len), Some(1));
        assert_eq!(body["metadata"]["event_payload"]["turn_id"], "turn-1");
    }

    #[tokio::test]
    async fn a_decision_authored_view_is_the_final_message() {
        let (api_base, calls) = fake_slack().await;
        let (bot, ws) = bot_for(api_base);
        bot.streams.set_final(
            key("turn-1"),
            View {
                text: "custom answer".into(),
                blocks: vec![section_block("custom answer")],
            },
        );

        let t = TurnCompleted {
            turn_id: "turn-1".into(),
            data: Value::String("ignored".into()),
            turn_cost: Default::default(),
            turn_token_usage: Default::default(),
            error: None,
        };
        bot.complete_turn(&ws, &thread(), SESSION, &t, chrono::Utc::now(), 0)
            .await
            .expect("posts");

        let body = calls
            .to("chat.postMessage")
            .first()
            .cloned()
            .expect("one message posted");
        assert_eq!(
            body["text"], "custom answer",
            "the worker's view, not the default"
        );
        assert!(
            bot.streams.take_message(&key("turn-1")).is_none(),
            "the view is spent"
        );
    }

    #[tokio::test]
    async fn a_failed_turn_writes_its_error_over_the_view() {
        let (api_base, calls) = fake_slack().await;
        let (bot, ws) = bot_for(api_base);
        bot.streams.set_view(
            key("turn-1"),
            View {
                text: "streamed".into(),
                blocks: vec![section_block("a card")],
            },
        );

        let t = failed_turn(ErrorInfo::new(ErrorCode::BudgetExceeded, "budget spent"));
        bot.complete_turn(&ws, &thread(), SESSION, &t, chrono::Utc::now(), 0)
            .await
            .expect("posts");

        let body = calls
            .to("chat.postMessage")
            .first()
            .cloned()
            .expect("one message posted");
        assert_eq!(body["text"], "Error: budget spent");
        let blocks = body["blocks"].as_array().unwrap();
        assert_eq!(blocks[0]["text"]["text"], "a card", "the work is kept");
        assert_eq!(blocks[1]["text"]["text"], "Error: budget spent");
    }

    /// The stream will not take the answer. The answer goes to the thread —
    /// and the stream above it still has to stop, or it spins over the reply.
    #[tokio::test]
    async fn a_stream_that_refuses_the_answer_is_closed_behind_it() {
        let (api_base, calls) = fake_slack_with(
            serde_json::json!({"ok": false, "error": "invalid_arguments"}),
            serde_json::json!({"ok": true}),
        )
        .await;
        let (bot, ws) = bot_for(api_base);
        bot.streams.insert(stream("turn-1", Some("1.1")));

        let t = failed_turn(ErrorInfo::new(ErrorCode::InvalidResponse, "bad decision"));
        bot.complete_turn(&ws, &thread(), SESSION, &t, chrono::Utc::now(), 0)
            .await
            .expect("the reply lands as a message");

        let posted = calls.to("chat.postMessage");
        assert_eq!(posted.len(), 1, "the answer is not lost");
        assert!(posted[0]["text"]
            .as_str()
            .is_some_and(|t| t.starts_with("Error: bad decision")));
        // Once with the answer, once bare after it was refused.
        assert_eq!(calls.to("chat.stopStream").len(), 2);
    }

    /// The finished message does not stream. A stream that took it would say
    /// the answer twice.
    #[tokio::test]
    async fn the_finished_message_is_not_streamed() {
        let (api_base, calls) = fake_slack().await;
        let (bot, ws) = bot_for(api_base);
        bot.streams.insert(stream("turn-1", None));

        let finished = channels_event(true, "the answer");
        let EventPayload::ChannelsUpdated(c) = finished.payload.clone() else {
            unreachable!()
        };
        bot.apply_channels(&ws, &thread(), &finished, &c)
            .await
            .expect("applies");
        bot.stream_turn(&ws, &key("turn-1")).await;
        assert!(
            calls.to("chat.startStream").is_empty(),
            "no message opens for the answer"
        );
        assert!(calls.to("chat.appendStream").is_empty());

        // Work in progress still streams.
        let working = channels_event(false, "working");
        let EventPayload::ChannelsUpdated(c) = working.payload.clone() else {
            unreachable!()
        };
        bot.apply_channels(&ws, &thread(), &working, &c)
            .await
            .expect("applies");
        bot.stream_turn(&ws, &key("turn-1")).await;
        assert_eq!(calls.to("chat.appendStream").len(), 1);

        // The answer is still there for the reply.
        let view = bot.streams.take_message(&key("turn-1")).expect("a message");
        assert_eq!(view.text, "the answer");
    }

    /// A decision's channels, as the engine records them.
    fn channels_event(finishes_turn: bool, text: &str) -> SessionEvent {
        let view = serde_json::json!({ "text": text, "blocks": [section_block(text)] });
        let meta = crate::session::state::EventMeta {
            status: crate::session::state::SessionStatus::Idle,
            wake_at: None,
            owner: None,
            agent_id: None,
            ancestry: Vec::new(),
            turn_id: Some("turn-1".into()),
            cost: Default::default(),
            sub_agent_cost: Default::default(),
            head_id: None,
            calls: Vec::new(),
            decisions: Vec::new(),
        };
        SessionEvent {
            id: uuid::Uuid::nil(),
            tenant_id: "t".into(),
            session_id: SESSION.into(),
            seq: 1,
            span: crate::span::SpanContext::root(),
            occurred_at: chrono::Utc::now(),
            payload: EventPayload::ChannelsUpdated(crate::session::events::ChannelsUpdated {
                decision_id: "d1".into(),
                finishes_turn,
                channels: [("slack".to_string(), serde_json::json!({ "view": view }))].into(),
            }),
            meta,
            start_time: chrono::Utc::now(),
            end_time: chrono::Utc::now(),
        }
    }

    /// A cancel completes no turn, so nothing else comes along to finish the
    /// message it left open.
    #[tokio::test]
    async fn a_cancelled_session_settles_and_closes_every_message_it_held() {
        let (api_base, calls) = fake_slack().await;
        let (bot, ws) = bot_for(api_base);

        let open = vec![stream("turn-1", Some("1.1")), stream("turn-2", None)];
        bot.close_turns(&ws, SESSION, &thread(), open).await;

        let stopped = calls.to("chat.stopStream");
        assert_eq!(stopped.len(), 1, "only the turn that opened a message");
        assert_eq!(stopped[0]["ts"], "1.1");
        assert_eq!(calls.to("chat.update")[0]["text"], "Cancelled.");
        // And the thread stops claiming it is working.
        let status = calls.to("assistant.threads.setStatus");
        assert_eq!(status.len(), 1);
        assert_eq!(status[0]["status"], "");
    }

    /// The indicator is the thread's, and a queued turn inherits it: only the
    /// last turn to end may say the thread has stopped working.
    #[test]
    fn a_turn_ending_hands_the_indicator_to_the_one_behind_it() {
        let streams = Streams::default();
        streams.insert(stream("turn-1", Some("1.1")));
        streams.insert(stream("turn-2", None));

        assert_eq!(
            streams.live_elsewhere("t", SESSION, Some("turn-1")),
            Some("turn-2".to_string()),
            "the queued turn is still working"
        );
        // The turn that ends is not itself a reason to keep it lit.
        streams.remove(&key("turn-2"));
        assert_eq!(streams.live_elsewhere("t", SESSION, Some("turn-1")), None);
        // Nor is one Slack refused: it will never render again.
        streams.insert(stream("turn-2", None));
        streams.kill(&key("turn-2"));
        assert_eq!(streams.live_elsewhere("t", SESSION, Some("turn-1")), None);
        // Another session's turn is another thread's indicator.
        assert_eq!(streams.live_elsewhere("t", "slack:C9:9.0", None), None);
    }

    /// A turn can start twice, and the second start builds a slot with no
    /// message. The row still names the one the turn opened.
    #[tokio::test]
    async fn a_turn_whose_slot_lost_its_message_takes_it_back() {
        let (api_base, calls) = fake_slack().await;
        let path = std::env::temp_dir().join(format!("core-slack-bot-{}.db", uuid::Uuid::now_v7()));
        let db = crate::providers::sqlite::SqliteDb::open(
            path.to_str().expect("a path"),
            std::time::Duration::from_secs(5),
        )
        .expect("a db");
        let store = StreamStore::new(db).expect("a store");
        let slot = store
            .upsert_turn("t", SESSION, "turn-1", 1, chrono::Utc::now())
            .await
            .expect("a row");
        store
            .set_ts("t", SESSION, "turn-1", "1.1", slot.version)
            .await
            .expect("the message is recorded");

        let ws = Arc::new(Workspace::new(
            "xoxb-test".into(),
            "t".into(),
            Routing::new().dm(Some("a".into())),
        ));
        let bot = SlackBot::new(
            Arc::new(OneWorkspace(ws.clone())),
            api_base,
            Some(store),
            None,
        );
        bot.streams.insert(stream("turn-1", None));
        bot.streams.set_view(
            key("turn-1"),
            View {
                text: "working".into(),
                blocks: vec![section_block("working")],
            },
        );

        bot.stream_turn(&ws, &key("turn-1")).await;

        assert!(
            calls.to("chat.startStream").is_empty(),
            "no second message opens"
        );
        let appended = calls.to("chat.appendStream");
        assert_eq!(appended.len(), 1);
        assert_eq!(appended[0]["ts"], "1.1", "the message the row names");
    }

    /// The second start must not take the first one's place: its slot holds
    /// no message, and the next tick would open one.
    #[test]
    fn a_turn_that_starts_twice_keeps_the_message_it_opened() {
        let streams = Streams::default();
        streams.track_new(stream("turn-1", None));
        streams.commit(&key("turn-1"), "1.1".into(), 1, vec![]);

        streams.track_new(stream("turn-1", None));
        let slot = streams.get(&key("turn-1")).expect("the slot");
        assert_eq!(slot.ts.as_deref(), Some("1.1"));
    }

    /// A stream that refuses a card takes nothing more. It still has to stop,
    /// or Slack streams it until it expires.
    #[tokio::test]
    async fn a_stream_that_refuses_a_card_is_closed() {
        let (api_base, calls) = fake_slack_with(
            serde_json::json!({"ok": true}),
            serde_json::json!({"ok": false, "error": "invalid_blocks"}),
        )
        .await;
        let (bot, ws) = bot_for(api_base);
        bot.streams.insert(stream("turn-1", Some("1.1")));
        bot.streams.set_view(
            key("turn-1"),
            View {
                text: "working".into(),
                blocks: vec![section_block("working")],
            },
        );

        bot.stream_turn(&ws, &key("turn-1")).await;

        assert_eq!(calls.to("chat.appendStream").len(), 1);
        let stopped = calls.to("chat.stopStream");
        assert_eq!(
            stopped.len(),
            1,
            "the message it cannot append to is stopped"
        );
        assert_eq!(stopped[0]["ts"], "1.1");
        assert!(bot.streams.get(&key("turn-1")).expect("the slot").dead);
    }

    /// How a writer learns the turn settled under it.
    #[test]
    fn a_commit_after_the_reply_took_the_slot_says_so() {
        let streams = Streams::default();
        streams.insert(stream("turn-1", None));
        assert!(streams.commit(&key("turn-1"), "1.1".into(), 1, vec![]));

        streams.take(&key("turn-1"));
        assert!(!streams.commit(&key("turn-1"), "1.2".into(), 2, vec![]));
    }

    /// The reply goes out first and the indicator after it, so a card still
    /// in flight cannot light it again.
    #[tokio::test]
    async fn the_indicator_goes_out_after_the_reply_not_before_it() {
        let (api_base, calls) = fake_slack().await;
        let (bot, _ws) = bot_for(api_base);

        let done = EventPayload::TurnCompleted(TurnCompleted {
            turn_id: "turn-1".into(),
            data: Value::Null,
            turn_cost: Default::default(),
            turn_token_usage: Default::default(),
            error: None,
        });
        bot.apply(turn_event(done, None, None))
            .await
            .expect("applies");

        let status = calls.to("assistant.threads.setStatus");
        assert_eq!(status.len(), 1);
        assert_eq!(status[0]["status"], "");
        assert_eq!(
            calls.methods().last().map(String::as_str),
            Some("assistant.threads.setStatus"),
            "the thread's last word is that it stopped working"
        );
    }

    /// A card is a write, and a write clears the indicator. The turn goes on,
    /// so the card lands with the indicator behind it.
    #[tokio::test]
    async fn a_card_lands_with_the_indicator_behind_it() {
        let (api_base, calls) = fake_slack().await;
        let (bot, ws) = bot_for(api_base);
        bot.streams.insert(stream("turn-1", Some("1.1")));
        bot.streams.set_view(
            key("turn-1"),
            View {
                text: "working".into(),
                blocks: vec![section_block("working")],
            },
        );

        bot.stream_turn(&ws, &key("turn-1")).await;

        assert_eq!(calls.to("chat.appendStream").len(), 1);
        let status = calls.to("assistant.threads.setStatus");
        assert_eq!(status.len(), 1, "the card cleared it; the turn lights it");
        assert_eq!(status[0]["status"], render::WORKING_STATUS);
    }

    /// Only the end of a turn puts the indicator out. A prompt's resume, and
    /// the work after it, set it again.
    #[tokio::test]
    async fn a_running_turn_shows_the_indicator_on_every_event() {
        let (api_base, calls) = fake_slack().await;
        let (bot, _ws) = bot_for(api_base);

        // The prompt took the turn's slot with it; the resume speaks for itself.
        let resumed = EventPayload::InterruptResumed(crate::session::events::InterruptResumed {
            interrupt_id: "i1".into(),
            payload: Value::Null,
        });
        bot.apply(turn_event(resumed, Some("turn-1"), None))
            .await
            .expect("applies");

        // The work that follows is the running turn's, and it holds a slot.
        bot.streams.insert(stream("turn-1", Some("1.1")));
        let work = EventPayload::DecisionCompleted(crate::session::events::DecisionCompleted {
            id: "d1".into(),
        });
        bot.apply(turn_event(work, Some("turn-1"), None))
            .await
            .expect("applies");

        let status = calls.to("assistant.threads.setStatus");
        assert_eq!(status.len(), 2, "the resume, and the work after it");
        assert!(status.iter().all(|s| s["status"] == render::WORKING_STATUS));
    }

    /// `session.done` carries the turn id of the turn that just ended, and it
    /// lands after the reply put the indicator out. Nothing follows it.
    #[tokio::test]
    async fn the_events_after_a_turn_do_not_light_it_again() {
        let (api_base, calls) = fake_slack().await;
        let (bot, _ws) = bot_for(api_base);
        bot.streams.insert(stream("turn-1", None));

        let done = EventPayload::TurnCompleted(TurnCompleted {
            turn_id: "turn-1".into(),
            data: Value::Null,
            turn_cost: Default::default(),
            turn_token_usage: Default::default(),
            error: None,
        });
        bot.apply(turn_event(done, None, None))
            .await
            .expect("applies");
        // As the log writes it: the turn id outlives the turn.
        let after = EventPayload::SessionDone(crate::session::events::SessionDone {});
        bot.apply(turn_event(after, Some("turn-1"), None))
            .await
            .expect("applies");

        let status = calls.to("assistant.threads.setStatus");
        assert_eq!(status.len(), 1, "the reply put it out, and it stayed out");
        assert_eq!(status[0]["status"], "");
    }

    /// A parked session waits for a person, and an event outside a turn is
    /// nobody's work.
    #[tokio::test]
    async fn a_parked_or_turnless_event_leaves_the_indicator_out() {
        let (api_base, calls) = fake_slack().await;
        let (bot, _ws) = bot_for(api_base);

        let event = || {
            EventPayload::DecisionCompleted(crate::session::events::DecisionCompleted {
                id: "d1".into(),
            })
        };
        bot.apply(turn_event(event(), Some("turn-1"), Some("i1")))
            .await
            .expect("applies");
        bot.apply(turn_event(event(), None, None))
            .await
            .expect("applies");

        assert!(calls.to("assistant.threads.setStatus").is_empty());
    }

    /// An event of a Slack-owned session: its turn, and the interrupt that
    /// parks it, if one does.
    fn turn_event(
        payload: EventPayload,
        turn_id: Option<&str>,
        parked: Option<&str>,
    ) -> SessionEvent {
        let status = match parked {
            Some(interrupt_id) => crate::session::state::SessionStatus::Interrupted {
                interrupt_id: interrupt_id.into(),
                origin: crate::protocol::InterruptOrigin::Machine,
                reason: "confirmation".into(),
            },
            None => crate::session::state::SessionStatus::Idle,
        };
        let meta = crate::session::state::EventMeta {
            status,
            wake_at: None,
            owner: Some(SessionOwner {
                tenant_id: "t".into(),
                requester: Requester::machine(),
                metadata: [
                    ("slack_channel".to_string(), "C1".to_string()),
                    ("slack_thread_ts".to_string(), "1.0".to_string()),
                ]
                .into(),
            }),
            agent_id: None,
            ancestry: Vec::new(),
            turn_id: turn_id.map(str::to_string),
            cost: Default::default(),
            sub_agent_cost: Default::default(),
            head_id: None,
            calls: Vec::new(),
            decisions: Vec::new(),
        };
        SessionEvent {
            id: uuid::Uuid::nil(),
            tenant_id: "t".into(),
            session_id: SESSION.into(),
            seq: 1,
            span: crate::span::SpanContext::root(),
            occurred_at: chrono::Utc::now(),
            payload,
            meta,
            start_time: chrono::Utc::now(),
            end_time: chrono::Utc::now(),
        }
    }

    /// Every live turn is refreshed, not only the ones with nothing on screen
    /// yet: Slack drops a status after two minutes, and the turns that take
    /// longest are the ones that need it most.
    #[test]
    fn the_status_is_refreshed_for_every_turn_still_running() {
        let streams = Streams::default();
        streams.insert(stream("turn-1", Some("1.1")));
        streams.insert(stream("turn-2", None));
        assert_eq!(streams.live().len(), 2);

        // Nor one Slack refused: it will never render again.
        streams.kill(&key("turn-1"));
        let live = streams.live();
        assert_eq!(live.len(), 1);
        assert_eq!(live[0].key(), key("turn-2"));
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
            thread: thread(),
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

    /// Any session whose owner records a thread is Slack's, whatever its id.
    #[test]
    fn delivery_is_gated_on_the_owners_address_not_the_id() {
        let meta = |metadata: &[(&str, &str)]| crate::session::state::EventMeta {
            status: crate::session::state::SessionStatus::Idle,
            wake_at: None,
            owner: Some(SessionOwner {
                tenant_id: "t".into(),
                requester: Requester::machine(),
                metadata: metadata
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect(),
            }),
            agent_id: None,
            ancestry: Vec::new(),
            turn_id: None,
            cost: Default::default(),
            sub_agent_cost: Default::default(),
            head_id: None,
            calls: Vec::new(),
            decisions: Vec::new(),
        };
        let addressed = meta(&[("slack_channel", "C1"), ("slack_thread_ts", "1.0")]);
        let thread = owner_thread(&addressed).expect("addressed at a thread");
        assert_eq!(thread.channel, "C1");
        assert_eq!(thread.ts, "1.0");
        assert!(owner_thread(&meta(&[("slack_channel", "C1")])).is_none());
        assert!(owner_thread(&meta(&[])).is_none());
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

        // A cancel takes the whole session, and hands back every message it
        // has to close.
        let taken = streams.take_session("t", SESSION);
        assert_eq!(taken.len(), 2);
        assert!(taken.iter().any(|s| s.ts.as_deref() == Some("100.1")));
        assert!(streams.get(&key("turn-1")).is_none());
        assert!(streams.get(&key("turn-2")).is_none());
    }
}
