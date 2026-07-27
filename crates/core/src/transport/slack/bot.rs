use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::Duration;

use serde_json::Value;
use tokio::sync::Notify;

use super::activity::{self, TurnActivity};
use super::state::StreamStore;
use super::{
    app_mention, block_action, build_batch, clip, display_of, dm_message, draft, prompt_blocks,
    resolution_text, section_block, settled_blocks, slack_session, turn_result_text,
    unstamped_ours, with_footer, Click, Inbound, ReplyMeta, MAX_FALLBACK, MAX_MARKDOWN,
    REPLY_EVENT_TYPE,
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
const STATUS: &str = "is thinking…";
/// Slack removes a status after two minutes. Set it again on this cadence.
const STATUS_REFRESH: Duration = Duration::from_secs(90);

/// One Slack install.
pub struct Workspace {
    pub bot_token: String,
    /// Must be unique for each install.
    pub tenant_id: String,
    pub agent_id: String,
    identity: tokio::sync::OnceCell<Identity>,
}

impl Workspace {
    pub fn new(bot_token: String, tenant_id: String, agent_id: String) -> Self {
        Self {
            bot_token,
            tenant_id,
            agent_id,
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
/// `by_tenant` must return the same workspace that `by_team` supplies.
#[async_trait::async_trait]
pub trait WorkspaceResolver: Send + Sync {
    async fn by_team(&self, team_id: Option<&str>) -> Option<Arc<Workspace>>;
    async fn by_tenant(&self, tenant_id: &str) -> Option<Arc<Workspace>>;
}

/// Session ids are unique only in one workspace.
#[derive(Clone, PartialEq, Eq, Hash)]
struct StreamKey {
    tenant_id: String,
    session_id: String,
}

impl StreamKey {
    fn new(tenant_id: &str, session_id: &str) -> Self {
        Self {
            tenant_id: tenant_id.to_string(),
            session_id: session_id.to_string(),
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
        StreamKey::new(&self.tenant_id, &self.session_id)
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

    /// Record what was sent, unless the turn ended during the append.
    fn commit(
        &self,
        key: &StreamKey,
        turn_id: &str,
        ts: String,
        version: u64,
        sent: Vec<(String, Value)>,
    ) {
        let mut open = lock(&self.open);
        let Some(stream) = open.get_mut(key) else {
            return;
        };
        if stream.turn_id != turn_id {
            return;
        }
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
        }
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
        let Some(ws) = self.resolver.by_team(team).await else {
            tracing::warn!(team = %team.unwrap_or(""), "slack: event for unknown workspace");
            return;
        };
        self.submit(&ctx, &ws, inbound).await;
    }

    /// Handle an `interactive` payload (a `block_actions` body).
    pub async fn handle_interaction(&self, payload: &Value) {
        let Some(click) = block_action(payload) else {
            return;
        };
        let Some(ctx) = self.ctx.get().cloned() else {
            tracing::warn!("slack: interaction before start; dropped");
            return;
        };
        let team = payload["team"]["id"]
            .as_str()
            .or_else(|| payload["user"]["team_id"].as_str());
        let Some(ws) = self.resolver.by_team(team).await else {
            tracing::warn!(team = %team.unwrap_or(""), "slack: click for unknown workspace");
            return;
        };
        self.resolve_click(&ctx, &ws, click).await;
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

        // Set the status while the fetch runs.
        let (fetched, _) = tokio::join!(
            self.fetch_thread(ws, &thread, cursor),
            self.set_status(ws, &thread, STATUS),
        );

        // If the fetch fails, append the message alone with a note.
        let input = match fetched {
            Ok(replies) => ClientInput::Append {
                agent_id: ws.agent_id.clone(),
                turn_id: turn_id.clone(),
                messages: build_batch(&path, &replies, &inbound),
                stream: false,
                client: Default::default(),
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
                    agent_id: ws.agent_id.clone(),
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
        // Clear the indicator unless a turn is active.
        if !matches!(
            submitted,
            Ok(_)
                | Err(RuntimeError::Session(
                    SessionError::TurnAlreadyActive { .. }
                ))
        ) {
            self.set_status(ws, &thread, "").await;
        }
        match submitted {
            Ok(_) => {}
            // A redelivery, or a message while a prompt is open. The next
            // turn gets it from the thread delta.
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
            let blocks = settled_blocks(
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
        // Clear the indicator before the final message.
        if matches!(
            &event.payload,
            EventPayload::TurnCompleted(_) | EventPayload::SessionInterrupted(_)
        ) {
            self.set_status(&ws, &thread, "").await;
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
                self.post_interrupt(&ws, &thread, &event.session_id, p, event.seq)
                    .await
            }
            EventPayload::InterruptResumed(p) => self.settle_prompt(&ws, &thread, p).await,
            _ => return Ok(()),
        };
        // The turn is settled (or undeliverable) either way: drop its row.
        if matches!(result, Ok(()) | Err(Error::Terminal(_))) {
            if let (EventPayload::TurnCompleted(t), Some(store)) =
                (&event.payload, self.store.as_deref())
            {
                if let Err(e) = store
                    .clear_turn(&event.tenant_id, &event.session_id, &t.turn_id)
                    .await
                {
                    tracing::warn!(session_id = %event.session_id, error = %e, "slack: stream state not cleared");
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
        let key = StreamKey::new(&ws.tenant_id, session_id);
        let live = match self.streams.take(&key).filter(|s| s.turn_id == t.turn_id) {
            Some(live) => Some(live),
            None => self
                .recover(ws, &key, seq)
                .await
                .filter(|s| s.turn_id == t.turn_id),
        };
        let footer = live.as_ref().map(|s| activity::elapsed(s.started_at, at));
        let Some(ts) = live.as_ref().and_then(|s| s.ts.clone()) else {
            return self
                .post_turn(ws, thread, session_id, t, footer.as_deref())
                .await;
        };
        if self.already_replied(ws, thread, t).await {
            return Ok(());
        }
        let answer = turn_result_text(t);
        let meta = self.reply_meta(ws, session_id, t).await;
        let text = with_footer(&answer, footer.as_deref());
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
                if let Some(blocks) = self
                    .expanded_blocks(ws, live.as_ref(), &answer, footer.as_deref())
                    .await
                {
                    if let Err(e) = self
                        .update(ws, &thread.channel, &ts, &text, blocks, &meta)
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
                let blocks = self
                    .expanded_blocks(ws, live.as_ref(), &answer, footer.as_deref())
                    .await
                    .unwrap_or_else(|| vec![section_block(&text)]);
                match self
                    .update(ws, &thread.channel, &ts, &text, blocks, &meta)
                    .await
                {
                    Ok(()) => Ok(()),
                    Err(e) => {
                        tracing::warn!(error = %e, %session_id, "slack: in-place finalize failed; posting reply");
                        self.post_turn(ws, thread, session_id, t, footer.as_deref())
                            .await
                    }
                }
            }
            // Do not lose the answer to a bad stream.
            Err(e) => {
                tracing::warn!(error = %e, %session_id, "slack: stream finalize failed; posting reply");
                self.post_turn(ws, thread, session_id, t, footer.as_deref())
                    .await
            }
        }
    }

    async fn post_turn(
        &self,
        ws: &Workspace,
        thread: &Thread,
        session_id: &str,
        t: &crate::session::events::TurnCompleted,
        footer: Option<&str>,
    ) -> Result<(), Error> {
        if self.already_replied(ws, thread, t).await {
            return Ok(());
        }
        let meta = self.reply_meta(ws, session_id, t).await;
        let text = with_footer(&turn_result_text(t), footer);
        self.post(ws, thread, &text, &meta).await
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
            Some(display) => (
                display.message.clone(),
                prompt_blocks(&display, &p.interrupt_id),
            ),
            None => {
                let text = format!("Paused: {}", p.reason);
                let block = section_block(&text);
                (text, vec![block])
            }
        };
        // A prompt can wait for a long time; do not hold the stream open.
        let key = StreamKey::new(&ws.tenant_id, session_id);
        let open = match self.streams.take(&key) {
            Some(open) => Some(open),
            None => self.recover(ws, &key, seq).await,
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
        let blocks = settled_blocks(&msg.blocks, &msg.text, &resolution);
        self.update(ws, &thread.channel, &msg.ts, &text, blocks, &meta)
            .await
    }
}

/// Activity: events mark a session dirty; one worker renders the cards.
impl SlackBot {
    /// Track the turn's slot; durable when a store is attached. A store
    /// failure is retryable so the row always lands before the checkpoint.
    async fn track(&self, ws: &Workspace, event: &SessionEvent) -> Result<(), Error> {
        let key = StreamKey::new(&event.tenant_id, &event.session_id);
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
                        if let Some(recovered) = self.recover(ws, &key, event.seq).await {
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
            // A cancelled turn frees its slot.
            EventPayload::SessionCancelled => {
                self.streams.remove(&key);
                if let Some(store) = self.store.as_deref() {
                    if let Err(e) = store.clear(&event.tenant_id, &event.session_id).await {
                        tracing::warn!(session_id = %event.session_id, error = %e, "slack: stream state not cleared");
                    }
                }
                return Ok(());
            }
            _ => return Ok(()),
        }
        self.streams.mark_dirty(key);
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
                self.set_status(&ws, &Thread::new(channel, thread_ts), STATUS)
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
        let Some(turn) = TurnActivity::fold(&events, Some(open.turn_id.clone())) else {
            return;
        };
        // A newer turn owns its own message.
        if turn.turn_id != open.turn_id {
            return;
        }
        let changed: Vec<(String, Value)> = turn
            .chunks()
            .into_iter()
            .filter(|(id, chunk)| open.sent.get(id) != Some(chunk))
            .collect();
        // No changes: only keep the status up.
        if changed.is_empty() {
            if open.ts.is_none() {
                self.set_status(ws, &thread, STATUS).await;
            }
            return;
        }

        let (ts, version) = match open.ts.clone() {
            Some(ts) => (ts, open.version),
            None => {
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
        self.streams
            .commit(key, &open.turn_id, ts, version, changed);
    }

    /// The finished turn rendered as blocks, ready to replace the stream.
    async fn expanded_blocks(
        &self,
        ws: &Workspace,
        stream: Option<&Stream>,
        answer: &str,
        footer: Option<&str>,
    ) -> Option<Vec<Value>> {
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
        Some(turn.blocks(answer, footer))
    }

    /// Abandon a rejected stream. The answer posts as a message.
    fn kill_stream(&self, key: &StreamKey, error: Error) {
        tracing::warn!(error = %error, session_id = %key.session_id, "slack: activity stream failed");
        self.streams.kill(key);
    }

    /// Rebuild a lost slot from its durable row: the log replays the cards,
    /// the row carries the message. `before` bounds the staleness scan; the
    /// caller's own redelivered event sits at it.
    async fn recover(&self, ws: &Workspace, key: &StreamKey, before: u64) -> Option<Stream> {
        let (store, ctx) = (self.store.as_deref()?, self.ctx.get()?);
        let row = store
            .load(&key.tenant_id, &key.session_id)
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
                // The turn is over; the row is leftover.
                EventPayload::TurnStarted(_) | EventPayload::SessionCancelled => {
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
                turn.chunks()
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
    /// last unstamped message after the turn's trigger.
    async fn heal_ts(&self, ws: &Workspace, key: &StreamKey, stream: &mut Stream) {
        let Some((_, trigger)) = slack_session(&stream.turn_id) else {
            return;
        };
        let Some((channel, thread_ts)) = slack_session(&key.session_id) else {
            return;
        };
        let thread = Thread::new(channel, thread_ts);
        let Ok(resp) = self.fetch_replies_raw(ws, &thread, Some(trigger)).await else {
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
                Ok(false) => match store.load(&open.tenant_id, &open.session_id).await {
                    Ok(Some(row)) if row.turn_id == open.turn_id => match row.ts {
                        Some(theirs) if theirs != ts => {
                            return TsPersist::Adopted(theirs, row.version)
                        }
                        Some(_) => return TsPersist::Kept(row.version),
                        None => expected = row.version,
                    },
                    Ok(_) => return TsPersist::TurnOver,
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
    /// The row is gone or re-owned: the turn ended under us.
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
