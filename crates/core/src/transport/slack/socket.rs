use std::sync::Arc;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use tokio_tungstenite::tungstenite::Message as WsMessage;

use super::bot::{SlackBot, Workspace, WorkspaceResolver};
use crate::transport::channel::{Channel, ChannelContext};

const RECONNECT_DELAY: Duration = Duration::from_secs(3);

/// Required environment variables that were not set, and what they are for.
#[derive(Debug)]
pub struct MissingEnv(Vec<(&'static str, &'static str)>);

impl std::fmt::Display for MissingEnv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "missing required environment variable(s):")?;
        for (name, desc) in &self.0 {
            write!(f, "\n  - {name}: {desc}")?;
        }
        write!(f, "\n\nSet them and try again, e.g.:")?;
        for (name, _) in &self.0 {
            write!(f, "\n  export {name}=...")?;
        }
        Ok(())
    }
}

impl std::error::Error for MissingEnv {}

/// Socket Mode transport: one workspace from env, behavior in [`SlackBot`].
#[derive(Clone)]
pub struct SlackChannel {
    agent_id: String,
    app_token: String,
    api_base: String,
    http: reqwest::Client,
    bot: SlackBot,
}

struct StaticResolver(Arc<Workspace>);

#[async_trait::async_trait]
impl WorkspaceResolver for StaticResolver {
    async fn by_install(
        &self,
        _team_id: Option<&str>,
        _app_id: Option<&str>,
        _channel: &str,
    ) -> Option<Arc<Workspace>> {
        Some(self.0.clone())
    }

    async fn by_tenant(&self, tenant_id: &str) -> Option<Arc<Workspace>> {
        (tenant_id == self.0.tenant_id).then(|| self.0.clone())
    }
}

impl SlackChannel {
    /// `store` holds durable stream state so a restart resumes open
    /// streaming messages.
    pub fn new(
        agent_id: String,
        app_token: String,
        bot_token: String,
        tenant_id: String,
        api_base: String,
        store: Option<super::StreamStore>,
    ) -> Self {
        let workspace = Arc::new(Workspace::new(bot_token, tenant_id, agent_id.clone()));
        Self {
            agent_id,
            app_token,
            api_base: api_base.clone(),
            http: reqwest::Client::new(),
            bot: SlackBot::new(Arc::new(StaticResolver(workspace)), api_base, store),
        }
    }

    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// Reads SLACK_APP_TOKEN and SLACK_BOT_TOKEN. SLACK_API_BASE overrides
    /// the API origin (tests).
    pub fn from_env(
        agent_id: String,
        tenant_id: String,
        store: Option<super::StreamStore>,
    ) -> Result<Self, MissingEnv> {
        let mut missing = Vec::new();
        let mut var = |name: &'static str, desc: &'static str| {
            std::env::var(name).unwrap_or_else(|_| {
                missing.push((name, desc));
                String::new()
            })
        };
        let app_token = var(
            "SLACK_APP_TOKEN",
            "App-level token with connections:write, for Socket Mode (xapp-…)",
        );
        let bot_token = var(
            "SLACK_BOT_TOKEN",
            "Bot token with app_mentions:read, chat:write, channels:history, im:history, reactions:write (xoxb-…)",
        );
        if !missing.is_empty() {
            return Err(MissingEnv(missing));
        }
        let api_base =
            std::env::var("SLACK_API_BASE").unwrap_or_else(|_| "https://slack.com/api".to_string());
        Ok(Self::new(
            agent_id, app_token, bot_token, tenant_id, api_base, store,
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

    async fn connect_and_listen(&self) -> anyhow::Result<()> {
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
                        self.bot.handle_event(&envelope["payload"]).await;
                    }
                    if envelope["type"].as_str() == Some("interactive") {
                        self.bot.handle_interaction(&envelope["payload"]).await;
                    }
                    // Ack after the submit; Slack sends an unacked event again.
                    if let Some(id) = envelope["envelope_id"].as_str() {
                        let ack = serde_json::json!({ "envelope_id": id }).to_string();
                        ws.send(WsMessage::text(ack)).await?;
                    }
                    // Slack rotates sockets; connect again with a new url.
                    if envelope["type"].as_str() == Some("disconnect") {
                        return Ok(());
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl Channel for SlackChannel {
    fn kind(&self) -> &'static str {
        "slack"
    }

    async fn run(&self, ctx: ChannelContext) {
        if self.bot.start(&ctx).await.is_err() {
            return;
        }
        loop {
            tokio::select! {
                _ = ctx.shutdown.cancelled() => return,
                r = self.connect_and_listen() => {
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
