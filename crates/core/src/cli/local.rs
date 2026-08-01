use std::sync::Arc;

use clap::Args;
use tokio::net::TcpListener;

use crate::cli::auth::AuthWiring;
use crate::cli::cloud::project_config::{self, ProjectConfig};
use crate::cli::env::{EnvVars, ProviderEnv, ProviderKind};
use crate::cli::DEFAULT_TENANT;
use crate::connectors::oauth::StoredCredentials;
use crate::connectors::registry::{Connections, LocalRegistry};
use crate::llm::{LlmProviderRegistry, LlmProviderTrait, LlmTask};
use crate::providers::anthropic::{AnthropicConfig, AnthropicProvider};
use crate::providers::memory_queue::{ShardedInMemoryQueue, TaskQueue};
use crate::providers::openai::{OpenAiConfig, OpenAiProvider};
use crate::providers::openrouter::{OpenRouterConfig, OpenRouterProvider};
use crate::providers::sqlite::{
    SqliteCheckpointStore, SqliteDb, SqliteEventStore, SqliteSessionIndexStore, SqliteTokenStore,
    SqliteWakeStore, SqliteWorkerQueue,
};
use crate::runtime::connector::ConnectorTask;
use crate::sub_agent::SubAgentTask;
use crate::transport::admin_http::{self, AdminHttpState};
use crate::transport::ag_ui::channel::AgUiChannel;
use crate::transport::channel::{start_channels, Channel, ChannelContext};
use crate::transport::client_http::{self, ClientHttpState};
use crate::transport::http_push::http_transport;
use crate::transport::push::PushAdapter;
use crate::transport::server::SubstructureServer;
use crate::transport::slack::{Routing, SlackChannel, StreamStore};
use crate::transport::worker_http::{self, WorkerHttpState};
use crate::worker::push::TransportRegistry;
use crate::worker::StaticAgentDirectory;
use crate::{start, Runtime, RuntimeConfig, RuntimeDeps};

#[derive(Args)]
pub struct ServeArgs {
    /// [default: 127.0.0.1]
    #[arg(long)]
    host: Option<String>,
    /// [default: 8080]
    #[arg(long)]
    port: Option<u16>,
    /// [default: substructure.db]
    #[arg(long)]
    db: Option<String>,
    /// Environment file (default: walks up from cwd looking for
    /// `substructure.toml`).
    #[arg(short = 'c', long)]
    config: Option<std::path::PathBuf>,
    /// Serve without client or worker authentication. For a server nothing off
    /// this machine can reach.
    #[arg(long = "no-auth", alias = "dev")]
    no_auth: bool,
    /// Serve a Slack Socket Mode bot driving this agent for DMs and for any
    /// channel `[slack.channel.<id>]` does not name. Requires SLACK_APP_TOKEN
    /// and SLACK_BOT_TOKEN.
    #[arg(long, value_name = "AGENT_ID")]
    slack_agent: Option<String>,
}

impl ServeArgs {
    pub fn config_path(&self) -> Option<&std::path::Path> {
        self.config.as_deref()
    }
}

/// Who the bot answers as, where. `None` leaves it off.
///
/// `--slack-agent` is the one-flag way to serve a whole workspace, so it sets
/// both DMs and unnamed channels; the file's channel table still stands, and
/// `off` on a channel still wins.
fn slack_routing(cfg: &ProjectConfig, flag: Option<String>) -> Option<Routing> {
    let slack = cfg.slack.clone().unwrap_or_default();
    let mut routing = Routing::new()
        .dm(flag.clone().or(slack.dm))
        .any_channel(flag.or(slack.any_channel));
    for (id, channel) in &slack.channel {
        routing = routing.channel(id, channel.agent().map(str::to_string));
    }
    (!routing.is_empty()).then_some(routing)
}

pub async fn serve(args: ServeArgs) -> anyhow::Result<()> {
    start_server(args).await
}

async fn start_server(args: ServeArgs) -> anyhow::Result<()> {
    // Anything argv omits can be pinned in the project file. Precedence is
    // flag > environment > file > default, applied one field at a time.
    let cfg = project_config::load(args.config.as_deref())?;

    let slack_routing = slack_routing(&cfg, args.slack_agent);
    let server = cfg.server.clone().unwrap_or_default();

    let host = args
        .host
        .or(server.host)
        .unwrap_or_else(|| "127.0.0.1".into());
    let port = args.port.or(server.port).unwrap_or(8080);
    let db_path = args.db.unwrap_or_else(|| cfg.db_path());
    let authenticate = !args.no_auth && cfg.server_auth();

    let env = match EnvVars::load(cfg.provider_bindings(), authenticate) {
        Some(e) => e,
        None => std::process::exit(2),
    };
    let db = SqliteDb::open(&db_path, std::time::Duration::from_secs(5))?;
    let slack = match slack_routing {
        Some(routing) => {
            let store = StreamStore::new(db.clone())?;
            match SlackChannel::from_env(routing, DEFAULT_TENANT.to_string(), Some(store)) {
                Ok(s) => Some(s),
                Err(e) => {
                    eprintln!("error: {e}");
                    std::process::exit(2)
                }
            }
        }
        None => None,
    };

    // Held for the process's life: dropping it aborts the decision loops.
    let (rt, _adapter) = start_engine(db, env.providers, &cfg).await?;
    announce_agents(&cfg);

    let auth = match env.auth {
        Some(a) => AuthWiring::from_env(a)?,
        None => AuthWiring::unauthenticated(),
    };

    let shutdown = tokio_util::sync::CancellationToken::new();
    let signal_token = shutdown.clone();

    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("shutdown signal received");
        signal_token.cancel();
    });

    let admin_routes = admin_http::router(AdminHttpState {
        runtime: rt.clone(),
        auth: auth.admin.clone(),
        shutdown: shutdown.clone(),
    });
    let v1_routes = admin_http::v1_router(AdminHttpState {
        runtime: rt.clone(),
        auth: auth.admin,
        shutdown: shutdown.clone(),
    });
    let client_routes = client_http::router(ClientHttpState {
        runtime: rt.clone(),
        auth: auth.client.clone(),
        shutdown: shutdown.clone(),
    });
    let worker_routes = worker_http::router(WorkerHttpState {
        runtime: rt.clone(),
        auth: auth.worker,
        client_token_issuer: auth.issuer,
        shutdown: shutdown.clone(),
    });

    let mut channels: Vec<Arc<dyn Channel>> = vec![Arc::new(AgUiChannel::new(auth.client))];
    if let Some(slack) = slack {
        tracing::info!(routing = %slack.routing(), "slack channel enabled");
        channels.push(Arc::new(slack));
    }
    let channel_ctx = ChannelContext::new(rt.clone(), shutdown.clone());

    let mut routers = vec![admin_routes, client_routes, worker_routes, v1_routes];
    routers.extend(start_channels(channels, channel_ctx));
    let server = SubstructureServer::new(routers);

    let addr = format!("{host}:{port}");
    if !authenticate {
        eprintln!();
        eprintln!("  AUTHENTICATION IS DISABLED.");
        eprintln!("  Do not use in production or expose to untrusted networks.");
        eprintln!();
    }

    tracing::info!(%addr, "listening");
    let listener = TcpListener::bind(&addr).await?;
    server.serve(listener, shutdown).await
}

/// What this engine serves, once, at startup: an agent's hosting is a property
/// of the file, so it should be readable without reading the file.
fn announce_agents(cfg: &ProjectConfig) {
    if cfg.agent.is_empty() {
        tracing::warn!(
            "substructure.toml declares no [agent.*]; every decision will fail until one does"
        );
        return;
    }
    for (id, section) in &cfg.agent {
        match &section.worker {
            Some(url) => tracing::info!(agent = %id, worker = %url, "agent hosted by a worker"),
            None => tracing::info!(agent = %id, "agent hosted by the engine"),
        }
    }
}

/// Open the SQLite-backed stores, start the in-process engine, and build a
/// started push adapter. Shared by `serve` (which then mounts HTTP routers) and
/// `run` (which drives a single turn without any HTTP server).
pub(crate) async fn start_engine(
    db: SqliteDb,
    providers: Vec<ProviderEnv>,
    cfg: &ProjectConfig,
) -> anyhow::Result<(Arc<Runtime>, Arc<PushAdapter>)> {
    let connectors = cfg.connections();
    let event_store = Arc::new(SqliteEventStore::new(db.clone())?);
    let worker_queue = Arc::new(SqliteWorkerQueue::new(db.clone())?);
    let checkpoint_store = Arc::new(SqliteCheckpointStore::new(db.clone())?);
    let wake_store = Arc::new(SqliteWakeStore::new(db.clone())?);
    let session_index_store = Arc::new(SqliteSessionIndexStore::new(db.clone())?);
    let token_store = Arc::new(SqliteTokenStore::new(db)?);

    let config = RuntimeConfig::default();
    let llm_task_queue: Arc<dyn TaskQueue<LlmTask>> = Arc::new(ShardedInMemoryQueue::new(
        config.llm_executor_workers as u32,
    ));
    let sub_agent_task_queue: Arc<dyn TaskQueue<SubAgentTask>> = Arc::new(
        ShardedInMemoryQueue::new(config.sub_agent_executor_workers as u32),
    );
    let connector_task_queue: Arc<dyn TaskQueue<ConnectorTask>> = Arc::new(
        ShardedInMemoryQueue::new(config.connector_executor_workers as u32),
    );
    // Connections come from `substructure.toml`; the file holds only names and
    // env-var references, never a token. What `subs mcp login` authorized is in
    // this same database, so a login and the engine that uses it cannot drift
    // apart.
    let connections = Some(connectors)
        .filter(|c| !c.is_empty())
        .map(|connectors| {
            tracing::info!(
                connections = connectors.len(),
                "loaded connections from substructure.toml"
            );
            Arc::new(Connections::new(
                Arc::new(LocalRegistry::new(connectors)),
                Arc::new(StoredCredentials::new(token_store)),
            ))
        });

    if providers.is_empty() {
        tracing::info!(
            "no engine-run [llm.*] block declared; every model call belongs to a worker"
        );
    }
    let llm_providers = Arc::new(LlmProviderRegistry::new(
        providers
            .into_iter()
            .map(|p| (p.name.clone(), client(p)))
            .collect(),
    ));

    let agents = Arc::new(StaticAgentDirectory::new(
        DEFAULT_TENANT.to_string(),
        cfg.agents(),
        cfg.llm_blocks(),
    ));

    let token_delta_transport = Arc::new(crate::llm::InMemoryTokenDeltaTransport::new());
    let rt = start(
        RuntimeDeps {
            store: event_store,
            agents: agents.clone(),
            llm: llm_providers,
            llm_task_queue,
            sub_agent_task_queue,
            connections,
            connector_task_queue,
            worker_queue,
            session_index_store,
            checkpoint_store,
            wake_store,
            token_delta_transport,
        },
        config,
    );

    let adapter = Arc::new(PushAdapter::new(
        rt.clone(),
        agents,
        TransportRegistry::new(vec![http_transport()]),
        16,
    ));
    adapter.start();

    Ok((rt, adapter))
}

/// The client one engine-run block calls with. `base_url` is the file's when it
/// sets one, then the vendor's own variable, then the vendor default.
fn client(p: ProviderEnv) -> Arc<dyn LlmProviderTrait> {
    let base_url = p
        .base_url
        .clone()
        .or_else(|| p.kind.base_url_env().and_then(|v| std::env::var(v).ok()));
    match p.kind {
        ProviderKind::Openrouter => Arc::new(OpenRouterProvider::new(OpenRouterConfig {
            base_url: base_url.unwrap_or_else(|| "https://openrouter.ai/api".to_string()),
            api_key: p.api_key,
        })),
        ProviderKind::Anthropic => {
            let mut config = AnthropicConfig::new(p.api_key);
            if let Some(base_url) = base_url {
                config.base_url = base_url;
            }
            Arc::new(AnthropicProvider::new(config))
        }
        ProviderKind::Openai => {
            let mut config = OpenAiConfig::new(p.api_key);
            if let Some(base_url) = base_url {
                config.base_url = base_url;
            }
            config.organization = std::env::var("OPENAI_ORG_ID").ok();
            config.project = std::env::var("OPENAI_PROJECT_ID").ok();
            Arc::new(OpenAiProvider::new(config))
        }
        // Never reached: `provider_bindings` keeps worker blocks out of the
        // registry, since the engine never calls one.
        ProviderKind::Worker => unreachable!("a worker block has no engine-side client"),
    }
}
