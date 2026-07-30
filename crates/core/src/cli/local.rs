use std::collections::BTreeMap;
use std::sync::Arc;

use clap::Args;
use tokio::net::TcpListener;

use crate::cli::auth::AuthWiring;
use crate::cli::cloud::project_config;
use crate::cli::env::{EnvVars, LlmProviderArg, ProviderEnv};
use crate::cli::{register_startup_worker, DEFAULT_TENANT};
use crate::connectors::oauth::StoredCredentials;
use crate::connectors::registry::{ConnectionSpec, Connections, LocalRegistry};
use crate::llm::{LlmProviderTrait, LlmTask};
use crate::providers::anthropic::{AnthropicConfig, AnthropicProvider};
use crate::providers::memory_queue::{ShardedInMemoryQueue, TaskQueue};
use crate::providers::openai::{OpenAiConfig, OpenAiProvider};
use crate::providers::openrouter::{OpenRouterConfig, OpenRouterProvider};
use crate::providers::sqlite::{
    SqliteCheckpointStore, SqliteDb, SqliteEventStore, SqlitePushStore, SqliteSessionIndexStore,
    SqliteTokenStore, SqliteWakeStore, SqliteWorkerQueue,
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
use crate::transport::slack::{SlackChannel, StreamStore};
use crate::transport::worker_http::{self, WorkerHttpState};
use crate::worker::push::{PushRegistry, TransportRegistry};
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
    /// `substructure.toml`). Must declare `target = "local"`.
    #[arg(short = 'c', long)]
    config: Option<std::path::PathBuf>,
    /// Pre-register an HTTP worker at startup.
    #[arg(long)]
    worker_url: Option<String>,
    /// Signing secret for the pre-registered HTTP worker (auto-generated if omitted).
    #[arg(long, requires = "worker_url")]
    signing_secret: Option<String>,
    /// LLM provider. Determines which API key env var is required
    /// (e.g. `openrouter` needs OPENROUTER_API_KEY). Optional: omit it to
    /// run a server that only handles worker-side LLM calls.
    #[arg(long, value_enum)]
    llm_provider: Option<LlmProviderArg>,
    /// Disable client and worker authentication. For local development only.
    #[arg(long)]
    dev: bool,
    /// Serve a Slack Socket Mode bot driving this agent. Requires
    /// SLACK_APP_TOKEN and SLACK_BOT_TOKEN.
    #[arg(long, value_name = "AGENT_ID")]
    slack_agent: Option<String>,
}

impl ServeArgs {
    pub fn config_path(&self) -> Option<&std::path::Path> {
        self.config.as_deref()
    }
}

pub async fn serve(args: ServeArgs) -> anyhow::Result<()> {
    start_server(args).await
}

async fn start_server(args: ServeArgs) -> anyhow::Result<()> {
    // Anything argv omits can be pinned in the project file. Precedence is
    // flag > environment > file > default, applied one field at a time.
    let cfg = project_config::local(args.config.as_deref(), "`subs serve`")?;

    let connections = cfg.connections();
    let slack_agent = args.slack_agent.or_else(|| cfg.slack_agent());
    let server = cfg.server.clone().unwrap_or_default();

    let host = args
        .host
        .or(server.host)
        .unwrap_or_else(|| "127.0.0.1".into());
    let port = args.port.or(server.port).unwrap_or(8080);
    let db_path = args.db.unwrap_or_else(|| cfg.db_path());
    let worker_url = args.worker_url.or_else(|| cfg.worker_url());
    let signing_secret = args.signing_secret.or_else(|| cfg.signing_secret());
    let dev = args.dev || server.dev.unwrap_or(false);

    let env = match EnvVars::load(args.llm_provider.or_else(|| cfg.llm_provider()), dev) {
        Some(e) => e,
        None => std::process::exit(2),
    };
    let db = SqliteDb::open(&db_path, std::time::Duration::from_secs(5))?;
    let slack = match slack_agent {
        Some(agent_id) => {
            let store = StreamStore::new(db.clone())?;
            match SlackChannel::from_env(agent_id, DEFAULT_TENANT.to_string(), Some(store)) {
                Ok(s) => Some(s),
                Err(e) => {
                    eprintln!("error: {e}");
                    std::process::exit(2)
                }
            }
        }
        None => None,
    };

    let (rt, adapter) = start_engine(db, env.provider, connections).await?;

    let auth = match env.auth {
        Some(a) => AuthWiring::from_env(a)?,
        None => AuthWiring::dev(),
    };

    if let Some(ref url) = worker_url {
        register_startup_worker(&adapter, url, signing_secret).await?;
        tracing::info!(url, "startup worker registered (signing enabled)");
    }

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
    let v1_routes = admin_http::v1_router(
        AdminHttpState {
            runtime: rt.clone(),
            auth: auth.admin,
            shutdown: shutdown.clone(),
        },
        adapter.clone(),
    );
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
        tracing::info!(agent_id = %slack.agent_id(), "slack channel enabled");
        channels.push(Arc::new(slack));
    }
    let channel_ctx = ChannelContext::new(rt.clone(), shutdown.clone());

    let mut routers = vec![admin_routes, client_routes, worker_routes, v1_routes];
    routers.extend(start_channels(channels, channel_ctx));
    let server = SubstructureServer::new(routers);

    let addr = format!("{host}:{port}");
    if dev {
        eprintln!();
        eprintln!("  DEV MODE - authentication is disabled.");
        eprintln!("  Do not use in production or expose to untrusted networks.");
        eprintln!();
    }

    tracing::info!(%addr, "listening");
    let listener = TcpListener::bind(&addr).await?;
    server.serve(listener, shutdown).await
}

/// Open the SQLite-backed stores, start the in-process engine, and build a
/// started push adapter. Shared by `serve` (which then mounts HTTP routers) and
/// `run` (which drives a single turn without any HTTP server).
pub(crate) async fn start_engine(
    db: SqliteDb,
    provider_env: Option<ProviderEnv>,
    connectors: BTreeMap<String, ConnectionSpec>,
) -> anyhow::Result<(Arc<Runtime>, Arc<PushAdapter>)> {
    let event_store = Arc::new(SqliteEventStore::new(db.clone())?);
    let worker_queue = Arc::new(SqliteWorkerQueue::new(db.clone())?);
    let checkpoint_store = Arc::new(SqliteCheckpointStore::new(db.clone())?);
    let wake_store = Arc::new(SqliteWakeStore::new(db.clone())?);
    let session_index_store = Arc::new(SqliteSessionIndexStore::new(db.clone())?);
    let token_store = Arc::new(SqliteTokenStore::new(db.clone())?);
    let push_store = Arc::new(SqlitePushStore::new(db)?);

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

    let llm_provider: Option<Arc<dyn LlmProviderTrait>> = match provider_env {
        Some(ProviderEnv::Openrouter { api_key }) => {
            Some(Arc::new(OpenRouterProvider::new(OpenRouterConfig {
                base_url: std::env::var("OPENROUTER_BASE_URL")
                    .unwrap_or_else(|_| "https://openrouter.ai/api".to_string()),
                api_key,
            })))
        }
        Some(ProviderEnv::Anthropic { api_key }) => {
            let mut config = AnthropicConfig::new(api_key);
            if let Ok(base_url) = std::env::var("ANTHROPIC_BASE_URL") {
                config.base_url = base_url;
            }
            Some(Arc::new(AnthropicProvider::new(config)))
        }
        Some(ProviderEnv::Openai { api_key }) => {
            let mut config = OpenAiConfig::new(api_key);
            if let Ok(base_url) = std::env::var("OPENAI_BASE_URL") {
                config.base_url = base_url;
            }
            config.organization = std::env::var("OPENAI_ORG_ID").ok();
            config.project = std::env::var("OPENAI_PROJECT_ID").ok();
            Some(Arc::new(OpenAiProvider::new(config)))
        }
        None => {
            tracing::info!("no LLM provider configured; server-side LLM execution is disabled (worker-handled calls only)");
            None
        }
    };

    let token_delta_transport = Arc::new(crate::llm::InMemoryTokenDeltaTransport::new());
    let rt = start(
        RuntimeDeps {
            store: event_store,
            llm_provider,
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

    let transports = TransportRegistry::new(vec![http_transport()]);
    let registry = PushRegistry::new(push_store, transports);
    let adapter = Arc::new(PushAdapter::new(rt.clone(), registry, 16));
    adapter.start().await;

    Ok((rt, adapter))
}
