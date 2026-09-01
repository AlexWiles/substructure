use std::sync::Arc;

use clap::Args;
use tokio::net::TcpListener;

use crate::cli::auth::AuthWiring;
use crate::cli::cloud::project_config::{self, ProjectConfig};
use crate::cli::cloud::{credentials, print};
use crate::cli::env::{EnvVars, ProviderEnv, ProviderKind};
use crate::cli::DEFAULT_TENANT;
use crate::connectors::credential::StoredCredentials;
use crate::connectors::registry::{ConnectionPath, Connections, CredentialScope, LocalRegistry};
use crate::llm::{LlmProviderRegistry, LlmProviderTrait, LlmTask};

use crate::providers::anthropic::{AnthropicConfig, AnthropicProvider};
use crate::providers::memory_queue::{ShardedInMemoryQueue, TaskQueue};
use crate::providers::openai::{OpenAiConfig, OpenAiProvider};
use crate::providers::openrouter::{OpenRouterConfig, OpenRouterProvider};
use crate::providers::sqlite::{
    SqliteAuthFlows, SqliteBlobStore, SqliteCredentialStore, SqliteCursorStore, SqliteDb,
    SqliteEventStore, SqliteSecretStore, SqliteSessionIndexStore, SqliteWakeStore,
    SqliteWorkerQueue,
};
use crate::runtime::blob::{BlobResolvingLlm, BlobStore};
use crate::runtime::connector::ConnectorTask;
use crate::runtime::secret::SecretCipher;
use crate::subagent::SubagentTask;
use crate::transport::admin_http::{self, AdminHttpState};
use crate::transport::ag_ui::channel::AgUiChannel;
use crate::transport::channel::{start_channels, Channel, ChannelContext};
use crate::transport::client_http::{self, ClientHttpState};
use crate::transport::mcp_auth::{self, AuthorizeLinks, McpAuthDeps};
use crate::transport::push::PushAdapter;
use crate::transport::server::SubstructureServer;
use crate::transport::slack::{SlackChannel, StreamStore};
use crate::transport::worker_http::{self, WorkerHttpState};
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
    /// [default: `db` in `subs.toml`, else
    /// `~/.config/subs/subs.db`]
    #[arg(long)]
    db: Option<String>,
    /// Environment file (default: walks up from cwd looking for
    /// `subs.toml`).
    #[arg(short = 'c', long)]
    config: Option<std::path::PathBuf>,
    /// Serve without client or worker authentication. For a server nothing off
    /// this machine can reach.
    #[arg(long = "no-auth", alias = "dev")]
    no_auth: bool,
}

impl ServeArgs {
    pub fn config_path(&self) -> Option<&std::path::Path> {
        self.config.as_deref()
    }
}

fn authorize_page(cfg: &ProjectConfig) -> Option<String> {
    let project = cfg.project()?;
    let api_url = credentials::resolve_api_url(cfg.remote_url());
    let hosted = api_url.trim_end_matches('/') == credentials::DEFAULT_API_URL;
    hosted.then(|| format!("{}/overview", print::admin_url(&api_url, project)))
}

fn consent(
    mcp_auth: &Option<Arc<McpAuthDeps>>,
    cfg: &ProjectConfig,
) -> Arc<dyn crate::transport::consent::Consent> {
    use crate::transport::consent::{CliConsent, DashboardConsent, EngineConsent};
    match mcp_auth {
        Some(deps) => Arc::new(EngineConsent(deps.links.clone())),
        None => match authorize_page(cfg) {
            Some(page) => Arc::new(DashboardConsent(page)),
            None => Arc::new(CliConsent),
        },
    }
}

pub async fn serve(args: ServeArgs) -> anyhow::Result<()> {
    start_server(args).await
}

async fn start_server(args: ServeArgs) -> anyhow::Result<()> {
    let cfg = project_config::load(args.config.as_deref())?;

    let server = cfg.serve.clone().unwrap_or_default();

    let host = args
        .host
        .or(server.host)
        .unwrap_or_else(|| "127.0.0.1".into());
    let port = args.port.or(server.port).unwrap_or(8080);
    let db_path = args.db.unwrap_or_else(|| cfg.db_path());
    let authenticate = !args.no_auth && cfg.serve_auth();

    let env = match EnvVars::load(cfg.provider_bindings(), authenticate) {
        Some(e) => e,
        None => std::process::exit(2),
    };
    project_config::ensure_parent(&db_path)?;
    let db = SqliteDb::open(&db_path, std::time::Duration::from_secs(5))?;
    let blobs: Arc<dyn BlobStore> = Arc::new(SqliteBlobStore::new(db.clone()));
    let mut slack = Vec::new();
    for (agent_id, app) in cfg.manifest().slack_apps() {
        let store = StreamStore::new(db.clone())?;
        match SlackChannel::from_env(
            agent_id.to_string(),
            app.answers,
            DEFAULT_TENANT.to_string(),
            Some(store),
            Some(blobs.clone()),
        ) {
            Ok(channel) => slack.push(channel),
            Err(e) => {
                eprintln!("error: [agent.{agent_id}.slack]: {e}");
                std::process::exit(2)
            }
        }
    }

    let (rt, _adapter, mcp_auth) = start_engine(db, blobs.clone(), env.providers, &cfg).await?;

    let slack: Vec<SlackChannel> = slack
        .into_iter()
        .map(|s| s.with_consent(consent(&mcp_auth, &cfg)))
        .collect();
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
        blobs: Some(blobs.clone()),
    });
    let worker_routes = worker_http::router(WorkerHttpState {
        runtime: rt.clone(),
        auth: auth.worker,
        client_token_issuer: auth.issuer,
        shutdown: shutdown.clone(),
    });

    let mut channels: Vec<Arc<dyn Channel>> = vec![Arc::new(AgUiChannel::new(auth.client))];
    for channel in slack {
        tracing::info!(agent = channel.agent_id(), "slack app connected");
        channels.push(Arc::new(channel));
    }
    let channel_ctx = ChannelContext::new(rt.clone(), shutdown.clone());

    let mut routers = vec![admin_routes, client_routes, worker_routes, v1_routes];
    if let Some(deps) = mcp_auth {
        routers.push(mcp_auth::router(deps.clone()));
        mcp_auth::spawn_sweeper(deps, shutdown.clone());
    }
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

fn announce_agents(cfg: &ProjectConfig) {
    if cfg.agent.is_empty() {
        tracing::warn!("subs.toml declares no [agent.*]; every decision will fail until one does");
        return;
    }
    for (id, section) in &cfg.agent {
        match &section.worker {
            Some(url) => tracing::info!(agent = %id, worker = %url, "agent hosted by a worker"),
            None => tracing::info!(agent = %id, "agent hosted by the engine"),
        }
    }
}

pub(crate) async fn start_engine(
    db: SqliteDb,
    blobs: Arc<dyn BlobStore>,
    providers: Vec<ProviderEnv>,
    cfg: &ProjectConfig,
) -> anyhow::Result<(Arc<Runtime>, Arc<PushAdapter>, Option<Arc<McpAuthDeps>>)> {
    let (manifest, mut resolved) = cfg.resolved_manifest()?;
    for notice in &resolved.notices {
        tracing::warn!("{notice}");
    }
    for notice in manifest.scope_notices() {
        eprintln!("{notice}");
    }
    let mut bundles: crate::plugins::PluginSet = manifest
        .plugin
        .iter()
        .filter_map(|(id, spec)| Some((id.clone(), spec.bundle.clone()?)))
        .collect();

    for (id, bundle) in &mut bundles {
        let Some(pending) = resolved.pending.remove(id) else {
            continue;
        };
        let notices =
            crate::plugins::store_binaries(bundle, pending, DEFAULT_TENANT, blobs.as_ref()).await;
        for notice in notices {
            tracing::warn!("[plugin.{id}]: {notice}");
        }
    }
    let plugins: Arc<dyn crate::plugins::PluginResolver> =
        Arc::new(crate::plugins::StaticPlugins::new(bundles));
    let connectors = manifest.connections();
    let cipher = SecretCipher::from_env()
        .map_err(|e| anyhow::anyhow!(e))?
        .map(Arc::new);

    if cipher.is_none() {
        if let Some((id, _)) = connectors
            .iter()
            .find(|(_, spec)| spec.effective_scope() == CredentialScope::User)
        {
            tracing::warn!(
                connection = %id,
                "a `credential = \"user\"` connection stores personal credentials in plain \
                 text; set ${} to seal them",
                crate::runtime::secret::KEYS_ENV
            );
        }
    }
    let event_store = Arc::new(SqliteEventStore::new(db.clone())?);
    let worker_queue = Arc::new(SqliteWorkerQueue::new(db.clone())?);
    let cursor_store = Arc::new(SqliteCursorStore::new(db.clone())?);
    let wake_store = Arc::new(SqliteWakeStore::new(db.clone())?);
    let session_index_store = Arc::new(SqliteSessionIndexStore::new(db.clone())?);
    let secret_store = Arc::new(SqliteSecretStore::new(db.clone(), cipher));
    let token_store = Arc::new(SqliteCredentialStore::new(
        db.clone(),
        secret_store.clone(),
    )?);

    let declared: Vec<String> = connectors.keys().map(ConnectionPath::to_string).collect();
    for forgotten in token_store.retain(DEFAULT_TENANT, &declared).await? {
        tracing::info!(
            connection = %forgotten,
            "forgot the credential: the file no longer declares this connection"
        );
    }

    for (id, spec) in &connectors {
        let holders = token_store.holders(DEFAULT_TENANT, &id.to_string()).await?;
        let scope = spec.effective_scope();
        let stranded = match scope {
            CredentialScope::User => holders.contains(&crate::connectors::Slot::Shared),
            CredentialScope::Shared => holders
                .iter()
                .any(|h| matches!(h, crate::connectors::Slot::Of(_))),
        };
        if stranded {
            tracing::warn!(
                connection = %id,
                scope = scope.as_str(),
                "the connection changed scope; credentials from the old scope are never \
                 adopted, so each holder under the new scope must connect again"
            );
        }
    }

    let config = RuntimeConfig::default();
    let llm_task_queue: Arc<dyn TaskQueue<LlmTask>> = Arc::new(ShardedInMemoryQueue::new(
        config.llm_executor_workers as u32,
    ));
    let subagent_task_queue: Arc<dyn TaskQueue<SubagentTask>> = Arc::new(
        ShardedInMemoryQueue::new(config.subagent_executor_workers as u32),
    );
    let connector_task_queue: Arc<dyn TaskQueue<ConnectorTask>> = Arc::new(
        ShardedInMemoryQueue::new(config.connector_executor_workers as u32),
    );

    let connections = Some(connectors.clone())
        .filter(|c| !c.is_empty())
        .map(|connectors| {
            tracing::info!(
                connections = connectors.len(),
                "loaded connections from subs.toml"
            );
            Arc::new(Connections::new(
                Arc::new(LocalRegistry::new(connectors)),
                Arc::new(StoredCredentials::new(token_store.clone())),
                blobs.clone(),
            ))
        });

    let mcp_auth = match cfg.serve.as_ref().and_then(|s| s.public_url.clone()) {
        Some(public_url) if !connectors.is_empty() => {
            let flows = Arc::new(SqliteAuthFlows::new(db));
            Some(Arc::new(McpAuthDeps {
                links: Arc::new(AuthorizeLinks::new(&public_url, flows.clone(), connectors)),
                flows,
                secrets: secret_store.clone(),
                credentials: token_store,
                http: reqwest::Client::new(),
            }))
        }
        _ => None,
    };

    if providers.is_empty() {
        tracing::info!(
            "no engine-run [llm.*] block declared; every model call belongs to a worker"
        );
    }

    let llm_providers = Arc::new(BlobResolvingLlm::new(
        Arc::new(LlmProviderRegistry::new(
            providers
                .into_iter()
                .map(|p| (p.name.clone(), client(p)))
                .collect(),
        )),
        blobs.clone(),
    ));

    let agents = Arc::new(
        StaticAgentDirectory::new(
            DEFAULT_TENANT.to_string(),
            manifest.agents(),
            cfg.llm_blocks(),
        )
        .with_workers(manifest.worker.clone(), manifest.default_worker.clone()),
    );

    let token_delta_transport = Arc::new(crate::llm::InMemoryTokenDeltaTransport::new());
    let rt = start(
        RuntimeDeps {
            store: event_store,
            agents: agents.clone(),
            llm: llm_providers,
            llm_task_queue,
            subagent_task_queue,
            connections,
            plugins,
            connector_task_queue,
            worker_queue,
            channel_proposers: vec![Arc::new(crate::transport::slack::SlackProposer::new())],
            session_index_store,
            cursor_store,
            wake_store,
            blobs,
            token_delta_transport,
        },
        config,
    );

    let adapter = Arc::new(PushAdapter::new(rt.clone(), agents, secret_store, 16));
    adapter.start();

    Ok((rt, adapter, mcp_auth))
}

fn client(p: ProviderEnv) -> Arc<dyn LlmProviderTrait> {
    let base_url = p
        .base_url
        .clone()
        .or_else(|| p.kind.base_url_env().and_then(|v| std::env::var(v).ok()));
    match p.kind {
        ProviderKind::Openrouter => Arc::new(OpenRouterProvider::new(OpenRouterConfig {
            base_url: base_url.unwrap_or_else(|| "https://openrouter.ai/api".to_string()),
            api_key: p.api_key,
            cache_ttl: p.cache_ttl,
        })),
        ProviderKind::Anthropic => {
            let mut config = AnthropicConfig::new(p.api_key);
            if let Some(base_url) = base_url {
                config.base_url = base_url;
            }
            config.cache_ttl = p.cache_ttl;
            Arc::new(AnthropicProvider::new(config))
        }
        ProviderKind::Openai => {
            let mut config = OpenAiConfig::new(p.api_key);
            if let Some(base_url) = base_url {
                config.base_url = base_url;
            }
            config.organization = std::env::var("OPENAI_ORG_ID").ok();
            config.project = std::env::var("OPENAI_PROJECT_ID").ok();
            config.cache_retention = p.cache_ttl;
            Arc::new(OpenAiProvider::new(config))
        }

        ProviderKind::Worker => unreachable!("a worker block has no engine-side client"),
    }
}
