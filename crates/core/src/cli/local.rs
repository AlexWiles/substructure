use std::sync::Arc;

use clap::Args;
use tokio::net::TcpListener;

use crate::cli::auth::AuthWiring;
use crate::cli::env::{EnvVars, LlmProviderArg, ProviderEnv};
use crate::cli::register_startup_worker;
use crate::llm::{LlmProviderTrait, LlmTask};
use crate::providers::anthropic::{AnthropicConfig, AnthropicProvider};
use crate::providers::memory_queue::{ShardedInMemoryQueue, TaskQueue};
use crate::providers::openai::{OpenAiConfig, OpenAiProvider};
use crate::providers::openrouter::{OpenRouterConfig, OpenRouterProvider};
use crate::providers::sqlite::{
    SqliteCheckpointStore, SqliteDb, SqliteEventStore, SqlitePushStore, SqliteSessionIndexStore,
    SqliteWakeStore, SqliteWorkerQueue,
};
use crate::sub_agent::SubAgentTask;
use crate::transport::admin_http::{self, AdminHttpState};
use crate::transport::client_http::{self, ClientHttpState};
use crate::transport::http_push::http_transport;
use crate::transport::push::PushAdapter;
use crate::transport::server::SubstructureServer;
use crate::transport::worker_http::{self, WorkerHttpState};
use crate::worker::push::{PushRegistry, TransportRegistry};
use crate::{start, RuntimeConfig};

#[derive(Args)]
pub struct ServeArgs {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    #[arg(long, default_value = "data.db")]
    db: String,
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
    provider: Option<LlmProviderArg>,
    /// Disable client and worker authentication. For local development only.
    #[arg(long)]
    dev: bool,
}

pub async fn serve(args: ServeArgs) -> anyhow::Result<()> {
    start_server(
        args.host,
        args.port,
        args.db,
        args.worker_url,
        args.signing_secret,
        args.provider,
        args.dev,
    )
    .await
}

async fn start_server(
    host: String,
    port: u16,
    db_path: String,
    worker_url: Option<String>,
    signing_secret: Option<String>,
    provider: Option<LlmProviderArg>,
    dev: bool,
) -> anyhow::Result<()> {
    let env = match EnvVars::load(provider, dev) {
        Ok(e) => e,
        Err(_) => std::process::exit(2),
    };

    let db = SqliteDb::open(&db_path, std::time::Duration::from_secs(5))?;

    let event_store = Arc::new(SqliteEventStore::new(db.clone())?);
    let worker_queue = Arc::new(SqliteWorkerQueue::new(db.clone())?);
    let checkpoint_store = Arc::new(SqliteCheckpointStore::new(db.clone())?);
    let wake_store = Arc::new(SqliteWakeStore::new(db.clone())?);
    let session_index_store = Arc::new(SqliteSessionIndexStore::new(db.clone())?);
    let push_store = Arc::new(SqlitePushStore::new(db)?);

    let config = RuntimeConfig::default();
    let llm_task_queue: Arc<dyn TaskQueue<LlmTask>> = Arc::new(ShardedInMemoryQueue::new(
        config.llm_executor_workers as u32,
    ));
    let sub_agent_task_queue: Arc<dyn TaskQueue<SubAgentTask>> = Arc::new(
        ShardedInMemoryQueue::new(config.sub_agent_executor_workers as u32),
    );
    let llm_provider: Option<Arc<dyn LlmProviderTrait>> = match env.provider {
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
        event_store.clone(),
        llm_provider,
        llm_task_queue,
        sub_agent_task_queue,
        worker_queue,
        session_index_store,
        checkpoint_store,
        wake_store,
        token_delta_transport,
        config,
    );

    let transports = TransportRegistry::new(vec![http_transport()]);
    let registry = PushRegistry::new(push_store, transports);
    let adapter = Arc::new(PushAdapter::new(rt.clone(), registry, 16));

    let auth = match env.auth {
        Some(a) => AuthWiring::from_env(a)?,
        None => AuthWiring::dev(),
    };

    adapter.start().await;

    if let Some(ref url) = worker_url {
        register_startup_worker(&adapter, url, signing_secret).await?;
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
        auth: auth.client,
        shutdown: shutdown.clone(),
    });
    let worker_routes = worker_http::router(WorkerHttpState {
        runtime: rt.clone(),
        auth: auth.worker,
        client_token_issuer: auth.issuer,
        shutdown: shutdown.clone(),
    });

    let server =
        SubstructureServer::new(vec![admin_routes, client_routes, worker_routes, v1_routes]);

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
