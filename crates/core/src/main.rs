use std::sync::Arc;

use clap::Parser;
use tokio::net::TcpListener;

use substructure_core::cli::auth::AuthWiring;
use substructure_core::cli::env::{EnvVars, LlmProviderArg, ProviderEnv};
use substructure_core::cli::register_startup_worker;
use substructure_core::llm::LlmTask;
use substructure_core::providers::memory_queue::{ShardedInMemoryQueue, TaskQueue};
use substructure_core::providers::openrouter::{OpenRouterConfig, OpenRouterProvider};
use substructure_core::providers::sqlite::{
    SqliteCheckpointStore, SqliteDb, SqliteEventStore, SqlitePushStore, SqliteSessionIndexStore,
    SqliteWakeStore, SqliteWorkerQueue,
};
use substructure_core::sub_agent::SubAgentTask;
use substructure_core::transport::admin_http::{self, AdminHttpState};
use substructure_core::transport::client_http::{self, ClientHttpState};
use substructure_core::transport::http_push::http_transport;
use substructure_core::transport::push::PushAdapter;
use substructure_core::transport::server::SubstructureServer;
use substructure_core::transport::worker_http::{self, WorkerHttpState};
use substructure_core::worker::push::{PushRegistry, TransportRegistry};
use substructure_core::{start, RuntimeConfig};

#[derive(Parser)]
#[command(name = "substructure", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
enum Command {
    /// Start the server
    Start {
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        #[arg(long, default_value = "data.db")]
        db: String,
        /// Pre-register an HTTP worker at startup
        #[arg(long)]
        worker_url: Option<String>,
        /// Signing secret for the pre-registered HTTP worker (auto-generated if omitted)
        #[arg(long, requires = "worker_url")]
        signing_secret: Option<String>,
        /// LLM provider. Determines which API key env var is required
        /// (e.g. `openrouter` needs OPENROUTER_API_KEY).
        #[arg(long, value_enum)]
        provider: LlmProviderArg,
        /// Disable client and worker authentication. For local development only.
        #[arg(long)]
        dev: bool,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Start {
            host,
            port,
            db,
            worker_url,
            signing_secret,
            provider,
            dev,
        } => {
            let env = match EnvVars::load(provider, dev) {
                Ok(e) => e,
                Err(_) => std::process::exit(2),
            };

            let db = SqliteDb::open(&db, std::time::Duration::from_secs(5))?;

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
            let llm_provider = match env.provider {
                ProviderEnv::Openrouter { api_key } => {
                    Arc::new(OpenRouterProvider::new(OpenRouterConfig {
                        base_url: std::env::var("OPENROUTER_BASE_URL")
                            .unwrap_or_else(|_| "https://openrouter.ai/api".to_string()),
                        api_key,
                    }))
                }
            };

            let rt = start(
                event_store.clone(),
                llm_provider,
                llm_task_queue,
                sub_agent_task_queue,
                worker_queue,
                session_index_store,
                checkpoint_store,
                wake_store,
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
                auth: auth.admin,
                shutdown: shutdown.clone(),
            });

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

            let server = SubstructureServer::new(vec![admin_routes, client_routes, worker_routes]);

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
    }
}
