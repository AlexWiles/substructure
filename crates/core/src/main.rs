use std::sync::Arc;

use clap::Parser;
use tokio::net::TcpListener;

use substructure_core::providers::memory_queue::{ShardedInMemoryQueue, TaskQueue};
use substructure_core::providers::openrouter::{OpenRouterConfig, OpenRouterProvider};
use substructure_core::providers::sqlite::{SqliteConfig, SqliteStore};
use substructure_core::providers::worker_queue::SqliteWorkerQueue;
use substructure_core::sub_agent::SubAgentTask;
use substructure_core::transport::admin_http;
use substructure_core::transport::auth::{
    ApiKeyBinding, AuthResolver, BearerHashedApiKeyAuthResolver, JwtHs256ClientTokenAuthResolver,
};
use substructure_core::transport::client_http::{self, ClientHttpState};
use substructure_core::transport::dashboard;
use substructure_core::transport::http_push::http_transport;
use substructure_core::transport::push::PushAdapter;
use substructure_core::transport::server::SubstructureServer;
use substructure_core::transport::worker_http::{self, WorkerHttpState};
use substructure_core::worker::push::{PushRegistry, TransportRegistry};
use substructure_core::{start, RuntimeConfig};
use substructure_core::llm::LlmTask;

#[derive(Parser)]
#[command(name = "substructure", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
enum Command {
    /// Start the server
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        #[arg(long, default_value = "data.db")]
        db: String,
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
        Command::Serve { host, port, db } => {
            let store = Arc::new(SqliteStore::new(SqliteConfig {
                path: db.clone(),
                busy_timeout: std::time::Duration::from_secs(5),
            })?);
            let config = RuntimeConfig::default();
            let queue = Arc::new(SqliteWorkerQueue::new(&db).map_err(anyhow::Error::msg)?);
            let llm_task_queue: Arc<dyn TaskQueue<LlmTask>> =
                Arc::new(ShardedInMemoryQueue::new(config.llm_executor_workers as u32));
            let sub_agent_task_queue: Arc<dyn TaskQueue<SubAgentTask>> =
                Arc::new(ShardedInMemoryQueue::new(config.sub_agent_executor_workers as u32));
            let llm_provider = Arc::new(OpenRouterProvider::new(OpenRouterConfig {
                base_url: std::env::var("OPENROUTER_BASE_URL")
                    .unwrap_or_else(|_| "https://openrouter.ai/api".to_string()),
                api_key: std::env::var("OPENROUTER_API_KEY").unwrap_or_default(),
            }));

            let rt = start(
                store.clone(),
                llm_provider,
                llm_task_queue,
                sub_agent_task_queue,
                queue,
                store.clone(),
                store.clone(),
                store.clone(),
                config,
            );

            let transports = TransportRegistry::new(vec![http_transport()]);
            let registry = PushRegistry::new(store, transports);
            let adapter = Arc::new(PushAdapter::new(rt.clone(), registry, 16));
            let client_token_issuer = Arc::new(JwtHs256ClientTokenAuthResolver::new(
                std::env::var("CLIENT_TOKEN_ISSUER")
                    .unwrap_or_else(|_| "substructure".to_string()),
                std::env::var("CLIENT_TOKEN_AUDIENCE")
                    .unwrap_or_else(|_| "substructure-client".to_string()),
                std::env::var("CLIENT_TOKEN_HS256_SECRET")
                    .unwrap_or_else(|_| "dev-client-token-secret-change-me".to_string()),
            ));
            let bindings = vec![ApiKeyBinding::new(
                "default",
                "0b42357e3654716d9915e42b3b44d9c762169d7c4c972906b45a1d8b28dbad2e",
                "default-dev-key",
            )];
            let client_auth: Arc<dyn AuthResolver> = client_token_issuer.clone();
            let worker_auth: Arc<dyn AuthResolver> = Arc::new(
                BearerHashedApiKeyAuthResolver::new(bindings)
                    .map_err(anyhow::Error::msg)?,
            );
            adapter.start().await;

            let admin_routes = admin_http::router(rt.clone());
            let client_routes = client_http::router(ClientHttpState {
                runtime: rt.clone(),
                auth: client_auth,
            });
            let worker_routes = worker_http::router(WorkerHttpState {
                adapter,
                auth: worker_auth,
                client_token_issuer,
            });
            let dashboard_routes = dashboard::router();
            let server = SubstructureServer::new(vec![
                admin_routes,
                client_routes,
                worker_routes,
                dashboard_routes,
            ]);

            let addr = format!("{host}:{port}");
            tracing::info!(%addr, "listening");
            let listener = TcpListener::bind(&addr).await?;
            server.serve(listener).await
        }
    }
}
