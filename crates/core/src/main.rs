use std::sync::Arc;

use clap::Parser;
use tokio::net::TcpListener;

use substructure_core::providers::memory_queue::{ShardedInMemoryQueue, TaskQueue};
use substructure_core::providers::openrouter::{OpenRouterConfig, OpenRouterProvider};
use substructure_core::providers::sqlite::SqliteStore;
use substructure_core::providers::worker_queue::InMemoryWorkerQueue;
use substructure_core::transport::http_push::http_transport;
use substructure_core::transport::push::PushAdapter;
use substructure_core::transport::server::SubstructureServer;
use substructure_core::worker::push::{PushRegistry, TransportRegistry};
use substructure_core::RuntimeConfig;

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
            let store = Arc::new(SqliteStore::new(&db)?);
            let config = RuntimeConfig::default();
            let queue = Arc::new(InMemoryWorkerQueue::new());
            let llm_task_queue: Arc<dyn TaskQueue<substructure_core::llm::LlmTask>> =
                Arc::new(ShardedInMemoryQueue::new(config.llm_executor_workers as u32));
            let sub_agent_task_queue: Arc<dyn TaskQueue<substructure_core::sub_agent::SubAgentTask>> =
                Arc::new(ShardedInMemoryQueue::new(config.sub_agent_executor_workers as u32));
            let llm_provider = Arc::new(OpenRouterProvider::new(OpenRouterConfig {
                base_url: std::env::var("OPENROUTER_BASE_URL")
                    .unwrap_or_else(|_| "https://openrouter.ai/api".to_string()),
                api_key: std::env::var("OPENROUTER_API_KEY").unwrap_or_default(),
            }));

            let rt = substructure_core::start(
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
            adapter.start().await;

            let admin_routes = substructure_core::transport::admin_http::router(rt.clone());
            let client_routes = substructure_core::transport::client_http::router(rt);
            let worker_routes = substructure_core::transport::worker_http::router(adapter);
            let dashboard_routes = substructure_core::transport::dashboard::router();
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
