mod providers;
mod push;
mod runtime;
mod server;
mod transport;

use std::sync::Arc;

use clap::Parser;
use tokio::net::TcpListener;

use providers::llm::openrouter::{OpenRouterConfig, OpenRouterProvider};
use providers::sqlite::SqliteStore;
use providers::worker::http_push::http_transport;
use providers::worker::memory_queue::InMemoryWorkerQueue;
use push::PushAdapter;
use runtime::worker::push::{PushRegistry, TransportRegistry};
use server::SubstructureServer;

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
            let queue = Arc::new(InMemoryWorkerQueue::new());
            let llm_provider = Arc::new(OpenRouterProvider::new(OpenRouterConfig {
                base_url: std::env::var("OPENROUTER_BASE_URL")
                    .unwrap_or_else(|_| "https://openrouter.ai/api".to_string()),
                api_key: std::env::var("OPENROUTER_API_KEY").unwrap_or_default(),
            }));

            let rt = runtime::start(store.clone(), llm_provider, queue, store.clone(), Default::default());

            let transports = TransportRegistry::new(vec![http_transport()]);
            let registry = PushRegistry::new(store, transports);
            let adapter = Arc::new(PushAdapter::new(rt.clone(), registry, 16));
            adapter.start().await;

            let admin_routes = transport::admin_http::router(rt.clone());
            let client_routes = transport::client_http::router(rt);
            let worker_routes = transport::worker_http::router(adapter);
            let server = SubstructureServer::new(vec![admin_routes, client_routes, worker_routes]);

            let addr = format!("{host}:{port}");
            tracing::info!(%addr, "listening");
            let listener = TcpListener::bind(&addr).await?;
            server.serve(listener).await
        }
    }
}
