mod providers;
mod runtime;
mod transport;

use std::sync::Arc;

use clap::Parser;
use tokio::net::TcpListener;

use providers::event_store::sqlite::SqliteEventStore;
use providers::llm::openrouter::{OpenRouterConfig, OpenRouterProvider};
use providers::worker::memory_queue::InMemoryWorkerQueue;

#[derive(Parser)]
#[command(name = "substructure2", version)]
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
            let store = Arc::new(SqliteEventStore::new(&db)?);
            let queue = Arc::new(InMemoryWorkerQueue::new());
            let llm_provider = Arc::new(OpenRouterProvider::new(OpenRouterConfig {
                base_url: std::env::var("OPENROUTER_BASE_URL")
                    .unwrap_or_else(|_| "https://openrouter.ai/api".to_string()),
                api_key: std::env::var("OPENROUTER_API_KEY").unwrap_or_default(),
            }));

            let rt = Arc::new(runtime::start(store, llm_provider, queue, Default::default()));

            let app = transport::worker_http::router(rt.clone())
                .merge(transport::client_http::router(rt));
            let addr = format!("{host}:{port}");
            tracing::info!(%addr, "listening");
            let listener = TcpListener::bind(&addr).await?;
            axum::serve(listener, app).await?;

            Ok(())
        }
    }
}
