use std::sync::Arc;

use clap::Parser;
use sha2::{Digest, Sha256};
use tokio::net::TcpListener;

use substructure_core::llm::LlmTask;
use substructure_core::providers::memory_queue::{ShardedInMemoryQueue, TaskQueue};
use substructure_core::providers::openrouter::{OpenRouterConfig, OpenRouterProvider};
use substructure_core::providers::sqlite::{
    SqliteCheckpointStore, SqliteDb, SqliteEventStore, SqlitePushStore, SqliteSessionIndexStore,
    SqliteWakeStore, SqliteWorkerQueue,
};
use substructure_core::sub_agent::SubAgentTask;
use substructure_core::transport::admin_http;
use substructure_core::transport::auth::{
    ApiKeyBinding, AuthResolver, BearerHashedApiKeyAuthResolver, JwtHs256ClientTokenAuthResolver,
    NoopAuthResolver,
};
use substructure_core::transport::client_http::{self, ClientHttpState};
use substructure_core::transport::http_push::http_transport;
use substructure_core::transport::push::PushAdapter;
use substructure_core::transport::server::SubstructureServer;
use substructure_core::transport::worker_http::{self, WorkerHttpState};
use substructure_core::worker::push::{PushRegistrationRecord, PushRegistry, TransportRegistry};
use substructure_core::{start, RuntimeConfig};

fn required_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} environment variable is required"))
}

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
        } => {
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
            let llm_provider = Arc::new(OpenRouterProvider::new(OpenRouterConfig {
                base_url: std::env::var("OPENROUTER_BASE_URL")
                    .unwrap_or_else(|_| "https://openrouter.ai/api".to_string()),
                api_key: required_env("OPENROUTER_API_KEY"),
            }));

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
            let client_token_issuer = Arc::new(JwtHs256ClientTokenAuthResolver::new(
                required_env("CLIENT_TOKEN_ISSUER"),
                required_env("CLIENT_TOKEN_AUDIENCE"),
                required_env("CLIENT_TOKEN_HS256_SECRET"),
            ));

            let worker_api_key = required_env("WORKER_API_KEY");
            let worker_key_hash = hex::encode(Sha256::digest(worker_api_key.as_bytes()));
            let bindings = vec![ApiKeyBinding::new("default", worker_key_hash, "worker")];
            let client_auth: Arc<dyn AuthResolver> = client_token_issuer.clone();
            let worker_auth: Arc<dyn AuthResolver> = Arc::new(
                BearerHashedApiKeyAuthResolver::new(bindings).map_err(anyhow::Error::msg)?,
            );
            adapter.start().await;

            if let Some(ref url) = worker_url {
                let secret =
                    signing_secret.unwrap_or_else(|| hex::encode(rand::random::<[u8; 32]>()));
                adapter
                    .register(PushRegistrationRecord {
                        tenant_id: "default".into(),
                        transport_type: "http".into(),
                        config: serde_json::json!({
                            "endpoint_url": url,
                            "signing_secret": secret,
                        }),
                    })
                    .await
                    .expect("failed to register startup worker");
                tracing::info!(url, "startup worker registered (signing enabled)");
            }

            let admin_routes = admin_http::router(admin_http::AdminHttpState {
                runtime: rt.clone(),
                auth: Arc::new(NoopAuthResolver),
            });
            let client_routes = client_http::router(ClientHttpState {
                runtime: rt.clone(),
                auth: client_auth,
            });
            let worker_routes = worker_http::router(WorkerHttpState {
                runtime: rt.clone(),
                auth: worker_auth,
                client_token_issuer,
            });
            let server = SubstructureServer::new(vec![admin_routes, client_routes, worker_routes]);

            let addr = format!("{host}:{port}");
            tracing::info!(%addr, "listening");
            let listener = TcpListener::bind(&addr).await?;
            server.serve(listener).await
        }
    }
}
