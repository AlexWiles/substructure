use std::sync::Arc;

use clap::{Parser, ValueEnum};
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

#[derive(Copy, Clone, ValueEnum)]
enum LlmProviderArg {
    Openrouter,
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
        /// LLM provider. Determines which API key env var is required
        /// (e.g. `openrouter` needs OPENROUTER_API_KEY).
        #[arg(long, value_enum)]
        provider: LlmProviderArg,
        /// Disable client and worker authentication. For local development only.
        #[arg(long)]
        dev: bool,
    },
}

enum ProviderEnv {
    Openrouter { api_key: String },
}

struct AuthEnvVars {
    client_token_issuer: String,
    client_token_audience: String,
    client_token_hs256_secret: String,
    worker_api_key: String,
}

struct EnvVars {
    provider: ProviderEnv,
    auth: Option<AuthEnvVars>,
}

impl EnvVars {
    fn load(provider: LlmProviderArg, dev: bool) -> Result<Self, ()> {
        let provider_specs: &[(&str, &str)] = match provider {
            LlmProviderArg::Openrouter => &[(
                "OPENROUTER_API_KEY",
                "API key for OpenRouter (https://openrouter.ai/keys)",
            )],
        };

        let auth_specs: &[(&str, &str)] = &[
            ("CLIENT_TOKEN_ISSUER", "JWT 'iss' claim for client tokens"),
            ("CLIENT_TOKEN_AUDIENCE", "JWT 'aud' claim for client tokens"),
            (
                "CLIENT_TOKEN_HS256_SECRET",
                "HS256 secret used to sign client tokens",
            ),
            (
                "WORKER_API_KEY",
                "Bearer API key that workers present to the server",
            ),
        ];

        let mut missing: Vec<(&'static str, &'static str)> = Vec::new();
        let mut provider_values: Vec<String> = Vec::with_capacity(provider_specs.len());
        let mut auth_values: Vec<String> = Vec::with_capacity(auth_specs.len());

        for (name, desc) in provider_specs {
            match std::env::var(name) {
                Ok(v) => provider_values.push(v),
                Err(_) => missing.push((name, desc)),
            }
        }
        if !dev {
            for (name, desc) in auth_specs {
                match std::env::var(name) {
                    Ok(v) => auth_values.push(v),
                    Err(_) => missing.push((name, desc)),
                }
            }
        }

        if !missing.is_empty() {
            eprintln!("error: missing required environment variable(s):");
            for (name, desc) in &missing {
                eprintln!("  - {name}: {desc}");
            }
            eprintln!("\nSet them and try again, e.g.:");
            for (name, _) in &missing {
                eprintln!("  export {name}=...");
            }
            return Err(());
        }

        let provider = match provider {
            LlmProviderArg::Openrouter => ProviderEnv::Openrouter {
                api_key: provider_values.into_iter().next().unwrap(),
            },
        };

        let auth = if dev {
            None
        } else {
            let mut it = auth_values.into_iter();
            Some(AuthEnvVars {
                client_token_issuer: it.next().unwrap(),
                client_token_audience: it.next().unwrap(),
                client_token_hs256_secret: it.next().unwrap(),
                worker_api_key: it.next().unwrap(),
            })
        };

        Ok(Self { provider, auth })
    }
}

struct AuthWiring {
    client: Arc<dyn AuthResolver>,
    worker: Arc<dyn AuthResolver>,
    issuer: Arc<JwtHs256ClientTokenAuthResolver>,
}

impl AuthWiring {
    fn dev() -> Self {
        let secret = hex::encode(rand::random::<[u8; 32]>());
        let issuer = Arc::new(JwtHs256ClientTokenAuthResolver::new(
            "substructure-dev",
            "substructure-dev",
            &secret,
        ));
        Self {
            client: Arc::new(NoopAuthResolver),
            worker: Arc::new(NoopAuthResolver),
            issuer,
        }
    }

    fn from_env(env: AuthEnvVars) -> anyhow::Result<Self> {
        let issuer = Arc::new(JwtHs256ClientTokenAuthResolver::new(
            env.client_token_issuer,
            env.client_token_audience,
            env.client_token_hs256_secret,
        ));
        let worker_key_hash = hex::encode(Sha256::digest(env.worker_api_key.as_bytes()));
        let bindings = vec![ApiKeyBinding::new("default", worker_key_hash, "worker")];
        let worker: Arc<dyn AuthResolver> = Arc::new(
            BearerHashedApiKeyAuthResolver::new(bindings).map_err(anyhow::Error::msg)?,
        );
        Ok(Self {
            client: issuer.clone(),
            worker,
            issuer,
        })
    }
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

            let shutdown = tokio_util::sync::CancellationToken::new();
            let signal_token = shutdown.clone();
            tokio::spawn(async move {
                let _ = tokio::signal::ctrl_c().await;
                tracing::info!("shutdown signal received");
                signal_token.cancel();
            });

            let admin_routes = admin_http::router(admin_http::AdminHttpState {
                runtime: rt.clone(),
                auth: Arc::new(NoopAuthResolver),
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
                eprintln!("  DEV MODE -- authentication is disabled.");
                eprintln!("  Do not use in production or expose to untrusted networks.");
                eprintln!();
            }
            tracing::info!(%addr, "listening");
            let listener = TcpListener::bind(&addr).await?;
            server.serve(listener, shutdown).await
        }
    }
}
