use std::collections::HashMap;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

use substructure::domain::config::{LoggingConfig, SystemConfig};
use substructure::domain::event::{ClientIdentity, EventPayload, SpanContext};
use substructure::domain::secret::resolve_secrets;
use substructure::domain::session::{CommandPayload, IncomingMessage, SessionCommand};
#[cfg(feature = "http")]
use substructure::http::start_server;
use substructure::runtime::Runtime;
use substructure::runtime::SessionUpdate;

#[derive(Parser)]
#[command(name = "substructure", about = "Substructure agent runtime CLI")]
struct Cli {
    /// Path to TOML config file
    #[arg(long, global = true)]
    config: Option<String>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Send a one-shot message to a named agent
    Run {
        /// Agent name (must match a key in [agents])
        #[arg(long)]
        agent: String,
        /// Message to send
        #[arg(long)]
        message: String,
        /// Optional session ID to resume (UUID)
        #[arg(long)]
        session: Option<Uuid>,
    },
    /// Start the HTTP server
    #[cfg(feature = "http")]
    Serve {
        /// Bind host
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Bind port
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
}

fn init_tracing(config: &LoggingConfig) {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.level));

    match config.format.as_str() {
        "json" => tracing_subscriber::fmt()
            .json()
            .with_span_events(FmtSpan::CLOSE)
            .with_env_filter(filter)
            .init(),
        "pretty" => tracing_subscriber::fmt()
            .pretty()
            .with_span_events(FmtSpan::CLOSE)
            .with_env_filter(filter)
            .init(),
        "full" => tracing_subscriber::fmt()
            .with_span_events(FmtSpan::CLOSE)
            .with_env_filter(filter)
            .init(),
        _ => tracing_subscriber::fmt()
            .compact()
            .with_span_events(FmtSpan::CLOSE)
            .with_env_filter(filter)
            .init(),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let config_path = cli
        .config
        .ok_or_else(|| anyhow::anyhow!("--config <path> is required"))?;

    let raw = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("failed to read {config_path}: {e}"))?;
    let config: SystemConfig =
        toml::from_str(&raw).map_err(|e| anyhow::anyhow!("failed to parse config: {e}"))?;
    let config = resolve_secrets(config)?;

    init_tracing(&config.logging);

    tracing::info!(
        agents = config.agents.len(),
        llm_clients = config.llm_clients.len(),
        budgets = config.budgets.len(),
        "config loaded from {config_path}",
    );

    match cli.command {
        Command::Run {
            agent,
            message,
            session,
        } => run_one_shot(config, &agent, &message, session).await,
        #[cfg(feature = "http")]
        Command::Serve { host, port } => run_http(config, &host, port).await,
    }
}

async fn run_one_shot(
    config: SystemConfig,
    agent_name: &str,
    message: &str,
    session_id: Option<Uuid>,
) -> anyhow::Result<()> {
    let runtime = Runtime::start(&config).await?;

    let auth = ClientIdentity {
        tenant_id: "cli".into(),
        sub: None,
        attrs: HashMap::from([("source".into(), "cli".into())]),
    };

    // Start the session — resumes if it exists, creates otherwise
    let session = match session_id {
        Some(id) => runtime.start_session(id, agent_name, auth.clone()).await?,
        None => runtime.create_session_for(agent_name, auth.clone()).await?,
    };
    let session_id = session.session_id;

    // Connect a client with a callback that captures the final assistant reply
    let notify = Arc::new(tokio::sync::Notify::new());

    let client = {
        let notify = notify.clone();
        runtime
            .connect(
                session_id,
                auth,
                Some(Box::new(move |update: &SessionUpdate| {
                    if let SessionUpdate::Event(event) = update {
                        if let Ok(ev) = serde_json::to_string_pretty(&event.payload) {
                            println!("{}", ev);
                        }

                        if matches!(&event.payload, EventPayload::SessionDone(_)) {
                            notify.notify_one();
                        }
                    }
                })),
            )
            .await?
    };

    // Send the user message, continuing the session's trace
    let span = match session.trace_id {
        Some(tid) => SpanContext::in_trace(tid, "cli.message"),
        None => SpanContext::root().with_name("cli.message"),
    };
    client
        .send_command(SessionCommand {
            span,
            occurred_at: chrono::Utc::now(),
            payload: CommandPayload::SendMessage {
                message: IncomingMessage::User {
                    content: message.into(),
                },
                stream: false,
            },
        })
        .await?;

    // Wait for the callback to fire
    let _result = notify.notified().await;

    client.shutdown();
    session.shutdown();
    runtime.shutdown();
    Ok(())
}

#[cfg(feature = "http")]
async fn run_http(config: SystemConfig, host: &str, port: u16) -> anyhow::Result<()> {
    let addr: std::net::SocketAddr = format!("{host}:{port}")
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid host/port: {e}"))?;

    start_server(config, addr).await
}
