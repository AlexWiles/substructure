use std::sync::Arc;

use clap::{Parser, Subcommand};
use uuid::Uuid;

use substructure::domain::aggregate::DomainEvent;
use substructure::domain::config::SystemConfig;
use substructure::domain::event::{EventPayload, SessionAuth, SpanContext};
use substructure::domain::secret::resolve_secrets;
use substructure::domain::session::{AgentState, CommandPayload, IncomingMessage, SessionCommand};
#[cfg(feature = "http")]
use substructure::http::start_server;
use substructure::runtime::Runtime;

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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let config = cli
        .config
        .ok_or_else(|| anyhow::anyhow!("--config <path> is required"))?;

    match cli.command {
        Command::Run {
            agent,
            message,
            session,
        } => run_one_shot(&config, &agent, &message, session).await,
        #[cfg(feature = "http")]
        Command::Serve { host, port } => run_http(&config, &host, port).await,
    }
}

async fn run_one_shot(
    config_path: &str,
    agent_name: &str,
    message: &str,
    session_id: Option<Uuid>,
) -> anyhow::Result<()> {
    let raw = std::fs::read_to_string(config_path)
        .map_err(|e| anyhow::anyhow!("failed to read {config_path}: {e}"))?;

    let config: SystemConfig =
        toml::from_str(&raw).map_err(|e| anyhow::anyhow!("failed to parse config: {e}"))?;

    let config = resolve_secrets(config)?;

    let runtime = Runtime::start(&config).await?;

    let auth = SessionAuth {
        tenant_id: "cli".into(),
        client_id: "substructure-cli".into(),
        sub: None,
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
                Some(Box::new(move |event: &DomainEvent<AgentState>| {
                    if let Ok(ev) = serde_json::to_string_pretty(&event.payload) {
                        println!("{}", ev);
                    }

                    if matches!(&event.payload, EventPayload::SessionDone(_)) {
                        notify.notify_one();
                    }
                })),
            )
            .await?
    };

    // Send the user message
    client
        .send_command(SessionCommand {
            span: SpanContext::root(),
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
async fn run_http(config_path: &str, host: &str, port: u16) -> anyhow::Result<()> {
    let raw = std::fs::read_to_string(config_path)
        .map_err(|e| anyhow::anyhow!("failed to read {config_path}: {e}"))?;
    let config: SystemConfig =
        toml::from_str(&raw).map_err(|e| anyhow::anyhow!("failed to parse config: {e}"))?;
    let config = resolve_secrets(config)?;

    let addr: std::net::SocketAddr = format!("{host}:{port}")
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid host/port: {e}"))?;

    start_server(config, addr).await
}
