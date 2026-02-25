use std::sync::{Arc, Mutex};

use clap::{Parser, Subcommand};
use uuid::Uuid;

use substructure::domain::config::SystemConfig;
use substructure::domain::event::{EventPayload, SessionAuth, SpanContext};
use substructure::domain::secret::resolve_secrets;
use substructure::domain::session::{CommandPayload, SessionCommand};
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
    let reply = Arc::new(Mutex::new(None::<String>));
    let notify = Arc::new(tokio::sync::Notify::new());

    let client = {
        let reply = reply.clone();
        let notify = notify.clone();
        runtime
            .connect(
                session_id,
                auth,
                Some(Box::new(move |event| {
                    let ev = serde_json::to_string_pretty(event).unwrap();

                    println!("{}", ev);

                    if let EventPayload::MessageAssistant(payload) = &event.payload {
                        if payload.message.content.is_some()
                            && payload.message.tool_calls.is_empty()
                        {
                            *reply.lock().unwrap() = payload.message.content.clone();
                            notify.notify_one();
                        }
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
            payload: CommandPayload::SendUserMessage {
                content: message.into(),
                stream: false,
            },
        })
        .await?;

    // Wait for the callback to fire
    let _result =
        tokio::time::timeout(tokio::time::Duration::from_secs(30), notify.notified()).await;

    client.shutdown();
    session.shutdown();
    runtime.shutdown();
    Ok(())
}
