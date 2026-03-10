mod providers;
mod runtime;

use clap::Parser;

#[derive(Parser)]
#[command(name = "substructure2", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
enum Command {
    /// Run a one-shot message against an agent
    Run {
        /// Agent name
        #[arg(long)]
        agent: String,
        /// Message to send
        #[arg(long)]
        message: String,
    },
    /// Start the server
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Run { agent, message } => {
            tracing::info!(agent, message, "running");
            todo!("run")
        }
        Command::Serve { host, port } => {
            tracing::info!(host, port, "serving");
            todo!("serve")
        }
    }
}
