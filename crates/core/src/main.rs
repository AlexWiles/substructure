use clap::Parser;

use substructure_core::cli::cloud::{self, CloudCommand};
use substructure_core::cli::local::{self, LocalCommand};

#[derive(Parser)]
#[command(name = "subs", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
enum Command {
    /// Run a local Substructure server.
    Local {
        #[command(subcommand)]
        command: LocalCommand,
    },
    /// Manage your hosted Substructure cloud account.
    #[command(after_help = substructure_core::cli::cloud::GLOBAL_FLAGS_HELP)]
    Cloud {
        #[command(subcommand)]
        command: CloudCommand,
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
        Command::Local { command } => local::run(command).await,
        Command::Cloud { command } => cloud::run(command).await,
    }
}
