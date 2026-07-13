use clap::Parser;

use substructure_core::cli::{self, Command};

#[derive(Parser)]
#[command(name = "subs", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let default_level = match &cli.command {
        Command::Run(_) => "error",
        _ => "info",
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| default_level.into()),
        )
        .init();

    cli::run(cli.command).await
}
