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

    let configured = cli::project_log_filter(cli.command.config_path());
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                configured
                    .unwrap_or_else(|| cli.command.default_log().to_string())
                    .into()
            }),
        )
        .init();

    match cli::run(cli.command).await {
        Err(e) if is_broken_pipe(&e) => Ok(()),
        result => result,
    }
}

fn is_broken_pipe(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|e| e.kind() == std::io::ErrorKind::BrokenPipe)
    })
}
