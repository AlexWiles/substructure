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

    // Precedence: $RUST_LOG > `log` in substructure.toml > the command's
    // default. Read here rather than inside the command, because a setting that
    // only took effect after startup would miss what startup has to say.
    //
    // A file that will not parse is ignored at this point: the command reads it
    // again and reports that properly, and failing here would mean failing
    // before there is anywhere to print to.
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

/// Whether the reader went away — `subs sessions events … | head`, or a pager
/// someone quit. Nothing failed, so nothing is reported: the output ends where
/// the reader stopped reading.
fn is_broken_pipe(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|e| e.kind() == std::io::ErrorKind::BrokenPipe)
    })
}
