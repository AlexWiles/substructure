mod apps;
mod context;
pub mod credentials;
mod http;
mod init;
mod keys;
mod login;
mod logout;
mod orgs;
mod pickers;
mod print;
mod project_config;
mod sessions;
mod webhook;
mod whoami;

use std::path::PathBuf;

use clap::Subcommand;

/// Footer listing flags that propagate to every leaf subcommand. Clap's
/// `global = true` only surfaces these in leaf help, so we paste them
/// into the parent help screens manually.
pub const GLOBAL_FLAGS_HELP: &str = "\
Global Options:
      --url <URL>          Override the cloud API URL.
  -c, --config <PATH>      Project-local subs.toml override.
      --credentials <PATH> User-level credentials file override.
      --json               Emit machine-readable JSON.
  -n, --no-interaction     Never prompt; fail if input is required.";

#[derive(Debug, clap::Args, Clone, Default)]
pub struct CloudGlobals {
    /// Override the cloud API URL (precedence: flag > $SUBS_API_URL > prod).
    #[arg(long, global = true)]
    pub url: Option<String>,
    /// Project-local config file (default: walks up from cwd looking for
    /// `subs.toml`). Pins which org/app commands target without flags.
    #[arg(short = 'c', long, global = true)]
    pub config: Option<PathBuf>,
    /// User-level credentials file holding the bearer token
    /// (default: ~/.config/subs/credentials.toml).
    #[arg(long, global = true)]
    pub credentials: Option<PathBuf>,
    /// Emit machine-readable JSON instead of the human-readable table/text.
    #[arg(long, global = true)]
    pub json: bool,
    /// Never prompt; fail if input is required.
    #[arg(long, short = 'n', global = true)]
    pub no_interaction: bool,
}

#[derive(Debug, clap::Args, Clone)]
pub struct OrgScope {
    #[arg(long)]
    pub org: Option<String>,
    #[command(flatten)]
    pub globals: CloudGlobals,
}

#[derive(Debug, clap::Args, Clone)]
pub struct AppScope {
    #[arg(long)]
    pub org: Option<String>,
    #[arg(long)]
    pub app: Option<String>,
    #[command(flatten)]
    pub globals: CloudGlobals,
}

#[derive(Subcommand)]
pub enum CloudCommand {
    /// Authenticate via the OAuth device flow and persist the token locally.
    Login {
        /// Don't try to open the verification URL in a browser.
        #[arg(long)]
        no_browser: bool,
        #[command(flatten)]
        globals: CloudGlobals,
    },
    /// Clear local credentials and revoke the server-side session.
    Logout {
        #[command(flatten)]
        globals: CloudGlobals,
    },
    /// Show the currently logged-in user and default org/app.
    Whoami {
        #[command(flatten)]
        globals: CloudGlobals,
    },
    /// Manage organizations.
    #[command(after_help = GLOBAL_FLAGS_HELP)]
    Orgs {
        #[command(subcommand)]
        command: orgs::OrgsCommand,
    },
    /// Manage apps and their per-app resources (keys, sessions, webhook).
    #[command(after_help = GLOBAL_FLAGS_HELP)]
    Apps {
        #[command(subcommand)]
        command: apps::AppsCommand,
    },
    /// Write a `subs.toml` in the current directory pinning org (and app)
    /// so commands run from this tree pick them up automatically.
    Init(init::InitCommand),
}

pub async fn run(command: CloudCommand) -> anyhow::Result<()> {
    match command {
        CloudCommand::Login {
            no_browser,
            globals,
        } => {
            // --no-interaction implies --no-browser; opening a browser is
            // never appropriate in non-interactive contexts (CI, scripts).
            let no_browser = no_browser || globals.no_interaction;
            login::run(globals.url, globals.credentials, no_browser).await
        }
        CloudCommand::Logout { globals } => logout::run(globals.url, globals.credentials).await,
        CloudCommand::Whoami { globals } => whoami::run(globals).await,
        CloudCommand::Orgs { command } => orgs::run(command).await,
        CloudCommand::Apps { command } => apps::run(command).await,
        CloudCommand::Init(cmd) => init::run(cmd).await,
    }
}
