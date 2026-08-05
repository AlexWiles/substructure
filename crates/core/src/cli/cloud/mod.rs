pub mod agents;
pub mod apply;
pub mod context;
pub mod credentials;
mod http;
pub mod keys;
pub mod link;
pub mod llm;
pub mod login;
pub mod logout;
pub mod notices;
pub mod open;
pub mod orgs;
mod pickers;
pub(crate) mod print;
pub(crate) mod project_config;
pub mod projects;
pub mod sessions;
pub mod slack;
pub mod telemetry;
pub mod whoami;

use std::path::PathBuf;

/// Footer listing flags that propagate to every leaf subcommand. Clap's
/// `global = true` only surfaces these in leaf help, so we paste them
/// into the parent help screens manually.
pub const GLOBAL_FLAGS_HELP: &str = "\
Global Options:
      --url <URL>          Override the cloud API URL.
  -c, --config <PATH>      Environment file override (substructure.toml).
      --credentials <PATH> User-level credentials file override.
      --json               Emit machine-readable JSON.
  -n, --no-interaction     Never prompt; fail if input is required.";

#[derive(Debug, clap::Args, Clone, Default)]
pub struct CloudGlobals {
    /// Override the cloud API URL (precedence: flag > the environment file's
    /// `[remote].url` > $SUBS_API_URL > prod).
    #[arg(long, global = true)]
    pub url: Option<String>,
    /// Environment file (default: walks up from cwd looking for
    /// `substructure.toml`). Its `[remote]` section names the server and
    /// pins which org/project commands target without flags.
    #[arg(short = 'c', long, global = true)]
    pub config: Option<PathBuf>,
    /// User-level credentials file holding the bearer token
    /// (default: ~/.config/substructure/credentials.toml).
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
pub struct ProjectScope {
    #[arg(long)]
    pub org: Option<String>,
    #[arg(long)]
    pub project: Option<String>,
    #[command(flatten)]
    pub globals: CloudGlobals,
}
