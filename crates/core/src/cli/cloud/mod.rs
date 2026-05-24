mod apps;
mod context;
pub mod credentials;
mod http;
mod keys;
mod link;
mod login;
mod logout;
mod open;
mod orgs;
mod pickers;
mod print;
mod project_config;
mod sessions;
mod telemetry;
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
    /// Manage apps.
    #[command(after_help = GLOBAL_FLAGS_HELP)]
    Apps {
        #[command(subcommand)]
        command: apps::AppsCommand,
    },
    /// Manage API keys for an app.
    #[command(after_help = GLOBAL_FLAGS_HELP)]
    Keys {
        #[command(subcommand)]
        command: keys::KeysCommand,
    },
    /// Inspect sessions for an app (list, stream events).
    #[command(after_help = GLOBAL_FLAGS_HELP)]
    Sessions {
        #[command(subcommand)]
        command: sessions::SessionsCommand,
    },
    /// Manage the webhook (worker) config for an app.
    #[command(after_help = GLOBAL_FLAGS_HELP)]
    Webhook {
        #[command(subcommand)]
        command: webhook::WebhookCommand,
    },
    /// Open an app's admin page in your browser.
    Open {
        app_id: Option<String>,
        /// Print the URL instead of opening a browser.
        #[arg(long)]
        no_browser: bool,
        #[command(flatten)]
        scope: AppScope,
    },
    /// Link the current directory to an org (and app) by writing a
    /// `subs.toml`, so commands run from this tree pick them up automatically.
    Link(link::LinkCommand),
}

pub async fn run(command: CloudCommand) -> anyhow::Result<()> {
    telemetry::init(command_path(&command));
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
        CloudCommand::Keys { command } => keys::run(command).await,
        CloudCommand::Sessions { command } => sessions::run(command).await,
        CloudCommand::Webhook { command } => webhook::run(command).await,
        CloudCommand::Open {
            app_id,
            no_browser,
            scope,
        } => open::run(app_id, no_browser, scope).await,
        CloudCommand::Link(cmd) => link::run(cmd).await,
    }
}

// Leaf command path for telemetry headers (e.g. "cloud webhook set").
// Kept manually in sync with the enum rather than scraping argv so we never
// leak secrets users pass on the command line.
fn command_path(cmd: &CloudCommand) -> &'static str {
    match cmd {
        CloudCommand::Login { .. } => "cloud login",
        CloudCommand::Logout { .. } => "cloud logout",
        CloudCommand::Whoami { .. } => "cloud whoami",
        CloudCommand::Open { .. } => "cloud open",
        CloudCommand::Link(_) => "cloud link",
        CloudCommand::Orgs { command } => match command {
            orgs::OrgsCommand::List { .. } => "cloud orgs list",
        },
        CloudCommand::Apps { command } => match command {
            apps::AppsCommand::List { .. } => "cloud apps list",
            apps::AppsCommand::Create { .. } => "cloud apps create",
            apps::AppsCommand::Show { .. } => "cloud apps show",
            apps::AppsCommand::Rename { .. } => "cloud apps rename",
            apps::AppsCommand::Delete { .. } => "cloud apps delete",
        },
        CloudCommand::Keys { command } => match command {
            keys::KeysCommand::List { .. } => "cloud keys list",
            keys::KeysCommand::Create { .. } => "cloud keys create",
            keys::KeysCommand::Revoke { .. } => "cloud keys revoke",
        },
        CloudCommand::Sessions { command } => match command {
            sessions::SessionsCommand::List(_) => "cloud sessions list",
            sessions::SessionsCommand::Events(_) => "cloud sessions events",
        },
        CloudCommand::Webhook { command } => match command {
            webhook::WebhookCommand::Show { .. } => "cloud webhook show",
            webhook::WebhookCommand::Set { .. } => "cloud webhook set",
            webhook::WebhookCommand::Disable { .. } => "cloud webhook disable",
            webhook::WebhookCommand::Secret { .. } => "cloud webhook secret",
            webhook::WebhookCommand::RotateSecret { .. } => "cloud webhook rotate-secret",
        },
    }
}
