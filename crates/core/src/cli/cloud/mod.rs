// Hosted-cloud commands: auth + org/app/key/session/webhook management.
// Talks to /api/v1/* on the cloud backend with a bearer token persisted in
// ~/.config/subs/config.toml.

pub mod config;
mod apps;
mod context;
mod http;
mod keys;
mod login;
mod logout;
mod orgs;
mod sessions;
mod webhook;
mod whoami;

use std::path::PathBuf;

use clap::Subcommand;

/// Global flags accepted by every cloud subcommand.
#[derive(Debug, clap::Args, Clone, Default)]
pub struct CloudGlobals {
    /// Override the cloud API URL (default: configured value, else prod).
    #[arg(long, global = true)]
    pub url: Option<String>,
    /// Override the config file path (default: ~/.config/subs/config.toml).
    #[arg(long, global = true)]
    pub config: Option<PathBuf>,
}

#[derive(Subcommand)]
pub enum CloudCommand {
    /// Authenticate via the OAuth device flow and persist the token locally.
    Login {
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
    Orgs {
        #[command(subcommand)]
        command: orgs::OrgsCommand,
    },
    /// Manage apps.
    Apps {
        #[command(subcommand)]
        command: apps::AppsCommand,
    },
    /// Manage API keys (per-app).
    Keys {
        #[command(subcommand)]
        command: keys::KeysCommand,
    },
    /// List debug sessions (per-app).
    Sessions(sessions::SessionsCommand),
    /// Manage the webhook (worker) config (per-app).
    Webhook {
        #[command(subcommand)]
        command: webhook::WebhookCommand,
    },
}

pub async fn run(command: CloudCommand) -> anyhow::Result<()> {
    match command {
        CloudCommand::Login { globals } => login::run(globals.url, globals.config).await,
        CloudCommand::Logout { globals } => logout::run(globals.url, globals.config).await,
        CloudCommand::Whoami { globals } => whoami::run(globals.url, globals.config).await,
        CloudCommand::Orgs { command } => orgs::run(command).await,
        CloudCommand::Apps { command } => apps::run(command).await,
        CloudCommand::Keys { command } => keys::run(command).await,
        CloudCommand::Sessions(cmd) => sessions::run(cmd).await,
        CloudCommand::Webhook { command } => webhook::run(command).await,
    }
}
