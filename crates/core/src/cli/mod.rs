pub mod auth;
pub mod cloud;
pub mod env;
pub mod local;
pub mod run;

use clap::Subcommand;

use crate::transport::push::PushAdapter;
use crate::worker::push::PushRegistrationRecord;

use cloud::{AppScope, CloudGlobals, GLOBAL_FLAGS_HELP};

pub(crate) const DEFAULT_TENANT: &str = "default";

#[derive(Subcommand)]
pub enum Command {
    /// Run a local Substructure server.
    Serve(local::ServeArgs),
    /// Run a single turn against a worker in-process and stream events, then exit.
    /// For local development and testing example agents.
    Run(run::RunArgs),
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
        command: cloud::orgs::OrgsCommand,
    },
    /// Manage apps.
    #[command(after_help = GLOBAL_FLAGS_HELP)]
    Apps {
        #[command(subcommand)]
        command: cloud::apps::AppsCommand,
    },
    /// Manage API keys for an app.
    #[command(after_help = GLOBAL_FLAGS_HELP)]
    Keys {
        #[command(subcommand)]
        command: cloud::keys::KeysCommand,
    },
    /// Inspect sessions for an app (list, stream events).
    #[command(after_help = GLOBAL_FLAGS_HELP)]
    Sessions {
        #[command(subcommand)]
        command: cloud::sessions::SessionsCommand,
    },
    /// Manage the webhook (worker) config for an app.
    #[command(after_help = GLOBAL_FLAGS_HELP)]
    Webhook {
        #[command(subcommand)]
        command: cloud::webhook::WebhookCommand,
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
    /// `substructure.toml`, so commands run from this tree pick them up automatically.
    Link(cloud::link::LinkCommand),
}

pub async fn run(command: Command) -> anyhow::Result<()> {
    cloud::telemetry::init(command_path(&command));
    match command {
        Command::Serve(args) => local::serve(args).await,
        Command::Run(args) => run::run(args).await,
        Command::Login {
            no_browser,
            globals,
        } => {
            // --no-interaction implies --no-browser; opening a browser is
            // never appropriate in non-interactive contexts (CI, scripts).
            let no_browser = no_browser || globals.no_interaction;
            cloud::login::run(globals.url, globals.credentials, no_browser).await
        }
        Command::Logout { globals } => cloud::logout::run(globals.url, globals.credentials).await,
        Command::Whoami { globals } => cloud::whoami::run(globals).await,
        Command::Orgs { command } => cloud::orgs::run(command).await,
        Command::Apps { command } => cloud::apps::run(command).await,
        Command::Keys { command } => cloud::keys::run(command).await,
        Command::Sessions { command } => cloud::sessions::run(command).await,
        Command::Webhook { command } => cloud::webhook::run(command).await,
        Command::Open {
            app_id,
            no_browser,
            scope,
        } => cloud::open::run(app_id, no_browser, scope).await,
        Command::Link(cmd) => cloud::link::run(cmd).await,
    }
}

// Leaf command path for telemetry headers (e.g. "webhook set"). Kept manually
// in sync with the enum rather than scraping argv so we never leak secrets
// users pass on the command line.
fn command_path(cmd: &Command) -> &'static str {
    use cloud::{apps::AppsCommand, keys::KeysCommand, orgs::OrgsCommand};
    use cloud::{sessions::SessionsCommand, webhook::WebhookCommand};
    match cmd {
        Command::Serve(_) => "serve",
        Command::Run(_) => "run",
        Command::Login { .. } => "login",
        Command::Logout { .. } => "logout",
        Command::Whoami { .. } => "whoami",
        Command::Open { .. } => "open",
        Command::Link(_) => "link",
        Command::Orgs { command } => match command {
            OrgsCommand::List { .. } => "orgs list",
        },
        Command::Apps { command } => match command {
            AppsCommand::List { .. } => "apps list",
            AppsCommand::Create { .. } => "apps create",
            AppsCommand::Show { .. } => "apps show",
            AppsCommand::Rename { .. } => "apps rename",
            AppsCommand::Delete { .. } => "apps delete",
        },
        Command::Keys { command } => match command {
            KeysCommand::List { .. } => "keys list",
            KeysCommand::Create { .. } => "keys create",
            KeysCommand::Revoke { .. } => "keys revoke",
        },
        Command::Sessions { command } => match command {
            SessionsCommand::List(_) => "sessions list",
            SessionsCommand::Events(_) => "sessions events",
        },
        Command::Webhook { command } => match command {
            WebhookCommand::Show { .. } => "webhook show",
            WebhookCommand::Set { .. } => "webhook set",
            WebhookCommand::Disable { .. } => "webhook disable",
            WebhookCommand::Secret { .. } => "webhook secret",
            WebhookCommand::RotateSecret { .. } => "webhook rotate-secret",
        },
    }
}

pub async fn register_startup_worker(
    adapter: &PushAdapter,
    url: &str,
    signing_secret: Option<String>,
) -> anyhow::Result<()> {
    let secret = signing_secret.unwrap_or_else(|| hex::encode(rand::random::<[u8; 32]>()));
    adapter
        .register(PushRegistrationRecord {
            tenant_id: DEFAULT_TENANT.into(),
            transport_type: "http".into(),
            config: serde_json::json!({
                "endpoint_url": url,
                "signing_secret": secret,
            }),
        })
        .await
        .map_err(|e| anyhow::anyhow!("failed to register startup worker: {e}"))?;
    tracing::info!(url, "startup worker registered (signing enabled)");
    Ok(())
}
