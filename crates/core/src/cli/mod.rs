pub mod auth;
pub mod cloud;
pub mod doctor;
pub mod env;
pub mod local;
pub mod mcp;
mod pretty;
pub mod run;
pub mod run_remote;
pub mod sessions;
pub mod target;

use clap::Subcommand;

use cloud::{CloudGlobals, ProjectScope, GLOBAL_FLAGS_HELP};

/// Read a secret the project file names rather than holds. An unset or blank
/// variable reads as absent, so a stale name never becomes an empty secret.
pub(crate) fn env_value(var: &str) -> Option<String> {
    std::env::var(var)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// The `log` setting from the project file, for setting up tracing before the
/// command runs. Silent on any failure — the command reads the same file and
/// reports a real problem with it, and there is no logger yet to report to.
pub fn project_log_filter(config: Option<&std::path::Path>) -> Option<String> {
    let found = cloud::project_config::resolve(config).ok().flatten()?;
    found.config.log
}

pub(crate) const DEFAULT_TENANT: &str = "default";

/// The one person at an installation nothing authenticates. `subs run` acts
/// as this, and `subs mcp login` fills its slot. Prefixed so it cannot
/// collide with a subject an identity source issues.
pub(crate) const LOCAL_SUBJECT: &str = "local";

impl Command {
    /// The project file this invocation will read, for callers that need it
    /// before the command runs. Only the engine commands: a cloud command
    /// neither runs an engine nor logs at a level worth pinning.
    pub fn config_path(&self) -> Option<&std::path::Path> {
        match self {
            Command::Run(a) => a.config_path(),
            Command::Serve(a) => a.config_path(),
            _ => None,
        }
    }

    /// The log filter to use when `$RUST_LOG` is unset: `run` streams a turn to
    /// stdout, so only errors belong on stderr beside it.
    pub fn default_log(&self) -> &'static str {
        match self {
            Command::Run(_) => "error",
            _ => "info",
        }
    }
}

#[derive(Subcommand)]
pub enum Command {
    /// Run a local Substructure server.
    Serve(local::ServeArgs),
    /// Run a single turn against a worker in-process and stream events, then exit.
    /// For local development and testing example agents.
    Run(run::RunArgs),
    /// Authenticate via the OAuth device flow and persist the token locally.
    /// Targets the server the environment file names, so `subs login -c
    /// subs.prod.toml` logs in to a self-hosted deployment.
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
    /// Show the currently logged-in user and default org/project.
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
    /// Manage projects.
    #[command(after_help = GLOBAL_FLAGS_HELP)]
    Projects {
        #[command(subcommand)]
        command: cloud::projects::ProjectsCommand,
    },
    /// Inspect the agents a project declares, and read or rotate their signing
    /// secrets.
    #[command(after_help = GLOBAL_FLAGS_HELP)]
    Agents {
        #[command(subcommand)]
        command: cloud::agents::AgentsCommand,
    },
    /// Bind the keys behind a project's `[llm.*]` blocks.
    #[command(after_help = GLOBAL_FLAGS_HELP)]
    Llm {
        #[command(subcommand)]
        command: cloud::llm::LlmCommand,
    },
    /// Manage API keys for a project.
    #[command(after_help = GLOBAL_FLAGS_HELP)]
    Keys {
        #[command(subcommand)]
        command: cloud::keys::KeysCommand,
    },
    /// Inspect sessions (list, stream events). Reads the deployment the
    /// `[remote]` names, or the database an engine here writes.
    #[command(after_help = GLOBAL_FLAGS_HELP)]
    Sessions {
        #[command(subcommand)]
        command: sessions::SessionsCommand,
    },
    /// Open a project's admin page in your browser.
    Open {
        project_id: Option<String>,
        /// Print the URL instead of opening a browser.
        #[arg(long)]
        no_browser: bool,
        #[command(flatten)]
        scope: ProjectScope,
    },
    /// Link the current directory to an org (and project) by writing a
    /// `substructure.toml`, so commands run from this tree pick them up automatically.
    Link(cloud::link::LinkCommand),
    /// Push the environment file to the deployment, creating the project it
    /// describes if nothing is pinned yet.
    #[command(after_help = GLOBAL_FLAGS_HELP)]
    Apply(cloud::apply::ApplyCommand),
    /// Inspect a project's configuration history.
    #[command(after_help = GLOBAL_FLAGS_HELP)]
    Config {
        #[command(subcommand)]
        command: cloud::apply::ConfigCommand,
    },
    /// Authorize the MCP connections this project declares.
    Mcp {
        #[command(subcommand)]
        command: mcp::McpCommand,
    },
    /// Connect a Slack workspace to the deployment that answers for it.
    #[command(after_help = GLOBAL_FLAGS_HELP)]
    Slack {
        #[command(subcommand)]
        command: cloud::slack::SlackCommand,
    },
    /// Show what this project still needs before it works: the keys, the
    /// consents, and the workspaces nobody has set up yet.
    #[command(after_help = GLOBAL_FLAGS_HELP)]
    Doctor {
        #[command(flatten)]
        scope: ProjectScope,
    },
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
            cloud::login::run(&globals, no_browser).await
        }
        Command::Logout { globals } => cloud::logout::run(&globals).await,
        Command::Whoami { globals } => cloud::whoami::run(globals).await,
        Command::Orgs { command } => cloud::orgs::run(command).await,
        Command::Projects { command } => cloud::projects::run(command).await,
        Command::Agents { command } => cloud::agents::run(command).await,
        Command::Llm { command } => cloud::llm::run(command).await,
        Command::Keys { command } => cloud::keys::run(command).await,
        Command::Sessions { command } => sessions::run(command).await,
        Command::Open {
            project_id,
            no_browser,
            scope,
        } => cloud::open::run(project_id, no_browser, scope).await,
        Command::Link(cmd) => cloud::link::run(cmd).await,
        Command::Apply(cmd) => cloud::apply::run(cmd).await,
        Command::Config { command } => cloud::apply::config(command).await,
        Command::Mcp { command } => mcp::run(command).await,
        Command::Slack { command } => cloud::slack::run(command).await,
        Command::Doctor { scope } => doctor::run(scope).await,
    }
}

// Leaf command path for telemetry headers (e.g. "webhook set"). Kept manually
// in sync with the enum rather than scraping argv so we never leak secrets
// users pass on the command line.
fn command_path(cmd: &Command) -> &'static str {
    use cloud::agents::AgentsCommand;
    use cloud::llm::LlmCommand;
    use cloud::{keys::KeysCommand, orgs::OrgsCommand, projects::ProjectsCommand};
    use sessions::SessionsCommand;
    match cmd {
        Command::Serve(_) => "serve",
        Command::Run(_) => "run",
        Command::Login { .. } => "login",
        Command::Logout { .. } => "logout",
        Command::Whoami { .. } => "whoami",
        Command::Open { .. } => "open",
        Command::Link(_) => "link",
        Command::Apply(_) => "apply",
        Command::Config { command } => match command {
            cloud::apply::ConfigCommand::Log { .. } => "config log",
        },
        Command::Mcp { command } => match command {
            mcp::McpCommand::Login { .. } => "mcp login",
            mcp::McpCommand::SetToken { .. } => "mcp set-token",
            mcp::McpCommand::Logout { .. } => "mcp logout",
            mcp::McpCommand::List { .. } => "mcp list",
        },
        Command::Slack { command } => match command {
            cloud::slack::SlackCommand::Connect { .. } => "slack connect",
        },
        Command::Doctor { .. } => "doctor",
        Command::Orgs { command } => match command {
            OrgsCommand::List { .. } => "orgs list",
        },
        Command::Projects { command } => match command {
            ProjectsCommand::List { .. } => "projects list",
            ProjectsCommand::Show { .. } => "projects show",
            ProjectsCommand::Delete { .. } => "projects delete",
        },
        Command::Agents { command } => match command {
            AgentsCommand::List { .. } => "agents list",
            AgentsCommand::Show { .. } => "agents show",
            AgentsCommand::Secret { .. } => "agents secret",
            AgentsCommand::RotateSecret { .. } => "agents rotate-secret",
        },
        Command::Llm { command } => match command {
            LlmCommand::List { .. } => "llm list",
            LlmCommand::SetKey { .. } => "llm set-key",
            LlmCommand::DeleteKey { .. } => "llm delete-key",
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
    }
}
