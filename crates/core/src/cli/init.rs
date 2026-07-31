use std::fs;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context as _, Result};
use dialoguer::{theme::ColorfulTheme, Confirm, Input, Select};

use crate::cli::env::ProviderKind;
use crate::cli::env_value;

use super::cloud::project_config::FILENAME;

#[derive(Debug, clap::Args)]
pub struct InitCommand {
    /// What the file is for [default: engine]. `deployment` and `both` add the
    /// `[deployment]` section, which `subs link` and `subs apply` would
    /// otherwise write themselves the first time you deploy.
    #[arg(value_enum)]
    pub role: Option<Role>,
    /// Where to write it [default: substructure.toml].
    pub path: Option<PathBuf>,
    /// Overwrite an existing file.
    #[arg(long)]
    pub force: bool,
    /// Never prompt: write the starter file.
    #[arg(long, short = 'n')]
    pub no_interaction: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Role {
    /// An engine in this process over a SQLite file: `subs run`, `subs serve`.
    #[value(alias = "local")]
    Engine,
    /// A server speaking `/api/v1`: `subs apply`, `subs sessions`.
    #[value(alias = "remote")]
    Deployment,
    /// Serve it here and administer it from the same file.
    Both,
}

impl Role {
    fn engine(self) -> bool {
        matches!(self, Role::Engine | Role::Both)
    }

    fn deployment(self) -> bool {
        matches!(self, Role::Deployment | Role::Both)
    }
}

/// The answers a file is rendered from.
///
/// It holds only what `init` cannot guess: which provider (which decides the
/// key you have to export) and what the agent is called (which decides what you
/// type after `--agent`). Everything else the starter needs has a defensible
/// default, a flag, or is a one-line edit, and a question whose answer barely
/// changes the file is a question worth not asking.
///
/// There is no worker question either: an agent the file declares runs on the
/// engine, and attaching a worker is a later edit to one `[agent.<id>]` — the
/// same kind of edit as adding a tool or a second agent.
#[derive(Debug, Clone, PartialEq)]
struct Plan {
    /// What `subs apply` creates the app as. An engine-only file has nobody to
    /// name it to.
    name: Option<String>,
    agent: AgentPlan,
    /// Whether the file describes an engine run here — the `[server]` block.
    engine: bool,
    deployment: Option<DeploymentPlan>,
}

/// The one agent a starter file declares, and the block it runs on.
#[derive(Debug, Clone, PartialEq)]
struct AgentPlan {
    id: String,
    llm: String,
    provider: ProviderKind,
    model: String,
}

#[derive(Debug, Clone, PartialEq)]
struct DeploymentPlan {
    /// Absent means the hosted default, which the file need not restate.
    url: Option<String>,
}

impl Default for AgentPlan {
    fn default() -> Self {
        Self {
            id: "assistant".into(),
            llm: "claude".into(),
            provider: ProviderKind::Anthropic,
            model: ProviderKind::Anthropic.default_model().into(),
        }
    }
}

impl Plan {
    fn starter(role: Role) -> Self {
        Self {
            name: role.deployment().then(default_name).flatten(),
            agent: AgentPlan::default(),
            engine: role.engine(),
            deployment: role.deployment().then_some(DeploymentPlan { url: None }),
        }
    }
}

/// The block name for a provider: what a reader would have called it.
fn default_llm_name(provider: ProviderKind) -> &'static str {
    match provider {
        ProviderKind::Anthropic => "claude",
        ProviderKind::Openai => "openai",
        ProviderKind::Openrouter => "openrouter",
        ProviderKind::Worker => "worker",
    }
}

/// The directory's own name, which is what a project is usually called.
fn default_name() -> Option<String> {
    std::env::current_dir()
        .ok()?
        .file_name()?
        .to_str()
        .map(str::to_string)
}

/// Starters, not schemas: what a first run needs, plus the settings most
/// likely to be changed next, so the file works before it is edited and
/// `docs/160-cli.md` holds the rest. Scalars lead, then a section per group —
/// a top-level key written after a section would belong to it.
fn render(p: &Plan) -> String {
    let mut s = String::new();
    if let Some(name) = &p.name {
        s.push_str(&format!("name = \"{name}\"\n\n"));
    }
    s.push_str(&format!(
        "[llm.{}]\ntype = \"{}\"\n",
        p.agent.llm,
        p.agent.provider.as_str()
    ));
    s.push_str(&format!(
        "\n[agent.{}]\nllm = \"{}\"\nmodel = \"{}\"\n",
        p.agent.id, p.agent.llm, p.agent.model
    ));

    // Written rather than asked, and written rather than left to the default:
    // `auth` defaults to *on*, so an unstated `[server]` would make a first
    // `subs serve` demand tokens. Stating it keeps the insecure setting visible
    // in the file instead of silent.
    if p.engine {
        s.push_str(
            "\n[server]\nhost = \"127.0.0.1\"\nport = 8080\n\
             auth = false   # no client or worker auth. Keep this off the network\n",
        );
    }

    if let Some(d) = &p.deployment {
        s.push_str("\n[deployment]\n");
        match &d.url {
            Some(url) => s.push_str(&format!("url = \"{url}\"\n")),
            // The section is what says a deployment is in play; `subs link`
            // fills in the pins.
            None => s.push_str("# the hosted cloud. `subs link` pins an org and project\n"),
        }
    }
    s
}

pub fn run(cmd: InitCommand) -> Result<()> {
    let interactive = !cmd.no_interaction && std::io::stdin().is_terminal();
    if !interactive {
        return starter(cmd);
    }
    match wizard(cmd) {
        Ok(()) => Ok(()),
        Err(e) if cancelled(&e) => {
            println!("\nCancelled. Nothing was written.");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// The scripted path: no questions, so the role has to be on the command line
/// and the answers are the defaults.
fn starter(cmd: InitCommand) -> Result<()> {
    let role = cmd.role.unwrap_or(Role::Engine);
    let path = cmd.path.unwrap_or_else(|| PathBuf::from(FILENAME));
    if path.exists() && !cmd.force {
        bail!(
            "{} already exists. Pass --force to overwrite it.",
            path.display()
        );
    }
    let plan = Plan::starter(role);
    write_file(&path, &render(&plan))?;
    println!("Wrote {}", path.display());
    next_steps(&plan);
    Ok(())
}

fn write_file(path: &Path, body: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
    }
    fs::write(path, body).with_context(|| format!("writing {}", path.display()))
}

// ---------------------------------------------------------------------------
// The wizard
// ---------------------------------------------------------------------------

fn theme() -> ColorfulTheme {
    ColorfulTheme::default()
}

/// A prompt hides the cursor, and Ctrl-C inside one re-raises SIGINT rather
/// than returning, so the default handler would end the process with the
/// cursor still hidden and the terminal needing a `reset`. Show it again on
/// the way out. Best-effort: a wizard is not worth failing to start over a
/// handler that could not be installed.
fn restore_cursor_on_interrupt() {
    let _ = ctrlc::set_handler(|| {
        eprint!("\x1b[?25h");
        let _ = std::io::Write::flush(&mut std::io::stderr());
        std::process::exit(130);
    });
}

/// Ctrl-C at a prompt is a decision, not a failure: `run` turns it into a
/// clean exit rather than an error and a backtrace.
fn cancelled(e: &anyhow::Error) -> bool {
    matches!(
        e.downcast_ref::<dialoguer::Error>(),
        Some(dialoguer::Error::IO(io)) if io.kind() == std::io::ErrorKind::Interrupted
    )
}

/// What each question is for, printed above it. The file is short enough that
/// a first-time reader can be told the whole shape of it as they fill it in.
fn explain(text: &str) {
    println!("\n{text}");
}

fn wizard(cmd: InitCommand) -> Result<()> {
    restore_cursor_on_interrupt();
    println!("substructure.toml declares your agents.");
    println!("Press Ctrl-C to stop. Nothing is written until you confirm.");

    // Never asked. A file gains `[deployment]` the moment you deploy — `subs
    // link` and `subs apply` both write the section themselves — so making it
    // the first question would demand an architecture decision the file does
    // not force, before the user has an agent to decide it about.
    let role = cmd.role.unwrap_or(Role::Engine);

    let path = match cmd.path {
        Some(p) => p,
        None => PathBuf::from(FILENAME),
    };
    if path.exists() && !cmd.force {
        explain(&format!("{} already exists.", path.display()));
        let overwrite = Confirm::with_theme(&theme())
            .with_prompt("Overwrite it?")
            .default(false)
            .interact()?;
        if !overwrite {
            println!("Left {} as it is.", path.display());
            return Ok(());
        }
    }

    let plan = ask(role)?;
    let body = render(&plan);

    println!("\n{}", path.display());
    println!("{}", "─".repeat(60));
    print!("{body}");
    println!("{}", "─".repeat(60));

    if !Confirm::with_theme(&theme())
        .with_prompt("Write it?")
        .default(true)
        .interact()?
    {
        println!("Nothing was written.");
        return Ok(());
    }
    write_file(&path, &body)?;
    println!("\nWrote {}", path.display());
    next_steps(&plan);
    Ok(())
}

fn ask(role: Role) -> Result<Plan> {
    // The agent leads: it is the whole point of the file, and the only part of
    // it whose answers cannot be guessed.
    let agent = ask_agent()?;

    let name = match role.deployment() {
        false => None,
        true => {
            explain(
                "The app's name. `subs apply` creates the app from it and renames when it\n\
                 changes, so the file stays the source of truth.",
            );
            let entered: String = Input::with_theme(&theme())
                .with_prompt("Project name")
                .default(default_name().unwrap_or_else(|| "my-project".into()))
                .interact_text()?;
            Some(entered)
        }
    };

    Ok(Plan {
        name,
        agent,
        engine: role.engine(),
        deployment: role.deployment().then(ask_deployment).transpose()?,
    })
}

/// The one thing every file needs: an agent, and the model it speaks to. The
/// engine decides for it until you attach a worker.
fn ask_agent() -> Result<AgentPlan> {
    let d = AgentPlan::default();

    // The provider first: it is the answer that decides which key you have to
    // export, and it supplies the default for the model question after it.
    explain(
        "Which LLM your agent talks to. The engine makes the call and reads the key\n\
         from the environment — the file names the variable and never holds the secret.",
    );
    let items: Vec<String> = ProviderKind::VENDORS
        .iter()
        .map(|p| {
            format!(
                "{}  ({})",
                p.as_str(),
                p.default_api_key_env().unwrap_or_default()
            )
        })
        .collect();
    let pick = Select::with_theme(&theme())
        .with_prompt("LLM provider")
        .items(&items)
        .default(0)
        .interact()?;
    let provider = ProviderKind::VENDORS[pick];
    if let Some(var) = provider.default_api_key_env() {
        if env_value(var).is_some() {
            println!("  ✓ {var} is set.");
        } else {
            println!(
                "  {var} is not set yet — get a key at {}",
                provider.console_url().unwrap_or_default()
            );
        }
    }

    explain(
        "Agents are named. Clients route to one by id, and `subs run --agent <id>`\n\
         picks which to run. The engine decides every step of its turns until you\n\
         give it a `worker` URL.",
    );
    let id: String = Input::with_theme(&theme())
        .with_prompt("Agent id")
        .default(d.id.clone())
        .interact_text()?;

    explain("Which model, from that provider.");
    let model: String = Input::with_theme(&theme())
        .with_prompt("Model")
        .default(provider.default_model().to_string())
        .interact_text()?;

    Ok(AgentPlan {
        id,
        llm: default_llm_name(provider).to_string(),
        provider,
        model,
    })
}

fn ask_deployment() -> Result<DeploymentPlan> {
    const HOSTED: &str = "https://api.substructure.ai";

    explain(
        "Which server. The hosted cloud is the default; a self-hosted deployment or\n\
         someone else's `subs serve` is a URL here. `subs login -c this-file` signs\n\
         in to whichever it is.",
    );
    let url: String = Input::with_theme(&theme())
        .with_prompt("API URL")
        .default(HOSTED.to_string())
        .validate_with(url_is_parseable)
        .interact_text()?;

    explain("The org and app are pinned by `subs link` or `subs apply` after you log in.");

    Ok(DeploymentPlan {
        url: (url != HOSTED).then_some(url),
    })
}

/// A typo caught at the prompt beats one caught at the first request.
#[allow(clippy::ptr_arg)] // dialoguer's validator takes &String.
fn url_is_parseable(input: &String) -> Result<(), String> {
    reqwest::Url::parse(input)
        .map(|_| ())
        .map_err(|e| format!("not a URL: {e}"))
}

fn next_steps(plan: &Plan) {
    println!("\nNext:");
    if let Some(var) = plan.agent.provider.default_api_key_env() {
        if env_value(var).is_none() {
            println!(
                "  export {var}=...   # {}",
                plan.agent.provider.console_url().unwrap_or_default()
            );
        }
    }
    if plan.engine {
        println!("  subs run --agent {} -o pretty 'hello'", plan.agent.id);
        println!("  subs serve   # the same engine behind an HTTP API");
    }
    if plan.deployment.is_some() {
        println!("  subs login");
        println!("  subs apply   # create the app this file describes, and push it");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::cloud::project_config::{self, ProjectConfig};

    fn tmpdir() -> PathBuf {
        // Timestamp alone collides across parallel tests; the counter disambiguates.
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("subs-init-test-{nanos}-{seq}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Every test drives the scripted path: a prompt in a test would block on
    /// whatever stdin happens to be.
    fn init(role: Role, path: PathBuf, force: bool) -> InitCommand {
        InitCommand {
            role: Some(role),
            path: Some(path),
            force,
            no_interaction: true,
        }
    }

    /// A rendered file has to survive the same parse a hand-written one does —
    /// `deny_unknown_fields` makes a misspelling an error, so a renderer can rot.
    fn parse(body: &str) -> ProjectConfig {
        let path = tmpdir().join("substructure.toml");
        fs::write(&path, body).unwrap();
        project_config::load_explicit(&path).unwrap().config
    }

    #[test]
    fn each_starter_declares_the_roles_it_was_asked_for() {
        let dir = tmpdir();
        for role in [Role::Engine, Role::Deployment, Role::Both] {
            let path = dir.join(format!("{role:?}.toml"));
            run(init(role, path.clone(), false)).unwrap();

            let cfg = project_config::load_explicit(&path).unwrap().config;
            assert_eq!(
                cfg.deployment.is_some(),
                role.deployment(),
                "{role:?} rendered {}",
                fs::read_to_string(&path).unwrap()
            );
            // An engine's settings are what `subs serve` reads; a
            // deployment-only file leaves them to the defaults.
            assert_eq!(cfg.server.is_some(), role.engine(), "{role:?}");
            // `[run]` pins which agent a bare `subs run` drives. A starter has
            // one agent and names it on the command line, so it is not written.
            assert!(cfg.run.is_none(), "{role:?}");
        }
    }

    /// The wizard's answers are the same file the starter is, so anything it
    /// can produce has to parse too — including the sections and the spellings
    /// only an answer reaches.
    #[test]
    fn every_answer_renders_a_file_that_parses() {
        let deployments = [
            None,
            Some(DeploymentPlan { url: None }),
            Some(DeploymentPlan {
                url: Some("https://subs.internal".into()),
            }),
        ];
        for provider in ProviderKind::VENDORS {
            for deployment in &deployments {
                let plan = Plan {
                    name: Some("support-bot".into()),
                    agent: AgentPlan {
                        id: "my-agent".into(),
                        llm: default_llm_name(provider).into(),
                        provider,
                        model: provider.default_model().into(),
                    },
                    engine: true,
                    deployment: deployment.clone(),
                };
                let body = render(&plan);
                let cfg = parse(&body);
                assert_eq!(cfg.name.as_deref(), Some("support-bot"));
                assert_eq!(cfg.agent_ids(), ["my-agent"], "{body}");
                assert_eq!(cfg.llm[default_llm_name(provider)].kind, provider, "{body}");
                assert_eq!(
                    cfg.deployment_url(),
                    deployment.as_ref().and_then(|d| d.url.as_deref()),
                    "{body}"
                );
            }
        }

        // A deployment on its own: no engine sections to carry the scalars.
        let cfg = parse(&render(&Plan {
            name: None,
            agent: AgentPlan::default(),
            engine: false,
            deployment: Some(DeploymentPlan { url: None }),
        }));
        assert!(cfg.deployment.is_some());
        assert!(cfg.run.is_none() && cfg.server.is_none());
    }

    /// The starter is the engine-hosted minimal file: one llm block, one agent,
    /// no worker. Attaching a worker is a later edit, like every other rung.
    #[test]
    fn the_starter_declares_an_engine_hosted_agent() {
        let cfg = parse(&render(&Plan::starter(Role::Engine)));
        let agent = cfg.agent.get("assistant").expect("one agent declared");
        assert_eq!(agent.llm.as_deref(), Some("claude"));
        assert!(agent.worker.is_none(), "the engine decides for it");
        assert_eq!(cfg.llm["claude"].kind, ProviderKind::Anthropic);
    }

    /// `auth` defaults to on, so an unwritten `[server]` would make a first
    /// `subs serve` demand tokens. The starter states it instead.
    #[test]
    fn the_starter_states_the_auth_it_relies_on() {
        let cfg = parse(&render(&Plan::starter(Role::Engine)));
        assert!(!cfg.server_auth());
        assert!(cfg.slack_agent().is_none(), "init declares no slack bot");
    }

    #[test]
    fn an_existing_file_is_not_overwritten_without_force() {
        let path = tmpdir().join("substructure.toml");
        fs::write(&path, "db = \"mine.db\"\n").unwrap();

        let err = run(init(Role::Deployment, path.clone(), false))
            .unwrap_err()
            .to_string();
        assert!(err.contains("already exists"), "got {err}");
        assert!(fs::read_to_string(&path).unwrap().contains("mine.db"));

        run(init(Role::Deployment, path.clone(), true)).unwrap();
        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            render(&Plan::starter(Role::Deployment))
        );
    }

    /// The path is taken as given, so `subs init engine api/substructure.toml`
    /// does not fail on a directory that does not exist yet.
    #[test]
    fn a_missing_parent_directory_is_created() {
        let path = tmpdir().join("nested/deeper/substructure.toml");
        run(init(Role::Engine, path.clone(), false)).unwrap();
        assert!(path.exists());
    }

    /// The role is never a question — a file gains `[deployment]` when you
    /// deploy — so omitting it writes the engine file rather than failing.
    #[test]
    fn a_missing_role_writes_the_engine_file() {
        let path = tmpdir().join("substructure.toml");
        run(InitCommand {
            role: None,
            path: Some(path.clone()),
            force: false,
            no_interaction: true,
        })
        .unwrap();

        let cfg = project_config::load_explicit(&path).unwrap().config;
        assert_eq!(cfg.agent_ids(), ["assistant"]);
        assert!(cfg.server.is_some(), "an engine to run it");
        assert!(cfg.deployment.is_none(), "nothing to administer yet");
    }
}
