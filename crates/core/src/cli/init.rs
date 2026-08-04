use std::fs;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context as _, Result};
use dialoguer::{theme::ColorfulTheme, Confirm, Input, MultiSelect, Select};

use crate::cli::env::ProviderKind;
use crate::manifest::{check_id, check_url};

use super::cloud::project_config::FILENAME;
// The Slack app a local engine needs is built where `subs slack connect` says
// it is, so both paths name one page.
use super::cloud::slack::{SLACK_DOCS, SLACK_NEW_APP};

#[derive(Debug, clap::Args)]
pub struct InitCommand {
    /// Where to write it [default: substructure.toml].
    pub path: Option<PathBuf>,
    /// Overwrite an existing file.
    #[arg(long)]
    pub force: bool,
    /// Never prompt: write the starter file.
    #[arg(long, short = 'n')]
    pub no_interaction: bool,
}

/// The name to hold a first project under. A placeholder rather than the
/// directory's own name: a directory is called all sorts of things a project is
/// not — `src`, `tmp`, the CLI's own name — and a default you have to read
/// before you can accept it is not much of a default.
const DEFAULT_NAME: &str = "my-agent";

/// The hosted cloud. Written out rather than left to the default it happens to
/// equal: a file that says where it deploys can be read, and a reader of
/// somebody else's checkout should not have to know what an absent `url` means.
const HOSTED_URL: &str = "https://api.substructure.ai";

/// Where the agent runs. The last question, and the only one that is about
/// running rather than about the agent.
///
/// It is asked last because nothing before it changes with the answer, and it
/// is asked at all because it decides which single path the next steps print.
/// Self-hosting is deliberately not a third answer: it is `[remote].url`
/// pointing at your own `subs serve`, a one-line edit that a first file does
/// not have to decide about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Place {
    Cloud,
    Local,
}

/// The answers a file is rendered from.
///
/// One topic per field, in the order they are asked: what the project is
/// called, the agent and the model it speaks to, what that agent can reach,
/// where it answers, and where it runs. Everything else a starter needs has a
/// defensible default or is a one-line edit, and a question whose answer barely
/// changes the file is a question worth not asking.
///
/// A cloud answer writes `[remote]` rather than leaving it to the first
/// `subs apply`, because the section is also what sends `subs mcp login` to the
/// deployment: without it, authorizing a connection before deploying stores the
/// credential in a database here — the wrong place, silently. With it, the same
/// mistake is an error that names the login you have not done yet. A local
/// answer writes no `[remote]` for the same reason read the other way: its
/// connections really are authorized here.
///
/// There is no worker question either: an agent the file declares runs on the
/// engine, and attaching a worker is a later edit to one `[agent.<id>]` — the
/// same kind of edit as adding a tool or a second agent.
#[derive(Debug, Clone, PartialEq)]
struct Plan {
    name: String,
    agent: AgentPlan,
    /// The connections the agent draws tools from, in the order they were
    /// picked. Declared *and* named by the agent: a `[mcp.<id>]` nobody
    /// references reaches nothing.
    mcp: Vec<McpPlan>,
    /// Where the bot answers, or `None` where there is no bot. Never a section
    /// that routes nowhere: answering no to both leaves `[slack]` unwritten.
    slack: Option<SlackPlan>,
    place: Place,
}

/// The two Slack questions, asked apart because they have very different blast
/// radii: a DM is one person talking to the agent, a channel is a room.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SlackPlan {
    dm: bool,
    mentions: bool,
}

/// One `[mcp.<id>]`: where the server is and what the agent calls it. No
/// `auth` — every server the wizard offers authorizes with `subs mcp login`,
/// and a URL you bring is asked for as one too.
#[derive(Debug, Clone, PartialEq)]
struct McpPlan {
    id: String,
    url: String,
}

/// The remote MCP servers a first file is likely to name.
///
/// A handful, not a registry: the list exists so the common case is a
/// keystroke rather than a URL to go and look up, and `other` covers the rest.
/// Every entry is OAuth — one `subs mcp login <id>` and nothing to export —
/// because a token-backed server needs an `auth` line naming a variable, and
/// that is the engine's half alone: `subs apply` refuses a connection carrying
/// one. Offering it here would hand a deployment file a section it cannot push.
const CATALOG: &[Server] = &[
    Server::new("sentry", "https://mcp.sentry.dev/mcp"),
    Server::new("linear", "https://mcp.linear.app/mcp"),
    Server::new("notion", "https://mcp.notion.com/mcp"),
    Server::new("stripe", "https://mcp.stripe.com"),
];

struct Server {
    id: &'static str,
    url: &'static str,
}

impl Server {
    const fn new(id: &'static str, url: &'static str) -> Self {
        Self { id, url }
    }

    fn plan(&self) -> McpPlan {
        McpPlan {
            id: self.id.into(),
            url: self.url.into(),
        }
    }
}

/// The one agent a starter file declares, and the block it runs on.
#[derive(Debug, Clone, PartialEq)]
struct AgentPlan {
    id: String,
    llm: String,
    provider: ProviderKind,
    model: String,
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
    /// Every answer at its default. `Local` rather than `Cloud`, because a
    /// path that cannot ask cannot sign anybody in either: the local file is
    /// the one that needs nothing but a key to run.
    fn starter() -> Self {
        Self {
            name: DEFAULT_NAME.into(),
            agent: AgentPlan::default(),
            mcp: Vec::new(),
            slack: None,
            place: Place::Local,
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

/// Starters, not schemas: what a first run needs, plus the settings most
/// likely to be changed next, so the file works before it is edited and
/// `docs/160-cli.md` holds the rest. Scalars lead, then a section per group —
/// a top-level key written after a section would belong to it.
fn render(p: &Plan) -> String {
    let mut s = String::new();
    s.push_str(&format!("name = \"{}\"\n\n", p.name));
    s.push_str(&format!(
        "[llm.{}]\ntype = \"{}\"\n",
        p.agent.llm,
        p.agent.provider.as_str()
    ));
    s.push_str(&format!(
        "\n[agent.{}]\nllm = \"{}\"\nmodel = \"{}\"\n",
        p.agent.id, p.agent.llm, p.agent.model
    ));
    // The reference is what puts the connection's tools in front of the model;
    // the section below only says where the server is.
    if !p.mcp.is_empty() {
        let refs: Vec<String> = p.mcp.iter().map(|m| format!("\"{}\"", m.id)).collect();
        s.push_str(&format!("mcp = [{}]\n", refs.join(", ")));
    }

    for m in &p.mcp {
        s.push_str(&format!(
            "\n[mcp.{}]\nurl = \"{}\"   # authorize with `subs mcp login {}`\n",
            m.id, m.url, m.id
        ));
    }

    // A key per answer, so the section says exactly what was agreed to. The
    // unwritten half is a later edit rather than a line to go and delete.
    if let Some(slack) = &p.slack {
        let mut keys = Vec::new();
        if slack.dm {
            keys.push((format!("dm = \"{}\"", p.agent.id), "direct messages"));
        }
        if slack.mentions {
            keys.push((
                format!("mentions = \"{}\"", p.agent.id),
                "@mentions, in any channel it is invited to",
            ));
        }
        // Comments line up whichever keys are present, and an id of any length
        // keeps them lined up.
        let w = keys.iter().map(|(key, _)| key.len()).max().unwrap_or(0);
        s.push_str("\n[slack]\n");
        for (key, what) in keys {
            s.push_str(&format!("{key:<w$}   # {what}\n"));
        }
    }

    // Written rather than asked, and written rather than left to the default:
    // `auth` defaults to *on*, so an unstated `[serve]` would make a first
    // `subs serve` demand tokens. Stating it keeps the insecure setting visible
    // in the file instead of silent — and it is only defensible here, on a
    // loopback engine nothing else can reach, which is why a cloud file has no
    // `[serve]` at all rather than one that says something reassuring.
    match p.place {
        Place::Local => s.push_str(
            "\n[serve]\nhost = \"127.0.0.1\"\nport = 8080\n\
             auth = false   # no client or worker auth. Keep this off the network\n",
        ),
        Place::Cloud => s.push_str(&format!(
            "\n[remote]\nurl = \"{HOSTED_URL}\"   # `subs apply` pins the org and project\n"
        )),
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

/// The scripted path: no questions, so every answer is the default. An engine
/// here is the one that needs nothing else to run, and a file gains
/// `[remote]` from `subs link` or the first `subs apply` anyway.
fn starter(cmd: InitCommand) -> Result<()> {
    let path = cmd.path.unwrap_or_else(|| PathBuf::from(FILENAME));
    if path.exists() && !cmd.force {
        bail!(
            "{} already exists. Pass --force to overwrite it.",
            path.display()
        );
    }
    let plan = Plan::starter();
    write_file(&path, &render(&plan))?;
    println!("Wrote {}", path.display());
    next_steps(&plan, &path);
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
    println!("This creates substructure.toml, the file that describes your agent.");
    println!("Press Ctrl-C to stop. Nothing is written until you confirm.");

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

    let plan = ask()?;
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
    next_steps(&plan, &path);
    Ok(())
}

fn ask() -> Result<Plan> {
    // The name first: it is the one question with an answer already in hand,
    // so the wizard opens by confirming something rather than by asking for a
    // decision.
    let name = ask_name()?;
    let agent = ask_agent()?;
    // What it can reach and where it answers are both about the agent just
    // named, so they are asked while it is still the subject.
    let mcp = ask_mcp()?;
    let slack = ask_slack(&agent.id)?;
    // Where it runs comes last: nothing above changes with the answer, so
    // asking it first would spend the first question on the one topic that is
    // about running rather than about the agent.
    let place = ask_place(slack.is_some())?;

    Ok(Plan {
        name,
        agent,
        mcp,
        slack,
        place,
    })
}

/// What the project is called. `subs apply` creates the cloud project from it,
/// so answering it here is what makes deploying a single command later.
fn ask_name() -> Result<String> {
    explain("What do you want to call this project? You can rename it later.");
    Ok(Input::with_theme(&theme())
        .with_prompt("Project name")
        .default(DEFAULT_NAME.to_string())
        .interact_text()?)
}

/// The one thing every file needs: an agent, and the model it speaks to. The
/// engine decides for it until you attach a worker.
///
/// No key is mentioned in any of it. Where the key goes depends on where the
/// agent runs, which is not known yet, so both the variable and the check that
/// it is set belong to the next steps instead — where the answer is in hand.
fn ask_agent() -> Result<AgentPlan> {
    let d = AgentPlan::default();

    // The provider first: it supplies the default for the model question.
    explain("Which provider do you want to use? You can change this later, or use several.");
    let items: Vec<&str> = ProviderKind::VENDORS.iter().map(|p| p.as_str()).collect();
    let pick = Select::with_theme(&theme())
        .with_prompt("Provider")
        .items(&items)
        .default(0)
        .interact()?;
    let provider = ProviderKind::VENDORS[pick];

    println!();
    let model: String = Input::with_theme(&theme())
        .with_prompt("Model")
        .default(provider.default_model().to_string())
        .interact_text()?;

    explain("What id do you want for your agent? You use it to talk to your agent.");
    let id: String = Input::with_theme(&theme())
        .with_prompt("Agent id")
        .default(d.id.clone())
        .validate_with(|input: &String| check_id(input).map_err(|e| e.to_string()))
        .interact_text()?;

    Ok(AgentPlan {
        id,
        llm: default_llm_name(provider).to_string(),
        provider,
        model,
    })
}

/// What the agent can reach. Selecting nothing is the common answer and costs
/// one keystroke, so the question is cheap enough to ask everyone.
fn ask_mcp() -> Result<Vec<McpPlan>> {
    explain("Which MCP servers do you want to connect? Space selects, Enter accepts.");

    let mut items: Vec<String> = CATALOG
        .iter()
        .map(|s| format!("{:<7}  {}", s.id, s.url))
        .collect();
    items.push(format!("{:<7}  a URL you have", "other"));
    let picks = MultiSelect::with_theme(&theme())
        .with_prompt("MCP servers")
        .items(&items)
        .interact()?;

    let mut chosen: Vec<McpPlan> = picks
        .iter()
        .filter_map(|i| CATALOG.get(*i))
        .map(Server::plan)
        .collect();
    if picks.contains(&CATALOG.len()) {
        chosen.push(ask_custom_mcp(&chosen)?);
    }
    Ok(chosen)
}

/// A server the list does not carry. The URL leads: it is what the id is
/// guessed from, and the one part nobody can default.
fn ask_custom_mcp(taken: &[McpPlan]) -> Result<McpPlan> {
    println!();
    let url: String = Input::with_theme(&theme())
        .with_prompt("MCP server URL")
        .validate_with(|input: &String| check_url(input).map_err(|e| e.to_string()))
        .interact_text()?;

    let used: Vec<String> = taken.iter().map(|m| m.id.clone()).collect();
    let id: String = Input::with_theme(&theme())
        .with_prompt("Call it")
        .default(id_from_url(&url))
        .validate_with(move |input: &String| match used.contains(input) {
            true => Err(format!("`{input}` is already declared")),
            false => check_id(input).map_err(|e| e.to_string()),
        })
        .interact_text()?;

    Ok(McpPlan { id, url })
}

/// A first guess at what to call a server, from the host that serves it:
/// `https://mcp.sentry.dev/mcp` is `sentry`. The labels that say *how* it is
/// served say nothing about which service it is.
fn id_from_url(url: &str) -> String {
    let host = reqwest::Url::parse(url)
        .ok()
        .and_then(|u| u.host_str().map(str::to_string))
        .unwrap_or_default();
    host.split('.')
        .find(|label| !matches!(*label, "mcp" | "api" | "www" | "docs" | "server"))
        // A name, not an address: `127.0.0.1` has no service in it to borrow.
        .filter(|label| label.starts_with(|c: char| c.is_ascii_alphabetic()))
        .filter(|label| check_id(label).is_ok())
        .unwrap_or("connection")
        .to_string()
}

/// Where the bot answers, asked as a tree: one question for everybody, and the
/// two that widen it only for whoever wants a bot at all.
///
/// A DM and a channel are separately answered because they are separately
/// consequential — a DM is one person, a channel is a room — and each defaults
/// to what it would take to be surprised by: DMs on, channels off.
fn ask_slack(agent: &str) -> Result<Option<SlackPlan>> {
    explain("Do you want a Slack bot?");
    if !Confirm::with_theme(&theme())
        .with_prompt("Slack bot?")
        .default(false)
        .interact()?
    {
        return Ok(None);
    }

    let dm = Confirm::with_theme(&theme())
        .with_prompt(format!("  Should `{agent}` answer direct messages?"))
        .default(true)
        .interact()?;
    // The question says what happens, not what the key is called: a channel
    // reaches the bot only by mentioning it, which is what `mentions` records.
    let mentions = Confirm::with_theme(&theme())
        .with_prompt(format!(
            "  Should `{agent}` answer when mentioned in a channel?"
        ))
        .default(false)
        .interact()?;

    // Neither is a bot that would answer nowhere. Say so rather than write a
    // section whose every key is missing.
    if !dm && !mentions {
        println!("  Nowhere to answer, so no Slack section.");
        return Ok(None);
    }
    Ok(Some(SlackPlan { dm, mentions }))
}

/// Where it runs. Cloud leads because it is the shorter path to a bot you can
/// talk to: the workspace install carries the credential, so there is no Slack
/// app to build and no token to export.
fn ask_place(slack: bool) -> Result<Place> {
    explain("Last thing: where do you want to run it?");
    let items = match slack {
        true => [
            "Substructure cloud   `subs apply` deploys it. Add the Slack bot in one click.",
            "local development    `subs serve` runs it here. You create the Slack app yourself.",
        ],
        false => [
            "Substructure cloud   `subs apply` deploys it.",
            "local development    `subs serve` runs it here.",
        ],
    };
    let pick = Select::with_theme(&theme())
        .with_prompt("Run it")
        .items(&items)
        .default(0)
        .interact()?;
    Ok([Place::Cloud, Place::Local][pick])
}

/// One numbered path, for the answer that was given. Both paths at once would
/// leave a reader to work out which half is theirs, which is the work the last
/// question was asked to do for them.
///
/// The file is named on every command. It is the default one most of the time,
/// but `subs init subs.prod.toml` is exactly when a copied line has to work.
fn next_steps(plan: &Plan, path: &Path) {
    let c = format!("-c {}", path.display());
    let mut n = 0;
    let mut step = |line: String| {
        n += 1;
        println!("  {n}. {line}");
    };

    println!("\nNext:");
    match plan.place {
        Place::Cloud => {
            step(format!("subs login {c}"));
            step(format!("subs apply {c}"));
            // The key is uploaded rather than exported: the call runs there.
            step(format!("subs llm set-key {} {c}", plan.agent.llm));
            // After apply, so the credential lands in the deployment that
            // will dial the connection rather than in the database here.
            for m in &plan.mcp {
                step(format!("subs mcp login {} {c}", m.id));
            }
            // After apply, for the same reason a connection is: the workspace
            // is connected to the deployment that will answer from it.
            if plan.slack.is_some() {
                step(format!("subs slack connect {c}"));
            }
            step(format!("subs open {c}"));
        }
        Place::Local => {
            if let Some(var) = plan.agent.provider.default_api_key_env() {
                step(format!(
                    "export {var}=...   # {}",
                    plan.agent.provider.console_url().unwrap_or_default()
                ));
            }
            for m in &plan.mcp {
                step(format!("subs mcp login {} {c}", m.id));
            }
            match plan.slack.is_some() {
                true => {
                    step(format!("Create your Slack app: {SLACK_NEW_APP}"));
                    println!("     the manifest to paste: {SLACK_DOCS}");
                    println!("     export SLACK_APP_TOKEN=xapp-...");
                    println!("     export SLACK_BOT_TOKEN=xoxb-...");
                    step(format!("subs serve {c}"));
                }
                false => step(format!(
                    "subs run {c} --agent {} -o pretty 'hello'",
                    plan.agent.id
                )),
            }
        }
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
    fn init(path: PathBuf, force: bool) -> InitCommand {
        InitCommand {
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

    /// One section each, and never both: `[serve]` is what `subs serve` reads
    /// and `auth = false` in it is only defensible on loopback, while
    /// `[remote]` is what says the credentials and the agent live
    /// elsewhere. A file carrying both would be answering a question that was
    /// asked once.
    #[test]
    fn each_place_writes_only_the_sections_it_runs_on() {
        for place in [Place::Cloud, Place::Local] {
            let body = render(&Plan {
                place,
                ..Plan::starter()
            });
            let cfg = parse(&body);

            let local = place == Place::Local;
            assert_eq!(cfg.serve.is_some(), local, "{body}");
            assert_eq!(cfg.remote.is_some(), !local, "{body}");
            // Stated rather than implied, so the file can be read for where it
            // goes instead of for what an absent key would have meant.
            assert_eq!(cfg.remote_url(), (!local).then_some(HOSTED_URL), "{body}");
            // `[run]` pins which agent a bare `subs run` drives. A starter has
            // one agent and names it on the command line, so it is not written.
            assert!(cfg.run.is_none(), "{body}");
        }
    }

    /// The wizard's answers are the same file the starter is, so anything it
    /// can produce has to parse too — including the sections and the spellings
    /// only an answer reaches.
    #[test]
    fn every_answer_renders_a_file_that_parses() {
        // The maximal file: every section an answer can add, so the sections
        // are checked in each other's company rather than one at a time.
        let mut mcp: Vec<McpPlan> = CATALOG.iter().map(Server::plan).collect();
        mcp.push(McpPlan {
            id: "internal".into(),
            url: "https://mcp.internal.test/mcp".into(),
        });

        for provider in ProviderKind::VENDORS {
            for place in [Place::Cloud, Place::Local] {
                let plan = Plan {
                    name: "support-bot".into(),
                    agent: AgentPlan {
                        id: "my-agent".into(),
                        llm: default_llm_name(provider).into(),
                        provider,
                        model: provider.default_model().into(),
                    },
                    mcp: mcp.clone(),
                    slack: Some(SlackPlan {
                        dm: true,
                        mentions: true,
                    }),
                    place,
                };
                let body = render(&plan);
                let cfg = parse(&body);
                assert_eq!(cfg.name.as_deref(), Some("support-bot"));
                assert_eq!(cfg.agent_ids(), ["my-agent"], "{body}");
                assert_eq!(cfg.llm[default_llm_name(provider)].kind, provider, "{body}");
                // A declared connection reaches nothing until the agent names
                // it, so the two halves are checked together.
                let declared: Vec<&str> = cfg.mcp.keys().map(String::as_str).collect();
                let named: Vec<&str> = cfg.agent["my-agent"].mcp.iter().map(|s| s.id()).collect();
                assert_eq!(declared.len(), mcp.len(), "{body}");
                for m in &mcp {
                    assert!(declared.contains(&m.id.as_str()), "{body}");
                    assert!(named.contains(&m.id.as_str()), "{body}");
                    assert_eq!(cfg.mcp[&m.id].url, m.url, "{body}");
                    assert!(cfg.mcp[&m.id].auth.is_none(), "login, not a variable");
                }
                assert_eq!(cfg.slack_dm_agent().as_deref(), Some("my-agent"), "{body}");
            }
        }
    }

    /// Each Slack answer writes the key it agreed to and no other, so a bot
    /// invited to a channel it was not opened up to still answers nowhere.
    #[test]
    fn slack_writes_the_answer_it_was_given() {
        for (dm, mentions) in [(true, false), (false, true), (true, true)] {
            let body = render(&Plan {
                slack: Some(SlackPlan { dm, mentions }),
                ..Plan::starter()
            });
            let slack = parse(&body).slack.expect("a section");
            assert_eq!(slack.dm.is_some(), dm, "{body}");
            assert_eq!(slack.mentions.is_some(), mentions, "{body}");
        }
    }

    /// A URL is named after the service it reaches, not after how it is served.
    #[test]
    fn a_server_is_named_after_its_host() {
        for (url, id) in [
            ("https://mcp.sentry.dev/mcp", "sentry"),
            ("https://api.githubcopilot.com/mcp/", "githubcopilot"),
            ("https://huggingface.co/mcp", "huggingface"),
            ("http://127.0.0.1:4445/mcp", "connection"),
            ("nonsense", "connection"),
        ] {
            assert_eq!(id_from_url(url), id, "{url}");
        }
    }

    /// Every offered server has to survive the file's own checks, so a bad
    /// entry fails here rather than at somebody's first `subs init`.
    #[test]
    fn every_offered_server_is_one_the_file_accepts() {
        for s in CATALOG {
            check_id(s.id).unwrap();
            check_url(s.url).unwrap();
        }
    }

    /// The starter is the engine-hosted minimal file: one llm block, one agent,
    /// no worker. Attaching a worker is a later edit, like every other rung.
    #[test]
    fn the_starter_declares_an_engine_hosted_agent() {
        let cfg = parse(&render(&Plan::starter()));
        let agent = cfg.agent.get("assistant").expect("one agent declared");
        assert_eq!(agent.llm.as_deref(), Some("claude"));
        assert!(agent.worker.is_none(), "the engine decides for it");
        assert_eq!(cfg.llm["claude"].kind, ProviderKind::Anthropic);
        assert_eq!(cfg.name.as_deref(), Some(DEFAULT_NAME));
    }

    /// `auth` defaults to on, so an unwritten `[serve]` would make a first
    /// `subs serve` demand tokens. The starter states it instead.
    #[test]
    fn the_starter_states_the_auth_it_relies_on() {
        let cfg = parse(&render(&Plan::starter()));
        assert!(!cfg.serve_auth());
        assert!(cfg.slack_dm_agent().is_none(), "init declares no slack bot");
    }

    #[test]
    fn an_existing_file_is_not_overwritten_without_force() {
        let path = tmpdir().join("substructure.toml");
        fs::write(&path, "db = \"mine.db\"\n").unwrap();

        let err = run(init(path.clone(), false)).unwrap_err().to_string();
        assert!(err.contains("already exists"), "got {err}");
        assert!(fs::read_to_string(&path).unwrap().contains("mine.db"));

        run(init(path.clone(), true)).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), render(&Plan::starter()));
    }

    /// The path is taken as given, so `subs init api/substructure.toml` does
    /// not fail on a directory that does not exist yet.
    #[test]
    fn a_missing_parent_directory_is_created() {
        let path = tmpdir().join("nested/deeper/substructure.toml");
        run(init(path.clone(), false)).unwrap();
        assert!(path.exists());
    }

    /// Where it runs is a question, not an argument, so the path that cannot
    /// ask writes the file that needs nothing else to run.
    #[test]
    fn the_scripted_path_writes_the_engine_file() {
        let path = tmpdir().join("substructure.toml");
        run(init(path.clone(), false)).unwrap();

        let cfg = project_config::load_explicit(&path).unwrap().config;
        assert_eq!(cfg.agent_ids(), ["assistant"]);
        assert!(cfg.serve.is_some(), "an engine to run it");
        assert!(cfg.remote.is_none(), "nothing to administer yet");
    }
}
