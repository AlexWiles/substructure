use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use toml_edit::{DocumentMut, Item, Table, Value};

use crate::cli::env::{OutputFormat, ProviderBinding, ProviderKind};
use crate::connectors::registry::ConnectionSpec;
use crate::manifest::{AgentSection, Manifest, ProviderSpec, SlackConfig};
use crate::runtime::llm::LlmBlocks;
use crate::runtime::worker::AgentEntry;

pub const FILENAME: &str = "substructure.toml";
pub const DEFAULT_DB: &str = "substructure.db";

/// One project, described once. One file is one project: a second environment
/// is a second file (`subs apply -c substructure.staging.toml`), deployed as a
/// second project with its own wallet, quota, and keys.
///
/// A file carries two roles, either or both: **an engine you run** (`db`, `log`,
/// `[run]`, `[serve]`) and **a remote you administer** (`[remote]`). What the
/// project *is* — `name`, `[agent.<id>]`, `[llm.<id>]`, `[slack]`, `[mcp.<id>]`
/// — is one declaration whichever role reads it, and it is exactly
/// [`Manifest`], so a self-hosted system is served and administered from the
/// same file rather than two that have to agree.
///
/// The manifest's sections are spelled out here rather than nested behind
/// `#[serde(flatten)]` because flattening would cost `deny_unknown_fields`, and
/// a typo'd section name being loud matters more than the repetition.
///
/// A role is present when its keys are: `[remote]` is what `subs apply` and
/// `subs sessions` act on, and it is also what decides that a connection's
/// credential belongs to the remote rather than to the engine here.
///
/// Precedence for anything the CLI also accepts as a flag is
/// **flag > environment > this > default**, so pinning something here never
/// takes an override away.
///
/// Per-invocation arguments are deliberately absent: `--input`, `--session`, and
/// `-c` itself say what one run is doing, not how the environment is set up.
/// Secrets are absent for the same reason they are absent from `[mcp]` — a
/// committed file must not be able to hold one, so the signing secret is named
/// rather than written.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProjectConfig {
    /// Where this was read from, which is what `db` defaults from and what a
    /// relative `db` is resolved against. Not a setting: it is the file's own
    /// path, so it is never written back.
    #[serde(skip)]
    source: PathBuf,
    /// The project's name. `subs apply` creates the project from it when
    /// nothing is pinned, and renames when it changes: the file is the source
    /// of truth.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Engine state: events, sessions, and the credentials `subs mcp login`
    /// authorized. Defaults to this file's own name — `substructure.toml` keeps
    /// `substructure.db`, and `subs.staging.toml` gets `subs.staging.db`, so two
    /// files in one directory are two engines rather than one they share.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub db: Option<String>,
    /// Log filter in `RUST_LOG` syntax: a bare level (`info`) or per-target
    /// directives (`substructure_core=debug,warn`). `$RUST_LOG` still wins.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log: Option<String>,
    /// The LLM blocks this project declares, keyed by the name an agent names.
    /// The declaration travels with the project; the credential binds per
    /// environment.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub llm: BTreeMap<String, ProviderSpec>,
    /// The agents this project declares, keyed by the id a client routes on.
    /// Each section is the wire `AgentConfig` plus where its decisions go.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub agent: BTreeMap<String, AgentSection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run: Option<RunConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub serve: Option<ServeConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slack: Option<SlackConfig>,
    /// MCP servers this project may reach, keyed by the id an agent names. An
    /// engine here dials them itself; a remote is told the id and URL and
    /// holds the credential, so `auth` is the engine's half alone.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub mcp: BTreeMap<String, ConnectionSpec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remote: Option<Remote>,
}

/// The deployment this file administers — the hosted cloud, a self-hosted one,
/// or someone else's `subs serve` — and what it is pinned to there.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Remote {
    /// The API to talk to [default: `https://api.substructure.ai`]. A `--url`
    /// flag still overrides it; `$SUBS_API_URL` only fills in when neither is
    /// set. `subs login` reads it too, so the token is stored under the same
    /// server the rest of the file's commands reach.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub org: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
}

/// Defaults for `subs run`.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RunConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<OutputFormat>,
}

/// `subs serve` only.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServeConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,
    /// Client and worker authentication [default: true]. `false` serves
    /// without issuing tokens, which is for a server nothing outside this
    /// machine can reach.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth: Option<bool>,
}

impl ProjectConfig {
    /// The half of this file a remote holds. Cheap enough to build on
    /// demand: it is read at startup and at `subs apply`, never per decision.
    pub fn manifest(&self) -> Manifest {
        Manifest {
            name: self.name.clone(),
            llm: self.llm.clone(),
            agent: self.agent.clone(),
            mcp: self.mcp.clone(),
            slack: self.slack.clone(),
        }
    }

    pub fn connections(&self) -> BTreeMap<String, ConnectionSpec> {
        self.manifest().connections()
    }

    /// The engine's database, beside the file that names it. One file is one
    /// project, so two files in a directory hold two engines rather than
    /// writing each other's sessions and credentials.
    pub fn db_path(&self) -> String {
        let named = self.db.clone().unwrap_or_else(|| default_db(&self.source));
        match self.source.parent() {
            // `join` keeps an absolute `db` absolute.
            Some(dir) if !dir.as_os_str().is_empty() => dir.join(named).display().to_string(),
            _ => named,
        }
    }

    pub fn slack_dm_agent(&self) -> Option<String> {
        self.slack.as_ref()?.dm.clone()
    }

    /// The declared blocks as the engine reads them: venue and wire shape,
    /// never a credential.
    pub fn llm_blocks(&self) -> LlmBlocks {
        self.manifest().llm_blocks()
    }

    /// The blocks the engine runs itself, each with the variable its key comes
    /// from. `worker` blocks are absent: they never need a credential here.
    pub fn provider_bindings(&self) -> Vec<ProviderBinding> {
        self.llm
            .iter()
            .filter(|(_, spec)| spec.kind != ProviderKind::Worker)
            .filter_map(|(name, spec)| {
                Some(ProviderBinding {
                    name: name.clone(),
                    kind: spec.kind,
                    api_key_env: spec.api_key_env()?,
                    base_url: spec.base_url.clone(),
                })
            })
            .collect()
    }

    /// Every agent this project declares, keyed by the id a client routes on.
    pub fn agents(&self) -> BTreeMap<String, AgentEntry> {
        self.manifest().agents()
    }

    /// The declared agent ids, for the error that says what could have been named.
    pub fn agent_ids(&self) -> Vec<String> {
        self.agent.keys().cloned().collect()
    }

    /// Whether an engine here authenticates its clients and workers
    /// [default: yes].
    pub fn serve_auth(&self) -> bool {
        self.serve.as_ref().and_then(|s| s.auth).unwrap_or(true)
    }

    pub fn remote_url(&self) -> Option<&str> {
        self.remote.as_ref()?.url.as_deref()
    }

    pub fn org(&self) -> Option<&str> {
        self.remote.as_ref()?.org.as_deref()
    }

    pub fn project(&self) -> Option<&str> {
        self.remote.as_ref()?.project.as_deref()
    }

    /// The remote section, creating it if the file has none — for the
    /// commands that pin (`subs link`, `subs apply`).
    pub fn remote_mut(&mut self) -> &mut Remote {
        self.remote.get_or_insert_with(Remote::default)
    }

    fn parse(s: &str, path: &Path) -> Result<Self> {
        let at = path.display();
        let value: toml::Value = toml::from_str(s).map_err(|e| anyhow!("parsing {at}: {e}"))?;
        moved_keys(&value, &at)?;
        let mut config: ProjectConfig =
            value.try_into().map_err(|e| anyhow!("parsing {at}: {e}"))?;
        config
            .manifest()
            .validate()
            .map_err(|e| anyhow!("{at}: {e}"))?;
        config.source = path.to_path_buf();
        Ok(config)
    }
}

/// `substructure.toml` → `substructure.db`, `subs.staging.toml` →
/// `subs.staging.db`. A config with no file behind it keeps the plain default,
/// which is what an engine running on flags alone uses.
fn default_db(source: &Path) -> String {
    match source.file_stem().and_then(|s| s.to_str()) {
        Some(stem) if !stem.is_empty() => format!("{stem}.db"),
        _ => DEFAULT_DB.to_string(),
    }
}

/// The keys and sections that moved, reported where they went rather than as
/// `deny_unknown_fields`' "unknown field", which says nothing about the file
/// this one is.
fn moved_keys(value: &toml::Value, at: &impl std::fmt::Display) -> Result<()> {
    // Both sections were named for a server, which left the file with two of
    // them and no way to tell which one a line was about.
    if value.get("deployment").is_some() {
        bail!("{at}: `[deployment]` is now `[remote]`.");
    }
    if value.get("server").is_some() {
        bail!("{at}: `[server]` is now `[serve]`, beside `[run]`.");
    }
    // The unit is a project now, so `app` anywhere is named rather than left to
    // `deny_unknown_fields`, which would only say the field is unknown.
    if value.get("remote").and_then(|d| d.get("app")).is_some() {
        bail!(
            "{at}: `[remote].app` is now `[remote].project`. One file is one project; \
             a second environment is a second file (`subs apply -c substructure.staging.toml`)."
        );
    }
    let pins: Vec<&str> = ["url", "org", "project", "app"]
        .into_iter()
        .filter(|k| value.get(k).is_some())
        .map(|k| match k {
            "app" => "project",
            k => k,
        })
        .collect();
    if value.get("target").is_some() {
        let and_pins = match pins.is_empty() {
            true => String::new(),
            false => format!(", and move `{}` under `[remote]`", pins.join("`, `")),
        };
        bail!(
            "{at}: `target` is no longer a setting. Delete it{and_pins} — a file describes an \
             engine you run (`db`, `[run]`, `[serve]`), a remote you administer \
             (`[remote]`), or both."
        );
    }
    if !pins.is_empty() {
        bail!(
            "{at}: `{}` belongs under `[remote]`, with the server's `url`.",
            pins.join("`, `")
        );
    }
    // `[slack].agent` meant "DMs, and any channel not named" — one key for two
    // decisions with very different blast radii, so it became two.
    if value.get("slack").and_then(|s| s.get("agent")).is_some() {
        bail!(
            "{at}: `[slack].agent` is now two settings, because it answered two questions: \
             `dm` for direct messages, and `mentions` for being mentioned in a channel no \
             `[slack.channel.<id>]` names. Set whichever you meant — setting neither serves \
             only the channels you name."
        );
    }
    if value.get("worker").is_some() {
        bail!(
            "{at}: `[worker]` is no longer a setting: a worker belongs to one agent, not to \
             the project. Write `worker = \"<url>\"` under the `[agent.<id>]` it decides for, and \
             leave it off the agents the engine decides for."
        );
    }
    // `[llm] provider = "anthropic"` versus `[llm.claude] type = "anthropic"`:
    // the old form's values are scalars where the new form's are tables.
    if let Some(llm) = value.get("llm").and_then(toml::Value::as_table) {
        if llm.values().any(|v| !v.is_table()) {
            bail!(
                "{at}: `[llm]` now declares named blocks, so that a second one can be added \
                 and an agent can say which it uses. Write `[llm.<name>]` with a `type`, e.g. \
                 `[llm.claude]` / `type = \"anthropic\"`, and name it from `[agent.<id>]`."
            );
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct Found {
    pub config: ProjectConfig,
    pub path: PathBuf,
}

/// The environment file at `path`, or the nearest one above the working
/// directory. An explicit path that does not resolve is an error; discovery
/// finding nothing is not.
pub fn resolve(path: Option<&Path>) -> Result<Option<Found>> {
    match path {
        Some(p) => load_explicit(p).map(Some),
        None => find(),
    }
}

/// What the file says, or the defaults when there is none. For the commands
/// that work without one — an engine runs on defaults, and every setting is
/// also a flag.
pub fn load(path: Option<&Path>) -> Result<ProjectConfig> {
    Ok(resolve(path)?.map(|found| found.config).unwrap_or_default())
}

/// The file in this directory, and only this one. Ancestors are not searched:
/// a command run in a subdirectory acting on a project two levels up is a
/// surprise, and `-c` already says when the file is somewhere else.
pub fn find_from(dir: &Path) -> Result<Option<Found>> {
    let candidate = dir.join(FILENAME);
    match candidate.is_file() {
        true => load_explicit(&candidate).map(Some),
        false => Ok(None),
    }
}

pub fn find() -> Result<Option<Found>> {
    let cwd = env::current_dir().context("could not determine cwd for substructure.toml lookup")?;
    find_from(&cwd)
}

pub fn load_explicit(path: &Path) -> Result<Found> {
    let s = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(Found {
        config: ProjectConfig::parse(&s, path)?,
        path: path.to_path_buf(),
    })
}

/// Write `config` back, keeping everything about the file that is not a
/// setting. A machine edit must not cost a reader their comments or their
/// layout, so the parsed document is edited in place rather than replaced.
pub fn write(path: &Path, config: &ProjectConfig) -> Result<()> {
    let mut rendered: DocumentMut =
        toml_edit::ser::to_document(config).context("serializing substructure.toml")?;
    for (_, item) in rendered.as_table_mut().iter_mut() {
        expand(item, 2);
    }

    let mut doc = match fs::read_to_string(path) {
        Ok(existing) => existing
            .parse::<DocumentMut>()
            .map_err(|e| anyhow!("parsing {}: {e}", path.display()))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => DocumentMut::new(),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    merge(doc.as_table_mut(), rendered.as_table());

    fs::write(path, doc.to_string()).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Serde renders every struct as an inline table. Sections are what a reader
/// expects of the two outermost levels — `[worker]`, `[mcp.sentry]` — and an
/// inline table is what they expect of anything below.
fn expand(item: &mut Item, depth: usize) {
    if depth == 0 {
        return;
    }
    if let Item::Value(Value::InlineTable(inline)) = item {
        *item = Item::Table(std::mem::take(inline).into_table());
    }
    if let Item::Table(table) = item {
        for (_, child) in table.iter_mut() {
            expand(child, depth - 1);
        }
        // `[mcp]` holds nothing of its own, so it is a header nobody needs.
        if !table.is_empty() && table.iter().all(|(_, child)| child.is_table()) {
            table.set_implicit(true);
        }
    }
}

/// Overwrite `target` with `source`, key by key. A key `source` does not carry
/// is a setting that was removed, so it goes; a value that changed keeps its
/// decoration, which is where a trailing comment lives.
fn merge(target: &mut Table, source: &Table) {
    target.retain(|key, _| source.contains_key(key));
    for (key, item) in source.iter() {
        match (target.get_mut(key), item) {
            (Some(Item::Table(existing)), Item::Table(next)) => merge(existing, next),
            // A section someone wrote inline stays inline.
            (Some(Item::Value(Value::InlineTable(existing))), Item::Table(next)) => {
                let mut merged = next.clone().into_inline_table();
                *merged.decor_mut() = existing.decor().clone();
                *existing = merged;
            }
            (Some(Item::Value(existing)), Item::Value(next)) => {
                let mut next = next.clone();
                *next.decor_mut() = existing.decor().clone();
                *existing = next;
            }
            (Some(existing), next) => *existing = next.clone(),
            (None, next) => {
                target.insert(key, next.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::LlmFormat;
    use crate::runtime::llm::LlmBlock;

    fn tmpdir() -> PathBuf {
        // Timestamp alone collides across parallel tests; the counter disambiguates.
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("subs-project-test-{nanos}-{seq}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn parse(s: &str) -> Result<ProjectConfig> {
        ProjectConfig::parse(s, Path::new("substructure.toml"))
    }

    fn ok(s: &str) -> ProjectConfig {
        parse(s).unwrap()
    }

    #[test]
    fn a_file_carries_either_role_or_both() {
        let engine = ok("db = \"dev.db\"\n[serve]\nport = 9000\n");
        assert_eq!(engine.db_path(), "dev.db");
        assert!(engine.remote.is_none());

        let deployment = ok("[remote]\norg = \"org_1\"\n");
        assert_eq!(deployment.org(), Some("org_1"));
        assert_eq!(deployment.db_path(), DEFAULT_DB);

        // One system: served here, administered there, declared once.
        let both = ok(r#"
            name = "support-bot"
            db = "prod.db"

            [llm.claude]
            type = "anthropic"

            [agent.support]
            llm = "claude"
            model = "claude-sonnet-4-5"
            worker = "https://bot.example.com/agent"

            [remote]
            url = "https://subs.internal"
            project = "proj_1"
        "#);
        assert_eq!(both.name.as_deref(), Some("support-bot"));
        assert_eq!(both.db_path(), "prod.db");
        assert_eq!(both.remote_url(), Some("https://subs.internal"));
        assert_eq!(both.project(), Some("proj_1"));
    }

    #[test]
    fn an_empty_file_is_valid_and_is_the_defaults() {
        assert_eq!(
            ok(""),
            ProjectConfig {
                source: FILENAME.into(),
                ..Default::default()
            }
        );
        // No file behind it at all: an engine running on flags alone.
        assert_eq!(ProjectConfig::default().db_path(), DEFAULT_DB);
        assert!(ProjectConfig::default().serve_auth());
    }

    /// One file is one project, so a second file in a directory is a second
    /// engine rather than one that writes the first's sessions and credentials.
    #[test]
    fn the_database_defaults_to_the_files_own_name() {
        let dir = tmpdir();
        let named = |name: &str| {
            let path = dir.join(name);
            fs::write(&path, "[serve]\nport = 9000\n").unwrap();
            load_explicit(&path).unwrap().config.db_path()
        };

        // The usual file keeps the name it always had.
        assert_eq!(
            named(FILENAME),
            dir.join("substructure.db").to_str().unwrap()
        );
        assert_eq!(
            named("subs.staging.toml"),
            dir.join("subs.staging.db").to_str().unwrap()
        );

        // An explicit `db` still wins, and sits beside the file that names it.
        let path = dir.join("explicit.toml");
        fs::write(&path, "db = \"engine.db\"\n").unwrap();
        assert_eq!(
            load_explicit(&path).unwrap().config.db_path(),
            dir.join("engine.db").to_str().unwrap()
        );

        // An absolute one is taken as given.
        let elsewhere = tmpdir().join("far.db");
        fs::write(&path, format!("db = {:?}\n", elsewhere.to_str().unwrap())).unwrap();
        assert_eq!(
            load_explicit(&path).unwrap().config.db_path(),
            elsewhere.to_str().unwrap()
        );
    }

    #[test]
    fn the_engine_groups_read_back() {
        let cfg = ok(r#"
            db = "dev.substructure.db"
            log = "substructure_core=debug,warn"

            [llm.claude]
            type = "anthropic"

            [agent.support]
            llm = "claude"
            model = "claude-sonnet-4-5"
            worker = "http://localhost:4444"
            signing_secret_env = "SUBS_SIGNING_SECRET"

            [run]
            agent = "support"
            output = "pretty"

            [serve]
            port = 9000
            auth = false
        "#);
        assert_eq!(cfg.db_path(), "dev.substructure.db");
        assert_eq!(cfg.log.as_deref(), Some("substructure_core=debug,warn"));
        assert_eq!(
            cfg.agent["support"].worker.as_deref(),
            Some("http://localhost:4444")
        );
        assert_eq!(cfg.llm["claude"].kind, ProviderKind::Anthropic);
        assert_eq!(cfg.run.as_ref().unwrap().agent.as_deref(), Some("support"));
        assert_eq!(cfg.run.clone().unwrap().output, Some(OutputFormat::Pretty));
        assert!(!cfg.serve_auth());
        let serve = cfg.serve.unwrap();
        assert_eq!(serve.port, Some(9000));
        // Absent is absent, not a default the flag would then have to beat.
        assert_eq!(serve.host, None);
    }

    /// Both sections were named for a server, so a line about "the server" said
    /// nothing about which one. Renamed, and reported by name rather than as an
    /// unknown field.
    #[test]
    fn the_two_server_sections_say_where_they_went() {
        let err = parse("[deployment]\norg = \"acme\"\n")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("`[deployment]` is now `[remote]`"),
            "got {err}"
        );

        let err = parse("[server]\nport = 9000\n").unwrap_err().to_string();
        assert!(err.contains("`[server]` is now `[serve]`"), "got {err}");
    }

    #[test]
    fn target_says_where_it_went() {
        let err = parse("target = \"local\"\ndb = \"dev.db\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("`target` is no longer"), "got {err}");
        assert!(err.contains("[remote]"), "got {err}");

        // The pins moved with it, so one message covers the whole edit — and a
        // top-level `app` is reported under the name it now has.
        let err = parse("target = \"remote\"\nurl = \"https://x\"\norg = \"o\"\napp = \"a\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("`url`, `org`, `project`"), "got {err}");

        // And on their own, without a target to hang the message on.
        let err = parse("org = \"acme\"\n").unwrap_err().to_string();
        assert!(
            err.contains("`org`") && err.contains("[remote]"),
            "got {err}"
        );
    }

    #[test]
    fn a_misspelled_key_is_a_parse_error_not_a_silent_no_op() {
        // An agent section is the wire config plus two hosting keys, and
        // `flatten` blinds serde's own check, so the key set is checked by hand.
        let err = parse(
            "[llm.claude]\ntype = \"anthropic\"\n\n\
             [agent.a]\nllm = \"claude\"\nmodel = \"m\"\nsytem = \"be brief\"\n",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("sytem"), "got {err}");

        let err = parse("[remote]\nnmae = \"x\"\n").unwrap_err().to_string();
        assert!(err.contains("nmae"), "got {err}");

        // There is no catalog key: a connection always declares a URL.
        let err = parse("[mcp.sentry]\ncatalog = \"sentry\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("catalog"), "got {err}");
    }

    #[test]
    fn a_connection_is_checked_where_it_was_typed() {
        // An id prefixes every tool name the model sees.
        let err = parse("[mcp.\"my server\"]\nurl = \"https://x/mcp\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("cannot prefix a tool name"), "got {err}");

        // A credential would cross the network in the clear.
        let err = parse("[mcp.sentry]\nurl = \"http://mcp.sentry.dev/mcp\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("not https"), "got {err}");

        // Loopback is exempt: nothing off-host sees it.
        ok("[mcp.issues]\nurl = \"http://localhost:4445/mcp\"\n");
    }

    #[test]
    fn an_inline_secret_is_a_parse_error() {
        let err = parse(
            "[llm.claude]\ntype = \"anthropic\"\n\n\
             [agent.a]\nllm = \"claude\"\nmodel = \"m\"\nsigning_secret = \"s3cret\"\n",
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("signing_secret"),
            "a committed file must not be able to hold a secret; got {err}"
        );

        let err = parse("[llm.claude]\ntype = \"anthropic\"\napi_key = \"sk-1\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("api_key"), "got {err}");

        let err = parse("[mcp.sentry]\nurl = \"https://x/mcp\"\nauth = { token = \"t\" }\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("token"), "got {err}");
    }

    /// An agent whose worker authors its whole config declares nothing but the
    /// URL — the file has no business naming an `llm` or a `model` it will
    /// never use.
    #[test]
    fn an_agent_may_delegate_everything_to_its_worker() {
        let cfg = ok(r#"
            [agent.reggu]
            worker = "http://localhost:4000/substructure/agent"
        "#);
        let entry = &cfg.agents()["reggu"];
        assert!(entry.config.is_none(), "nothing to seed");
        assert!(entry.worker.is_some(), "the worker authors it");
    }

    /// …but only a worker can author one, so an agent that declares neither a
    /// config nor a worker is a declaration nothing can act on.
    #[test]
    fn an_agent_that_declares_nothing_at_all_is_an_error() {
        let err = parse("[agent.a]\n").unwrap_err().to_string();
        assert!(err.contains("declares nothing"), "got {err}");
    }

    /// A half-declared config would seed a proposal the engine cannot call
    /// with, so it fails here instead of at the first model call.
    #[test]
    fn a_partly_declared_config_is_an_error() {
        let err = parse(
            "[llm.claude]\ntype = \"anthropic\"\n\n\
             [agent.a]\nllm = \"claude\"\nworker = \"https://a/agent\"\n",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("no `model`"), "got {err}");

        let err = parse("[agent.a]\nsystem = \"be brief\"\nworker = \"https://a/agent\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("no `llm`"), "got {err}");
    }

    /// Every name an agent uses is declared in this same file, so a typo is a
    /// parse error rather than a decision that fails on the first turn.
    #[test]
    fn an_agents_references_are_checked_against_the_file() {
        let err = parse("[agent.a]\nmodel = \"m\"\n").unwrap_err().to_string();
        assert!(err.contains("no `llm`"), "got {err}");

        let err = parse(
            "[llm.claude]\ntype = \"anthropic\"\n\n[agent.a]\nllm = \"clade\"\nmodel = \"m\"\n",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("names no block"), "got {err}");
        assert!(
            err.contains("claude"),
            "and says what is declared; got {err}"
        );

        let err = parse(
            "[llm.claude]\ntype = \"anthropic\"\n\n\
             [agent.a]\nllm = \"claude\"\nmodel = \"m\"\nmcp = [\"sentry\"]\n",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("no connection `sentry`"), "got {err}");
    }

    /// A `worker` block's calls are made by a worker, so an agent on one that
    /// has no worker could never make a call at all.
    #[test]
    fn an_agent_on_a_worker_block_needs_a_worker() {
        let err = parse(
            "[llm.byo]\ntype = \"worker\"\nformat = \"anthropic\"\n\n\
             [agent.a]\nllm = \"byo\"\nmodel = \"m\"\n",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("needs a `worker`"), "got {err}");

        // With one attached it parses.
        ok("[llm.byo]\ntype = \"worker\"\n\n\
            [agent.a]\nllm = \"byo\"\nmodel = \"m\"\nworker = \"https://a/agent\"\n");
    }

    /// A field belonging to the other venue is a misunderstanding of the block,
    /// not a detail to ignore.
    #[test]
    fn a_block_is_checked_against_its_own_type() {
        let err = parse("[llm.claude]\ntype = \"anthropic\"\nformat = \"anthropic\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("only applies to"), "got {err}");

        let err = parse("[llm.byo]\ntype = \"worker\"\napi_key_env = \"K\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("never leaves your worker"), "got {err}");
    }

    /// A file can only declare tools the browser runs: a worker-run tool is
    /// worker code, and nothing here could execute one.
    #[test]
    fn a_file_declared_tool_must_be_client_handled() {
        let err = parse(
            "[llm.claude]\ntype = \"anthropic\"\n\n\
             [agent.a]\nllm = \"claude\"\nmodel = \"m\"\n\
             tools = [{ name = \"get_time\" }]\n",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("handler = \"client\""), "got {err}");
    }

    /// Signing a request nobody receives is a setting with no effect, which is
    /// worse than an error.
    #[test]
    fn a_signing_secret_without_a_worker_is_an_error() {
        let err = parse(
            "[llm.claude]\ntype = \"anthropic\"\n\n\
             [agent.a]\nllm = \"claude\"\nmodel = \"m\"\nsigning_secret_env = \"S\"\n",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("no `worker`"), "got {err}");
    }

    /// The two sections that moved say where they went, rather than reading as
    /// "unknown field".
    #[test]
    fn the_old_worker_and_llm_forms_say_what_replaced_them() {
        let err = parse("[worker]\nurl = \"http://localhost:4444\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("belongs to one agent"), "got {err}");
        assert!(err.contains("[agent.<id>]"), "got {err}");

        let err = parse("[llm]\nprovider = \"anthropic\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("named blocks"), "got {err}");
        assert!(err.contains("[llm.<name>]"), "got {err}");
    }

    /// What the engine reads off the file: the venue per block, and one binding
    /// per block it runs itself.
    #[test]
    fn the_engine_reads_venues_and_bindings_off_the_blocks() {
        let cfg = ok(r#"
            [llm.claude]
            type = "anthropic"

            [llm.cheap]
            type = "openai"
            api_key_env = "MY_OPENAI_KEY"

            [llm.byo]
            type = "worker"
            format = "anthropic"
        "#);

        let blocks = cfg.llm_blocks();
        assert_eq!(blocks.get("claude"), Some(LlmBlock::engine()));
        assert_eq!(
            blocks.get("byo"),
            Some(LlmBlock::worker(Some(LlmFormat::Anthropic)))
        );
        assert_eq!(blocks.declared(), "byo, cheap, claude");

        // A worker block needs no credential where the engine runs.
        let bindings: Vec<(String, String)> = cfg
            .provider_bindings()
            .into_iter()
            .map(|b| (b.name, b.api_key_env))
            .collect();
        assert_eq!(
            bindings,
            [
                ("cheap".to_string(), "MY_OPENAI_KEY".to_string()),
                ("claude".to_string(), "ANTHROPIC_API_KEY".to_string()),
            ]
        );
    }

    /// The directory the engine routes on: config without hosting, hosting
    /// without config.
    #[test]
    fn agents_become_directory_entries() {
        let cfg = ok(r#"
            [llm.claude]
            type = "anthropic"

            [agent.assistant]
            llm = "claude"
            model = "claude-sonnet-4-5"

            [agent.triage]
            llm = "claude"
            model = "claude-haiku-4-5"
            worker = "https://triage.internal/agent"
        "#);
        let agents = cfg.agents();
        assert!(agents["assistant"].worker.is_none(), "the engine decides");
        assert!(agents["assistant"].config.is_some(), "and needs a config");
        assert_eq!(
            agents["triage"].worker.as_ref().map(|w| w.url.as_str()),
            Some("https://triage.internal/agent")
        );
        // The hosting never crosses the wire.
        let wire = serde_json::to_value(agents["triage"].config.as_ref().unwrap()).unwrap();
        assert!(wire.get("worker").is_none(), "got {wire}");
        assert_eq!(wire["llm"], "claude");
    }

    #[test]
    fn an_output_mode_that_does_not_exist_is_a_parse_error() {
        // It used to fall back to `ag-ui`, so a typo silently changed the mode.
        let err = parse("[run]\noutput = \"pretyy\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("pretyy"), "got {err}");
    }

    #[test]
    fn everything_set_survives_a_round_trip() {
        let cfg = ok(r#"
            name = "support-bot"
            db = "dev.db"
            log = "info"

            [llm.cheap]
            type = "openai"
            api_key_env = "MY_OPENAI_KEY"
            base_url = "https://openai.internal"

            [llm.byo]
            type = "worker"
            format = "anthropic"

            [agent.support]
            llm = "cheap"
            model = "gpt-5-mini"
            system = "Be brief."
            worker = "http://localhost:4444"
            signing_secret_env = "S"
            mcp = ["sentry"]
            sub_agents = ["researcher"]
            tools = [{ name = "confirm", description = "Ask", handler = "client" }]

            [agent.researcher]
            description = "Finds sources"
            llm = "cheap"
            model = "gpt-5-mini"

            [run]
            agent = "support"
            output = "jsonl"

            [serve]
            host = "0.0.0.0"
            port = 9000
            auth = false

            [slack]
            dm = "support"

            [slack.channel.C0ENGOPS]
            agent = "researcher"

            [slack.channel.C0RANDOM]
            off = true

            [mcp.sentry]
            url = "https://mcp.sentry.dev/mcp"
            prefix_tools = false

            [remote]
            url = "https://subs.internal"
            org = "org_1"
            project = "proj_1"
        "#);
        let written = toml::to_string_pretty(&cfg).unwrap();
        assert_eq!(ok(&written), cfg, "written back as {written}");
    }

    /// Serde renders in declaration order, and a top-level scalar written after
    /// a section would parse back as that section's key.
    #[test]
    fn a_written_file_keeps_its_scalars_above_its_sections() {
        let path = tmpdir().join(FILENAME);
        let mut cfg = ok("db = \"dev.db\"\n[llm.claude]\ntype = \"anthropic\"\n\n\
             [agent.a]\nllm = \"claude\"\nmodel = \"m\"\n");
        cfg.name = Some("support-bot".into());
        cfg.remote_mut().project = Some("proj_1".into());
        write(&path, &cfg).unwrap();

        // Same settings, read from where it was written: `source` is the file's
        // own path rather than something the file declares.
        assert_eq!(
            load_explicit(&path).unwrap().config,
            ProjectConfig {
                source: path,
                ..cfg
            }
        );
    }

    #[test]
    fn unset_settings_are_not_written_back() {
        let cfg = ok("[remote]\norg = \"acme\"\n");
        let out = toml::to_string_pretty(&cfg).unwrap();
        assert_eq!(out.trim(), "[remote]\norg = \"acme\"", "got {out}");
    }

    #[test]
    fn an_empty_slack_section_is_not_a_configured_bot() {
        assert_eq!(ok("[slack]\n").slack_dm_agent(), None);
        assert_eq!(ProjectConfig::default().slack_dm_agent(), None);
        assert!(!ok("[slack]\n").slack.unwrap().is_configured());

        // The old bare key is gone, and says so rather than doing nothing.
        let err = parse("slack_agent = \"helper\"\n").unwrap_err().to_string();
        assert!(err.contains("slack_agent"), "got {err}");
    }

    fn agents() -> String {
        "[llm.claude]\ntype = \"anthropic\"\n\n\
         [agent.support]\nllm = \"claude\"\nmodel = \"m\"\n\n\
         [agent.oncall]\nllm = \"claude\"\nmodel = \"m\"\n\n"
            .to_string()
    }

    fn slack(s: &str) -> Result<ProjectConfig> {
        parse(&(agents() + s))
    }

    /// A channel names an agent rather than restating a prompt or a tool list:
    /// `[agent.<id>]` already is that bundle.
    #[test]
    fn a_channel_names_the_agent_that_answers_there() {
        let cfg = slack(
            "[slack]\ndm = \"support\"\nmentions = \"support\"\n\n\
             [slack.channel.C0ENGOPS]\nagent = \"oncall\"\n\n\
             [slack.channel.C0RANDOM]\noff = true\n",
        )
        .unwrap();
        let s = cfg.slack.unwrap();
        assert_eq!(s.dm.as_deref(), Some("support"));
        assert_eq!(s.mentions.as_deref(), Some("support"));
        assert_eq!(s.channel["C0ENGOPS"].agent(), Some("oncall"));
        // `off` is the absence of an agent, however the section spelled it.
        assert_eq!(s.channel["C0RANDOM"].agent(), None);
        assert!(s.is_configured());
    }

    /// Naming channels alone is the allowlist: without `mentions` the bot
    /// can be invited anywhere and still answers only where it was named.
    #[test]
    fn channels_without_mentions_are_a_complete_section() {
        let cfg = slack("[slack.channel.C0ENGOPS]\nagent = \"oncall\"\n").unwrap();
        let s = cfg.slack.unwrap();
        assert_eq!(s.dm, None);
        assert_eq!(s.mentions, None);
        assert!(s.is_configured(), "the bot is on, in one channel");
    }

    /// The old key answered two questions at once, so it is named rather than
    /// reported as an unknown field.
    #[test]
    fn the_old_slack_agent_key_says_what_it_became() {
        let err = slack("[slack]\nagent = \"support\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("`dm`"), "got {err}");
        assert!(err.contains("`mentions`"), "got {err}");
    }

    /// Every name the bot routes to is declared in this same file, so a typo
    /// is caught here rather than as a bot that answers nowhere.
    #[test]
    fn a_channels_agent_is_checked_against_the_file() {
        let err = slack("[slack]\ndm = \"suport\"\n").unwrap_err().to_string();
        assert!(err.contains("names no agent"), "got {err}");
        assert!(err.contains("oncall, support"), "and says which; got {err}");

        let err = slack("[slack.channel.C0ENGOPS]\nagent = \"on-call\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("[slack.channel.C0ENGOPS]"), "got {err}");
        assert!(err.contains("names no agent"), "got {err}");
    }

    /// The likeliest mistake, and one that would otherwise match no event and
    /// report nothing.
    #[test]
    fn a_channel_name_is_not_a_channel_id() {
        let err = slack("[slack.channel.\"#eng-oncall\"]\nagent = \"oncall\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("is a name, not a channel id"), "got {err}");
        assert!(err.contains("About tab"), "and where to get one; got {err}");
    }

    /// A channel says one of two things. Both, or neither, is a setting with
    /// no meaning rather than one to resolve by precedence.
    #[test]
    fn a_channel_that_says_both_or_neither_is_an_error() {
        let err = slack("[slack.channel.C0RANDOM]\nagent = \"oncall\"\noff = true\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("contradict"), "got {err}");

        let err = slack("[slack.channel.C0RANDOM]\n").unwrap_err().to_string();
        assert!(err.contains("declares nothing"), "got {err}");
    }

    /// A bot that connects, listens, and can answer nowhere.
    #[test]
    fn a_section_that_serves_nowhere_is_an_error() {
        let err = slack("[slack.channel.C0RANDOM]\noff = true\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("nothing to answer with"), "got {err}");
    }

    #[test]
    fn writing_keeps_comments_layout_and_everything_it_did_not_change() {
        let path = tmpdir().join(FILENAME);
        fs::write(
            &path,
            "# how this project is deployed\n\
             name = \"support-bot\"\n\
             \n\
             [llm.claude]\n\
             type = \"anthropic\"\n\
             \n\
             [agent.support]\n\
             llm = \"claude\"\n\
             model = \"claude-sonnet-4-5\"\n\
             # where the agent runs\n\
             worker = \"https://bot.example.com/agent\"\n\
             \n\
             [mcp.sentry]\n\
             url = \"https://mcp.sentry.dev/mcp\"\n\
             \n\
             [remote]\n\
             org = \"old\"        # pinned by hand\n",
        )
        .unwrap();

        let mut cfg = load_explicit(&path).unwrap().config;
        cfg.remote_mut().org = Some("new".into());
        cfg.remote_mut().project = Some("proj_1".into());
        write(&path, &cfg).unwrap();

        let after = fs::read_to_string(&path).unwrap();
        assert!(after.contains("# how this project is deployed"), "{after}");
        assert!(after.contains("# where the agent runs"), "{after}");
        assert!(
            after.contains("org = \"new\"        # pinned by hand"),
            "{after}"
        );
        assert!(after.contains("project = \"proj_1\""), "{after}");
        assert!(after.contains("[mcp.sentry]"), "{after}");
    }

    #[test]
    fn writing_removes_a_setting_that_is_no_longer_set() {
        let path = tmpdir().join(FILENAME);
        fs::write(&path, "[remote]\norg = \"acme\"\nproject = \"proj_1\"\n").unwrap();

        let mut cfg = load_explicit(&path).unwrap().config;
        cfg.remote_mut().project = None;
        write(&path, &cfg).unwrap();

        let after = fs::read_to_string(&path).unwrap();
        assert!(!after.contains("project"), "{after}");
        assert!(after.contains("org = \"acme\""), "{after}");
    }

    #[test]
    fn an_explicit_config_path_that_is_missing_is_an_error() {
        let missing = tmpdir().join("nope.toml");
        assert!(resolve(Some(&missing)).is_err());
    }

    /// Every example's config parses and validates. One that does not load is
    /// worse than no example: it is the first file a reader copies.
    #[test]
    fn every_example_config_parses_and_validates() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples")
            .canonicalize()
            .expect("examples directory");
        let mut checked = 0;
        for entry in fs::read_dir(&root).expect("read examples") {
            let path = entry.expect("entry").path().join(FILENAME);
            if !path.exists() {
                continue;
            }
            let found = load_explicit(&path)
                .unwrap_or_else(|e| panic!("{} does not parse: {e:#}", path.display()));
            found
                .config
                .manifest()
                .validate()
                .unwrap_or_else(|e| panic!("{} does not validate: {e}", path.display()));
            checked += 1;
        }
        assert!(checked > 10, "found only {checked} example configs");
    }

    #[test]
    fn load_without_a_file_is_the_defaults() {
        let root = tmpdir().join("isolated");
        fs::create_dir_all(&root).unwrap();
        assert!(find_from(&root).unwrap().is_none());
    }

    /// Only this directory. A command run in a subdirectory acting on a project
    /// two levels up is a surprise, and `-c` says when the file is elsewhere.
    #[test]
    fn find_does_not_look_at_ancestors() {
        let root = tmpdir();
        let nested = root.join("a/b/c");
        fs::create_dir_all(&nested).unwrap();
        let cfg_path = root.join(FILENAME);
        fs::write(
            &cfg_path,
            "[remote]\norg = \"org-x\"\nproject = \"project-y\"\n",
        )
        .unwrap();

        assert!(find_from(&nested).unwrap().is_none());

        let found = find_from(&root).unwrap().expect("the file is right here");
        assert_eq!(found.path, cfg_path);
        assert_eq!(found.config.org(), Some("org-x"));
        assert_eq!(found.config.project(), Some("project-y"));

        // And a file of its own is the one a directory uses.
        fs::write(nested.join(FILENAME), "[remote]\norg = \"inner\"\n").unwrap();
        assert_eq!(
            find_from(&nested).unwrap().unwrap().config.org(),
            Some("inner")
        );
    }
}
