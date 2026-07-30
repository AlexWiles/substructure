use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use toml_edit::{DocumentMut, Item, Table, Value};

use crate::cli::env::{LlmProviderArg, OutputFormat};
use crate::connectors::registry::ConnectionSpec;
use crate::protocol::ConnectorProtocol;

pub const FILENAME: &str = "substructure.toml";
pub const DEFAULT_DB: &str = "substructure.db";

/// One environment: one file, one engine.
///
/// `target` says which engine the rest of the file describes — an embedded one
/// over a SQLite file, or a server reached over HTTP — and the two halves share
/// no keys, so a setting that means nothing here is a parse error rather than
/// something that silently does nothing.
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
#[derive(Debug, Clone, PartialEq)]
pub enum EnvConfig {
    Local(LocalEnv),
    Remote(RemoteEnv),
}

/// An engine running in this process against a SQLite file: `subs run`,
/// `subs serve`.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LocalEnv {
    pub target: LocalTag,
    /// Engine state: events, sessions, and the credentials `subs mcp login`
    /// authorized [default: `substructure.db`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub db: Option<String>,
    /// Log filter in `RUST_LOG` syntax: a bare level (`info`) or per-target
    /// directives (`substructure_core=debug,warn`). `$RUST_LOG` still wins.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker: Option<WorkerConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm: Option<LlmConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run: Option<RunConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server: Option<ServerConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slack: Option<SlackConfig>,
    /// MCP servers this engine can reach, keyed by the id an agent config
    /// names.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub mcp: BTreeMap<String, ConnectionSpec>,
}

/// A server speaking `/api/v1`: the hosted cloud, a self-hosted deployment, or
/// someone else's `subs serve`.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RemoteEnv {
    pub target: RemoteTag,
    /// The API to talk to [default: `https://api.substructure.ai`]. A `--url`
    /// flag or `$SUBS_API_URL` still overrides it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub org: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub app: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker: Option<RemoteWorkerConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slack: Option<SlackConfig>,
    /// MCP servers this app may reach. Only a URL: the credential is held by
    /// the deployment, so there is no `token_env` to name here.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub mcp: BTreeMap<String, RemoteConnectionSpec>,
}

/// The tag that selects [`LocalEnv`]. A field rather than serde's internal
/// tagging, which cannot be combined with `deny_unknown_fields`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LocalTag {
    #[default]
    #[serde(rename = "local")]
    Local,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RemoteTag {
    #[default]
    #[serde(rename = "remote")]
    Remote,
}

/// Where the engine POSTs decisions.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WorkerConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Environment variable holding the signing secret. Named, never written —
    /// same rule as a connection's `token_env`. Unset means a random secret
    /// per start.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signing_secret_env: Option<String>,
}

/// The remote half: the deployment mints the signing secret, so only the
/// endpoint is the manifest's to state.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RemoteWorkerConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LlmConfig {
    /// Provider for engine-executed calls. The key comes from the matching
    /// `*_API_KEY` variable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<LlmProviderArg>,
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
pub struct ServerConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,
    /// Disable client and worker authentication. Local development only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dev: Option<bool>,
}

/// The `[slack]` section: what the Socket Mode bot needs that is not a secret.
///
/// The tokens are absent for the same reason they are absent from `[mcp]` — a
/// committed file must not be able to hold one — so `SLACK_APP_TOKEN` and
/// `SLACK_BOT_TOKEN` stay in the environment.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SlackConfig {
    /// Agent id the bot drives. Absent leaves the channel off.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
}

/// A connection in a remote environment. The deployment holds the credential
/// and decides whether the URL is one it will send it to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteConnectionSpec {
    pub url: String,
}

impl LocalEnv {
    /// Every connection this environment declares, keyed by the id an agent
    /// names.
    ///
    /// The one place the per-protocol sections become one registry, and so the
    /// one place a second protocol is added: `[a2a]` folds in here with its own
    /// `ConnectorProtocol`, and ids must stay unique across sections because an
    /// agent references a bare id and tool names are prefixed from it.
    pub fn connections(&self) -> BTreeMap<String, ConnectionSpec> {
        self.mcp
            .iter()
            .map(|(id, spec)| {
                let spec = ConnectionSpec {
                    protocol: ConnectorProtocol::Mcp,
                    ..spec.clone()
                };
                (id.clone(), spec)
            })
            .collect()
    }

    pub fn db_path(&self) -> String {
        self.db.clone().unwrap_or_else(|| DEFAULT_DB.to_string())
    }

    pub fn slack_agent(&self) -> Option<String> {
        self.slack.as_ref()?.agent.clone()
    }

    pub fn worker_url(&self) -> Option<String> {
        self.worker.as_ref()?.url.clone()
    }

    /// The signing secret the named variable holds, if the file names one and
    /// it is set.
    pub fn signing_secret(&self) -> Option<String> {
        crate::cli::env_value(self.worker.as_ref()?.signing_secret_env.as_deref()?)
    }

    pub fn llm_provider(&self) -> Option<LlmProviderArg> {
        self.llm.as_ref()?.provider
    }
}

impl RemoteEnv {
    pub fn worker_url(&self) -> Option<String> {
        self.worker.as_ref()?.url.clone()
    }
}

impl EnvConfig {
    pub fn log(&self) -> Option<&str> {
        match self {
            EnvConfig::Local(local) => local.log.as_deref(),
            EnvConfig::Remote(_) => None,
        }
    }

    fn target(&self) -> &'static str {
        match self {
            EnvConfig::Local(_) => "local",
            EnvConfig::Remote(_) => "remote",
        }
    }

    /// Every connection declared here, as the id an agent names and the URL it
    /// reaches.
    fn declared(&self) -> Box<dyn Iterator<Item = (&str, &str)> + '_> {
        match self {
            EnvConfig::Local(local) => Box::new(
                local
                    .mcp
                    .iter()
                    .map(|(id, spec)| (id.as_str(), spec.url.as_str())),
            ),
            EnvConfig::Remote(remote) => Box::new(
                remote
                    .mcp
                    .iter()
                    .map(|(id, spec)| (id.as_str(), spec.url.as_str())),
            ),
        }
    }

    fn parse(s: &str, path: &Path) -> Result<Self> {
        let at = path.display();
        // Two steps because serde's internal tagging cannot be combined with
        // `deny_unknown_fields`: read the tag, then deserialize the whole
        // document into the struct it selects, which carries the tag itself so
        // the unknown-key check accounts for it.
        let value: toml::Value = toml::from_str(s).map_err(|e| anyhow!("parsing {at}: {e}"))?;
        let config = match value.get("target").and_then(|t| t.as_str()) {
            Some("local") => {
                EnvConfig::Local(value.try_into().map_err(|e| anyhow!("parsing {at}: {e}"))?)
            }
            Some("remote") => {
                EnvConfig::Remote(value.try_into().map_err(|e| anyhow!("parsing {at}: {e}"))?)
            }
            Some(other) => bail!(
                "{at}: `{other}` is not a target; use target = \"local\" or target = \"remote\""
            ),
            None => bail!("{at} must declare target = \"local\" or target = \"remote\""),
        };
        for (id, url) in config.declared() {
            check_id(id).map_err(|e| anyhow!("{at}: [mcp.{id}]: {e}"))?;
            check_url(url).map_err(|e| anyhow!("{at}: [mcp.{id}]: {e}"))?;
        }
        Ok(config)
    }
}

/// An id becomes the prefix on every tool name the model sees, so it is held to
/// what a tool name may contain rather than to what TOML accepts as a key.
fn check_id(id: &str) -> Result<()> {
    if id.is_empty() {
        bail!("the id is empty");
    }
    if let Some(c) = id
        .chars()
        .find(|c| !c.is_ascii_alphanumeric() && *c != '_' && *c != '-')
    {
        bail!("`{id}` cannot prefix a tool name: `{c}` is not a letter, digit, `_`, or `-`");
    }
    Ok(())
}

/// Rejected while reading the file rather than at the first fetch, where it
/// would surface as a discovery failure against a URL nobody meant to write.
/// A token-backed connection never reaches the OAuth resolver's own check:
/// `EnvCredentials` sends `$token_env` wherever the URL points.
fn check_url(url: &str) -> Result<()> {
    let parsed = reqwest::Url::parse(url).with_context(|| format!("`{url}` is not a URL"))?;
    if parsed.scheme() == "https" || crate::connectors::oauth::is_loopback(url) {
        return Ok(());
    }
    bail!("`{url}` is not https: a credential would cross the network in the clear")
}

#[derive(Debug, Clone)]
pub struct Found {
    pub config: EnvConfig,
    pub path: PathBuf,
}

impl Found {
    pub fn into_local(self, command: &str) -> Result<LocalEnv> {
        match self.config {
            EnvConfig::Local(local) => Ok(local),
            other => bail!(
                "{} declares target = \"{}\"; {command} requires target = \"local\"",
                self.path.display(),
                other.target()
            ),
        }
    }

    pub fn into_remote(self, command: &str) -> Result<RemoteEnv> {
        match self.config {
            EnvConfig::Remote(remote) => Ok(remote),
            other => bail!(
                "{} declares target = \"{}\"; {command} requires target = \"remote\"",
                self.path.display(),
                other.target()
            ),
        }
    }
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

/// The local environment for a command that runs an engine, or the defaults
/// when no file is in play.
pub fn local(path: Option<&Path>, command: &str) -> Result<LocalEnv> {
    Ok(resolve(path)?
        .map(|found| found.into_local(command))
        .transpose()?
        .unwrap_or_default())
}

pub fn find_from(start: &Path) -> Result<Option<Found>> {
    let mut dir: &Path = start;
    loop {
        let candidate = dir.join(FILENAME);
        if candidate.is_file() {
            return load_explicit(&candidate).map(Some);
        }
        match dir.parent() {
            Some(parent) => dir = parent,
            None => return Ok(None),
        }
    }
}

pub fn find() -> Result<Option<Found>> {
    let cwd = env::current_dir().context("could not determine cwd for substructure.toml lookup")?;
    find_from(&cwd)
}

pub fn load_explicit(path: &Path) -> Result<Found> {
    let s = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(Found {
        config: EnvConfig::parse(&s, path)?,
        path: path.to_path_buf(),
    })
}

/// Write `config` back, keeping everything about the file that is not a
/// setting. A machine edit must not cost a reader their comments or their
/// layout, so the parsed document is edited in place rather than replaced.
pub fn write(path: &Path, config: &EnvConfig) -> Result<()> {
    let mut rendered: DocumentMut = match config {
        EnvConfig::Local(local) => toml_edit::ser::to_document(local),
        EnvConfig::Remote(remote) => toml_edit::ser::to_document(remote),
    }
    .context("serializing substructure.toml")?;
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

    fn parse(s: &str) -> Result<EnvConfig> {
        EnvConfig::parse(s, Path::new("substructure.toml"))
    }

    fn local_of(s: &str) -> LocalEnv {
        match parse(s).unwrap() {
            EnvConfig::Local(local) => local,
            other => panic!("expected a local environment, got {other:?}"),
        }
    }

    fn remote_of(s: &str) -> RemoteEnv {
        match parse(s).unwrap() {
            EnvConfig::Remote(remote) => remote,
            other => panic!("expected a remote environment, got {other:?}"),
        }
    }

    #[test]
    fn a_file_without_a_target_says_so() {
        let err = parse("db = \"dev.db\"\n").unwrap_err().to_string();
        assert!(err.contains("must declare target"), "got {err}");

        let err = parse("target = \"cloud\"\n").unwrap_err().to_string();
        assert!(err.contains("local") && err.contains("remote"), "got {err}");
    }

    #[test]
    fn a_local_environment_reads_its_groups() {
        let cfg = local_of(
            r#"
            target = "local"
            db = "dev.substructure.db"
            log = "substructure_core=debug,warn"

            [worker]
            url = "http://localhost:4444"
            signing_secret_env = "SUBS_SIGNING_SECRET"

            [llm]
            provider = "anthropic"

            [run]
            agent = "support"
            output = "pretty"

            [server]
            port = 9000
            dev = true
        "#,
        );
        assert_eq!(cfg.db_path(), "dev.substructure.db");
        assert_eq!(cfg.log.as_deref(), Some("substructure_core=debug,warn"));
        assert_eq!(cfg.worker_url().as_deref(), Some("http://localhost:4444"));
        assert_eq!(cfg.llm_provider(), Some(LlmProviderArg::Anthropic));
        assert_eq!(cfg.run.as_ref().unwrap().agent.as_deref(), Some("support"));
        assert_eq!(cfg.run.unwrap().output, Some(OutputFormat::Pretty));
        let server = cfg.server.unwrap();
        assert_eq!(server.port, Some(9000));
        assert_eq!(server.dev, Some(true));
        // Absent is absent, not a default the flag would then have to beat.
        assert_eq!(server.host, None);
    }

    #[test]
    fn a_remote_environment_reads_its_identity_and_worker() {
        let cfg = remote_of(
            r#"
            target = "remote"
            url = "https://api.substructure.ai"
            org = "org_1"
            app = "app_1"

            [worker]
            url = "https://bot.example.com/agent"

            [mcp.sentry]
            url = "https://mcp.sentry.dev/mcp"
        "#,
        );
        assert_eq!(cfg.org.as_deref(), Some("org_1"));
        assert_eq!(cfg.app.as_deref(), Some("app_1"));
        assert_eq!(
            cfg.worker_url().as_deref(),
            Some("https://bot.example.com/agent")
        );
        assert_eq!(cfg.mcp["sentry"].url, "https://mcp.sentry.dev/mcp");
    }

    #[test]
    fn a_target_only_file_is_valid_and_empty() {
        assert_eq!(local_of("target = \"local\"\n"), LocalEnv::default());
        assert_eq!(remote_of("target = \"remote\"\n"), RemoteEnv::default());
    }

    #[test]
    fn a_misspelled_key_is_a_parse_error_not_a_silent_no_op() {
        let err = parse("target = \"local\"\n[worker]\nuri = \"http://x\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("uri"), "got {err}");
    }

    #[test]
    fn a_key_belonging_to_the_other_target_is_rejected() {
        // `db` is an engine setting: a remote has none.
        let err = parse("target = \"remote\"\ndb = \"dev.db\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("db"), "got {err}");

        // `org` names a cloud app, which a local engine knows nothing about.
        let err = parse("target = \"local\"\norg = \"acme\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("org"), "got {err}");

        // Credentials are the deployment's, so a remote connection names none.
        let err = parse(
            "target = \"remote\"\n[mcp.sentry]\nurl = \"https://x/mcp\"\nauth = { token_env = \"T\" }\n",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("auth"), "got {err}");

        // There is no catalog key: a connection always declares a URL.
        let err = parse("target = \"remote\"\n[mcp.sentry]\ncatalog = \"sentry\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("catalog"), "got {err}");
    }

    #[test]
    fn a_connection_is_checked_where_it_was_typed() {
        // An id prefixes every tool name the model sees.
        let err = parse("target = \"local\"\n[mcp.\"my server\"]\nurl = \"https://x/mcp\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("cannot prefix a tool name"), "got {err}");

        // A credential would cross the network in the clear.
        let err = parse("target = \"remote\"\n[mcp.sentry]\nurl = \"http://mcp.sentry.dev/mcp\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("not https"), "got {err}");

        // Loopback is exempt: nothing off-host sees it.
        parse("target = \"local\"\n[mcp.issues]\nurl = \"http://localhost:4445/mcp\"\n").unwrap();
    }

    #[test]
    fn an_inline_secret_is_a_parse_error() {
        let err = parse("target = \"local\"\n[worker]\nsigning_secret = \"s3cret\"\n")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("signing_secret"),
            "a committed file must not be able to hold a secret; got {err}"
        );
    }

    #[test]
    fn an_output_mode_that_does_not_exist_is_a_parse_error() {
        // It used to fall back to `ag-ui`, so a typo silently changed the mode.
        let err = parse("target = \"local\"\n[run]\noutput = \"pretyy\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("pretyy"), "got {err}");
    }

    #[test]
    fn everything_set_survives_a_round_trip() {
        let cfg = local_of(
            r#"
            target = "local"
            db = "dev.db"
            log = "info"

            [worker]
            url = "http://localhost:4444"
            signing_secret_env = "S"

            [llm]
            provider = "openai"

            [run]
            agent = "support"
            output = "jsonl"

            [server]
            host = "0.0.0.0"
            port = 9000
            dev = true

            [slack]
            agent = "support"

            [mcp.sentry]
            url = "https://mcp.sentry.dev/mcp"
            prefix_tools = false
        "#,
        );
        let written = toml::to_string_pretty(&cfg).unwrap();
        assert_eq!(local_of(&written), cfg, "written back as {written}");

        let remote = remote_of(
            "target = \"remote\"\nurl = \"https://api.test\"\norg = \"o\"\napp = \"a\"\n\
             \n[worker]\nurl = \"https://w.test\"\n\n[slack]\nagent = \"s\"\n\
             \n[mcp.sentry]\nurl = \"https://mcp.sentry.dev/mcp\"\n",
        );
        let written = toml::to_string_pretty(&remote).unwrap();
        assert_eq!(remote_of(&written), remote, "written back as {written}");
    }

    #[test]
    fn unset_settings_are_not_written_back() {
        let cfg = remote_of("target = \"remote\"\norg = \"acme\"\n");
        let out = toml::to_string_pretty(&cfg).unwrap();
        assert_eq!(
            out.trim(),
            "target = \"remote\"\norg = \"acme\"",
            "got {out}"
        );
    }

    #[test]
    fn an_empty_slack_section_is_not_a_configured_bot() {
        assert_eq!(
            local_of("target = \"local\"\n[slack]\n").slack_agent(),
            None
        );
        assert_eq!(LocalEnv::default().slack_agent(), None);

        // The old bare key is gone, and says so rather than doing nothing.
        let err = parse("target = \"local\"\nslack_agent = \"helper\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("slack_agent"), "got {err}");
    }

    #[test]
    fn the_wrong_target_names_the_file_and_the_command() {
        let dir = tmpdir();
        let path = dir.join(FILENAME);
        fs::write(&path, "target = \"remote\"\norg = \"acme\"\n").unwrap();

        let err = load_explicit(&path)
            .unwrap()
            .into_local("`subs run`")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("subs run") && err.contains("\"local\""),
            "got {err}"
        );
        assert!(err.contains(&path.display().to_string()), "got {err}");

        assert!(load_explicit(&path).unwrap().into_remote("x").is_ok());
    }

    #[test]
    fn writing_keeps_comments_layout_and_everything_it_did_not_change() {
        let path = tmpdir().join(FILENAME);
        fs::write(
            &path,
            "# how this app is deployed\n\
             target = \"remote\"\n\
             org = \"old\"        # pinned by hand\n\
             \n\
             [worker]\n\
             # where the agent runs\n\
             url = \"https://bot.example.com/agent\"\n\
             \n\
             [mcp.sentry]\n\
             url = \"https://mcp.sentry.dev/mcp\"\n",
        )
        .unwrap();

        let mut cfg = load_explicit(&path).unwrap().into_remote("x").unwrap();
        cfg.org = Some("new".into());
        cfg.app = Some("app_1".into());
        write(&path, &EnvConfig::Remote(cfg)).unwrap();

        let after = fs::read_to_string(&path).unwrap();
        assert!(after.contains("# how this app is deployed"), "{after}");
        assert!(after.contains("# where the agent runs"), "{after}");
        assert!(
            after.contains("org = \"new\"        # pinned by hand"),
            "{after}"
        );
        assert!(after.contains("app = \"app_1\""), "{after}");
        assert!(after.contains("[mcp.sentry]"), "{after}");
    }

    #[test]
    fn writing_removes_a_setting_that_is_no_longer_set() {
        let path = tmpdir().join(FILENAME);
        fs::write(
            &path,
            "target = \"remote\"\norg = \"acme\"\napp = \"app_1\"\n",
        )
        .unwrap();

        let mut cfg = load_explicit(&path).unwrap().into_remote("x").unwrap();
        cfg.app = None;
        write(&path, &EnvConfig::Remote(cfg)).unwrap();

        let after = fs::read_to_string(&path).unwrap();
        assert!(!after.contains("app"), "{after}");
        assert!(after.contains("org = \"acme\""), "{after}");
    }

    #[test]
    fn an_explicit_config_path_that_is_missing_is_an_error() {
        let missing = tmpdir().join("nope.toml");
        assert!(resolve(Some(&missing)).is_err());
    }

    #[test]
    fn find_walks_up_from_cwd_to_first_match() {
        let root = tmpdir();
        let nested = root.join("a/b/c");
        fs::create_dir_all(&nested).unwrap();
        let cfg_path = root.join(FILENAME);
        fs::write(
            &cfg_path,
            "target = \"remote\"\norg = \"org-x\"\napp = \"app-y\"\n",
        )
        .unwrap();

        let found = find_from(&nested).unwrap().expect("should find ancestor");
        assert_eq!(found.path, cfg_path);
        let remote = found.into_remote("x").unwrap();
        assert_eq!(remote.org.as_deref(), Some("org-x"));
        assert_eq!(remote.app.as_deref(), Some("app-y"));
    }

    #[test]
    fn find_returns_none_when_no_subs_toml_anywhere() {
        let root = tmpdir().join("isolated");
        fs::create_dir_all(&root).unwrap();
        assert!(find_from(&root).unwrap().is_none());
    }

    #[test]
    fn nearest_subs_toml_wins_over_ancestor() {
        let root = tmpdir();
        let nested = root.join("inner");
        fs::create_dir_all(&nested).unwrap();
        fs::write(
            root.join(FILENAME),
            "target = \"remote\"\norg = \"outer\"\n",
        )
        .unwrap();
        fs::write(
            nested.join(FILENAME),
            "target = \"remote\"\norg = \"inner\"\n",
        )
        .unwrap();

        let found = find_from(&nested).unwrap().unwrap();
        assert_eq!(
            found.into_remote("x").unwrap().org.as_deref(),
            Some("inner")
        );
    }
}
