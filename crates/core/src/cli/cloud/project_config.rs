use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use toml_edit::{DocumentMut, Item, Table, Value};

use crate::cli::env::{ProviderBinding, ProviderKind};
use crate::connectors::registry::{ConnectionDecl, ConnectionPath, ConnectionSpec};
use crate::manifest::{AgentSection, Manifest, PluginSpec, ProviderSpec, ResolvedPlugins};
use crate::runtime::llm::LlmBlocks;
use crate::runtime::worker::AgentEntry;

pub const FILENAME: &str = "subs.toml";
pub const DEFAULT_DB: &str = "subs.db";

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProjectConfig {
    #[serde(skip)]
    source: PathBuf,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub db: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub log: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_subagent_depth: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_worker: Option<String>,

    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub llm: BTreeMap<String, ProviderSpec>,

    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub worker: BTreeMap<String, crate::worker::WorkerBlock>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub agent: BTreeMap<String, AgentSection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub serve: Option<ServeConfig>,

    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub mcp: BTreeMap<String, ConnectionDecl>,

    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub plugin: BTreeMap<String, PluginSpec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remote: Option<Remote>,
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Remote {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub org: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServeConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth: Option<bool>,

    #[serde(
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::size::de"
    )]
    pub max_body: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub public_url: Option<String>,
}

/// Base64 is 4/3 the bytes, so this carries a file of about 6 MB.
pub const MAX_BODY: u64 = 8 * 1024 * 1024;

impl ProjectConfig {
    pub fn manifest(&self) -> Manifest {
        Manifest {
            name: self.name.clone(),
            worker: self.worker.clone(),
            default_worker: self.default_worker.clone(),
            max_subagent_depth: self.max_subagent_depth,
            llm: self.llm.clone(),
            agent: self.agent.clone(),
            mcp: self.mcp.clone(),
            plugin: self.plugin.clone(),
        }
    }

    pub fn resolved_manifest(&self) -> anyhow::Result<(Manifest, ResolvedPlugins)> {
        let mut manifest = self.manifest();
        let base = self
            .source
            .parent()
            .filter(|d| !d.as_os_str().is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        let resolved = manifest.resolve_plugins(&base)?;
        Ok((manifest, resolved))
    }

    pub fn resolved_connections(&self) -> anyhow::Result<BTreeMap<ConnectionPath, ConnectionSpec>> {
        Ok(self.resolved_manifest()?.0.connections())
    }

    pub fn db_path(&self) -> String {
        let Some(named) = self.db.clone() else {
            return user_db_path();
        };
        match self.source.parent() {
            Some(dir) if !dir.as_os_str().is_empty() => dir.join(named).display().to_string(),
            _ => named,
        }
    }

    pub fn llm_blocks(&self) -> LlmBlocks {
        self.manifest().llm_blocks()
    }

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
                    cache_ttl: spec.cache_ttl.clone(),
                })
            })
            .collect()
    }

    pub fn agents(&self) -> BTreeMap<String, AgentEntry> {
        self.manifest().agents()
    }

    pub fn agent_ids(&self) -> Vec<String> {
        self.agent.keys().cloned().collect()
    }

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

    pub fn remote_mut(&mut self) -> &mut Remote {
        self.remote.get_or_insert_with(Remote::default)
    }

    pub(crate) fn parse(s: &str, path: &Path) -> Result<Self> {
        let at = path.display();
        let value: toml::Value = toml::from_str(s).map_err(|e| anyhow!("parsing {at}: {e}"))?;
        let mut config: ProjectConfig =
            value.try_into().map_err(|e| anyhow!("parsing {at}: {e}"))?;

        for (id, spec) in &config.plugin {
            if spec.bundle.is_some() {
                return Err(anyhow!(
                    "{at}: [plugin.{id}]: `bundle` is resolved data and does not belong in the \
                     file. Write `path` and let the CLI resolve it."
                ));
            }
        }
        config
            .manifest()
            .validate()
            .map_err(|e| anyhow!("{at}: {e}"))?;
        config.source = path.to_path_buf();
        Ok(config)
    }
}

pub fn ensure_parent(path: &str) -> Result<()> {
    let Some(dir) = Path::new(path)
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
    else {
        return Ok(());
    };
    match super::credentials::config_dir().is_ok_and(|c| c == dir) {
        true => super::credentials::ensure_config_dir(dir),
        false => fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display())),
    }
}

fn user_db_path() -> String {
    match super::credentials::config_dir() {
        Ok(dir) => dir.join(DEFAULT_DB).display().to_string(),
        Err(_) => DEFAULT_DB.to_string(),
    }
}

#[derive(Debug, Clone)]
pub struct Found {
    pub config: ProjectConfig,
    pub path: PathBuf,
}

pub fn resolve(path: Option<&Path>) -> Result<Option<Found>> {
    match path {
        Some(p) => load_explicit(p).map(Some),
        None => find(),
    }
}

pub fn load(path: Option<&Path>) -> Result<ProjectConfig> {
    Ok(resolve(path)?.map(|found| found.config).unwrap_or_default())
}

pub fn find_from(dir: &Path) -> Result<Option<Found>> {
    let candidate = dir.join(FILENAME);
    match candidate.is_file() {
        true => load_explicit(&candidate).map(Some),
        false => Ok(None),
    }
}

pub fn find() -> Result<Option<Found>> {
    let cwd = env::current_dir().context("could not determine cwd for subs.toml lookup")?;
    find_from(&cwd)
}

pub fn load_explicit(path: &Path) -> Result<Found> {
    let s = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(Found {
        config: ProjectConfig::parse(&s, path)?,
        path: path.to_path_buf(),
    })
}

pub fn write(path: &Path, config: &ProjectConfig) -> Result<()> {
    let mut rendered: DocumentMut =
        toml_edit::ser::to_document(config).context("serializing subs.toml")?;
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

        if !table.is_empty() && table.iter().all(|(_, child)| child.is_table()) {
            table.set_implicit(true);
        }
    }
}

fn merge(target: &mut Table, source: &Table) {
    target.retain(|key, _| source.contains_key(key));
    for (key, item) in source.iter() {
        match (target.get_mut(key), item) {
            (Some(Item::Table(existing)), Item::Table(next)) => merge(existing, next),

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
        ProjectConfig::parse(s, Path::new("subs.toml"))
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
        assert_eq!(deployment.db_path(), user_db_path());

        let both = ok(r#"
            name = "support-bot"
            db = "prod.db"

            [llm.claude]
            type = "anthropic"

            [worker.bot]
            url = "https://bot.example.com/agent"

            [agent.support]
            llm = "claude"
            model = "claude-sonnet-4-5"
            worker = "bot"

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
        assert!(ProjectConfig::default().serve_auth());
    }

    #[test]
    fn no_file_puts_the_database_beside_the_credentials() {
        let expected = super::super::credentials::config_dir()
            .unwrap()
            .join(DEFAULT_DB);
        assert_eq!(
            ProjectConfig::default().db_path(),
            expected.display().to_string()
        );
    }

    #[test]
    fn a_file_that_names_no_database_uses_the_one_beside_the_credentials() {
        let dir = tmpdir();
        let named = |name: &str| {
            let path = dir.join(name);
            fs::write(&path, "[serve]\nport = 9000\n").unwrap();
            load_explicit(&path).unwrap().config.db_path()
        };

        assert_eq!(named(FILENAME), user_db_path());
        assert_eq!(named("subs.staging.toml"), user_db_path());
    }

    #[test]
    fn a_named_database_sits_beside_the_file_that_names_it() {
        let dir = tmpdir();
        let path = dir.join("explicit.toml");
        fs::write(&path, "db = \"engine.db\"\n").unwrap();
        assert_eq!(
            load_explicit(&path).unwrap().config.db_path(),
            dir.join("engine.db").to_str().unwrap()
        );

        let elsewhere = tmpdir().join("far.db");
        fs::write(&path, format!("db = {:?}\n", elsewhere.to_str().unwrap())).unwrap();
        assert_eq!(
            load_explicit(&path).unwrap().config.db_path(),
            elsewhere.to_str().unwrap()
        );
    }

    #[test]
    fn max_body_takes_bytes_or_a_size_word() {
        let body = |toml: &str| ok(toml).serve.and_then(|s| s.max_body);
        assert_eq!(body("[serve]\nmax_body = \"32mb\"\n"), Some(32 << 20));
        assert_eq!(body("[serve]\nmax_body = 4096\n"), Some(4096));
        assert_eq!(body("[serve]\nport = 9000\n"), None, "the default applies");
        let err = parse("[serve]\nmax_body = \"huge\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("huge"), "{err}");
    }

    #[test]
    fn the_engine_groups_read_back() {
        let cfg = ok(r#"
            db = "dev.subs.db"
            log = "substructure_core=debug,warn"

            [llm.claude]
            type = "anthropic"

            [worker.local]
            url = "http://localhost:4444"

            [agent.support]
            llm = "claude"
            model = "claude-sonnet-4-5"
            worker = "local"

            [serve]
            port = 9000
            auth = false
        "#);
        assert_eq!(cfg.db_path(), "dev.subs.db");
        assert_eq!(cfg.log.as_deref(), Some("substructure_core=debug,warn"));
        assert_eq!(cfg.agent["support"].worker.as_deref(), Some("local"));
        assert_eq!(
            cfg.worker["local"].url.as_deref(),
            Some("http://localhost:4444")
        );
        assert_eq!(cfg.llm["claude"].kind, ProviderKind::Anthropic);
        assert!(!cfg.serve_auth());
        let serve = cfg.serve.unwrap();
        assert_eq!(serve.port, Some(9000));

        assert_eq!(serve.host, None);
    }

    #[test]
    fn a_misspelled_key_is_a_parse_error_not_a_silent_no_op() {
        let err = parse(
            "[llm.claude]\ntype = \"anthropic\"\n\n\
             [agent.a]\nllm = \"claude\"\nmodel = \"m\"\nsytem = \"be brief\"\n",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("sytem"), "got {err}");

        let err = parse("[remote]\nnmae = \"x\"\n").unwrap_err().to_string();
        assert!(err.contains("nmae"), "got {err}");

        let err = parse("[mcp.sentry]\ncatalog = \"sentry\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("catalog"), "got {err}");
    }

    #[test]
    fn a_connection_is_checked_where_it_was_typed() {
        let err = parse("[mcp.\"my server\"]\nurl = \"https://x/mcp\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("cannot prefix a tool name"), "got {err}");

        let err = parse("[mcp.sentry]\nurl = \"mcp.sentry.dev\"\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("is not a URL"), "got {err}");

        ok("[mcp.issues]\nurl = \"http://localhost:4445/mcp\"\n");
        ok("[mcp.issues]\nurl = \"http://mcp.internal:8080/mcp\"\n");
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

    #[test]
    fn an_agent_may_delegate_everything_to_its_worker() {
        let cfg = ok(r#"
            [worker.reggu]
            url = "http://localhost:4000/substructure/agent"

            [agent.reggu]
            worker = "reggu"
        "#);
        let entry = &cfg.agents()["reggu"];
        assert!(entry.config.is_none(), "nothing to seed");
        assert_eq!(
            entry.hosting,
            crate::worker::Hosting::Worker("reggu".to_string()),
            "the worker authors it"
        );
    }

    #[test]
    fn an_agent_that_declares_nothing_at_all_is_an_error() {
        let err = parse("[agent.a]\n").unwrap_err().to_string();
        assert!(err.contains("declares nothing"), "got {err}");
    }

    #[test]
    fn a_partly_declared_config_is_an_error() {
        let err = parse(
            "[worker.a]\nurl = \"https://a/agent\"\n[llm.claude]\ntype = \"anthropic\"\n\n\
             [agent.a]\nllm = \"claude\"\nworker = \"a\"\n",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("no `model`"), "got {err}");

        let err = parse(
            "[worker.a]\nurl = \"https://a/agent\"\n\
             [agent.a]\nsystem = \"be brief\"\nworker = \"a\"\n",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("no `llm`"), "got {err}");
    }

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
        assert!(err.contains("names no [mcp.sentry]"), "got {err}");
    }

    #[test]
    fn an_agent_on_a_worker_block_needs_a_worker() {
        let err = parse(
            "[llm.byo]\ntype = \"worker\"\nformat = \"anthropic\"\n\n\
             [agent.a]\nllm = \"byo\"\nmodel = \"m\"\n",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("needs a `worker`"), "got {err}");

        ok(
            "[llm.byo]\ntype = \"worker\"\n[worker.a]\nurl = \"https://a/agent\"\n\n\
            [agent.a]\nllm = \"byo\"\nmodel = \"m\"\nworker = \"a\"\n",
        );
    }

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

    #[test]
    fn an_undeclared_worker_reference_is_an_error() {
        let err = parse(
            "[llm.claude]\ntype = \"anthropic\"\n\n\
             [agent.a]\nllm = \"claude\"\nmodel = \"m\"\nworker = \"typo\"\n",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("names no block"), "got {err}");
    }

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

    #[test]
    fn a_default_worker_reaches_the_manifest() {
        let cfg = ok(
            "default_worker = \"main\"\n[worker.main]\nurl = \"https://api.example.com/agent\"\n",
        );
        let m = cfg.manifest();
        assert_eq!(m.default_worker.as_deref(), Some("main"));
        let main = m.worker.get("main").expect("declared");
        assert_eq!(main.url.as_deref(), Some("https://api.example.com/agent"));
    }

    #[test]
    fn agents_become_directory_entries() {
        let cfg = ok(r#"
            [llm.claude]
            type = "anthropic"

            [agent.assistant]
            llm = "claude"
            model = "claude-sonnet-4-5"

            [worker.triage]
            url = "https://triage.internal/agent"

            [agent.triage]
            llm = "claude"
            model = "claude-haiku-4-5"
            worker = "triage"
        "#);
        let agents = cfg.agents();
        assert_eq!(agents["assistant"].hosting, crate::worker::Hosting::Engine);
        assert!(agents["assistant"].config.is_some(), "and needs a config");
        assert_eq!(
            agents["triage"].hosting,
            crate::worker::Hosting::Worker("triage".to_string())
        );

        let wire = serde_json::to_value(agents["triage"].config.as_ref().unwrap()).unwrap();
        assert!(wire.get("worker").is_none(), "got {wire}");
        assert_eq!(wire["llm"], "claude");
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

            [worker.local]
            url = "http://localhost:4444"

            [agent.support]
            llm = "cheap"
            model = "gpt-5-mini"
            system = "Be brief."
            worker = "local"
            mcp = ["sentry"]
            subagents = ["researcher"]
            tools = [{ name = "confirm", description = "Ask", handler = "client" }]

            [agent.researcher]
            description = "Finds sources"
            llm = "cheap"
            model = "gpt-5-mini"

            [serve]
            host = "0.0.0.0"
            port = 9000
            auth = false

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

    #[test]
    fn a_written_file_keeps_its_scalars_above_its_sections() {
        let path = tmpdir().join(FILENAME);
        let mut cfg = ok("db = \"dev.db\"\n[llm.claude]\ntype = \"anthropic\"\n\n\
             [agent.a]\nllm = \"claude\"\nmodel = \"m\"\n");
        cfg.name = Some("support-bot".into());
        cfg.remote_mut().project = Some("proj_1".into());
        write(&path, &cfg).unwrap();

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
             [worker.bot]\n\
             url = \"https://bot.example.com/agent\"\n\
             \n\
             [agent.support]\n\
             llm = \"claude\"\n\
             model = \"claude-sonnet-4-5\"\n\
             # where the agent runs\n\
             worker = \"bot\"\n\
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

        fs::write(nested.join(FILENAME), "[remote]\norg = \"inner\"\n").unwrap();
        assert_eq!(
            find_from(&nested).unwrap().unwrap().config.org(),
            Some("inner")
        );
    }
}

#[cfg(test)]
mod plugin_file_tests {
    use super::*;

    #[test]
    fn a_committed_bundle_is_a_parse_error() {
        let err = ProjectConfig::parse(
            r#"
            [plugin.pdf]
            path = "./plugins/pdf"
            [plugin.pdf.bundle]
            name = "pdf-tools"
            "#,
            Path::new("subs.toml"),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("resolved data"), "{err}");
    }
}
