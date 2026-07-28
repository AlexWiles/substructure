use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::cli::env::LlmProviderArg;
use crate::connectors::registry::ConnectionSpec;
use crate::protocol::ConnectorProtocol;

pub const FILENAME: &str = "substructure.toml";

/// A project's settings: which cloud app it targets, what `subs run` and `subs
/// serve` would otherwise take on argv, and the connections a local engine can
/// reach.
///
/// Precedence for anything the CLI also accepts as a flag is
/// **flag > environment > this > default**, so pinning something here never
/// takes an override away.
///
/// Per-invocation arguments are deliberately absent: `--input`, `--session`, and
/// `--config` itself say what one run is doing, not how the project is set up.
/// Secrets are absent for the same reason they are absent from `[mcp]` —
/// a committed file must not be able to hold one, so the signing secret is named
/// rather than written.
///
/// `deny_unknown_fields` so a misspelled key is a parse error rather than a
/// setting that silently does nothing. `connectors` is declared last because
/// TOML puts every bare key above the first table.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProjectConfig {
    pub org: Option<String>,
    pub app: Option<String>,
    /// Override the cloud API URL for commands run from this tree. Written
    /// when `subs link --url <URL>` is used; respected by all later
    /// commands unless a `--url` flag or `$SUBS_API_URL` overrides it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Worker endpoint the engine POSTs decisions to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_url: Option<String>,
    /// LLM provider for engine-executed calls: `anthropic`, `openai`, `openrouter`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_provider: Option<LlmProviderArg>,
    /// SQLite database path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub db: Option<String>,
    /// Default agent id for `subs run`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
    /// Output mode for `subs run`: `ag-ui`, `jsonl`, or `pretty`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    /// Environment variable holding the worker signing secret. Named, never
    /// written — same rule as a connection's `token_env`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signing_secret_env: Option<String>,
    /// Log filter for `subs run` and `subs serve`, in `RUST_LOG` syntax:
    /// a bare level (`info`) or per-target directives
    /// (`substructure_core=debug,warn`). `$RUST_LOG` still wins.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log: Option<String>,
    /// Bind address for `subs serve`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,
    /// Disable client and worker authentication. Local development only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dev: Option<bool>,
    /// Agent id for the Slack Socket Mode bot.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slack_agent: Option<String>,
    /// MCP servers a local engine can reach, keyed by the id an agent config
    /// names. The cloud reads these from its own registry instead; the agent
    /// config is identical either way.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub mcp: BTreeMap<String, ConnectionSpec>,
}

impl ProjectConfig {
    /// Every connection this project declares, keyed by the id an agent names.
    ///
    /// The one place the per-protocol sections become one registry, and so the
    /// one place a second protocol is added: `[a2a]` folds in here with its own
    /// `ConnectorProtocol`, and ids must stay unique across sections because an
    /// agent references a bare id and tool names are prefixed from it. With a
    /// single section today a duplicate cannot be expressed, so there is nothing
    /// yet to reject.
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
}

/// The project file for `path`, or the nearest one above the working directory.
/// An explicit path that does not resolve is an error; discovery finding nothing
/// is not.
pub fn resolve(path: Option<&Path>) -> Result<Option<ProjectConfig>> {
    match path {
        Some(p) => load_explicit(p).map(|f| Some(f.config)),
        None => find().map(|f| f.map(|f| f.config)),
    }
}

#[derive(Debug, Clone)]
pub struct Found {
    pub config: ProjectConfig,
    #[allow(dead_code)]
    pub path: PathBuf,
}

pub fn find_from(start: &Path) -> Result<Option<Found>> {
    let mut dir: &Path = start;
    loop {
        let candidate = dir.join(FILENAME);
        if candidate.is_file() {
            let s = fs::read_to_string(&candidate)
                .with_context(|| format!("reading {}", candidate.display()))?;
            let config: ProjectConfig =
                toml::from_str(&s).with_context(|| format!("parsing {}", candidate.display()))?;
            return Ok(Some(Found {
                config,
                path: candidate,
            }));
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
    let config: ProjectConfig =
        toml::from_str(&s).with_context(|| format!("parsing {}", path.display()))?;
    Ok(Found {
        config,
        path: path.to_path_buf(),
    })
}

pub fn write(path: &Path, config: &ProjectConfig) -> Result<()> {
    let serialized = toml::to_string_pretty(config).context("serializing substructure.toml")?;
    fs::write(path, serialized).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
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

    #[test]
    fn settings_parse_from_the_top_level() {
        let cfg: ProjectConfig = toml::from_str(
            r#"
            org = "acme"
            worker_url = "http://localhost:4444"
            llm_provider = "anthropic"
            port = 9000
            dev = true
            log = "substructure_core=debug,warn"
        "#,
        )
        .unwrap();
        assert_eq!(cfg.worker_url.as_deref(), Some("http://localhost:4444"));
        assert_eq!(cfg.llm_provider, Some(LlmProviderArg::Anthropic));
        assert_eq!(cfg.port, Some(9000));
        assert_eq!(cfg.dev, Some(true));
        assert_eq!(cfg.log.as_deref(), Some("substructure_core=debug,warn"));
        // Absent is absent, not a default the flag would then have to beat.
        assert_eq!(cfg.db, None);
        assert_eq!(cfg.host, None);
    }

    #[test]
    fn unset_settings_are_not_written_back() {
        let cfg: ProjectConfig = toml::from_str("org = \"acme\"\n").unwrap();
        let out = toml::to_string_pretty(&cfg).unwrap();
        assert_eq!(out.trim(), "org = \"acme\"", "got {out}");
    }

    #[test]
    fn mcp_serializes_after_the_bare_keys() {
        // TOML puts every bare key above the first table, so a setting declared
        // after `mcp` would be written inside `[mcp.x]`.
        let cfg: ProjectConfig = toml::from_str(
            "org = \"acme\"\nport = 9000\n\n[mcp.sentry]\nurl = \"https://x/mcp\"\n",
        )
        .unwrap();
        let out = toml::to_string_pretty(&cfg).unwrap();
        let round: ProjectConfig = toml::from_str(&out).unwrap();
        assert_eq!(round, cfg, "written back as {out}");
    }

    #[test]
    fn a_misspelled_key_is_a_parse_error_not_a_silent_no_op() {
        let err = toml::from_str::<ProjectConfig>(
            r#"
            worker_uri = "http://localhost:4444"
        "#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("worker_uri"), "got {err}");
    }

    #[test]
    fn an_inline_signing_secret_is_a_parse_error() {
        let err = toml::from_str::<ProjectConfig>(
            r#"
            signing_secret = "s3cret"
        "#,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("signing_secret"),
            "a committed file must not be able to hold a secret; got {err}"
        );
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
        let cfg_path = root.join("substructure.toml");
        fs::write(&cfg_path, "org = \"org-x\"\napp = \"app-y\"\n").unwrap();

        let found = find_from(&nested).unwrap().expect("should find ancestor");
        assert_eq!(found.path, cfg_path);
        assert_eq!(found.config.org.as_deref(), Some("org-x"));
        assert_eq!(found.config.app.as_deref(), Some("app-y"));
    }

    #[test]
    fn find_returns_none_when_no_subs_toml_anywhere() {
        let root = tmpdir().join("isolated");
        fs::create_dir_all(&root).unwrap();
        let found = find_from(&root).unwrap();
        assert!(found.is_none());
    }

    #[test]
    fn nearest_subs_toml_wins_over_ancestor() {
        let root = tmpdir();
        let nested = root.join("inner");
        fs::create_dir_all(&nested).unwrap();
        fs::write(root.join("substructure.toml"), "org = \"outer\"\n").unwrap();
        fs::write(
            nested.join("substructure.toml"),
            "org = \"inner\"\napp = \"app-i\"\n",
        )
        .unwrap();

        let found = find_from(&nested).unwrap().unwrap();
        assert_eq!(found.config.org.as_deref(), Some("inner"));
        assert_eq!(found.config.app.as_deref(), Some("app-i"));
    }

    #[test]
    fn missing_fields_default_to_none() {
        let dir = tmpdir();
        fs::write(dir.join("substructure.toml"), "").unwrap();
        let found = find_from(&dir).unwrap().unwrap();
        assert!(found.config.org.is_none());
        assert!(found.config.app.is_none());
    }
}
