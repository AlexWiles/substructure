// User config + bearer token at ~/.config/subs/config.toml. Single file,
// 0600 perms, written atomically. No env-var or project-file fallbacks —
// every command takes flags (`--url`, `--org`, `--app`, `--config`) for
// one-off overrides.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Hardcoded fallback if the user hasn't pinned an `api_url` in config and
/// hasn't passed `--url`. Update when prod moves.
pub const DEFAULT_API_URL: &str = "https://api.substructure.ai";

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub api_url: Option<String>,
    pub token: Option<String>,
    pub default_org: Option<String>,
    pub orgs: BTreeMap<String, OrgConfig>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OrgConfig {
    pub default_app: Option<String>,
}

/// Default config path: `~/.config/subs/config.toml`. Honors `XDG_CONFIG_HOME`
/// on Linux via `dirs::config_dir()`.
pub fn default_path() -> Result<PathBuf> {
    let base = dirs::config_dir()
        .context("could not determine config dir (HOME/XDG_CONFIG_HOME unset)")?;
    Ok(base.join("subs").join("config.toml"))
}

/// Resolve the config path: explicit override, otherwise the default.
pub fn resolve_path(explicit: Option<PathBuf>) -> Result<PathBuf> {
    match explicit {
        Some(p) => Ok(p),
        None => default_path(),
    }
}

/// Load the config from `path`, returning `Default::default()` if the file is
/// missing. Errors only on read/parse failure.
pub fn load(path: &Path) -> Result<Config> {
    match fs::read_to_string(path) {
        Ok(s) => {
            toml::from_str::<Config>(&s).with_context(|| format!("parsing {}", path.display()))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Config::default()),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

/// Atomically persist `config` to `path` with 0600 perms. Creates the parent
/// dir (0700) on first write.
pub fn save(path: &Path, config: &Config) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        // Best-effort: tighten dir perms even on subsequent saves. Skip on
        // failure (e.g. user has a custom umask we shouldn't fight).
        let _ = fs::set_permissions(parent, fs::Permissions::from_mode(0o700));
    }

    let serialized = toml::to_string_pretty(config).context("serializing config")?;

    let tmp = path.with_extension("toml.tmp");
    {
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&tmp)
            .with_context(|| format!("opening {}", tmp.display()))?;
        f.write_all(serialized.as_bytes())?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path)
        .with_context(|| format!("renaming {} -> {}", tmp.display(), path.display()))?;
    // The mode flag on create above only applies on first creation; reset
    // explicitly so a pre-existing file with looser perms gets tightened.
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    Ok(())
}

impl Config {
    /// Resolved API URL: flag > config > hardcoded default.
    pub fn resolve_api_url(&self, flag: Option<&str>) -> String {
        flag.map(str::to_string)
            .or_else(|| self.api_url.clone())
            .unwrap_or_else(|| DEFAULT_API_URL.to_string())
    }

    /// Resolved bearer token. Errors if neither is set.
    pub fn require_token(&self) -> Result<&str> {
        self.token
            .as_deref()
            .context("no token found — run `subs cloud login`")
    }

    /// Resolved org id: flag > config.default_org. Returns None if unset.
    pub fn resolve_org(&self, flag: Option<&str>) -> Option<String> {
        flag.map(str::to_string)
            .or_else(|| self.default_org.clone())
    }

    /// Resolved app id for a given org: flag > orgs[org].default_app.
    pub fn resolve_app(&self, org_id: &str, flag: Option<&str>) -> Option<String> {
        flag.map(str::to_string)
            .or_else(|| self.orgs.get(org_id).and_then(|o| o.default_app.clone()))
    }

    pub fn set_default_app(&mut self, org_id: &str, app_id: String) {
        self.orgs.entry(org_id.to_string()).or_default().default_app = Some(app_id);
    }

    pub fn clear_default_app(&mut self, org_id: &str) {
        if let Some(o) = self.orgs.get_mut(org_id) {
            o.default_app = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::MetadataExt;

    fn tmpdir() -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("subs-config-test-{nanos}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn load_missing_returns_default() {
        let path = tmpdir().join("missing.toml");
        let cfg = load(&path).unwrap();
        assert!(cfg.token.is_none());
        assert!(cfg.default_org.is_none());
        assert!(cfg.orgs.is_empty());
    }

    #[test]
    fn save_round_trip_with_0600_perms() {
        let dir = tmpdir();
        let path = dir.join("config.toml");
        let mut cfg = Config {
            api_url: Some("https://api.example.com".into()),
            token: Some("ba_secret".into()),
            default_org: Some("org-1".into()),
            ..Default::default()
        };
        cfg.set_default_app("org-1", "app-1".into());

        save(&path, &cfg).unwrap();

        // perms
        let mode = fs::metadata(&path).unwrap().mode() & 0o777;
        assert_eq!(mode, 0o600, "expected 0600, got {:o}", mode);

        // round-trip
        let loaded = load(&path).unwrap();
        assert_eq!(loaded.api_url.as_deref(), Some("https://api.example.com"));
        assert_eq!(loaded.token.as_deref(), Some("ba_secret"));
        assert_eq!(loaded.default_org.as_deref(), Some("org-1"));
        assert_eq!(
            loaded
                .orgs
                .get("org-1")
                .and_then(|o| o.default_app.as_deref()),
            Some("app-1")
        );
    }

    #[test]
    fn save_tightens_existing_loose_perms() {
        let dir = tmpdir();
        let path = dir.join("config.toml");
        fs::write(&path, "").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();

        save(&path, &Config::default()).unwrap();
        let mode = fs::metadata(&path).unwrap().mode() & 0o777;
        assert_eq!(mode, 0o600);
    }

    #[test]
    fn resolve_precedence_flag_over_config_over_default() {
        let cfg = Config {
            api_url: Some("https://configured.example".into()),
            default_org: Some("cfg-org".into()),
            ..Default::default()
        };

        assert_eq!(
            cfg.resolve_api_url(Some("https://flag.example")),
            "https://flag.example"
        );
        assert_eq!(cfg.resolve_api_url(None), "https://configured.example");
        assert_eq!(Config::default().resolve_api_url(None), DEFAULT_API_URL);

        assert_eq!(
            cfg.resolve_org(Some("flag-org")).as_deref(),
            Some("flag-org")
        );
        assert_eq!(cfg.resolve_org(None).as_deref(), Some("cfg-org"));
    }
}
