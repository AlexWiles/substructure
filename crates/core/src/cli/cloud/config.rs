use std::fs;
use std::io::Write;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

pub const DEFAULT_API_URL: &str = "https://api.substructure.ai";

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub api_url: Option<String>,
    pub token: Option<String>,
}

pub fn default_path() -> Result<PathBuf> {
    let base = dirs::config_dir()
        .context("could not determine config dir (HOME/XDG_CONFIG_HOME unset)")?;
    Ok(base.join("subs").join("config.toml"))
}

pub fn resolve_path(explicit: Option<PathBuf>) -> Result<PathBuf> {
    match explicit {
        Some(p) => Ok(p),
        None => default_path(),
    }
}

pub fn load(path: &Path) -> Result<Config> {
    match fs::read_to_string(path) {
        Ok(s) => {
            toml::from_str::<Config>(&s).with_context(|| format!("parsing {}", path.display()))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Config::default()),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

pub fn save(path: &Path, config: &Config) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
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
    // O_CREAT mode only applies on first creation; reset so pre-existing loose perms get tightened.
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    Ok(())
}

impl Config {
    pub fn resolve_api_url(&self, flag: Option<&str>) -> String {
        flag.map(str::to_string)
            .or_else(|| self.api_url.clone())
            .unwrap_or_else(|| DEFAULT_API_URL.to_string())
    }

    pub fn require_token(&self) -> Result<&str> {
        self.token
            .as_deref()
            .context("not logged in. Run `subs cloud login` to authenticate.")
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
        assert!(cfg.api_url.is_none());
    }

    #[test]
    fn save_round_trip_with_0600_perms() {
        let dir = tmpdir();
        let path = dir.join("config.toml");
        let cfg = Config {
            api_url: Some("https://api.example.com".into()),
            token: Some("ba_secret".into()),
        };

        save(&path, &cfg).unwrap();

        let mode = fs::metadata(&path).unwrap().mode() & 0o777;
        assert_eq!(mode, 0o600, "expected 0600, got {:o}", mode);

        let loaded = load(&path).unwrap();
        assert_eq!(loaded.api_url.as_deref(), Some("https://api.example.com"));
        assert_eq!(loaded.token.as_deref(), Some("ba_secret"));
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
    fn resolve_api_url_precedence_flag_over_config_over_default() {
        let cfg = Config {
            api_url: Some("https://configured.example".into()),
            ..Default::default()
        };
        assert_eq!(
            cfg.resolve_api_url(Some("https://flag.example")),
            "https://flag.example"
        );
        assert_eq!(cfg.resolve_api_url(None), "https://configured.example");
        assert_eq!(Config::default().resolve_api_url(None), DEFAULT_API_URL);
    }
}
