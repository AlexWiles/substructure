use std::fs;
use std::io::Write;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

pub const DEFAULT_API_URL: &str = "https://api.substructure.ai";
pub const API_URL_ENV: &str = "SUBS_API_URL";

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Credentials {
    pub token: Option<String>,
}

pub fn resolve_api_url(flag: Option<&str>) -> String {
    if let Some(u) = flag {
        return u.to_string();
    }
    if let Ok(u) = std::env::var(API_URL_ENV) {
        if !u.is_empty() {
            return u;
        }
    }
    DEFAULT_API_URL.to_string()
}

pub fn default_path() -> Result<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|h| h.join(".config")))
        .context("could not determine config dir (HOME/XDG_CONFIG_HOME unset)")?;
    Ok(base.join("substructure").join("credentials.toml"))
}

pub fn resolve_path(explicit: Option<PathBuf>) -> Result<PathBuf> {
    match explicit {
        Some(p) => Ok(p),
        None => default_path(),
    }
}

pub fn load(path: &Path) -> Result<Credentials> {
    match fs::read_to_string(path) {
        Ok(s) => {
            toml::from_str::<Credentials>(&s).with_context(|| format!("parsing {}", path.display()))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Credentials::default()),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

pub fn save(path: &Path, creds: &Credentials) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        let _ = fs::set_permissions(parent, fs::Permissions::from_mode(0o700));
    }

    let serialized = toml::to_string_pretty(creds).context("serializing credentials")?;

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

impl Credentials {
    pub fn require_token(&self) -> Result<&str> {
        self.token
            .as_deref()
            .context("not logged in. Run `substructure cloud login` to authenticate.")
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
        let dir = std::env::temp_dir().join(format!("subs-credentials-test-{nanos}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn load_missing_returns_default() {
        let path = tmpdir().join("missing.toml");
        let creds = load(&path).unwrap();
        assert!(creds.token.is_none());
    }

    #[test]
    fn load_ignores_unknown_fields() {
        // Older credentials files carried an `api_url` key. After the move to
        // env-var-only URL config, leftover keys must not break loading.
        let dir = tmpdir();
        let path = dir.join("credentials.toml");
        fs::write(
            &path,
            "api_url = \"http://localhost:5173\"\ntoken = \"ba_secret\"\n",
        )
        .unwrap();
        let loaded = load(&path).unwrap();
        assert_eq!(loaded.token.as_deref(), Some("ba_secret"));
    }

    #[test]
    fn save_round_trip_with_0600_perms() {
        let dir = tmpdir();
        let path = dir.join("credentials.toml");
        let creds = Credentials {
            token: Some("ba_secret".into()),
        };

        save(&path, &creds).unwrap();

        let mode = fs::metadata(&path).unwrap().mode() & 0o777;
        assert_eq!(mode, 0o600, "expected 0600, got {:o}", mode);

        let loaded = load(&path).unwrap();
        assert_eq!(loaded.token.as_deref(), Some("ba_secret"));
    }

    #[test]
    fn save_tightens_existing_loose_perms() {
        let dir = tmpdir();
        let path = dir.join("credentials.toml");
        fs::write(&path, "").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();

        save(&path, &Credentials::default()).unwrap();
        let mode = fs::metadata(&path).unwrap().mode() & 0o777;
        assert_eq!(mode, 0o600);
    }

    #[test]
    fn resolve_api_url_precedence_flag_over_env_over_default() {
        // Snapshot + clear the env so this test is hermetic regardless of
        // what the surrounding shell has set.
        let prev = std::env::var(API_URL_ENV).ok();
        std::env::remove_var(API_URL_ENV);

        assert_eq!(resolve_api_url(None), DEFAULT_API_URL);

        std::env::set_var(API_URL_ENV, "https://env.example");
        assert_eq!(resolve_api_url(None), "https://env.example");
        assert_eq!(
            resolve_api_url(Some("https://flag.example")),
            "https://flag.example"
        );

        // Empty env var falls through to default.
        std::env::set_var(API_URL_ENV, "");
        assert_eq!(resolve_api_url(None), DEFAULT_API_URL);

        match prev {
            Some(v) => std::env::set_var(API_URL_ENV, v),
            None => std::env::remove_var(API_URL_ENV),
        }
    }
}
