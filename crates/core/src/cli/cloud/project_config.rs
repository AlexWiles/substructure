// Optional, committable project-local pointers: `subs.toml` at the project
// root holds `org` and `app` ids so teammates running `subs cloud apps
// keys list` in the repo all hit the same app without each setting their
// own user-level defaults. Contains NO credentials; only pointer fields.
//
// Discovery walks upward from cwd until a `subs.toml` is found or the
// filesystem root is reached, mirroring how git finds `.git/`.
//
// Precedence (used by Context::require_org / require_app):
//   1. explicit --org / --app flag
//   2. project subs.toml (this file)
//   3. user config (~/.config/subs/config.toml) defaults

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

pub const FILENAME: &str = "subs.toml";

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ProjectConfig {
    pub org: Option<String>,
    pub app: Option<String>,
}

/// Result of discovery: the loaded config and the path it came from. The
/// path is kept so future diagnostics (e.g. `whoami` could mention "from
/// ./subs.toml") can surface it; tests assert on it directly.
#[derive(Debug, Clone)]
pub struct Found {
    pub config: ProjectConfig,
    #[allow(dead_code)]
    pub path: PathBuf,
}

/// Walk upward from `start` looking for `subs.toml`. Returns the first hit
/// or `None` if we reach the filesystem root.
pub fn find_from(start: &Path) -> Result<Option<Found>> {
    let mut dir: &Path = start;
    loop {
        let candidate = dir.join(FILENAME);
        if candidate.is_file() {
            let s = fs::read_to_string(&candidate)
                .with_context(|| format!("reading {}", candidate.display()))?;
            let config: ProjectConfig = toml::from_str(&s)
                .with_context(|| format!("parsing {}", candidate.display()))?;
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

/// Convenience: discover from `std::env::current_dir()`.
pub fn find() -> Result<Option<Found>> {
    let cwd = env::current_dir().context("could not determine cwd for subs.toml lookup")?;
    find_from(&cwd)
}

/// Load a specific project config file (errors if missing). Used when the
/// caller passed `-c/--config` to override discovery.
pub fn load_explicit(path: &Path) -> Result<Found> {
    let s = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let config: ProjectConfig =
        toml::from_str(&s).with_context(|| format!("parsing {}", path.display()))?;
    Ok(Found {
        config,
        path: path.to_path_buf(),
    })
}

/// Write `config` to `path` as TOML. Used by `subs cloud init`.
pub fn write(path: &Path, config: &ProjectConfig) -> Result<()> {
    let serialized = toml::to_string_pretty(config).context("serializing subs.toml")?;
    fs::write(path, serialized).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir() -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("subs-project-test-{nanos}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn find_walks_up_from_cwd_to_first_match() {
        let root = tmpdir();
        let nested = root.join("a/b/c");
        fs::create_dir_all(&nested).unwrap();
        let cfg_path = root.join("subs.toml");
        fs::write(&cfg_path, "org = \"org-x\"\napp = \"app-y\"\n").unwrap();

        let found = find_from(&nested).unwrap().expect("should find ancestor");
        assert_eq!(found.path, cfg_path);
        assert_eq!(found.config.org.as_deref(), Some("org-x"));
        assert_eq!(found.config.app.as_deref(), Some("app-y"));
    }

    #[test]
    fn find_returns_none_when_no_subs_toml_anywhere() {
        // Use a deep tmp dir that won't accidentally collide with a real
        // subs.toml on the host.
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
        fs::write(root.join("subs.toml"), "org = \"outer\"\n").unwrap();
        fs::write(nested.join("subs.toml"), "org = \"inner\"\napp = \"app-i\"\n").unwrap();

        let found = find_from(&nested).unwrap().unwrap();
        assert_eq!(found.config.org.as_deref(), Some("inner"));
        assert_eq!(found.config.app.as_deref(), Some("app-i"));
    }

    #[test]
    fn missing_fields_default_to_none() {
        let dir = tmpdir();
        fs::write(dir.join("subs.toml"), "").unwrap();
        let found = find_from(&dir).unwrap().unwrap();
        assert!(found.config.org.is_none());
        assert!(found.config.app.is_none());
    }
}
