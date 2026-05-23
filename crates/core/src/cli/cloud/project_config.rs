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
    /// Override the cloud API URL for commands run from this tree. Written
    /// when `subs cloud init --url <URL>` is used; respected by all later
    /// commands unless a `--url` flag or `$SUBS_API_URL` overrides it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
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
    let cwd = env::current_dir().context("could not determine cwd for subs.toml lookup")?;
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
        fs::write(
            nested.join("subs.toml"),
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
        fs::write(dir.join("subs.toml"), "").unwrap();
        let found = find_from(&dir).unwrap().unwrap();
        assert!(found.config.org.is_none());
        assert!(found.config.app.is_none());
    }
}
