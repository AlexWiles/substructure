//! Where a command acts. The `[remote]` section decides.
//!
//! Credential and bootstrap commands do not use this rule. They must work
//! before a file has a `[remote]`.

use anyhow::{bail, Result};

use super::cloud::project_config::{self, ProjectConfig};
use super::cloud::CloudGlobals;

pub enum Target {
    Here(Box<ProjectConfig>),
    Deployment,
}

impl Target {
    pub fn here(self) -> Option<ProjectConfig> {
        match self {
            Target::Here(config) => Some(*config),
            Target::Deployment => None,
        }
    }
}

/// Precedence: `--url`, then `[remote]`, then here. No file reads as a
/// deployment.
pub fn target(globals: &CloudGlobals) -> Result<Target> {
    if globals.url.is_some() {
        return Ok(Target::Deployment);
    }
    let Some(found) = project_config::resolve(globals.config.as_deref())? else {
        return Ok(Target::Deployment);
    };
    Ok(match found.config.remote.is_none() {
        true => Target::Here(Box::new(found.config)),
        false => Target::Deployment,
    })
}

pub fn no_deployment(what: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "{what} acts on a deployment, and subs.toml has no [remote].\n\
         Point it at one:\n  \
         subs link                    pin an org and project in the file\n  \
         --url <url>                  target a deployment for this command"
    )
}

pub fn require_deployment(globals: &CloudGlobals, what: &str) -> Result<()> {
    match target(globals)? {
        Target::Deployment => Ok(()),
        Target::Here(_) => bail!(no_deployment(what)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tmpdir() -> PathBuf {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("subs-target-test-{nanos}-{seq}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    const ENGINE_HERE: &str = "[llm.byo]\ntype = \"worker\"\n\
         [agent.support]\nllm = \"byo\"\nmodel = \"m\"\nworker = \"https://w.test\"\n";
    const A_REMOTE: &str = "[remote]\nurl = \"https://subs.test\"\n";

    fn wrote(body: &str) -> CloudGlobals {
        let dir = tmpdir();
        let path = dir.join(project_config::FILENAME);
        std::fs::write(&path, body).unwrap();
        CloudGlobals {
            config: Some(path),
            ..Default::default()
        }
    }

    fn is_here(globals: &CloudGlobals) -> bool {
        matches!(target(globals).unwrap(), Target::Here(_))
    }

    #[test]
    fn the_remote_section_decides() {
        assert!(is_here(&wrote(ENGINE_HERE)));
        assert!(!is_here(&wrote(A_REMOTE)));
    }

    #[test]
    fn the_url_flag_wins_over_a_file_that_names_no_remote() {
        let mut globals = wrote(ENGINE_HERE);
        globals.url = Some("https://elsewhere.test".into());
        assert!(!is_here(&globals));
    }

    #[test]
    fn a_config_path_that_does_not_resolve_is_an_error() {
        let globals = CloudGlobals {
            config: Some(tmpdir().join("nowhere/subs.toml")),
            ..Default::default()
        };
        assert!(target(&globals).is_err());
    }

    #[test]
    fn a_deployment_only_command_says_how_to_get_one() {
        let err = require_deployment(&wrote(ENGINE_HERE), "subs keys")
            .unwrap_err()
            .to_string();
        assert!(err.contains("subs.toml has no [remote]"), "{err}");
        assert!(err.contains("subs link"), "{err}");
        assert!(err.contains("--url"), "{err}");
    }
}
