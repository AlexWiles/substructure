//! Where a command acts: an engine on this machine, or a deployment.
//!
//! One rule, in one place. **The `[remote]` section decides.** A file that
//! names one describes a deployment you administer, so that is what a command
//! reads and writes. A file that names none describes an engine you run here,
//! so this machine's environment and this machine's database are the whole
//! truth. `--url` and `--db` each name a target outright, for when the file is
//! not the one to ask.
//!
//! Three kinds of command follow from it:
//!
//! - **Both.** `sessions`, `agents`, `llm`, `mcp`, `doctor`, `slack` — the same
//!   question has an answer in either place.
//! - **Here only.** `run`, `serve` — a `[remote]` does not change what they do.
//! - **A deployment only.** `keys`, `open`, `config log`, and the commands that
//!   bind a secret. With no `[remote]` these have nothing to act on, and say so
//!   rather than reaching for the hosted cloud and asking you to log in to it.
//!
//! Deliberately outside the rule: `login`, `logout`, and `whoami` are about a
//! credential rather than a project, and `apply` and `link` are how a file gets
//! a `[remote]` in the first place. All five must work before there is one.

use anyhow::{bail, Result};

use super::cloud::project_config::{self, ProjectConfig};
use super::cloud::CloudGlobals;

/// What a command acts on.
pub enum Target {
    /// An engine you run here, described by this file.
    Here(Box<ProjectConfig>),
    /// The deployment `--url` or `[remote]` names — or, with no file at all,
    /// the default one.
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

/// Where this invocation acts: `--url` first, since naming a server is asking
/// for it; then the file's `[remote]`; then here. A file that does not exist
/// reads as a deployment, so a command run outside a project still reaches the
/// one it would have.
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

/// What a deployment-only command says when the file names no deployment. The
/// two ways forward are both in it, because "not configured" is not advice.
pub fn no_deployment(what: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "{what} acts on a deployment, and this project names no `[remote]`.\n\
         Point it at one:\n  \
         subs link                    pin an org and project in the file\n  \
         --url <url>                  target a deployment for this command"
    )
}

/// The guard a deployment-only command runs before it reaches for a server.
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

    /// Naming a server is asking for it, whatever the file says.
    #[test]
    fn the_url_flag_wins_over_a_file_that_names_no_remote() {
        let mut globals = wrote(ENGINE_HERE);
        globals.url = Some("https://elsewhere.test".into());
        assert!(!is_here(&globals));
    }

    /// A `-c` that names nothing is a mistake, not a reason to pick a target.
    #[test]
    fn a_config_path_that_does_not_resolve_is_an_error() {
        let globals = CloudGlobals {
            config: Some(tmpdir().join("nowhere/substructure.toml")),
            ..Default::default()
        };
        assert!(target(&globals).is_err());
    }

    /// "Not configured" is not advice: both ways forward are in the message.
    #[test]
    fn a_deployment_only_command_says_how_to_get_one() {
        let err = require_deployment(&wrote(ENGINE_HERE), "subs keys")
            .unwrap_err()
            .to_string();
        assert!(err.contains("names no `[remote]`"), "{err}");
        assert!(err.contains("subs link"), "{err}");
        assert!(err.contains("--url"), "{err}");
    }
}
