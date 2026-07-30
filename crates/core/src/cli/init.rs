use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context as _, Result};

use super::cloud::project_config::FILENAME;

#[derive(Debug, clap::Args)]
pub struct InitCommand {
    /// Which engine the file describes: one embedded over a SQLite file, or a
    /// server reached over HTTP.
    #[arg(value_enum)]
    pub target: Target,
    /// Where to write it [default: substructure.toml].
    pub path: Option<PathBuf>,
    /// Overwrite an existing file.
    #[arg(long)]
    pub force: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Target {
    Local,
    Remote,
}

/// Starters, not schemas: every key here is one a first run needs, so the file
/// works before it is edited and `docs/160-cli.md` holds the rest.
const LOCAL: &str = "\
target = \"local\"

[worker]
url = \"http://localhost:4444\"

[llm]
provider = \"anthropic\"

[run]
agent = \"my-agent\"
output = \"pretty\"
";

/// No org or app: `subs link` pins those, and guessing them here would only
/// have to be corrected.
const REMOTE: &str = "\
target = \"remote\"

[worker]
url = \"https://example.com/agent\"
";

pub fn run(cmd: InitCommand) -> Result<()> {
    let path = cmd.path.unwrap_or_else(|| PathBuf::from(FILENAME));
    if path.exists() && !cmd.force {
        bail!(
            "{} already exists. Pass --force to overwrite it.",
            path.display()
        );
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
    }
    let body = match cmd.target {
        Target::Local => LOCAL,
        Target::Remote => REMOTE,
    };
    fs::write(&path, body).with_context(|| format!("writing {}", path.display()))?;

    println!("Wrote {}", path.display());
    match cmd.target {
        Target::Local => println!("Point [worker].url at your worker, then `subs run`."),
        Target::Remote => println!("Run `subs link` to pin an org and app."),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::cloud::project_config::{self, EnvConfig};

    /// The starters have to survive the same parse a hand-written file does —
    /// `deny_unknown_fields` makes a key on the wrong side of the split an
    /// error, so a template can rot.
    #[test]
    fn each_starter_parses_as_the_target_it_declares() {
        let dir = tempfile::tempdir().unwrap();
        for (target, body) in [(Target::Local, LOCAL), (Target::Remote, REMOTE)] {
            let path = dir.path().join(format!("{target:?}.toml"));
            run(InitCommand {
                target,
                path: Some(path.clone()),
                force: false,
            })
            .unwrap();
            assert_eq!(fs::read_to_string(&path).unwrap(), body);

            let found = project_config::load_explicit(&path).unwrap();
            match (target, found.config) {
                (Target::Local, EnvConfig::Local(_)) | (Target::Remote, EnvConfig::Remote(_)) => {}
                (_, other) => panic!("{target:?} parsed as {other:?}"),
            }
        }
    }

    #[test]
    fn an_existing_file_is_not_overwritten_without_force() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("substructure.toml");
        fs::write(&path, "target = \"local\"\ndb = \"mine.db\"\n").unwrap();

        let init = |force| InitCommand {
            target: Target::Remote,
            path: Some(path.clone()),
            force,
        };
        let err = run(init(false)).unwrap_err().to_string();
        assert!(err.contains("already exists"), "got {err}");
        assert!(fs::read_to_string(&path).unwrap().contains("mine.db"));

        run(init(true)).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), REMOTE);
    }

    /// The path is taken as given, so `subs init local api/substructure.toml`
    /// does not fail on a directory that does not exist yet.
    #[test]
    fn a_missing_parent_directory_is_created() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/deeper/substructure.toml");
        run(InitCommand {
            target: Target::Local,
            path: Some(path.clone()),
            force: false,
        })
        .unwrap();
        assert!(path.exists());
    }
}
