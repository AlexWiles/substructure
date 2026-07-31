use std::env;
use std::path::PathBuf;

use anyhow::{bail, Context as _, Result};

use super::context::Context;
use super::pickers;
use super::project_config::{self, ProjectConfig, FILENAME};
use super::{print, CloudGlobals};

#[derive(Debug, clap::Args)]
pub struct LinkCommand {
    /// Org id to pin. Skips the org picker.
    #[arg(long)]
    pub org: Option<String>,
    /// Project id to pin. Skips the project picker. Pass `--project=` (empty)
    /// to omit pinning a project while still pinning an org.
    #[arg(long)]
    pub project: Option<String>,
    /// Repin a file that already names an org or project.
    #[arg(long)]
    pub force: bool,
    #[command(flatten)]
    pub globals: CloudGlobals,
}

/// The file to link: the one commands run from this tree already read, or a new
/// one here. Everything link does not own is carried across, so relinking keeps
/// the engine settings, connections, and worker the file declares.
fn target(globals: &CloudGlobals) -> Result<(PathBuf, ProjectConfig)> {
    let found = match globals.config.as_deref() {
        Some(path) if !path.exists() => None,
        path => project_config::resolve(path)?,
    };
    match found {
        Some(found) => Ok((found.path, found.config)),
        None => {
            let path = match globals.config.clone() {
                Some(path) => path,
                None => env::current_dir()
                    .context("could not determine cwd")?
                    .join(FILENAME),
            };
            Ok((path, ProjectConfig::default()))
        }
    }
}

pub async fn run(cmd: LinkCommand) -> Result<()> {
    let (path, existing) = target(&cmd.globals)?;
    if !cmd.force {
        if let Some(pinned) = existing.org().or(existing.project()) {
            bail!(
                "{} is already linked to {pinned}. Pass --force to relink.",
                path.display()
            );
        }
    }

    // The environment link resolved, not one discovered a second time: the file
    // may not exist yet, and an unrelated one above it is not this link's.
    let ctx = Context::with_config(&cmd.globals, Some(existing.clone()))?;
    let interactive = pickers::interactive(&cmd.globals);

    let org = if let Some(o) = cmd.org.clone() {
        o
    } else if let Some(o) = ctx.server_default_org().await {
        o
    } else if interactive {
        pickers::pick_org(&ctx).await?
    } else {
        bail!("no org to pin. Pass --org <id>.")
    };

    // `--project=` (empty string) pins only the org.
    let project: Option<String> = match cmd.project.clone() {
        Some(s) if s.is_empty() => None,
        Some(s) => Some(s),
        None => {
            if let Some(a) = ctx.server_default_project().await {
                Some(a)
            } else if interactive {
                pickers::pick_project(&ctx, &org).await?
            } else {
                None
            }
        }
    };

    // A `--url` this invocation did not pass leaves the file's own alone: the
    // API a linked tree talks to is the file's, not this command's.
    let mut config = existing;
    let deployment = config.deployment_mut();
    deployment.org = Some(org.clone());
    deployment.project = project.clone();
    deployment.url = cmd.globals.url.clone().or(deployment.url.take());
    let url = deployment.url.clone();
    project_config::write(&path, &config)?;

    if cmd.globals.json {
        return print::json(&serde_json::json!({
            "wrote": path,
            "org": org,
            "project": project,
            "url": url,
        }));
    }

    println!("Wrote {}", path.display());
    println!("  org = {org}");
    if let Some(a) = &project {
        println!("  project = {a}");
    }
    if let Some(u) = &url {
        println!("  url = {u}");
    }
    if project.is_none() {
        println!();
        println!("No project pinned. Commands that target a project will need --project.");
    }
    Ok(())
}
