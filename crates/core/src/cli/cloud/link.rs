use std::env;
use std::path::PathBuf;

use anyhow::{bail, Context as _, Result};

use super::context::Context;
use super::pickers;
use super::project_config::{self, EnvConfig, RemoteEnv, FILENAME};
use super::{print, CloudGlobals};

#[derive(Debug, clap::Args)]
pub struct LinkCommand {
    /// Org id to pin. Skips the org picker.
    #[arg(long)]
    pub org: Option<String>,
    /// App id to pin. Skips the app picker. Pass `--app=` (empty) to omit
    /// pinning an app while still pinning an org.
    #[arg(long)]
    pub app: Option<String>,
    /// Repin a file that already names an org or app.
    #[arg(long)]
    pub force: bool,
    #[command(flatten)]
    pub globals: CloudGlobals,
}

/// The file to link: the one commands run from this tree already read, or a new
/// one here. Everything link does not own is carried across, so relinking keeps
/// the connections and worker settings the environment declares.
fn target(globals: &CloudGlobals) -> Result<(PathBuf, RemoteEnv)> {
    let found = match globals.config.as_deref() {
        Some(path) if !path.exists() => None,
        path => project_config::resolve(path)?,
    };
    match found {
        Some(found) => {
            let path = found.path.clone();
            Ok((path, found.into_remote("`subs link`")?))
        }
        None => {
            let path = match globals.config.clone() {
                Some(path) => path,
                None => env::current_dir()
                    .context("could not determine cwd")?
                    .join(FILENAME),
            };
            Ok((path, RemoteEnv::default()))
        }
    }
}

pub async fn run(cmd: LinkCommand) -> Result<()> {
    let (path, existing) = target(&cmd.globals)?;
    if !cmd.force {
        if let Some(pinned) = existing.org.as_deref().or(existing.app.as_deref()) {
            bail!(
                "{} is already linked to {pinned}. Pass --force to relink.",
                path.display()
            );
        }
    }

    // The environment link resolved, not one discovered a second time: the file
    // may not exist yet, and an unrelated one above it is not this link's.
    let ctx = Context::with_project(&cmd.globals, Some(existing.clone()))?;
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

    // `--app=` (empty string) pins only the org.
    let app: Option<String> = match cmd.app.clone() {
        Some(s) if s.is_empty() => None,
        Some(s) => Some(s),
        None => {
            if let Some(a) = ctx.server_default_app().await {
                Some(a)
            } else if interactive {
                pickers::pick_app(&ctx, &org).await?
            } else {
                None
            }
        }
    };

    // A `--url` this invocation did not pass leaves the file's own alone: the
    // API a linked tree talks to is the environment's, not this command's.
    let project = RemoteEnv {
        org: Some(org.clone()),
        app: app.clone(),
        url: cmd.globals.url.clone().or(existing.url),
        ..existing
    };
    project_config::write(&path, &EnvConfig::Remote(project.clone()))?;

    if cmd.globals.json {
        return print::json(&serde_json::json!({
            "wrote": path,
            "org": org,
            "app": app,
            "url": project.url,
        }));
    }

    println!("Wrote {}", path.display());
    println!("  org = {org}");
    if let Some(a) = &app {
        println!("  app = {a}");
    }
    if let Some(u) = &project.url {
        println!("  url = {u}");
    }
    if app.is_none() {
        println!();
        println!("No app pinned. Commands that target an app will need --app.");
    }
    Ok(())
}
