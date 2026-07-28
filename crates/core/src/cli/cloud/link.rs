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
    /// App id to pin. Skips the app picker. Pass `--app=` (empty) to omit
    /// pinning an app while still pinning an org.
    #[arg(long)]
    pub app: Option<String>,
    /// Overwrite an existing substructure.toml in the current directory.
    #[arg(long)]
    pub force: bool,
    #[command(flatten)]
    pub globals: CloudGlobals,
}

pub async fn run(cmd: LinkCommand) -> Result<()> {
    let cwd = env::current_dir().context("could not determine cwd")?;
    let target: PathBuf = cwd.join(FILENAME);
    if target.exists() && !cmd.force {
        bail!(
            "{} already exists. Pass --force to overwrite.",
            target.display()
        );
    }

    let ctx = Context::load(&cmd.globals)?;
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

    // Relinking rewrites the file, so everything link does not own — the
    // connections and engine settings for this tree — is carried across.
    let existing = project_config::load_explicit(&target)
        .map(|f| f.config)
        .unwrap_or_default();
    let project = ProjectConfig {
        org: Some(org.clone()),
        app: app.clone(),
        url: cmd.globals.url.clone(),
        ..existing
    };
    project_config::write(&target, &project)?;

    if cmd.globals.json {
        return print::json(&serde_json::json!({
            "wrote": target,
            "org": org,
            "app": app,
            "url": project.url,
        }));
    }

    println!("Wrote {}", target.display());
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
