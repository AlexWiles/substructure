use std::env;
use std::io::IsTerminal;
use std::path::PathBuf;

use anyhow::{bail, Context as _, Result};
use dialoguer::{theme::ColorfulTheme, Input, Select};
use serde::{Deserialize, Serialize};

use super::context::Context;
use super::project_config::{self, ProjectConfig, FILENAME};
use super::{print, CloudGlobals};

#[derive(Debug, clap::Args)]
pub struct InitCommand {
    /// Org id to pin. Skips the org picker.
    #[arg(long)]
    pub org: Option<String>,
    /// App id to pin. Skips the app picker. Pass `--app=` (empty) to omit
    /// pinning an app while still pinning an org.
    #[arg(long)]
    pub app: Option<String>,
    /// Overwrite an existing subs.toml in the current directory.
    #[arg(long)]
    pub force: bool,
    /// Don't prompt. Fails if `--org` isn't enough to write the file.
    /// Intended for scripted or agent usage.
    #[arg(long, short = 'y')]
    pub yes: bool,
    #[command(flatten)]
    pub globals: CloudGlobals,
}

#[derive(Debug, Deserialize)]
struct OrgRef {
    id: String,
    name: String,
}

#[derive(Debug, Deserialize)]
struct AppRef {
    id: String,
    name: String,
}

pub async fn run(cmd: InitCommand) -> Result<()> {
    let cwd = env::current_dir().context("could not determine cwd")?;
    let target: PathBuf = cwd.join(FILENAME);
    if target.exists() && !cmd.force {
        bail!(
            "{} already exists. Pass --force to overwrite.",
            target.display()
        );
    }

    let ctx = Context::load(&cmd.globals)?;
    let interactive = !cmd.yes && std::io::stdin().is_terminal() && cmd.org.is_none();

    let org = if let Some(o) = cmd.org.clone() {
        o
    } else if interactive {
        pick_org(&ctx).await?
    } else {
        bail!(
            "no org to pin. Pass --org <id>, or re-run in an interactive terminal to pick from a list."
        )
    };

    // `--app=` (empty string) pins only the org.
    let app: Option<String> = match cmd.app.clone() {
        Some(s) if s.is_empty() => None,
        Some(s) => Some(s),
        None => {
            if interactive {
                pick_app(&ctx, &org).await?
            } else {
                None
            }
        }
    };

    let project = ProjectConfig {
        org: Some(org.clone()),
        app: app.clone(),
    };
    project_config::write(&target, &project)?;

    if cmd.globals.json {
        return print::json(&serde_json::json!({
            "wrote": target,
            "org": org,
            "app": app,
        }));
    }

    println!("Wrote {}", target.display());
    println!("  org = {org}");
    if let Some(a) = &app {
        println!("  app = {a}");
    } else {
        println!();
        println!("No app pinned. Commands that target an app will need --app.");
    }
    Ok(())
}

async fn pick_org(ctx: &Context) -> Result<String> {
    let orgs: Vec<OrgRef> = ctx.client.get("/api/v1/orgs").await?;
    if orgs.is_empty() {
        bail!("no organizations found for your account. Create one in the web UI first.");
    }
    if orgs.len() == 1 {
        println!("Using your only org: {} ({})", orgs[0].name, orgs[0].id);
        return Ok(orgs[0].id.clone());
    }

    let default_idx = ctx
        .project
        .as_ref()
        .and_then(|p| p.org.as_deref())
        .and_then(|d| orgs.iter().position(|o| o.id == d))
        .unwrap_or(0);

    let items: Vec<String> = orgs
        .iter()
        .map(|o| format!("{}  ({})", o.name, o.id))
        .collect();
    let pick = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Which org?")
        .items(&items)
        .default(default_idx)
        .interact()
        .context("org picker")?;
    Ok(orgs[pick].id.clone())
}

async fn pick_app(ctx: &Context, org_id: &str) -> Result<Option<String>> {
    let apps: Vec<AppRef> = ctx
        .client
        .get(&format!("/api/v1/orgs/{org_id}/apps"))
        .await?;

    const CREATE_LABEL: &str = "(create new app…)";
    const SKIP_LABEL: &str = "(skip)";

    if apps.is_empty() {
        let items = vec![CREATE_LABEL, SKIP_LABEL];
        let pick = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("No apps in this org yet")
            .items(&items)
            .default(0)
            .interact()
            .context("app picker")?;
        if pick == 0 {
            return create_app(ctx, org_id).await.map(Some);
        }
        return Ok(None);
    }

    let default_idx = ctx
        .project
        .as_ref()
        .and_then(|p| p.app.as_deref())
        .and_then(|d| apps.iter().position(|a| a.id == d))
        .unwrap_or(0);

    let mut items: Vec<String> = apps
        .iter()
        .map(|a| format!("{}  ({})", a.name, a.id))
        .collect();
    let create_idx = items.len();
    items.push(CREATE_LABEL.into());
    let skip_idx = items.len();
    items.push(SKIP_LABEL.into());

    let pick = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Which app?")
        .items(&items)
        .default(default_idx)
        .interact()
        .context("app picker")?;
    if pick == create_idx {
        create_app(ctx, org_id).await.map(Some)
    } else if pick == skip_idx {
        Ok(None)
    } else {
        Ok(Some(apps[pick].id.clone()))
    }
}

#[derive(Debug, Serialize)]
struct NamePayload<'a> {
    name: &'a str,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CreateAppResponse {
    app: AppRef,
    signing_secret: String,
}

async fn create_app(ctx: &Context, org_id: &str) -> Result<String> {
    let name: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("App name")
        .interact_text()
        .context("app name prompt")?;

    let res: CreateAppResponse = ctx
        .client
        .post_json(
            &format!("/api/v1/orgs/{org_id}/apps"),
            &NamePayload { name: &name },
        )
        .await?;

    println!();
    println!("App created");
    println!("  id:              {}", res.app.id);
    println!("  name:            {}", res.app.name);
    println!();
    println!("  signing_secret:  {}", res.signing_secret);
    println!();

    Ok(res.app.id)
}
