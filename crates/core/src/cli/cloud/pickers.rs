use std::io::IsTerminal;

use anyhow::{bail, Context as _, Result};
use dialoguer::{theme::ColorfulTheme, Input, Select};
use serde::{Deserialize, Serialize};

use super::context::Context;
use super::CloudGlobals;

#[derive(Debug, Deserialize)]
pub struct OrgRef {
    pub id: String,
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct AppRef {
    pub id: String,
    pub name: String,
}

/// True when we may show interactive prompts: stdin is a TTY and
/// `--no-interaction` is not set.
pub fn interactive(globals: &CloudGlobals) -> bool {
    !globals.no_interaction && std::io::stdin().is_terminal()
}

pub async fn pick_org(ctx: &Context) -> Result<String> {
    let orgs: Vec<OrgRef> = ctx.client.get("/api/v1/orgs").await?;
    if orgs.is_empty() {
        bail!("no organizations. Create one in the web UI first.");
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
        .with_prompt("Select org")
        .items(&items)
        .default(default_idx)
        .interact()
        .context("org picker")?;
    Ok(orgs[pick].id.clone())
}

pub async fn pick_app(ctx: &Context, org_id: &str) -> Result<Option<String>> {
    let apps: Vec<AppRef> = ctx
        .client
        .get(&format!("/api/v1/orgs/{org_id}/apps"))
        .await?;

    const CREATE_LABEL: &str = "(create new app)";
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
        .with_prompt("Select app")
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

#[derive(Debug, Deserialize)]
struct ApiKeyRow {
    key_id: String,
    label: String,
}

pub async fn pick_api_key(ctx: &Context, app_id: &str) -> Result<String> {
    let keys: Vec<ApiKeyRow> = ctx
        .client
        .get(&format!("/api/v1/apps/{app_id}/api-keys"))
        .await?;
    if keys.is_empty() {
        bail!("no API keys to revoke.");
    }

    let items: Vec<String> = keys
        .iter()
        .map(|k| format!("{}  ({})", k.label, k.key_id))
        .collect();
    let pick = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select key")
        .items(&items)
        .default(0)
        .interact()
        .context("api key picker")?;
    Ok(keys[pick].key_id.clone())
}

pub fn prompt_text(prompt: &str) -> Result<String> {
    Input::with_theme(&ColorfulTheme::default())
        .with_prompt(prompt)
        .interact_text()
        .context("text prompt")
}

pub async fn create_app(ctx: &Context, org_id: &str) -> Result<String> {
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
    println!("  signing_secret:  {}", res.signing_secret);
    println!();

    Ok(res.app.id)
}
