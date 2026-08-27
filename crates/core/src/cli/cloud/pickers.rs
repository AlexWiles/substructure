use std::io::IsTerminal;

use anyhow::{anyhow, bail, Context as _, Result};
use dialoguer::{theme::ColorfulTheme, Input, Select};
use inquire::ui::{Attributes, Color, RenderConfig, StyleSheet, Styled};
use inquire::InquireError;
use serde::{Deserialize, Serialize};

use crate::api::v1::{Org, Project};

use super::context::Context;
use super::CloudGlobals;

/// True when we may show interactive prompts: stdin is a TTY and
/// `--no-interaction` is not set.
pub fn interactive(globals: &CloudGlobals) -> bool {
    !globals.no_interaction && std::io::stdin().is_terminal()
}

pub async fn pick_org(ctx: &Context) -> Result<String> {
    let orgs: Vec<Org> = ctx.client.get("/api/v1/orgs").await?;
    if orgs.is_empty() {
        bail!("no organizations. Create one in the web UI first.");
    }

    let default_idx = ctx
        .config
        .as_ref()
        .and_then(|p| p.org())
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

/// Which project this file is: an existing one to adopt, or a new one.
///
/// One file is one project, so binding a file that has no project yet means
/// allocating one — the project is empty until `subs apply` says what it is.
pub async fn pick_project(ctx: &Context, org_id: &str) -> Result<Option<String>> {
    let projects: Vec<Project> = ctx
        .client
        .get(&format!("/api/v1/orgs/{org_id}/projects"))
        .await?;

    const CREATE_LABEL: &str = "(create new project)";
    const SKIP_LABEL: &str = "(skip)";

    if projects.is_empty() {
        let items = [CREATE_LABEL, SKIP_LABEL];
        let pick = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("No projects in this org yet")
            .items(&items)
            .default(0)
            .interact()
            .context("project picker")?;
        return match pick {
            0 => create_project(ctx, org_id).await.map(Some),
            _ => Ok(None),
        };
    }

    let default_idx = ctx
        .config
        .as_ref()
        .and_then(|p| p.project())
        .and_then(|d| projects.iter().position(|a| a.id == d))
        .unwrap_or(0);

    let mut items: Vec<String> = projects
        .iter()
        .map(|a| format!("{}  ({})", a.name, a.id))
        .collect();
    let create_idx = items.len();
    items.push(CREATE_LABEL.into());
    let skip_idx = items.len();
    items.push(SKIP_LABEL.into());

    let pick = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select project")
        .items(&items)
        .default(default_idx)
        .interact()
        .context("project picker")?;
    if pick == create_idx {
        create_project(ctx, org_id).await.map(Some)
    } else if pick == skip_idx {
        Ok(None)
    } else {
        Ok(Some(projects[pick].id.clone()))
    }
}

#[derive(Debug, Serialize)]
struct NamePayload<'a> {
    name: &'a str,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CreateProjectResponse {
    project: Project,
}

/// Allocate an empty project for this file to pin.
///
/// It carries no secret: signing secrets are per agent, minted by the first
/// apply that gives an agent a worker. Until `subs apply` runs, the project
/// exists and declares nothing.
pub async fn create_project(ctx: &Context, org_id: &str) -> Result<String> {
    let name: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Project name")
        .interact_text()
        .context("project name prompt")?;

    let res: CreateProjectResponse = ctx
        .client
        .post_json(
            &format!("/api/v1/orgs/{org_id}/projects"),
            &NamePayload { name: &name },
        )
        .await?;

    println!();
    println!("Project created");
    println!("  id:    {}", res.project.id);
    println!("  name:  {}", res.project.name);
    println!();

    Ok(res.project.id)
}

#[derive(Debug, Deserialize)]
struct ApiKeyRow {
    key_id: String,
    label: String,
}

pub async fn pick_api_key(ctx: &Context, project_id: &str) -> Result<String> {
    let keys: Vec<ApiKeyRow> = ctx
        .client
        .get(&format!("/api/v1/projects/{project_id}/api-keys"))
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

pub fn read_secret(globals: &CloudGlobals, prompt: &str) -> Result<String> {
    use std::io::Read as _;

    if std::io::stdin().is_terminal() {
        if !interactive(globals) {
            bail!("nothing on stdin. Pipe it in, or pass --env <VAR>.");
        }
        return prompt_secret(prompt);
    }
    let mut buf = String::new();
    std::io::stdin()
        .read_to_string(&mut buf)
        .context("reading from stdin")?;
    Ok(buf.trim().to_string())
}

pub fn prompt_secret(prompt: &str) -> Result<String> {
    inquire::Password::new(prompt)
        .with_display_mode(inquire::PasswordDisplayMode::Masked)
        .without_confirmation()
        .with_render_config(secret_theme())
        .prompt()
        .map_err(|err| match err {
            InquireError::OperationCanceled | InquireError::OperationInterrupted => {
                anyhow!("canceled.")
            }
            err => anyhow::Error::new(err).context("secret prompt"),
        })
}

fn secret_theme() -> RenderConfig<'static> {
    let mut theme = RenderConfig::default_colored()
        .with_prompt_prefix(Styled::new("?").with_fg(Color::LightYellow))
        .with_answered_prompt_prefix(Styled::new("✔").with_fg(Color::LightGreen))
        .with_answer(StyleSheet::new().with_fg(Color::LightGreen));
    theme.prompt = StyleSheet::new().with_attr(Attributes::BOLD);
    theme
}

pub fn prompt_text(prompt: &str) -> Result<String> {
    Input::with_theme(&ColorfulTheme::default())
        .with_prompt(prompt)
        .interact_text()
        .context("text prompt")
}
