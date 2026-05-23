// `subs cloud webhook {show,set,disable,rotate-secret}` — per-app worker
// (webhook) configuration.

use anyhow::Result;
use clap::Subcommand;
use serde::{Deserialize, Serialize};

use super::context::Context;
use super::print;
use super::CloudGlobals;

#[derive(Subcommand)]
pub enum WebhookCommand {
    /// Show the current webhook config.
    Show {
        #[arg(long)]
        org: Option<String>,
        #[arg(long)]
        app: Option<String>,
        #[command(flatten)]
        globals: CloudGlobals,
    },
    /// Set the webhook URL (implies state=enabled).
    Set {
        url: String,
        #[arg(long)]
        org: Option<String>,
        #[arg(long)]
        app: Option<String>,
        #[command(flatten)]
        globals: CloudGlobals,
    },
    /// Disable webhook delivery (does not delete the URL).
    Disable {
        #[arg(long)]
        org: Option<String>,
        #[arg(long)]
        app: Option<String>,
        #[command(flatten)]
        globals: CloudGlobals,
    },
    /// Rotate the signing secret (owner only). The new secret is shown once.
    RotateSecret {
        #[arg(long)]
        org: Option<String>,
        #[arg(long)]
        app: Option<String>,
        #[command(flatten)]
        globals: CloudGlobals,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct WorkerConfig {
    #[serde(default)]
    endpoint_url: Option<String>,
    #[serde(default)]
    state: Option<String>,
    #[serde(default)]
    signing_secret: Option<String>,
}

#[derive(Debug, Serialize)]
struct UpsertBody<'a> {
    #[serde(rename = "endpointUrl", skip_serializing_if = "Option::is_none")]
    endpoint_url: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    state: Option<&'a str>,
}

pub async fn run(command: WebhookCommand) -> Result<()> {
    match command {
        WebhookCommand::Show { org, app, globals } => show(org, app, globals).await,
        WebhookCommand::Set { url, org, app, globals } => set(url, org, app, globals).await,
        WebhookCommand::Disable { org, app, globals } => disable(org, app, globals).await,
        WebhookCommand::RotateSecret { org, app, globals } => rotate(org, app, globals).await,
    }
}

fn print_config(w: &WorkerConfig, show_secret: bool) {
    println!("endpoint_url:  {}", w.endpoint_url.as_deref().unwrap_or("(unset)"));
    println!("state:         {}", w.state.as_deref().unwrap_or("(unknown)"));
    if show_secret {
        if let Some(s) = &w.signing_secret {
            println!();
            println!("signing_secret: {s}");
            println!("^ save this — it is shown only once");
        }
    }
}

async fn show(org_flag: Option<String>, app_flag: Option<String>, globals: CloudGlobals) -> Result<()> {
    let ctx = Context::load(&globals)?;
    let org = ctx.require_org(org_flag.as_deref())?;
    let app = ctx.require_app(&org, app_flag.as_deref())?;
    let w: WorkerConfig = ctx.client.get(&format!("/api/v1/orgs/{org}/apps/{app}/worker")).await?;

    if globals.json {
        return print::json(&w);
    }
    print_config(&w, false);
    Ok(())
}

async fn set(url: String, org_flag: Option<String>, app_flag: Option<String>, globals: CloudGlobals) -> Result<()> {
    let ctx = Context::load(&globals)?;
    let org = ctx.require_org(org_flag.as_deref())?;
    let app = ctx.require_app(&org, app_flag.as_deref())?;
    let w: WorkerConfig = ctx
        .client
        .put_json(
            &format!("/api/v1/orgs/{org}/apps/{app}/worker"),
            &UpsertBody {
                endpoint_url: Some(&url),
                state: Some("enabled"),
            },
        )
        .await?;

    if globals.json {
        return print::json(&w);
    }
    print_config(&w, false);
    Ok(())
}

async fn disable(org_flag: Option<String>, app_flag: Option<String>, globals: CloudGlobals) -> Result<()> {
    let ctx = Context::load(&globals)?;
    let org = ctx.require_org(org_flag.as_deref())?;
    let app = ctx.require_app(&org, app_flag.as_deref())?;
    let w: WorkerConfig = ctx
        .client
        .put_json(
            &format!("/api/v1/orgs/{org}/apps/{app}/worker"),
            &UpsertBody {
                endpoint_url: None,
                state: Some("disabled"),
            },
        )
        .await?;

    if globals.json {
        return print::json(&w);
    }
    print_config(&w, false);
    Ok(())
}

async fn rotate(org_flag: Option<String>, app_flag: Option<String>, globals: CloudGlobals) -> Result<()> {
    let ctx = Context::load(&globals)?;
    let org = ctx.require_org(org_flag.as_deref())?;
    let app = ctx.require_app(&org, app_flag.as_deref())?;
    let empty: serde_json::Value = serde_json::json!({});
    let w: WorkerConfig = ctx
        .client
        .post_json(&format!("/api/v1/orgs/{org}/apps/{app}/worker/rotate-secret"), &empty)
        .await?;

    if globals.json {
        return print::json(&w);
    }
    print_config(&w, true);
    Ok(())
}
