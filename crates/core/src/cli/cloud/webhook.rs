use anyhow::Result;
use clap::Subcommand;
use serde::{Deserialize, Serialize};

use super::context::Context;
use super::print;
use super::AppScope;

#[derive(Subcommand)]
pub enum WebhookCommand {
    /// Show the current webhook config.
    Show {
        #[command(flatten)]
        scope: AppScope,
    },
    /// Set the webhook URL (implies state=enabled).
    Set {
        #[arg(value_name = "URL")]
        endpoint: String,
        #[command(flatten)]
        scope: AppScope,
    },
    /// Disable webhook delivery (does not delete the URL).
    Disable {
        #[command(flatten)]
        scope: AppScope,
    },
    /// Rotate the signing secret (owner only). The new secret is shown once.
    RotateSecret {
        #[command(flatten)]
        scope: AppScope,
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
#[serde(rename_all = "camelCase")]
struct UpsertBody<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    endpoint_url: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    state: Option<&'a str>,
}

pub async fn run(command: WebhookCommand) -> Result<()> {
    match command {
        WebhookCommand::Show { scope } => show(scope).await,
        WebhookCommand::Set { endpoint, scope } => set(endpoint, scope).await,
        WebhookCommand::Disable { scope } => disable(scope).await,
        WebhookCommand::RotateSecret { scope } => rotate(scope).await,
    }
}

fn print_config(w: &WorkerConfig) {
    println!(
        "endpoint_url:    {}",
        w.endpoint_url.as_deref().unwrap_or("(unset)")
    );
    println!(
        "state:           {}",
        w.state.as_deref().unwrap_or("(unknown)")
    );
    if let Some(s) = &w.signing_secret {
        println!("signing_secret:  {s}");
    }
}

async fn show(scope: AppScope) -> Result<()> {
    let (ctx, org, app) = Context::from_app(&scope)?;
    let w: WorkerConfig = ctx
        .client
        .get(&format!("/api/v1/orgs/{org}/apps/{app}/worker"))
        .await?;

    if scope.globals.json {
        return print::json(&w);
    }
    print_config(&w);
    Ok(())
}

async fn set(endpoint: String, scope: AppScope) -> Result<()> {
    let (ctx, org, app) = Context::from_app(&scope)?;
    let w: WorkerConfig = ctx
        .client
        .put_json(
            &format!("/api/v1/orgs/{org}/apps/{app}/worker"),
            &UpsertBody {
                endpoint_url: Some(&endpoint),
                state: Some("enabled"),
            },
        )
        .await?;

    if scope.globals.json {
        return print::json(&w);
    }
    print_config(&w);
    Ok(())
}

async fn disable(scope: AppScope) -> Result<()> {
    let (ctx, org, app) = Context::from_app(&scope)?;
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

    if scope.globals.json {
        return print::json(&w);
    }
    print_config(&w);
    Ok(())
}

async fn rotate(scope: AppScope) -> Result<()> {
    let (ctx, org, app) = Context::from_app(&scope)?;
    let w: WorkerConfig = ctx
        .client
        .post_empty(&format!(
            "/api/v1/orgs/{org}/apps/{app}/worker/rotate-secret"
        ))
        .await?;

    if scope.globals.json {
        return print::json(&w);
    }
    println!("Signing secret rotated");
    print_config(&w);
    Ok(())
}
