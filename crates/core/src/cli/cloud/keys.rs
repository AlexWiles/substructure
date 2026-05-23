// `subs cloud keys {list,create,revoke}` — per-app API keys.

use anyhow::Result;
use clap::Subcommand;
use serde::{Deserialize, Serialize};

use super::context::Context;
use super::CloudGlobals;

#[derive(Subcommand)]
pub enum KeysCommand {
    /// List API keys for the current app.
    #[command(name = "list", visible_alias = "ls")]
    List {
        #[arg(long)]
        org: Option<String>,
        #[arg(long)]
        app: Option<String>,
        #[command(flatten)]
        globals: CloudGlobals,
    },
    /// Issue a new API key. The plaintext is shown once.
    Create {
        label: String,
        #[arg(long)]
        org: Option<String>,
        #[arg(long)]
        app: Option<String>,
        #[command(flatten)]
        globals: CloudGlobals,
    },
    /// Revoke an API key (owner only).
    Revoke {
        key_id: String,
        #[arg(long)]
        org: Option<String>,
        #[arg(long)]
        app: Option<String>,
        #[command(flatten)]
        globals: CloudGlobals,
    },
}

#[derive(Debug, Deserialize)]
struct ApiKeyRow {
    key_id: String,
    label: String,
    #[serde(default)]
    created_at: Option<String>,
    #[serde(default)]
    last_used_at: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CreateKeyResponse {
    api_key: String,
}

#[derive(Debug, Serialize)]
struct LabelPayload<'a> {
    label: &'a str,
}

pub async fn run(command: KeysCommand) -> Result<()> {
    match command {
        KeysCommand::List { org, app, globals } => list(org, app, globals).await,
        KeysCommand::Create { label, org, app, globals } => create(label, org, app, globals).await,
        KeysCommand::Revoke { key_id, org, app, globals } => revoke(key_id, org, app, globals).await,
    }
}

async fn list(org_flag: Option<String>, app_flag: Option<String>, globals: CloudGlobals) -> Result<()> {
    let ctx = Context::load(&globals)?;
    let org = ctx.require_org(org_flag.as_deref())?;
    let app = ctx.require_app(&org, app_flag.as_deref())?;
    let keys: Vec<ApiKeyRow> = ctx
        .client
        .get(&format!("/api/v1/orgs/{org}/apps/{app}/api-keys"))
        .await?;

    println!("{:<40} {:<30} {:<25} {}", "KEY_ID", "LABEL", "CREATED", "LAST USED");
    for k in &keys {
        println!(
            "{:<40} {:<30} {:<25} {}",
            k.key_id,
            k.label,
            k.created_at.as_deref().unwrap_or("—"),
            k.last_used_at.as_deref().unwrap_or("—"),
        );
    }
    Ok(())
}

async fn create(label: String, org_flag: Option<String>, app_flag: Option<String>, globals: CloudGlobals) -> Result<()> {
    let ctx = Context::load(&globals)?;
    let org = ctx.require_org(org_flag.as_deref())?;
    let app = ctx.require_app(&org, app_flag.as_deref())?;
    let res: CreateKeyResponse = ctx
        .client
        .post_json(
            &format!("/api/v1/orgs/{org}/apps/{app}/api-keys"),
            &LabelPayload { label: &label },
        )
        .await?;
    println!("API key created");
    println!("  label:    {}", label);
    println!("  api_key:  {}", res.api_key);
    println!("  ^ save this — it is shown only once");
    Ok(())
}

async fn revoke(key_id: String, org_flag: Option<String>, app_flag: Option<String>, globals: CloudGlobals) -> Result<()> {
    let ctx = Context::load(&globals)?;
    let org = ctx.require_org(org_flag.as_deref())?;
    let app = ctx.require_app(&org, app_flag.as_deref())?;
    let _: serde_json::Value = ctx
        .client
        .delete(&format!("/api/v1/orgs/{org}/apps/{app}/api-keys/{key_id}"))
        .await?;
    println!("Key {key_id} revoked");
    Ok(())
}
