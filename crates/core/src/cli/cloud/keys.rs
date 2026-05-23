// `subs cloud apps keys {list,create,revoke}`: per-app API keys.

use anyhow::Result;
use clap::Subcommand;
use serde::{Deserialize, Serialize};

use super::context::Context;
use super::print;
use super::AppScope;

#[derive(Subcommand)]
pub enum KeysCommand {
    /// List API keys for the current app.
    #[command(name = "list", visible_alias = "ls")]
    List {
        #[command(flatten)]
        scope: AppScope,
    },
    /// Issue a new API key. The plaintext is shown once.
    Create {
        label: String,
        #[command(flatten)]
        scope: AppScope,
    },
    /// Revoke an API key (owner only).
    Revoke {
        key_id: String,
        #[command(flatten)]
        scope: AppScope,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct ApiKeyRow {
    key_id: String,
    label: String,
    #[serde(default)]
    created_at: Option<String>,
    #[serde(default)]
    last_used_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CreateKeyResponse {
    api_key: String,
}

#[derive(Debug, Serialize)]
struct CreateOutput<'a> {
    label: &'a str,
    api_key: &'a str,
}

#[derive(Debug, Serialize)]
struct RevokeResult<'a> {
    revoked: bool,
    key_id: &'a str,
}

#[derive(Debug, Serialize)]
struct LabelPayload<'a> {
    label: &'a str,
}

pub async fn run(command: KeysCommand) -> Result<()> {
    match command {
        KeysCommand::List { scope } => list(scope).await,
        KeysCommand::Create { label, scope } => create(label, scope).await,
        KeysCommand::Revoke { key_id, scope } => revoke(key_id, scope).await,
    }
}

async fn list(scope: AppScope) -> Result<()> {
    let (ctx, org, app) = Context::from_app(&scope)?;
    let keys: Vec<ApiKeyRow> = ctx
        .client
        .get(&format!("/api/v1/orgs/{org}/apps/{app}/api-keys"))
        .await?;

    if scope.globals.json {
        return print::json(&keys);
    }

    println!(
        "{:<40} {:<30} {:<25} {}",
        "KEY_ID", "LABEL", "CREATED", "LAST USED"
    );
    for k in &keys {
        println!(
            "{:<40} {:<30} {:<25} {}",
            k.key_id,
            k.label,
            k.created_at.as_deref().unwrap_or("-"),
            k.last_used_at.as_deref().unwrap_or("-"),
        );
    }
    Ok(())
}

async fn create(label: String, scope: AppScope) -> Result<()> {
    let (ctx, org, app) = Context::from_app(&scope)?;
    let res: CreateKeyResponse = ctx
        .client
        .post_json(
            &format!("/api/v1/orgs/{org}/apps/{app}/api-keys"),
            &LabelPayload { label: &label },
        )
        .await?;

    if scope.globals.json {
        return print::json(&CreateOutput {
            label: &label,
            api_key: &res.api_key,
        });
    }

    println!("API key created");
    println!("  label:    {}", label);
    println!("  api_key:  {}", res.api_key);
    println!();
    println!("Save this now. It will not be shown again.");
    Ok(())
}

async fn revoke(key_id: String, scope: AppScope) -> Result<()> {
    let (ctx, org, app) = Context::from_app(&scope)?;
    ctx.client
        .delete_discard(&format!("/api/v1/orgs/{org}/apps/{app}/api-keys/{key_id}"))
        .await?;

    if scope.globals.json {
        return print::json(&RevokeResult {
            revoked: true,
            key_id: &key_id,
        });
    }

    println!("Key {key_id} revoked");
    Ok(())
}
