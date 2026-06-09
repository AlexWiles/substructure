use anyhow::{bail, Result};
use clap::Subcommand;
use serde::{Deserialize, Serialize};

use super::context::Context;
use super::pickers;
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
    /// Issue a new API key. The plaintext is printed to stdout and shown once.
    /// Pipe-friendly: `substructure keys create --label foo | wrangler secret put X`.
    Create {
        label: Option<String>,
        #[command(flatten)]
        scope: AppScope,
    },
    /// Revoke an API key (owner only).
    Revoke {
        key_id: Option<String>,
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
    let (ctx, app) = Context::from_app(&scope).await?;
    let keys: Vec<ApiKeyRow> = ctx
        .client
        .get(&format!("/api/v1/apps/{app}/api-keys"))
        .await?;

    if scope.globals.json {
        return print::json(&keys);
    }

    let columns = [
        print::Column::left("KEY_ID"),
        print::Column::left("LABEL"),
        print::Column::left("CREATED"),
        print::Column::left("LAST USED"),
    ];
    let rows: Vec<Vec<String>> = keys
        .iter()
        .map(|k| {
            vec![
                k.key_id.clone(),
                k.label.clone(),
                k.created_at.clone().unwrap_or_else(|| "-".into()),
                k.last_used_at.clone().unwrap_or_else(|| "-".into()),
            ]
        })
        .collect();
    print::table(&columns, &rows);
    Ok(())
}

async fn create(label: Option<String>, scope: AppScope) -> Result<()> {
    let (ctx, app) = Context::from_app(&scope).await?;
    let label = match label {
        Some(l) => l,
        None => {
            if !pickers::interactive(&scope.globals) {
                bail!("missing <LABEL>");
            }
            pickers::prompt_text("Key label")?
        }
    };
    let res: CreateKeyResponse = ctx
        .client
        .post_json(
            &format!("/api/v1/apps/{app}/api-keys"),
            &LabelPayload { label: &label },
        )
        .await?;

    if scope.globals.json {
        return print::json(&CreateOutput {
            label: &label,
            api_key: &res.api_key,
        });
    }

    // stdout = the value (pipe-safe). stderr = human chatter.
    eprintln!("API key '{label}' created — save now, will not be shown again.");
    println!("{}", res.api_key);
    Ok(())
}

async fn revoke(key_id: Option<String>, scope: AppScope) -> Result<()> {
    let (ctx, app) = Context::from_app(&scope).await?;
    let key_id = match key_id {
        Some(id) => id,
        None => {
            if !pickers::interactive(&scope.globals) {
                bail!("missing <KEY_ID>");
            }
            pickers::pick_api_key(&ctx, &app).await?
        }
    };
    ctx.client
        .delete_discard(&format!("/api/v1/apps/{app}/api-keys/{key_id}"))
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
