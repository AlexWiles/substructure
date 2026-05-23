use anyhow::{bail, Result};
use clap::Subcommand;
use serde::{Deserialize, Serialize};

use super::context::Context;
use super::keys;
use super::pickers;
use super::print;
use super::sessions;
use super::webhook;
use super::{AppScope, OrgScope, GLOBAL_FLAGS_HELP};

#[derive(Subcommand)]
pub enum AppsCommand {
    /// List apps in the current org.
    #[command(name = "list", visible_alias = "ls")]
    List {
        #[command(flatten)]
        scope: OrgScope,
    },
    /// Create a new app. The signing secret is printed once. Save it now.
    Create {
        name: String,
        #[command(flatten)]
        scope: OrgScope,
    },
    /// Show an app's details.
    Show {
        app_id: Option<String>,
        #[command(flatten)]
        scope: AppScope,
    },
    /// Rename an app.
    Rename {
        app_id: Option<String>,
        new_name: Option<String>,
        #[command(flatten)]
        scope: AppScope,
    },
    /// Delete an app (owner only).
    Delete {
        app_id: Option<String>,
        #[command(flatten)]
        scope: AppScope,
    },
    /// Manage API keys for an app.
    #[command(after_help = GLOBAL_FLAGS_HELP)]
    Keys {
        #[command(subcommand)]
        command: keys::KeysCommand,
    },
    /// List debug sessions for an app.
    Sessions {
        #[command(flatten)]
        args: sessions::SessionsCommand,
    },
    /// Manage the webhook (worker) config for an app.
    #[command(after_help = GLOBAL_FLAGS_HELP)]
    Webhook {
        #[command(subcommand)]
        command: webhook::WebhookCommand,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AppRecord {
    id: String,
    organization_id: String,
    name: String,
    #[serde(default)]
    created_at: Option<String>,
    #[serde(default)]
    balance_usd: Option<String>,
    #[serde(default)]
    session_count: Option<i64>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CreateAppResponse {
    app: AppRecord,
    signing_secret: String,
}

#[derive(Debug, Serialize)]
struct DeleteResult<'a> {
    deleted: bool,
    id: &'a str,
}

#[derive(Debug, Serialize)]
struct NamePayload<'a> {
    name: &'a str,
}

pub async fn run(command: AppsCommand) -> Result<()> {
    match command {
        AppsCommand::List { scope } => list(scope).await,
        AppsCommand::Create { name, scope } => create(name, scope).await,
        AppsCommand::Show { app_id, scope } => show(app_id, scope).await,
        AppsCommand::Rename {
            app_id,
            new_name,
            scope,
        } => rename(app_id, new_name, scope).await,
        AppsCommand::Delete { app_id, scope } => delete(app_id, scope).await,
        AppsCommand::Keys { command } => keys::run(command).await,
        AppsCommand::Sessions { args } => sessions::run(args).await,
        AppsCommand::Webhook { command } => webhook::run(command).await,
    }
}

async fn list(scope: OrgScope) -> Result<()> {
    let (ctx, org) = Context::from_org(&scope).await?;
    let apps: Vec<AppRecord> = ctx.client.get(&format!("/api/v1/orgs/{org}/apps")).await?;

    if scope.globals.json {
        return print::json(&apps);
    }

    let pinned = ctx.project.as_ref().and_then(|p| p.app.as_deref());
    println!(
        "  {:<36} {:<30} {:>10} {:>12}",
        "ID", "NAME", "SESSIONS", "BALANCE"
    );
    for a in &apps {
        let marker = if pinned == Some(a.id.as_str()) {
            "*"
        } else {
            " "
        };
        let sessions = a
            .session_count
            .map(|n| n.to_string())
            .unwrap_or_else(|| "-".into());
        let balance = print::fmt_usd(a.balance_usd.as_deref().unwrap_or("0"));
        println!(
            "{marker} {:<36} {:<30} {:>10} {:>12}",
            a.id, a.name, sessions, balance
        );
    }
    Ok(())
}

async fn create(name: String, scope: OrgScope) -> Result<()> {
    let (ctx, org) = Context::from_org(&scope).await?;
    let res: CreateAppResponse = ctx
        .client
        .post_json(
            &format!("/api/v1/orgs/{org}/apps"),
            &NamePayload { name: &name },
        )
        .await?;

    if scope.globals.json {
        return print::json(&res);
    }

    println!("App created");
    println!("  id:              {}", res.app.id);
    println!("  name:            {}", res.app.name);
    println!("  organization_id: {}", res.app.organization_id);
    println!("  signing_secret:  {}", res.signing_secret);
    println!();
    print::warn_zero_balance(&res.app.name, ctx.client.base_url(), &res.app.id);
    Ok(())
}

async fn show(app_id: Option<String>, scope: AppScope) -> Result<()> {
    let scope = AppScope {
        app: app_id.or(scope.app.clone()),
        ..scope
    };
    let (ctx, app_id) = Context::from_app(&scope).await?;
    let a: AppRecord = ctx.client.get(&format!("/api/v1/apps/{app_id}")).await?;

    if scope.globals.json {
        return print::json(&a);
    }

    println!("id:              {}", a.id);
    println!("name:            {}", a.name);
    println!("organization_id: {}", a.organization_id);
    if let Some(ca) = &a.created_at {
        println!("created_at:      {ca}");
    }
    let balance_raw = a.balance_usd.as_deref().unwrap_or("0");
    println!("balance:         {}", print::fmt_usd(balance_raw));
    if let Some(s) = a.session_count {
        println!("session_count:   {s}");
    }
    Ok(())
}

async fn rename(app_id: Option<String>, new_name: Option<String>, scope: AppScope) -> Result<()> {
    let scope = AppScope {
        app: app_id.or(scope.app.clone()),
        ..scope
    };
    let (ctx, app_id) = Context::from_app(&scope).await?;
    let new_name = match new_name {
        Some(n) => n,
        None => {
            if !pickers::interactive(&scope.globals) {
                bail!("missing <NEW_NAME>");
            }
            pickers::prompt_text("New name")?
        }
    };
    let a: AppRecord = ctx
        .client
        .patch_json(
            &format!("/api/v1/apps/{app_id}"),
            &NamePayload { name: &new_name },
        )
        .await?;

    if scope.globals.json {
        return print::json(&a);
    }

    println!("App renamed to \"{}\"", a.name);
    Ok(())
}

async fn delete(app_id: Option<String>, scope: AppScope) -> Result<()> {
    let scope = AppScope {
        app: app_id.or(scope.app.clone()),
        ..scope
    };
    let (ctx, app_id) = Context::from_app(&scope).await?;
    ctx.client
        .delete_discard(&format!("/api/v1/apps/{app_id}"))
        .await?;

    if scope.globals.json {
        return print::json(&DeleteResult {
            deleted: true,
            id: &app_id,
        });
    }

    println!("App {app_id} deleted");
    Ok(())
}
