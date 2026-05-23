// `subs cloud apps {list,create,show,rename,delete,use}`.

use anyhow::{bail, Result};
use clap::Subcommand;
use serde::{Deserialize, Serialize};

use super::context::Context;
use super::print;
use super::CloudGlobals;

#[derive(Subcommand)]
pub enum AppsCommand {
    /// List apps in the current org.
    #[command(name = "list", visible_alias = "ls")]
    List {
        #[arg(long)]
        org: Option<String>,
        #[command(flatten)]
        globals: CloudGlobals,
    },
    /// Create a new app. The signing secret is printed once — save it.
    Create {
        name: String,
        #[arg(long)]
        org: Option<String>,
        #[command(flatten)]
        globals: CloudGlobals,
    },
    /// Show an app's details.
    Show {
        app_id: String,
        #[arg(long)]
        org: Option<String>,
        #[command(flatten)]
        globals: CloudGlobals,
    },
    /// Rename an app.
    Rename {
        app_id: String,
        new_name: String,
        #[arg(long)]
        org: Option<String>,
        #[command(flatten)]
        globals: CloudGlobals,
    },
    /// Delete an app (owner only).
    Delete {
        app_id: String,
        #[arg(long)]
        org: Option<String>,
        #[command(flatten)]
        globals: CloudGlobals,
    },
    /// Set the default app for subsequent commands (per-org).
    Use {
        app_id: String,
        #[arg(long)]
        org: Option<String>,
        #[command(flatten)]
        globals: CloudGlobals,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct AppRecord {
    id: String,
    #[serde(rename = "organizationId")]
    organization_id: String,
    name: String,
    #[serde(rename = "shardId", default)]
    shard_id: Option<i64>,
    #[serde(rename = "createdAt", default)]
    created_at: Option<String>,
    #[serde(rename = "balanceUsd", default)]
    balance_usd: Option<String>,
    #[serde(rename = "sessionCount", default)]
    session_count: Option<i64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CreateAppResponse {
    app: AppRecord,
    #[serde(rename = "signingSecret")]
    signing_secret: String,
}

#[derive(Debug, Serialize)]
struct DeleteResult<'a> {
    deleted: bool,
    id: &'a str,
}

#[derive(Debug, Serialize)]
struct UseResult<'a> {
    org_id: &'a str,
    default_app: &'a AppRecord,
}

#[derive(Debug, Serialize)]
struct NamePayload<'a> {
    name: &'a str,
}

pub async fn run(command: AppsCommand) -> Result<()> {
    match command {
        AppsCommand::List { org, globals } => list(org, globals).await,
        AppsCommand::Create { name, org, globals } => create(name, org, globals).await,
        AppsCommand::Show { app_id, org, globals } => show(app_id, org, globals).await,
        AppsCommand::Rename { app_id, new_name, org, globals } => rename(app_id, new_name, org, globals).await,
        AppsCommand::Delete { app_id, org, globals } => delete(app_id, org, globals).await,
        AppsCommand::Use { app_id, org, globals } => use_app(app_id, org, globals).await,
    }
}

async fn list(org_flag: Option<String>, globals: CloudGlobals) -> Result<()> {
    let ctx = Context::load(&globals)?;
    let org = ctx.require_org(org_flag.as_deref())?;
    let apps: Vec<AppRecord> = ctx.client.get(&format!("/api/v1/orgs/{org}/apps")).await?;

    if globals.json {
        return print::json(&apps);
    }

    let default = ctx.config.resolve_app(&org, None);
    println!("{:<38} {:<30} {:>10} {:>10}", "ID", "NAME", "SESSIONS", "BALANCE");
    for a in &apps {
        let marker = if default.as_deref() == Some(&a.id) { "*" } else { " " };
        let sessions = a.session_count.map(|n| n.to_string()).unwrap_or_else(|| "—".into());
        let balance = a.balance_usd.as_deref().unwrap_or("—");
        println!("{marker} {:<36} {:<30} {:>10} {:>10}", a.id, a.name, sessions, balance);
    }
    Ok(())
}

async fn create(name: String, org_flag: Option<String>, globals: CloudGlobals) -> Result<()> {
    let ctx = Context::load(&globals)?;
    let org = ctx.require_org(org_flag.as_deref())?;
    let res: CreateAppResponse = ctx
        .client
        .post_json(&format!("/api/v1/orgs/{org}/apps"), &NamePayload { name: &name })
        .await?;

    if globals.json {
        return print::json(&res);
    }

    println!("App created");
    println!("  id:              {}", res.app.id);
    println!("  name:            {}", res.app.name);
    println!("  organization_id: {}", res.app.organization_id);
    println!();
    println!("  signing_secret:  {}", res.signing_secret);
    println!("  ^ save this — it is shown only once");
    Ok(())
}

async fn show(app_id: String, org_flag: Option<String>, globals: CloudGlobals) -> Result<()> {
    let ctx = Context::load(&globals)?;
    let org = ctx.require_org(org_flag.as_deref())?;
    let a: AppRecord = ctx.client.get(&format!("/api/v1/orgs/{org}/apps/{app_id}")).await?;

    if globals.json {
        return print::json(&a);
    }

    println!("id:              {}", a.id);
    println!("name:            {}", a.name);
    println!("organization_id: {}", a.organization_id);
    if let Some(shard) = a.shard_id {
        println!("shard_id:        {shard}");
    }
    if let Some(ca) = &a.created_at {
        println!("created_at:      {ca}");
    }
    if let Some(b) = &a.balance_usd {
        println!("balance_usd:     {b}");
    }
    if let Some(s) = a.session_count {
        println!("session_count:   {s}");
    }
    Ok(())
}

async fn rename(app_id: String, new_name: String, org_flag: Option<String>, globals: CloudGlobals) -> Result<()> {
    let ctx = Context::load(&globals)?;
    let org = ctx.require_org(org_flag.as_deref())?;
    let a: AppRecord = ctx
        .client
        .patch_json(
            &format!("/api/v1/orgs/{org}/apps/{app_id}"),
            &NamePayload { name: &new_name },
        )
        .await?;

    if globals.json {
        return print::json(&a);
    }

    println!("App renamed to \"{}\"", a.name);
    Ok(())
}

async fn delete(app_id: String, org_flag: Option<String>, globals: CloudGlobals) -> Result<()> {
    let mut ctx = Context::load(&globals)?;
    let org = ctx.require_org(org_flag.as_deref())?;
    let _: serde_json::Value = ctx
        .client
        .delete(&format!("/api/v1/orgs/{org}/apps/{app_id}"))
        .await?;
    if ctx.config.resolve_app(&org, None).as_deref() == Some(&app_id) {
        ctx.config.clear_default_app(&org);
        ctx.save()?;
    }

    if globals.json {
        return print::json(&DeleteResult { deleted: true, id: &app_id });
    }

    println!("App {app_id} deleted");
    Ok(())
}

async fn use_app(app_id: String, org_flag: Option<String>, globals: CloudGlobals) -> Result<()> {
    let mut ctx = Context::load(&globals)?;
    let org = ctx.require_org(org_flag.as_deref())?;
    let a: AppRecord = ctx.client.get(&format!("/api/v1/orgs/{org}/apps/{app_id}")).await?;
    if a.organization_id != org {
        bail!("app {app_id} belongs to a different org");
    }
    ctx.config.set_default_app(&org, a.id.clone());
    ctx.save()?;

    if globals.json {
        return print::json(&UseResult { org_id: &org, default_app: &a });
    }

    println!("Default app for org {org} set to \"{}\" ({})", a.name, a.id);
    Ok(())
}
