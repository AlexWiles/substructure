// `subs cloud orgs` and `subs cloud orgs use <id>`.

use anyhow::{bail, Result};
use clap::Subcommand;
use serde::Deserialize;

use super::context::Context;
use super::CloudGlobals;

#[derive(Subcommand)]
pub enum OrgsCommand {
    /// List organizations you belong to.
    #[command(name = "list", visible_alias = "ls")]
    List {
        #[command(flatten)]
        globals: CloudGlobals,
    },
    /// Set the default organization for subsequent commands.
    Use {
        /// Organization id.
        org_id: String,
        #[command(flatten)]
        globals: CloudGlobals,
    },
}

#[derive(Debug, Deserialize)]
struct OrgRef {
    id: String,
    name: String,
    role: String,
}

pub async fn run(command: OrgsCommand) -> Result<()> {
    match command {
        OrgsCommand::List { globals } => list(globals).await,
        OrgsCommand::Use { org_id, globals } => use_org(org_id, globals).await,
    }
}

async fn list(globals: CloudGlobals) -> Result<()> {
    let ctx = Context::load(&globals)?;
    let orgs: Vec<OrgRef> = ctx.client.get("/api/v1/orgs").await?;
    let default = ctx.config.default_org.as_deref();
    println!("{:<38} {:<30} {}", "ID", "NAME", "ROLE");
    for o in &orgs {
        let marker = if Some(o.id.as_str()) == default { "*" } else { " " };
        println!("{marker} {:<36} {:<30} {}", o.id, o.name, o.role);
    }
    Ok(())
}

async fn use_org(org_id: String, globals: CloudGlobals) -> Result<()> {
    let mut ctx = Context::load(&globals)?;
    // Verify membership before pinning — friendlier than a 403 on the next call.
    let orgs: Vec<OrgRef> = ctx.client.get("/api/v1/orgs").await?;
    let Some(matched) = orgs.iter().find(|o| o.id == org_id) else {
        bail!("you are not a member of org {org_id}");
    };
    ctx.config.default_org = Some(org_id.clone());
    ctx.save()?;
    println!("Default org set to \"{}\" ({})", matched.name, matched.id);
    Ok(())
}
