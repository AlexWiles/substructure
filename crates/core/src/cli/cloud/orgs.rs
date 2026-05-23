// `subs cloud orgs list`: list the orgs you belong to. There is no
// `orgs use`: pinning happens via `subs cloud init`, which writes a
// project-local `subs.toml`. Pass `--org <id>` for one-off overrides.

use anyhow::Result;
use clap::Subcommand;
use serde::{Deserialize, Serialize};

use super::context::Context;
use super::print;
use super::CloudGlobals;

#[derive(Subcommand)]
pub enum OrgsCommand {
    /// List organizations you belong to.
    #[command(name = "list", visible_alias = "ls")]
    List {
        #[command(flatten)]
        globals: CloudGlobals,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct OrgRef {
    id: String,
    name: String,
    role: String,
}

pub async fn run(command: OrgsCommand) -> Result<()> {
    match command {
        OrgsCommand::List { globals } => list(globals).await,
    }
}

async fn list(globals: CloudGlobals) -> Result<()> {
    let ctx = Context::load(&globals)?;
    let orgs: Vec<OrgRef> = ctx.client.get("/api/v1/orgs").await?;

    if globals.json {
        return print::json(&orgs);
    }

    let pinned = ctx.project.as_ref().and_then(|p| p.org.as_deref());
    println!("{:<38} {:<30} {}", "ID", "NAME", "ROLE");
    for o in &orgs {
        let marker = if Some(o.id.as_str()) == pinned {
            "*"
        } else {
            " "
        };
        println!("{marker} {:<36} {:<30} {}", o.id, o.name, o.role);
    }
    Ok(())
}
