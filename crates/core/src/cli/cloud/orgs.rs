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
    let columns = [
        print::Column::left(""),
        print::Column::left("ID"),
        print::Column::left("NAME"),
        print::Column::left("ROLE"),
    ];
    let rows: Vec<Vec<String>> = orgs
        .iter()
        .map(|o| {
            let marker = if Some(o.id.as_str()) == pinned {
                "*"
            } else {
                ""
            };
            vec![marker.into(), o.id.clone(), o.name.clone(), o.role.clone()]
        })
        .collect();
    print::table(&columns, &rows);
    Ok(())
}
