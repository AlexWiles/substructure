use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::context::Context;
use super::print;
use super::CloudGlobals;

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Me {
    id: String,
    email: String,
    #[serde(default)]
    is_super_admin: bool,
    #[serde(default)]
    organizations: Vec<OrgRef>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OrgRef {
    id: String,
    name: String,
    role: String,
}

#[derive(Debug, Serialize)]
struct WhoamiOutput<'a> {
    #[serde(flatten)]
    me: &'a Me,
    pinned_org: Option<&'a str>,
    pinned_app: Option<&'a str>,
}

pub async fn run(globals: CloudGlobals) -> Result<()> {
    let ctx = Context::load(&globals)?;
    let me: Me = ctx.client.get("/api/v1/me").await?;

    let pinned_org = ctx.project.as_ref().and_then(|p| p.org.as_deref());
    let pinned_app = ctx.project.as_ref().and_then(|p| p.app.as_deref());

    if globals.json {
        return print::json(&WhoamiOutput {
            me: &me,
            pinned_org,
            pinned_app,
        });
    }

    println!("Logged in as {} ({})", me.email, me.id);
    if me.is_super_admin {
        println!("Role: super-admin");
    }

    let pinned_org_name = pinned_org.and_then(|id| {
        me.organizations
            .iter()
            .find(|o| o.id == id)
            .map(|o| o.name.as_str())
    });
    match (pinned_org, pinned_org_name) {
        (Some(id), Some(name)) => println!("Pinned org: {name} ({id})"),
        (Some(id), None) => println!("Pinned org: {id} (not in your memberships?)"),
        (None, _) => {
            println!("Pinned org: none");
            println!("Run `substructure cloud link` in your project to pin one.");
        }
    }
    if let Some(app_id) = pinned_app {
        println!("Pinned app: {app_id}");
    }

    println!();
    println!("Organizations ({}):", me.organizations.len());
    for o in &me.organizations {
        let marker = if Some(o.id.as_str()) == pinned_org {
            "* "
        } else {
            "  "
        };
        println!("{marker}{}  {}  ({})", o.id, o.name, o.role);
    }

    Ok(())
}
