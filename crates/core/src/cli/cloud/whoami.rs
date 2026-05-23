// `subs cloud whoami` — calls /api/v1/me and prints the user, plus the
// CLI's currently-defaulted org/app from config.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::config;
use super::http::CloudClient;
use super::print;
use super::CloudGlobals;

#[derive(Debug, Serialize, Deserialize)]
struct Me {
    id: String,
    email: String,
    #[serde(default, rename = "isSuperAdmin")]
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
    default_org: Option<&'a str>,
    default_app: Option<&'a str>,
}

pub async fn run(globals: CloudGlobals) -> Result<()> {
    let path = config::resolve_path(globals.config)?;
    let cfg = config::load(&path)?;
    let token = cfg.require_token()?.to_string();
    let api_url = cfg.resolve_api_url(globals.url.as_deref());

    let client = CloudClient::new(&api_url, Some(token));
    let me: Me = client.get("/api/v1/me").await?;

    let default_org = cfg.default_org.as_deref();
    let default_app = default_org.and_then(|org| {
        cfg.orgs
            .get(org)
            .and_then(|o| o.default_app.as_deref())
    });

    if globals.json {
        return print::json(&WhoamiOutput {
            me: &me,
            default_org,
            default_app,
        });
    }

    println!("Logged in as {} ({})", me.email, me.id);
    if me.is_super_admin {
        println!("Role: super-admin");
    }

    let default_org_name = default_org.and_then(|id| me.organizations.iter().find(|o| o.id == id).map(|o| o.name.as_str()));
    match (default_org, default_org_name) {
        (Some(id), Some(name)) => println!("Default org: {name} ({id})"),
        (Some(id), None) => println!("Default org: {id} (not in your memberships?)"),
        (None, _) => println!("Default org: (none — run `subs cloud orgs use <id>`)"),
    }

    if let Some(app_id) = default_app {
        println!("Default app: {app_id}");
    }

    println!();
    println!("Organizations ({}):", me.organizations.len());
    for o in &me.organizations {
        let marker = if Some(o.id.as_str()) == default_org { "* " } else { "  " };
        println!("{marker}{}  {}  ({})", o.id, o.name, o.role);
    }

    Ok(())
}
