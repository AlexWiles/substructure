// `subs cloud whoami` — calls /api/v1/me and prints the user, plus the
// CLI's currently-defaulted org/app from config.

use std::path::PathBuf;

use anyhow::Result;
use serde::Deserialize;

use super::config;
use super::http::CloudClient;

#[derive(Debug, Deserialize)]
struct Me {
    id: String,
    email: String,
    #[serde(default)]
    #[serde(rename = "isSuperAdmin")]
    is_super_admin: bool,
    #[serde(default)]
    organizations: Vec<OrgRef>,
}

#[derive(Debug, Deserialize)]
struct OrgRef {
    id: String,
    name: String,
    role: String,
}

pub async fn run(url_flag: Option<String>, config_path: Option<PathBuf>) -> Result<()> {
    let path = config::resolve_path(config_path)?;
    let cfg = config::load(&path)?;
    let token = cfg.require_token()?.to_string();
    let api_url = cfg.resolve_api_url(url_flag.as_deref());

    let client = CloudClient::new(&api_url, Some(token));
    let me: Me = client.get("/api/v1/me").await?;

    println!("Logged in as {} ({})", me.email, me.id);
    if me.is_super_admin {
        println!("Role: super-admin");
    }

    let default_org = cfg.default_org.as_deref();
    let default_org_name = default_org.and_then(|id| me.organizations.iter().find(|o| o.id == id).map(|o| o.name.as_str()));
    match (default_org, default_org_name) {
        (Some(id), Some(name)) => println!("Default org: {name} ({id})"),
        (Some(id), None) => println!("Default org: {id} (not in your memberships?)"),
        (None, _) => println!("Default org: (none — run `subs cloud orgs use <id>`)"),
    }

    if let Some(org_id) = default_org {
        if let Some(app_id) = cfg.resolve_app(org_id, None) {
            println!("Default app: {app_id}");
        }
    }

    println!();
    println!("Organizations ({}):", me.organizations.len());
    for o in &me.organizations {
        let marker = if Some(o.id.as_str()) == default_org { "* " } else { "  " };
        println!("{marker}{}  {}  ({})", o.id, o.name, o.role);
    }

    Ok(())
}
