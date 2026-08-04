use anyhow::Result;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

use super::context::Context;
use super::http;
use super::print;
use super::CloudGlobals;
use crate::api::v1::Meta;

/// The account behind the session token. Cloud-only: a single-tenant server
/// authenticates a machine, not a person, and answers `/meta` instead.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Me {
    id: String,
    email: String,
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
    me: Option<&'a Me>,
    server: &'a str,
    single_tenant: bool,
    features: &'a [String],
    pinned_org: Option<&'a str>,
    pinned_project: Option<&'a str>,
}

pub async fn run(globals: CloudGlobals) -> Result<()> {
    let ctx = Context::reporting(&globals)?;
    // A deployment that predates `/meta` is the hosted cloud, which has users.
    // Any other failure is the failure itself, not an old deployment.
    let meta: Meta = match ctx.client.get("/api/v1/meta").await {
        Ok(meta) => meta,
        Err(e) if http::status_of(&e) == Some(StatusCode::NOT_FOUND) => Meta::default(),
        Err(e) => return Err(e),
    };
    let me: Option<Me> = match meta.single_tenant {
        true => None,
        false => Some(ctx.client.get("/api/v1/me").await?),
    };

    let pinned_org = ctx.config.as_ref().and_then(|p| p.org());
    let pinned_project = ctx.config.as_ref().and_then(|p| p.project());

    if globals.json {
        return print::json(&WhoamiOutput {
            me: me.as_ref(),
            server: ctx.client.base_url(),
            single_tenant: meta.single_tenant,
            features: &meta.features,
            pinned_org,
            pinned_project,
        });
    }

    match &me {
        Some(me) => println!("Logged in as {} ({})", me.email, me.id),
        None => println!("Single-tenant server: no account to log in as."),
    }
    println!("Server: {}", ctx.client.base_url());
    if !meta.features.is_empty() {
        println!("Offers: {}", meta.features.join(", "));
    }

    let pinned_org_name = pinned_org.and_then(|id| {
        me.as_ref()?
            .organizations
            .iter()
            .find(|o| o.id == id)
            .map(|o| o.name.as_str())
    });
    match (pinned_org, pinned_org_name, &me) {
        (Some(id), Some(name), _) => println!("Pinned org: {name} ({id})"),
        (Some(id), None, Some(_)) => println!("Pinned org: {id} (not in your memberships?)"),
        (Some(id), None, None) => println!("Pinned org: {id}"),
        (None, _, _) => {
            println!("Pinned org: none");
            println!("Run `subs link` in your project to pin one.");
        }
    }
    if let Some(project_id) = pinned_project {
        println!("Pinned project: {project_id}");
    }

    if let Some(me) = &me {
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
    }

    Ok(())
}
