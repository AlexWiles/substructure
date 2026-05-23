// `subs cloud sessions` — list debug sessions for the current app.

use anyhow::Result;
use clap::Args;
use serde::Deserialize;

use super::context::Context;
use super::CloudGlobals;

#[derive(Args)]
pub struct SessionsCommand {
    #[arg(long)]
    pub org: Option<String>,
    #[arg(long)]
    pub app: Option<String>,
    #[arg(long)]
    pub cursor: Option<String>,
    #[arg(long, default_value_t = 50)]
    pub limit: u32,
    #[arg(long)]
    pub session_id: Option<String>,
    #[arg(long)]
    pub agent_id: Option<String>,
    #[command(flatten)]
    pub globals: CloudGlobals,
}

#[derive(Debug, Deserialize)]
struct SessionRow {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    agent_id: Option<String>,
    #[serde(default)]
    created_at: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Page {
    items: Vec<SessionRow>,
    #[serde(default)]
    next_cursor: Option<String>,
}

pub async fn run(cmd: SessionsCommand) -> Result<()> {
    let ctx = Context::load(&cmd.globals)?;
    let org = ctx.require_org(cmd.org.as_deref())?;
    let app = ctx.require_app(&org, cmd.app.as_deref())?;

    let mut qs: Vec<(&str, String)> = Vec::new();
    qs.push(("limit", cmd.limit.to_string()));
    if let Some(c) = cmd.cursor.as_ref() {
        qs.push(("cursor", c.clone()));
    }
    if let Some(s) = cmd.session_id.as_ref() {
        qs.push(("session_id", s.clone()));
    }
    if let Some(a) = cmd.agent_id.as_ref() {
        qs.push(("agent_id", a.clone()));
    }
    let qs_pairs: Vec<(&str, &str)> = qs.iter().map(|(k, v)| (*k, v.as_str())).collect();
    let query = serde_urlencoded::to_string(&qs_pairs).unwrap_or_default();
    let path = format!("/api/v1/orgs/{org}/apps/{app}/sessions?{query}");

    let page: Page = ctx.client.get(&path).await?;
    println!("{:<40} {:<30} {}", "SESSION_ID", "AGENT", "CREATED");
    for s in &page.items {
        let sid = s.session_id.as_deref().or(s.id.as_deref()).unwrap_or("—");
        println!(
            "{:<40} {:<30} {}",
            sid,
            s.agent_id.as_deref().unwrap_or("—"),
            s.created_at.as_deref().unwrap_or("—"),
        );
    }
    if let Some(c) = page.next_cursor {
        println!();
        println!("Next page: --cursor {c}");
    }
    Ok(())
}
