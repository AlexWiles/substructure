use anyhow::{Context as _, Result};
use clap::Args;
use serde::{Deserialize, Serialize};

use super::context::Context;
use super::print;
use super::AppScope;

#[derive(Args)]
pub struct SessionsCommand {
    #[arg(long)]
    pub cursor: Option<String>,
    #[arg(long, default_value_t = 50)]
    pub limit: u32,
    #[arg(long)]
    pub session_id: Option<String>,
    #[arg(long)]
    pub agent_id: Option<String>,
    #[command(flatten)]
    pub scope: AppScope,
}

#[derive(Debug, Serialize, Deserialize)]
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

#[derive(Debug, Serialize, Deserialize)]
struct Page {
    items: Vec<SessionRow>,
    #[serde(default)]
    next_cursor: Option<String>,
}

#[derive(Debug, Serialize)]
struct Query<'a> {
    limit: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    cursor: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    session_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    agent_id: Option<&'a str>,
}

pub async fn run(cmd: SessionsCommand) -> Result<()> {
    let (ctx, org, app) = Context::from_app(&cmd.scope)?;

    let query = serde_urlencoded::to_string(&Query {
        limit: cmd.limit,
        cursor: cmd.cursor.as_deref(),
        session_id: cmd.session_id.as_deref(),
        agent_id: cmd.agent_id.as_deref(),
    })
    .context("encoding query string")?;

    let page: Page = ctx
        .client
        .get(&format!("/api/v1/orgs/{org}/apps/{app}/sessions?{query}"))
        .await?;

    if cmd.scope.globals.json {
        return print::json(&page);
    }

    println!("{:<40} {:<30} {}", "SESSION_ID", "AGENT", "CREATED");
    for s in &page.items {
        let sid = s.session_id.as_deref().or(s.id.as_deref()).unwrap_or("-");
        println!(
            "{:<40} {:<30} {}",
            sid,
            s.agent_id.as_deref().unwrap_or("-"),
            s.created_at.as_deref().unwrap_or("-"),
        );
    }
    if let Some(c) = page.next_cursor {
        println!();
        println!("Next page: --cursor {c}");
    }
    Ok(())
}
