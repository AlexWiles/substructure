use anyhow::{bail, Context as _, Result};
use clap::{Args, Subcommand};
use serde::{Deserialize, Serialize};

use super::context::Context;
use super::print;
use super::AppScope;

#[derive(Subcommand)]
pub enum SessionsCommand {
    /// List debug sessions for an app.
    #[command(name = "list", visible_alias = "ls")]
    List(ListCommand),
    /// Stream events for a session as they arrive. Default starts from
    /// sequence 0 (full history + live). Ctrl-C to stop.
    Events(EventsCommand),
}

#[derive(Args)]
pub struct ListCommand {
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

#[derive(Args)]
pub struct EventsCommand {
    /// Session id. If omitted, you'll be prompted to pick from recent sessions.
    pub session_id: Option<String>,
    /// Only stream events with sequence > this value (defaults to 0 = full history).
    #[arg(long, default_value_t = 0)]
    pub from: u64,
    #[command(flatten)]
    pub scope: AppScope,
}

pub async fn run(command: SessionsCommand) -> Result<()> {
    match command {
        SessionsCommand::List(cmd) => list(cmd).await,
        SessionsCommand::Events(cmd) => events(cmd).await,
    }
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
    first_event_at: Option<String>,
    #[serde(default)]
    last_event_at: Option<String>,
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

async fn list(cmd: ListCommand) -> Result<()> {
    let (ctx, app) = Context::from_app(&cmd.scope).await?;

    let query = serde_urlencoded::to_string(&Query {
        limit: cmd.limit,
        cursor: cmd.cursor.as_deref(),
        session_id: cmd.session_id.as_deref(),
        agent_id: cmd.agent_id.as_deref(),
    })
    .context("encoding query string")?;

    let page: Page = ctx
        .client
        .get(&format!("/api/v1/apps/{app}/sessions?{query}"))
        .await?;

    if cmd.scope.globals.json {
        return print::json(&page);
    }

    println!(
        "{:<40} {:<30} {:<30} {}",
        "SESSION_ID", "AGENT", "FIRST_EVENT", "LAST_EVENT"
    );
    for s in &page.items {
        let sid = s.session_id.as_deref().or(s.id.as_deref()).unwrap_or("-");
        println!(
            "{:<40} {:<30} {:<30} {}",
            sid,
            s.agent_id.as_deref().unwrap_or("-"),
            s.first_event_at.as_deref().unwrap_or("-"),
            s.last_event_at.as_deref().unwrap_or("-"),
        );
    }
    if let Some(c) = page.next_cursor {
        println!();
        println!("Next page: --cursor {c}");
    }
    Ok(())
}

async fn events(cmd: EventsCommand) -> Result<()> {
    let (ctx, app) = Context::from_app(&cmd.scope).await?;
    let session_id = match cmd.session_id {
        Some(id) => id,
        None => bail!("missing <SESSION_ID>. (Session picker not yet implemented.)"),
    };
    let path = format!(
        "/api/v1/apps/{app}/sessions/{session_id}/events/stream?sequence_after={}",
        cmd.from
    );
    ctx.client
        .stream_sse(&path, |line| {
            // SSE frames: each event is a series of lines, blank-line separated.
            // Print as-is so users can pipe into jq / awk / less. Comments
            // (lines starting with `:`) and keep-alives are suppressed.
            if line.starts_with(':') {
                return;
            }
            println!("{line}");
        })
        .await
}
