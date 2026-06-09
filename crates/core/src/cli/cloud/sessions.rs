use anyhow::{bail, Context as _, Result};
use clap::{Args, Subcommand};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

use super::context::Context;
use super::print;
use super::AppScope;

#[derive(Subcommand)]
pub enum SessionsCommand {
    /// List debug sessions for an app.
    #[command(name = "list", visible_alias = "ls")]
    List(ListCommand),
    /// Print a session's events. Prints all events and exits; pass --stream to
    /// follow live (Ctrl-C to stop).
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
    /// Only include events with sequence > this value (0 = full history).
    #[arg(long, default_value_t = 0)]
    pub from: u64,
    /// Follow the session live instead of printing all events and exiting.
    #[arg(long, short = 'f')]
    pub stream: bool,
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

    let columns = [
        print::Column::left("SESSION_ID"),
        print::Column::left("AGENT"),
        print::Column::left("FIRST_EVENT"),
        print::Column::left("LAST_EVENT"),
    ];
    let rows: Vec<Vec<String>> = page
        .items
        .iter()
        .map(|s| {
            let sid = s.session_id.as_deref().or(s.id.as_deref()).unwrap_or("-");
            vec![
                sid.into(),
                s.agent_id.clone().unwrap_or_else(|| "-".into()),
                s.first_event_at.clone().unwrap_or_else(|| "-".into()),
                s.last_event_at.clone().unwrap_or_else(|| "-".into()),
            ]
        })
        .collect();
    print::table(&columns, &rows);
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

    if cmd.stream {
        let path = format!(
            "/api/v1/apps/{app}/sessions/{session_id}/events/stream?sequence_after={}",
            cmd.from
        );
        return ctx
            .client
            .stream_sse(&path, |line| {
                if let Some(data) = line.strip_prefix("data:") {
                    println!("{}", data.trim());
                }
            })
            .await;
    }

    let events: Vec<Box<RawValue>> = ctx
        .client
        .get(&format!(
            "/api/v1/apps/{app}/sessions/{session_id}/events?sequence_after={}",
            cmd.from
        ))
        .await?;
    for event in &events {
        println!("{}", event.get());
    }
    Ok(())
}
