//! `subs sessions`: what has happened, from whichever store holds it.
//!
//! Two sources, one output — the rule `subs doctor` already follows. A file
//! naming a `[remote]` asks the deployment, which holds the sessions. A file
//! naming none describes an engine you run here, whose sessions are in this
//! machine's database, so the list and the events are read straight off the
//! SQLite file `subs run` and `subs serve` write. Without that, a turn run here
//! leaves state no command can read.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context as _, Result};
use clap::{Args, Subcommand};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

use crate::event_store::Seq;
use crate::providers::sqlite::{SqliteDb, SqliteEventStore, SqliteSessionIndexStore};
use crate::session::index::{SessionCursor, SessionFilter, SessionItem, SessionSort};
use crate::session::read::SessionReader;
use crate::session::SessionEvent;
use crate::transport::ag_ui::translator::AgUiTranslator;
use crate::Caller;

use super::cloud::context::Context;
use super::cloud::{print, CloudGlobals, ProjectScope};
use super::env::OutputFormat;
use super::pretty::{self, write_json, Renderer};
use super::target::target;
use super::DEFAULT_TENANT;

/// How often a local `--stream` re-reads the file. There is no server to push,
/// so following one means asking it again.
const POLL: Duration = Duration::from_millis(250);

/// How long the database is held open for, matching `run` and `serve`, so a
/// read while an engine writes waits rather than failing.
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Subcommand)]
pub enum SessionsCommand {
    /// List debug sessions for a project.
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
    /// Read this SQLite database instead of a deployment. [default: the
    /// engine's, when the file names no `[remote]`]
    #[arg(long)]
    pub db: Option<String>,
    #[command(flatten)]
    pub scope: ProjectScope,
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
    /// Output mode. `jsonl` (default) prints the stored events; `pretty`
    /// replays the session as text, as `subs run` prints a live one; `ag-ui`
    /// translates it to protocol events.
    #[arg(long, short = 'o', value_enum)]
    pub output: Option<OutputFormat>,
    /// Read this SQLite database instead of a deployment. [default: the
    /// engine's, when the file names no `[remote]`]
    #[arg(long)]
    pub db: Option<String>,
    #[command(flatten)]
    pub scope: ProjectScope,
}

pub async fn run(command: SessionsCommand) -> Result<()> {
    match command {
        SessionsCommand::List(cmd) => match source(&cmd.scope.globals, cmd.db.clone())? {
            Source::Local(db) => local_list(&cmd, &db).await,
            Source::Remote => remote_list(&cmd).await,
        },
        SessionsCommand::Events(cmd) => match source(&cmd.scope.globals, cmd.db.clone())? {
            Source::Local(db) => local_events(&cmd, &db).await,
            Source::Remote => remote_events(&cmd).await,
        },
    }
}

/// Where a `sessions` command reads from.
enum Source {
    /// The deployment the file, the flag, or the default names.
    Remote,
    /// The database at this path, which an engine here writes.
    Local(String),
}

/// Which store answers. `--db` names one outright — the one case the file
/// cannot decide, since a database is not something a `[remote]` has an opinion
/// about. Everything else is [`target`], the rule every command follows.
fn source(globals: &CloudGlobals, db: Option<String>) -> Result<Source> {
    if let Some(db) = db {
        return Ok(Source::Local(db));
    }
    Ok(match target(globals)?.here() {
        Some(config) => Source::Local(config.db_path()),
        None => Source::Remote,
    })
}

/// The database an engine here wrote. Never created: this reads what happened,
/// and an empty file at a mistyped path answers every question with silence.
fn open_db(path: &str) -> Result<SqliteDb> {
    if !std::path::Path::new(path).exists() {
        bail!("no database at {path}. `subs run` or `subs serve` creates one.");
    }
    SqliteDb::open(path, BUSY_TIMEOUT).with_context(|| format!("opening {path}"))
}

/// The engine's own reader over a database on this machine — the code the API
/// serves its sessions with, without an engine around it. Reading needs the two
/// stores and nothing else, so nothing else is started.
fn reader(path: &str) -> Result<SessionReader> {
    let db = open_db(path)?;
    Ok(SessionReader::new(
        Arc::new(SqliteEventStore::new(db.clone())?),
        Arc::new(SqliteSessionIndexStore::new(db)?),
    ))
}

/// Who the CLI reads as. `run` acts as the same caller, and a system caller
/// answers to no session owner — this machine's database is already its own
/// permission.
fn caller() -> Caller {
    Caller::System {
        tenant_id: DEFAULT_TENANT.to_string(),
    }
}

// ---------------------------------------------------------------------------
// list
// ---------------------------------------------------------------------------

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

async fn remote_list(cmd: &ListCommand) -> Result<()> {
    let (ctx, project) = Context::from_project(&cmd.scope).await?;

    let query = serde_urlencoded::to_string(Query {
        limit: cmd.limit,
        cursor: cmd.cursor.as_deref(),
        session_id: cmd.session_id.as_deref(),
        agent_id: cmd.agent_id.as_deref(),
    })
    .context("encoding query string")?;

    let page: Page = ctx
        .client
        .get(&format!("/api/v1/projects/{project}/sessions?{query}"))
        .await?;

    render(&page, cmd.scope.globals.json)
}

async fn local_list(cmd: &ListCommand, db: &str) -> Result<()> {
    let page = local_page(cmd, db).await?;
    render(&page, cmd.scope.globals.json)
}

/// The same page the API builds, read off the file instead of asked for.
async fn local_page(cmd: &ListCommand, db: &str) -> Result<Page> {
    let reader = reader(db)?;

    let filter = SessionFilter {
        tenant_id: Some(DEFAULT_TENANT.to_string()),
        session_id: cmd.session_id.clone(),
        agent_id: cmd.agent_id.clone(),
        // What the API defaults to: a sub-agent's session is part of the
        // session that spawned it, not a row beside it.
        top_level: true,
        sort: SessionSort::LastEventDesc,
        limit: Some(cmd.limit as usize),
        cursor: cmd
            .cursor
            .as_deref()
            .map(SessionCursor::decode)
            .transpose()
            .map_err(|e| anyhow::anyhow!(e))?,
    };

    let page = reader.list(&filter).await?;
    Ok(Page {
        items: page.items.iter().map(row).collect(),
        next_cursor: page
            .next_cursor
            .as_ref()
            .map(SessionCursor::encode)
            .transpose()
            .map_err(|e| anyhow::anyhow!(e))?,
    })
}

fn row(item: &SessionItem) -> SessionRow {
    SessionRow {
        id: None,
        session_id: Some(item.session_id.clone()),
        agent_id: item.agent_id.clone(),
        first_event_at: item.first_event_at.map(|t| t.to_rfc3339()),
        last_event_at: item.last_event_at.map(|t| t.to_rfc3339()),
    }
}

fn render(page: &Page, json: bool) -> Result<()> {
    if json {
        return print::json(page);
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
    if let Some(c) = &page.next_cursor {
        println!();
        println!("Next page: --cursor {c}");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// events
// ---------------------------------------------------------------------------

/// Renders a stored stream the way `subs run` renders a live one, so reading a
/// session back looks like watching it happen.
///
/// A translator is one turn's — `terminated` flips at that turn's end and it
/// goes quiet — so a session of many turns gets one per turn. Every event in a
/// turn carries its id, so a turn opens on the first event belonging to it
/// rather than on `turn.started` alone: `--from` lands mid-turn, and the rest
/// of that turn is still worth reading. Events belonging to no turn belong to
/// no run, and translate to nothing.
struct Replay {
    session_id: String,
    renderer: Renderer,
    turn: Option<(String, AgUiTranslator)>,
}

impl Replay {
    fn new(session_id: String, output: OutputFormat) -> Self {
        Self {
            session_id,
            renderer: Renderer::new(output, pretty::color()),
            turn: None,
        }
    }

    /// Whether the caller prints the stored events itself. A translation cannot
    /// say what the engine wrote, so `jsonl` is not rendered from one.
    fn raw(&self) -> bool {
        self.renderer.is_raw()
    }

    fn push(&mut self, stdout: &mut std::io::Stdout, event: SessionEvent) -> Result<()> {
        if self.raw() {
            return write_json(stdout, &event);
        }

        let Some(turn_id) = event.meta.turn_id.clone() else {
            return Ok(());
        };

        // A turn that is not the open one opens its own: the previous turn
        // ended, or this is the first event this replay has seen.
        if self.turn.as_ref().is_none_or(|(open, _)| *open != turn_id) {
            let translator = AgUiTranslator::new(self.session_id.clone(), turn_id.clone());
            let started = translator.start();
            self.renderer.emit(stdout, started)?;
            self.turn = Some((turn_id, translator));
        }

        let Some((_, translator)) = self.turn.as_mut() else {
            return Ok(());
        };
        let events = translator.on_event(event.payload);
        self.renderer.emit(stdout, events)?;
        if translator.terminated {
            self.turn = None;
        }
        Ok(())
    }
}

async fn remote_events(cmd: &EventsCommand) -> Result<()> {
    let (ctx, project) = Context::from_project(&cmd.scope).await?;
    let session_id = require_session(cmd.session_id.as_deref())?;
    let mut replay = Replay::new(session_id.clone(), output(cmd));
    let mut stdout = std::io::stdout();

    if cmd.stream {
        let path = format!(
            "/api/v1/projects/{project}/sessions/{session_id}/events/stream?after_seq={}",
            cmd.from
        );
        // The line callback cannot fail, so the first failure is carried out
        // and reported once the stream ends rather than swallowed per line.
        let mut failed: Option<anyhow::Error> = None;
        ctx.client
            .stream_sse(&path, |line| {
                let Some(data) = line.strip_prefix("data:") else {
                    return;
                };
                let data = data.trim();
                if failed.is_some() || data.is_empty() {
                    return;
                }
                if let Err(e) = render_wire(&mut replay, &mut stdout, data) {
                    failed = Some(e);
                }
            })
            .await?;
        return match failed {
            Some(e) => Err(e),
            None => Ok(()),
        };
    }

    let events: Vec<Box<RawValue>> = ctx
        .client
        .get(&format!(
            "/api/v1/projects/{project}/sessions/{session_id}/events?after_seq={}",
            cmd.from
        ))
        .await?;
    for event in &events {
        render_wire(&mut replay, &mut stdout, event.get())?;
    }
    Ok(())
}

/// One event as the deployment sent it. `jsonl` prints those bytes rather than
/// a re-serialization of them, so what this shows is what the API answered.
fn render_wire(replay: &mut Replay, stdout: &mut std::io::Stdout, wire: &str) -> Result<()> {
    if replay.raw() {
        use std::io::Write as _;
        writeln!(stdout, "{wire}")?;
        return Ok(stdout.flush()?);
    }
    let event: SessionEvent =
        serde_json::from_str(wire).context("the deployment sent an event this CLI cannot read")?;
    replay.push(stdout, event)
}

/// What to print, defaulting to the stored events themselves. `[run].output` is
/// deliberately not read: it says how a turn you are watching should look, and
/// this command's job is what the engine wrote.
fn output(cmd: &EventsCommand) -> OutputFormat {
    cmd.output.unwrap_or(OutputFormat::Jsonl)
}

/// The stream as the file holds it. `--stream` polls, because nothing pushes
/// from a file: it shows what an engine writing to this database appends.
async fn local_events(cmd: &EventsCommand, db: &str) -> Result<()> {
    let session_id = require_session(cmd.session_id.as_deref())?;
    let reader = reader(db)?;
    let caller = caller();

    // A mistyped id would otherwise print nothing and, with `--stream`, print
    // nothing for as long as you left it running.
    let head = reader.events(&caller, &session_id, None, Some(1)).await?;
    if head.is_empty() {
        bail!("no session {session_id} in {db}");
    }

    let mut replay = Replay::new(session_id.clone(), output(cmd));
    let mut stdout = std::io::stdout();

    let mut after = Seq(cmd.from);
    loop {
        let events = reader
            .events(&caller, &session_id, Some(after), None)
            .await?;
        if let Some(last) = events.last() {
            after = Seq(last.seq);
        }
        for event in events {
            replay.push(&mut stdout, event)?;
        }
        if !cmd.stream {
            return Ok(());
        }
        tokio::time::sleep(POLL).await;
    }
}

fn require_session(session_id: Option<&str>) -> Result<String> {
    match session_id {
        Some(id) => Ok(id.to_string()),
        None => bail!("missing <SESSION_ID>. (Session picker not yet implemented.)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::cloud::project_config;
    use crate::session::events::EventPayload;
    use crate::session::index::{SessionIndexRecord, SessionIndexStore};
    use crate::session::state::EventMeta;
    use std::path::PathBuf;

    fn tmpdir() -> PathBuf {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("subs-sessions-test-{nanos}-{seq}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Writes a `substructure.toml` and returns globals pointing `-c` at it.
    fn wrote(body: &str) -> (CloudGlobals, PathBuf) {
        let dir = tmpdir();
        let path = dir.join(project_config::FILENAME);
        std::fs::write(&path, body).unwrap();
        (
            CloudGlobals {
                config: Some(path),
                ..Default::default()
            },
            dir,
        )
    }

    fn local_path(source: Source) -> String {
        match source {
            Source::Local(db) => db,
            Source::Remote => panic!("expected the local database"),
        }
    }

    fn is_remote(source: Source) -> bool {
        matches!(source, Source::Remote)
    }

    /// The rule `doctor` follows: no `[remote]` means the engine is here, so
    /// the sessions are too.
    const ENGINE_HERE: &str = "[llm.byo]\ntype = \"worker\"\n\
         [agent.support]\nllm = \"byo\"\nmodel = \"m\"\nworker = \"https://w.test\"\n";
    const A_REMOTE: &str = "[remote]\nurl = \"https://subs.test\"\n";

    #[test]
    fn a_file_naming_no_remote_reads_the_database_beside_it() {
        let (globals, dir) = wrote(ENGINE_HERE);
        assert_eq!(
            local_path(source(&globals, None).unwrap()),
            dir.join("substructure.db").display().to_string()
        );
    }

    #[test]
    fn a_file_naming_a_remote_asks_the_deployment() {
        let (globals, _dir) = wrote(A_REMOTE);
        assert!(is_remote(source(&globals, None).unwrap()));
    }

    /// Both flags name a store outright, so neither has to agree with the file:
    /// `--db` reads a database a `[remote]` file also has, and `--url` reaches a
    /// server the file never mentions.
    #[test]
    fn the_flags_name_a_store_the_file_does_not() {
        let (remote, _dir) = wrote(A_REMOTE);
        assert_eq!(
            local_path(source(&remote, Some("other.db".into())).unwrap()),
            "other.db"
        );

        let (mut here, _dir) = wrote(ENGINE_HERE);
        here.url = Some("http://127.0.0.1:8080".into());
        assert!(is_remote(source(&here, None).unwrap()));
    }

    /// A read never creates one: an empty database at a mistyped path would
    /// answer every question with an empty list.
    #[test]
    fn a_database_that_does_not_exist_is_an_error_not_a_new_one() {
        let path = tmpdir().join("absent.db");
        let err = match open_db(&path.display().to_string()) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("opened a database that does not exist"),
        };
        assert!(err.contains("no database at"), "{err}");
        assert!(!path.exists(), "the read created {}", path.display());
    }

    /// The page shape is the API's, whichever store filled it in.
    #[test]
    fn a_local_row_carries_the_same_fields_the_api_returns() {
        let item = SessionItem {
            session_id: "sess-1".into(),
            tenant_id: DEFAULT_TENANT.into(),
            seq: 4,
            first_event_at: Some(chrono::DateTime::UNIX_EPOCH),
            last_event_at: Some(chrono::DateTime::UNIX_EPOCH),
            wake_at: None,
            top_level: true,
            agent_id: Some("assistant".into()),
            cost: Default::default(),
            sub_agent_cost: Default::default(),
            status: crate::session::state::SessionStatus::Done,
            turn_id: None,
        };
        let row = row(&item);
        assert_eq!(row.session_id.as_deref(), Some("sess-1"));
        assert_eq!(row.agent_id.as_deref(), Some("assistant"));
        assert!(row.first_event_at.is_some());
        assert!(row.last_event_at.is_some());
    }

    fn list_command(db: &str, agent_id: Option<&str>) -> ListCommand {
        ListCommand {
            cursor: None,
            limit: 50,
            session_id: None,
            agent_id: agent_id.map(str::to_string),
            db: Some(db.to_string()),
            scope: ProjectScope {
                org: None,
                project: None,
                globals: CloudGlobals::default(),
            },
        }
    }

    fn indexed(session_id: &str, agent_id: &str, top_level: bool) -> SessionIndexRecord {
        SessionIndexRecord {
            tenant_id: DEFAULT_TENANT.into(),
            session_id: session_id.into(),
            seq: 3,
            first_event_at: Some(chrono::DateTime::UNIX_EPOCH),
            last_event_at: Some(chrono::DateTime::UNIX_EPOCH),
            wake_at: None,
            top_level,
            agent_id: Some(agent_id.into()),
            cost: Default::default(),
            sub_agent_cost: Default::default(),
            status: crate::session::state::SessionStatus::Done,
            turn_id: None,
        }
    }

    /// The whole point: a session an engine here wrote is one this reads back,
    /// off the file, with nothing running.
    #[tokio::test]
    async fn a_session_in_the_database_is_listed_from_it() {
        let db = tmpdir().join("t.db").display().to_string();
        let index =
            SqliteSessionIndexStore::new(SqliteDb::open(&db, BUSY_TIMEOUT).unwrap()).unwrap();
        index
            .upsert_session_index(indexed("sess-1", "assistant", true))
            .await
            .unwrap();
        // A sub-agent's session belongs to the one that spawned it, so the list
        // leaves it out — the same default the API applies.
        index
            .upsert_session_index(indexed("sess-2", "researcher", false))
            .await
            .unwrap();

        let page = local_page(&list_command(&db, None), &db).await.unwrap();
        let listed: Vec<_> = page
            .items
            .iter()
            .map(|r| r.session_id.clone().unwrap())
            .collect();
        assert_eq!(listed, vec!["sess-1".to_string()]);
        assert_eq!(page.items[0].agent_id.as_deref(), Some("assistant"));

        // The filters reach the store, rather than being dropped on the way.
        let none = local_page(&list_command(&db, Some("nobody")), &db)
            .await
            .unwrap();
        assert!(none.items.is_empty(), "{:?}", none.items);
    }

    /// A session of many turns is many runs. One translator for the whole
    /// stream would go quiet at the first turn's end, leaving every later turn
    /// unrendered.
    #[test]
    fn each_turn_in_a_replay_is_its_own_run() {
        let mut replay = Replay::new("sess-1".into(), OutputFormat::AgUi);
        let mut stdout = std::io::stdout();

        assert!(replay.turn.is_none());
        // An event belonging to no turn opens nothing.
        replay.push(&mut stdout, event(None)).unwrap();
        assert!(replay.turn.is_none());

        replay.push(&mut stdout, event(Some("turn-1"))).unwrap();
        assert_eq!(
            replay.turn.as_ref().map(|(id, _)| id.as_str()),
            Some("turn-1")
        );

        // The same turn keeps the run it opened...
        replay.push(&mut stdout, event(Some("turn-1"))).unwrap();
        assert_eq!(
            replay.turn.as_ref().map(|(id, _)| id.as_str()),
            Some("turn-1")
        );

        // ...and the next turn is a new one, whether or not the last one's end
        // was in the events this replay was given.
        replay.push(&mut stdout, event(Some("turn-2"))).unwrap();
        assert_eq!(
            replay.turn.as_ref().map(|(id, _)| id.as_str()),
            Some("turn-2")
        );
    }

    /// An event carrying only what the replay reads off it: which turn it
    /// belongs to.
    fn event(turn_id: Option<&str>) -> SessionEvent {
        let at = chrono::DateTime::UNIX_EPOCH;
        SessionEvent {
            id: uuid::Uuid::nil(),
            tenant_id: DEFAULT_TENANT.into(),
            session_id: "sess-1".into(),
            seq: 1,
            span: crate::span::SpanContext::root(),
            occurred_at: at,
            payload: EventPayload::DecisionQueued(crate::session::events::DecisionQueued {
                id: "d1".into(),
                trigger: crate::session::decision::Trigger::SessionStart,
            }),
            meta: EventMeta {
                status: crate::session::state::SessionStatus::Idle,
                wake_at: None,
                owner: None,
                agent_id: None,
                ancestry: Vec::new(),
                turn_id: turn_id.map(str::to_string),
                cost: Default::default(),
                sub_agent_cost: Default::default(),
                head_id: None,
                calls: Vec::new(),
                decisions: Vec::new(),
            },
            start_time: at,
            end_time: at,
        }
    }

    /// The cursor a page hands back is the one the next call hands in,
    /// whichever store produced it.
    #[test]
    fn a_cursor_survives_the_round_trip() {
        let cursor = SessionCursor {
            at: chrono::DateTime::UNIX_EPOCH,
            session_id: "sess-1".into(),
        };
        let decoded = SessionCursor::decode(&cursor.encode().unwrap()).unwrap();
        assert_eq!(decoded.session_id, "sess-1");
        assert_eq!(decoded.at, cursor.at);
    }
}
