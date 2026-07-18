use std::collections::HashMap;
use std::sync::Arc as StdArc;

use async_trait::async_trait;
use rusqlite::{Connection, OptionalExtension};
use sea_query::{Expr, ExprTrait, Iden, Order, Query, SqliteQueryBuilder};
use tokio::sync::broadcast;
use uuid::Uuid;

use crate::event_store::{AppendInput, EventFilter, EventStore, GlobalPosition, Seq, StoreError};
use crate::protocol::{Message, NewMessage};
use crate::runtime::session::events::EventPayload;
use crate::runtime::session::state::{AgentVersion, Logged, SessionState, StateVersion};
use crate::runtime::session::{NewSessionEvent, SessionAggregate, SessionEvent};

use super::{parse_dt, sea_params, spawn_err, SqliteDb};

/// The two version logs, split by kind, as loaded from `session_versions`.
type VersionLogs = (Vec<Logged<StateVersion>>, Vec<Logged<AgentVersion>>);

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS events (
    global_position  INTEGER PRIMARY KEY AUTOINCREMENT,
    id               TEXT NOT NULL,
    tenant_id        TEXT NOT NULL,
    session_id       TEXT NOT NULL,
    seq              INTEGER NOT NULL,
    occurred_at      TEXT NOT NULL,
    start_time       TEXT NOT NULL,
    end_time         TEXT NOT NULL,
    span             TEXT NOT NULL,
    payload          TEXT NOT NULL,
    meta             TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_events_session_seq ON events (tenant_id, session_id, seq);
CREATE INDEX IF NOT EXISTS idx_events_session ON events (tenant_id, session_id);

CREATE TABLE IF NOT EXISTS messages (
    tenant_id   TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    message_id  TEXT NOT NULL,
    parent_id   TEXT,
    seq         INTEGER NOT NULL,
    data        TEXT NOT NULL,
    PRIMARY KEY (tenant_id, session_id, message_id)
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (tenant_id, session_id, seq);

CREATE TABLE IF NOT EXISTS session_versions (
    tenant_id   TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    kind        TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    anchor      TEXT,
    data        TEXT NOT NULL,
    PRIMARY KEY (tenant_id, session_id, kind, seq)
);

CREATE TABLE IF NOT EXISTS llm_prompts (
    tenant_id   TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    call_id     TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    data        TEXT NOT NULL,
    PRIMARY KEY (tenant_id, session_id, call_id)
);

CREATE TABLE IF NOT EXISTS snapshots (
    tenant_id       TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    data            TEXT NOT NULL,
    wake_at         TEXT,
    first_event_at  TEXT,
    last_event_at   TEXT,
    PRIMARY KEY (tenant_id, session_id)
);
CREATE INDEX IF NOT EXISTS idx_snapshots_wake_at ON snapshots (wake_at);
CREATE INDEX IF NOT EXISTS idx_snapshots_last_event ON snapshots (last_event_at);
";

#[derive(Iden)]
enum Events {
    Table,
    GlobalPosition,
    Id,
    TenantId,
    SessionId,
    Seq,
    OccurredAt,
    StartTime,
    EndTime,
    Span,
    Payload,
    Meta,
}

pub struct SqliteEventStore {
    db: SqliteDb,
    tx: broadcast::Sender<StdArc<Vec<SessionEvent>>>,
}

impl SqliteEventStore {
    pub fn new(db: SqliteDb) -> Result<Self, StoreError> {
        db.run_schema(SCHEMA)?;
        let (tx, _) = broadcast::channel(1024);
        Ok(Self { db, tx })
    }
}

#[async_trait]
impl EventStore for SqliteEventStore {
    async fn append(&self, input: AppendInput) -> Result<(), StoreError> {
        let writer = self.db.writer.clone();
        let (positions, input) = tokio::task::spawn_blocking(move || {
            let mut conn = writer
                .lock()
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            do_append(&mut conn, &input).map(|positions| (positions, input))
        })
        .await
        .map_err(spawn_err)??;

        if input.events.len() != positions.len() {
            return Err(StoreError::Internal(
                "inserted event count did not match assigned positions".into(),
            ));
        }

        let events: Vec<SessionEvent> = input
            .events
            .into_iter()
            .zip(positions)
            .map(|(event, position)| event.into_event(GlobalPosition(position)))
            .collect();

        let _ = self.tx.send(StdArc::new(events));
        Ok(())
    }

    async fn load(
        &self,
        tenant_id: &str,
        session_id: &str,
    ) -> Result<SessionAggregate, StoreError> {
        let reader = self.db.reader.clone();
        let tenant_id = tenant_id.to_string();
        let session_id = session_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = reader.open()?;
            do_load(&conn, &tenant_id, &session_id)
        })
        .await
        .map_err(spawn_err)?
    }

    async fn query_events(&self, filter: &EventFilter) -> Result<Vec<SessionEvent>, StoreError> {
        let filter = filter.clone();
        let reader = self.db.reader.clone();
        tokio::task::spawn_blocking(move || {
            let conn = reader.open()?;
            do_query_events(&conn, filter)
        })
        .await
        .map_err(spawn_err)?
    }

    fn subscribe(&self) -> broadcast::Receiver<StdArc<Vec<SessionEvent>>> {
        self.tx.subscribe()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn internal(e: impl ToString) -> StoreError {
    StoreError::Internal(e.to_string())
}

fn do_append(conn: &mut Connection, input: &AppendInput) -> Result<Vec<u64>, StoreError> {
    let snap = &input.snapshot;
    let tx = conn.transaction().map_err(internal)?;

    let positions = input
        .events
        .iter()
        .map(|event| insert_event(&tx, snap, event, input.expected_version))
        .collect::<Result<Vec<_>, _>>()?;

    let snapshot_data = serde_json::to_string(&strip(&snap.state)).map_err(internal)?;

    tx.execute(
        "INSERT INTO snapshots (tenant_id, session_id, seq, data, wake_at, first_event_at, last_event_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
         ON CONFLICT(tenant_id, session_id) DO UPDATE SET
             seq = excluded.seq,
             data = excluded.data,
             wake_at = excluded.wake_at,
             first_event_at = COALESCE(snapshots.first_event_at, excluded.first_event_at),
             last_event_at = excluded.last_event_at",
        rusqlite::params![
            snap.tenant_id,
            snap.id,
            snap.seq,
            snapshot_data,
            snap.wake_at.map(|t| t.to_rfc3339()),
            snap.first_event_at.map(|t| t.to_rfc3339()),
            snap.last_event_at.map(|t| t.to_rfc3339()),
        ],
    )
    .map_err(internal)?;

    tx.commit().map_err(internal)?;

    Ok(positions)
}

/// One event: the CAS-guarded envelope insert, then its side-table rows.
/// Returns the assigned global position.
fn insert_event(
    tx: &rusqlite::Transaction<'_>,
    snap: &SessionAggregate,
    event: &NewSessionEvent,
    expected: Seq,
) -> Result<u64, StoreError> {
    let sid = &snap.id;
    let expected_i64 = i64::try_from(expected.0)
        .map_err(|_| StoreError::Internal("expected_version exceeds i64".into()))?;
    let payload = serde_json::to_string(&event.payload).map_err(internal)?;
    let meta = serde_json::to_string(&event.meta).map_err(internal)?;
    let span = serde_json::to_string(&event.span).map_err(internal)?;

    let rows = tx
        .execute(
            "INSERT INTO events (id, tenant_id, session_id, seq, occurred_at, start_time, end_time, span, payload, meta)
             SELECT ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10
             WHERE (
                 (?11 = 0 AND NOT EXISTS (SELECT 1 FROM snapshots WHERE tenant_id = ?2 AND session_id = ?3))
                 OR EXISTS (
                     SELECT 1 FROM snapshots
                     WHERE tenant_id = ?2 AND session_id = ?3 AND seq = ?11
                 )
             )",
            rusqlite::params![
                event.id.to_string(),
                snap.tenant_id,
                sid,
                event.seq,
                event.occurred_at.to_rfc3339(),
                event.start_time.to_rfc3339(),
                event.end_time.to_rfc3339(),
                span,
                payload,
                meta,
                expected_i64,
            ],
        )
        .map_err(internal)?;

    if rows == 0 {
        let actual_version: u64 = tx
            .query_row(
                "SELECT seq FROM snapshots WHERE tenant_id = ?1 AND session_id = ?2",
                rusqlite::params![&snap.tenant_id, &sid],
                |row| row.get(0),
            )
            .optional()
            .map_err(internal)?
            .unwrap_or(0);
        return Err(StoreError::VersionConflict {
            expected,
            actual: Seq(actual_version),
        });
    }

    let position = u64::try_from(tx.last_insert_rowid())
        .map_err(|_| StoreError::Internal("event position exceeds u64".into()))?;

    // OR IGNORE keeps appends idempotent under the conflict-retry loop.
    match &event.payload {
        EventPayload::NewMessage(node) => {
            let data = serde_json::to_string(&node.message).map_err(internal)?;
            tx.execute(
                "INSERT OR IGNORE INTO messages (tenant_id, session_id, message_id, parent_id, seq, data)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                rusqlite::params![
                    snap.tenant_id,
                    sid,
                    node.message.id,
                    node.parent_id,
                    event.seq,
                    data,
                ],
            )
            .map_err(internal)?;
        }
        // The worker-constructed prompt, verbatim; a re-request replaces
        // it, mirroring apply(). Stored from post-apply state so load
        // rejoins byte-identical bytes.
        EventPayload::LlmCallRequested(req) => {
            if let Some(call) = snap.state.llm_calls.get(&req.call_id) {
                let data = serde_json::to_string(&call.prompt).map_err(internal)?;
                tx.execute(
                    "INSERT INTO llm_prompts (tenant_id, session_id, call_id, seq, data)
                     VALUES (?1, ?2, ?3, ?4, ?5)
                     ON CONFLICT(tenant_id, session_id, call_id) DO UPDATE SET
                         seq = excluded.seq,
                         data = excluded.data",
                    rusqlite::params![snap.tenant_id, sid, req.call_id, event.seq, data],
                )
                .map_err(internal)?;
            }
        }
        EventPayload::WorkerStateUpdated(p) => {
            let data = serde_json::to_string(&p.state).map_err(internal)?;
            insert_version(
                tx,
                &snap.tenant_id,
                sid,
                "state",
                event.seq,
                &p.anchor,
                &data,
            )?;
        }
        EventPayload::AgentConfigUpdated(p) => {
            let data = serde_json::to_string(&p.config).map_err(internal)?;
            insert_version(
                tx,
                &snap.tenant_id,
                sid,
                "agent",
                event.seq,
                &p.anchor,
                &data,
            )?;
        }
        _ => {}
    }

    Ok(position)
}

#[allow(clippy::too_many_arguments)]
fn insert_version(
    tx: &rusqlite::Transaction<'_>,
    tenant_id: &str,
    session_id: &str,
    kind: &str,
    seq: u64,
    anchor: &Option<String>,
    data: &str,
) -> Result<(), StoreError> {
    tx.execute(
        "INSERT OR IGNORE INTO session_versions (tenant_id, session_id, kind, seq, anchor, data)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params![tenant_id, session_id, kind, seq, anchor, data],
    )
    .map_err(internal)?;
    Ok(())
}

fn do_load(
    conn: &Connection,
    tenant_id: &str,
    session_id: &str,
) -> Result<SessionAggregate, StoreError> {
    let (seq, data, wake_at, first_event_at, last_event_at) = conn
        .query_row(
            "SELECT seq, data, wake_at, first_event_at, last_event_at
             FROM snapshots WHERE tenant_id = ?1 AND session_id = ?2",
            rusqlite::params![tenant_id, session_id],
            |row| {
                Ok((
                    row.get::<_, u64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, Option<String>>(4)?,
                ))
            },
        )
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => StoreError::StreamNotFound,
            other => internal(other),
        })?;

    let state = hydrate(
        serde_json::from_str(&data).map_err(internal)?,
        load_nodes(conn, tenant_id, session_id)?,
        load_versions(conn, tenant_id, session_id)?,
        load_prompts(conn, tenant_id, session_id)?,
    );

    Ok(SessionAggregate {
        id: session_id.to_string(),
        tenant_id: tenant_id.to_string(),
        state,
        seq,
        first_event_at: first_event_at.as_deref().and_then(parse_dt),
        last_event_at: last_event_at.as_deref().and_then(parse_dt),
        wake_at: wake_at.as_deref().and_then(parse_dt),
    })
}

/// Tree, versions, and prompts live in their own tables; the snapshot holds
/// the rest.
fn strip(state: &SessionState) -> SessionState {
    let mut state = state.clone();
    state.nodes = Vec::new();
    state.state_versions = Vec::new();
    state.agent_versions = Vec::new();
    for call in state.llm_calls.values_mut() {
        call.prompt = Vec::new();
    }
    state
}

/// The inverse of `strip`: rejoin side-table rows onto the snapshot state.
fn hydrate(
    mut state: SessionState,
    nodes: Vec<Logged<NewMessage>>,
    versions: VersionLogs,
    prompts: HashMap<String, Vec<Message>>,
) -> SessionState {
    let (state_versions, agent_versions) = versions;
    state.nodes = nodes;
    state.state_versions = state_versions;
    state.agent_versions = agent_versions;
    for (call_id, prompt) in prompts {
        if let Some(call) = state.llm_calls.get_mut(&call_id) {
            call.prompt = prompt;
        }
    }
    state
}

fn load_nodes(
    conn: &Connection,
    tenant_id: &str,
    session_id: &str,
) -> Result<Vec<Logged<NewMessage>>, StoreError> {
    let mut stmt = conn
        .prepare(
            "SELECT seq, parent_id, data FROM messages
             WHERE tenant_id = ?1 AND session_id = ?2 ORDER BY seq",
        )
        .map_err(internal)?;
    let rows = stmt
        .query_map(rusqlite::params![tenant_id, session_id], |row| {
            Ok((
                row.get::<_, u64>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, String>(2)?,
            ))
        })
        .map_err(internal)?;

    rows.map(|row| {
        let (seq, parent_id, data) = row.map_err(internal)?;
        Ok(Logged {
            seq,
            entry: NewMessage {
                message: serde_json::from_str(&data).map_err(internal)?,
                parent_id,
            },
        })
    })
    .collect()
}

fn load_versions(
    conn: &Connection,
    tenant_id: &str,
    session_id: &str,
) -> Result<VersionLogs, StoreError> {
    let mut stmt = conn
        .prepare(
            "SELECT kind, seq, anchor, data FROM session_versions
             WHERE tenant_id = ?1 AND session_id = ?2 ORDER BY seq",
        )
        .map_err(internal)?;
    let rows = stmt
        .query_map(rusqlite::params![tenant_id, session_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, u64>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, String>(3)?,
            ))
        })
        .map_err(internal)?;

    let mut states = Vec::new();
    let mut agents = Vec::new();
    for row in rows {
        let (kind, seq, anchor, data) = row.map_err(internal)?;
        match kind.as_str() {
            "state" => states.push(Logged {
                seq,
                entry: StateVersion {
                    state: serde_json::from_str(&data).map_err(internal)?,
                    anchor,
                },
            }),
            "agent" => agents.push(Logged {
                seq,
                entry: AgentVersion {
                    agent: serde_json::from_str(&data).map_err(internal)?,
                    anchor,
                },
            }),
            other => {
                return Err(StoreError::Internal(format!(
                    "unknown version kind {other}"
                )))
            }
        }
    }
    Ok((states, agents))
}

fn load_prompts(
    conn: &Connection,
    tenant_id: &str,
    session_id: &str,
) -> Result<HashMap<String, Vec<Message>>, StoreError> {
    let mut stmt = conn
        .prepare(
            "SELECT call_id, data FROM llm_prompts
             WHERE tenant_id = ?1 AND session_id = ?2",
        )
        .map_err(internal)?;
    let rows = stmt
        .query_map(rusqlite::params![tenant_id, session_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(internal)?;

    rows.map(|row| {
        let (call_id, data) = row.map_err(internal)?;
        Ok((call_id, serde_json::from_str(&data).map_err(internal)?))
    })
    .collect()
}

fn do_query_events(
    conn: &Connection,
    filter: EventFilter,
) -> Result<Vec<SessionEvent>, StoreError> {
    let (sql, values) = Query::select()
        .columns([
            Events::GlobalPosition,
            Events::Id,
            Events::TenantId,
            Events::SessionId,
            Events::Seq,
            Events::OccurredAt,
            Events::StartTime,
            Events::EndTime,
            Events::Span,
            Events::Payload,
            Events::Meta,
        ])
        .from(Events::Table)
        .apply_if(filter.after_global_position, |q, pos| {
            q.and_where(Expr::col(Events::GlobalPosition).gt(pos.0 as i64));
        })
        .apply_if(filter.session_id.as_ref(), |q, id| {
            q.and_where(Expr::col(Events::SessionId).eq(id));
        })
        .apply_if(filter.tenant_id.as_ref(), |q, v| {
            q.and_where(Expr::col(Events::TenantId).eq(v));
        })
        .apply_if(filter.after_seq, |q, v| {
            q.and_where(Expr::col(Events::Seq).gt(v.0 as i64));
        })
        .apply_if(filter.limit, |q, n| {
            q.limit(n as u64);
        })
        .order_by(Events::GlobalPosition, Order::Asc)
        .build(SqliteQueryBuilder);

    let params = sea_params(values);
    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();

    let mut stmt = conn.prepare(&sql).map_err(internal)?;

    let rows = stmt
        .query_map(param_refs.as_slice(), |row| {
            Ok((
                row.get::<_, u64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, u64>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, String>(7)?,
                row.get::<_, String>(8)?,
                row.get::<_, String>(9)?,
                row.get::<_, String>(10)?,
            ))
        })
        .map_err(internal)?;

    rows.map(|row| {
        let (
            position,
            id,
            tenant_id,
            session_id,
            seq,
            occurred_at,
            start_time,
            end_time,
            span,
            payload,
            meta,
        ) = row.map_err(internal)?;
        Ok(SessionEvent {
            global_position: GlobalPosition(position),
            id: Uuid::parse_str(&id).map_err(internal)?,
            tenant_id,
            session_id,
            seq,
            span: serde_json::from_str(&span).map_err(internal)?,
            occurred_at: parse_dt(&occurred_at)
                .ok_or_else(|| StoreError::Internal("bad occurred_at".into()))?,
            payload: serde_json::from_str(&payload).map_err(internal)?,
            meta: serde_json::from_str(&meta).map_err(internal)?,
            start_time: parse_dt(&start_time)
                .ok_or_else(|| StoreError::Internal("bad start_time".into()))?,
            end_time: parse_dt(&end_time)
                .ok_or_else(|| StoreError::Internal("bad end_time".into()))?,
        })
    })
    .collect()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use chrono::Utc;
    use serde_json::json;
    use uuid::Uuid;

    use super::*;
    use crate::protocol::{
        Content, DraftMessage, LlmRequest, Message, RetryPolicy, Role, WorkerState,
    };
    use crate::runtime::session::decision::LlmHandler;
    use crate::runtime::session::events::{
        AgentConfigUpdated, LlmCallRequested, WorkerStateUpdated,
    };
    use crate::runtime::session::{CommitContext, SessionAggregate};
    use crate::runtime::span::SpanContext;

    fn temp_store() -> (SqliteEventStore, std::path::PathBuf) {
        let path = std::env::temp_dir().join(format!("core-event-store-{}.db", Uuid::now_v7()));
        let db = SqliteDb::open(path.to_str().unwrap(), Duration::from_secs(5)).unwrap();
        (SqliteEventStore::new(db).unwrap(), path)
    }

    fn cleanup(path: &std::path::Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(path.with_extension("db-wal"));
        let _ = std::fs::remove_file(path.with_extension("db-shm"));
    }

    fn message(id: &str, text: &str) -> Message {
        Message {
            id: id.to_string(),
            role: Role::User,
            content: Some(Content::Text(text.to_string())),
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
        }
    }

    fn agent_config(model: &str) -> crate::protocol::AgentConfig {
        crate::protocol::AgentConfig {
            format: None,
            model: model.to_string(),
            system: None,
            stream: false,
            handler: None,
            retry: None,
            tools: Vec::new(),
            sub_agents: Vec::new(),
        }
    }

    fn new_session(id: &str) -> SessionAggregate {
        SessionAggregate::new(
            id.to_string(),
            "t1".to_string(),
            SessionState::new(id.to_string()),
        )
    }

    fn commit(agg: &mut SessionAggregate, payloads: Vec<EventPayload>) -> AppendInput {
        let expected_version = agg.seq;
        let events = agg.commit(
            payloads,
            &CommitContext {
                span: SpanContext::root(),
                occurred_at: Utc::now(),
            },
        );
        AppendInput {
            events,
            snapshot: agg.clone(),
            expected_version: Seq(expected_version),
        }
    }

    fn payloads(agg_head: Option<&str>) -> Vec<EventPayload> {
        vec![
            EventPayload::NewMessage(NewMessage {
                message: message("m1", "hi"),
                parent_id: agg_head.map(str::to_string),
            }),
            EventPayload::WorkerStateUpdated(WorkerStateUpdated {
                state: WorkerState(json!({"v": 1})),
                anchor: Some("m1".to_string()),
            }),
            EventPayload::AgentConfigUpdated(AgentConfigUpdated {
                config: agent_config("model-a"),
                anchor: Some("m1".to_string()),
            }),
        ]
    }

    /// The snapshot strips tree/versions into their own tables; load must
    /// reassemble a state identical to the in-memory one.
    #[tokio::test]
    async fn append_then_load_round_trips_the_session() {
        let (store, path) = temp_store();
        let mut agg = new_session("s1");
        let input = commit(&mut agg, payloads(None));
        store.append(input).await.unwrap();

        let loaded = store.load("t1", "s1").await.unwrap();
        assert_eq!(loaded.seq, agg.seq);
        assert_eq!(
            serde_json::to_value(&loaded.state).unwrap(),
            serde_json::to_value(&agg.state).unwrap(),
        );

        // A second cycle: load → commit → append → load. The prompt diverges
        // from the tree (worker-constructed system message).
        let mut reloaded = loaded;
        let input = commit(
            &mut reloaded,
            vec![
                EventPayload::NewMessage(NewMessage {
                    message: message("m2", "again"),
                    parent_id: Some("m1".to_string()),
                }),
                // Same anchor: appended; newest wins resolution.
                EventPayload::WorkerStateUpdated(WorkerStateUpdated {
                    state: WorkerState(json!({"v": 2})),
                    anchor: Some("m1".to_string()),
                }),
                EventPayload::LlmCallRequested(LlmCallRequested {
                    call_id: "call-1".to_string(),
                    attempt: 0,
                    request: LlmRequest {
                        model: "m".to_string(),
                        messages: vec![
                            DraftMessage {
                                id: None,
                                role: Role::System,
                                content: Some(Content::Text("not a tree node".to_string())),
                                tool_calls: None,
                                tool_call_id: None,
                                name: None,
                            },
                            DraftMessage {
                                id: Some("m1".to_string()),
                                role: Role::User,
                                content: Some(Content::Text("hi".to_string())),
                                tool_calls: None,
                                tool_call_id: None,
                                name: None,
                            },
                        ],
                        tools: None,
                        temperature: None,
                        max_completion_tokens: None,
                        reasoning: None,
                    },
                    stream: false,
                    retry: RetryPolicy::no_retry(),
                    handler: LlmHandler::Worker,
                    format: None,
                }),
            ],
        );
        store.append(input).await.unwrap();

        let loaded = store.load("t1", "s1").await.unwrap();
        assert_eq!(loaded.state.nodes.len(), 2);
        assert_eq!(loaded.state.nodes[1].message.id, "m2");
        assert_eq!(loaded.state.state_versions.len(), 2);
        assert_eq!(loaded.state.state_versions[1].state.0, json!({"v": 2}));
        assert_eq!(
            loaded.state.llm_calls["call-1"].prompt.len(),
            2,
            "the verbatim prompt rejoins from its table"
        );
        assert_eq!(
            serde_json::to_value(&loaded.state).unwrap(),
            serde_json::to_value(&reloaded.state).unwrap(),
        );
        cleanup(&path);
    }

    #[tokio::test]
    async fn query_events_round_trips_payload_and_meta() {
        let (store, path) = temp_store();
        let mut agg = new_session("s1");
        let input = commit(&mut agg, payloads(None));
        let written = input.events.clone();
        store.append(input).await.unwrap();

        let events = store.query_events(&EventFilter::default()).await.unwrap();
        assert_eq!(events.len(), written.len());
        for (read, wrote) in events.iter().zip(&written) {
            assert_eq!(read.id, wrote.id);
            assert_eq!(read.seq, wrote.seq);
            assert_eq!(read.session_id, "s1");
            assert_eq!(
                serde_json::to_value(&read.payload).unwrap(),
                serde_json::to_value(&wrote.payload).unwrap(),
            );
            assert_eq!(
                serde_json::to_value(&read.meta).unwrap(),
                serde_json::to_value(&wrote.meta).unwrap(),
            );
        }
        assert!(events[0].global_position.0 < events[1].global_position.0);
        cleanup(&path);
    }

    /// A conflicting append must leave no rows in any table.
    #[tokio::test]
    async fn version_conflict_rolls_back_every_table() {
        let (store, path) = temp_store();
        let mut agg = new_session("s1");
        store
            .append(commit(&mut agg, payloads(None)))
            .await
            .unwrap();

        let mut stale = new_session("s1");
        let mut input = commit(&mut stale, payloads(None));
        input.expected_version = Seq(7); // wrong on purpose
        let err = store.append(input).await.unwrap_err();
        assert!(matches!(err, StoreError::VersionConflict { .. }));

        let loaded = store.load("t1", "s1").await.unwrap();
        assert_eq!(loaded.seq, agg.seq);
        assert_eq!(loaded.state.nodes.len(), 1);
        let events = store.query_events(&EventFilter::default()).await.unwrap();
        assert_eq!(events.len(), 3);
        cleanup(&path);
    }
}
