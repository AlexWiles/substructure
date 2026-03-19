use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rusqlite::{Connection, OpenFlags, OptionalExtension};
use sea_query::{Expr, ExprTrait, Iden, Order, Query, SqliteQueryBuilder};

use std::sync::Arc as StdArc;

use tokio::sync::broadcast;

use substructure_core::event_store::{
    AggregateFilter, AggregateSort, AggregateSummary, AppendInput, Event, EventFilter, EventStore,
    Snapshot, StoreError, Version,
};
use substructure_core::projection::{
    CheckpointError, ProjectionCheckpoint, ProjectionCheckpointStore,
};
use substructure_core::span::SpanContext;
use substructure_core::wake::{WakeScheduleItem, WakeScheduleStore};
use std::collections::HashMap;
use uuid::Uuid;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS events (
    global_sequence  INTEGER PRIMARY KEY AUTOINCREMENT,
    aggregate_type   TEXT NOT NULL,
    aggregate_id     TEXT NOT NULL,
    tenant_id        TEXT NOT NULL,
    sequence         INTEGER NOT NULL,
    occurred_at      TEXT NOT NULL,
    data             TEXT NOT NULL,
    trace_id         TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_occurred_at ON events (occurred_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_events_aggregate_seq ON events (tenant_id, aggregate_id, sequence);
CREATE INDEX IF NOT EXISTS idx_events_aggregate ON events (tenant_id, aggregate_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events (aggregate_type);
CREATE INDEX IF NOT EXISTS idx_events_trace_id ON events (trace_id);

CREATE TABLE IF NOT EXISTS snapshots (
    aggregate_id     TEXT NOT NULL,
    aggregate_type   TEXT NOT NULL,
    tenant_id        TEXT NOT NULL,
    stream_version   INTEGER NOT NULL,
    data             TEXT NOT NULL,
    wake_at          TEXT,
    first_event_at   TEXT,
    last_event_at    TEXT,
    PRIMARY KEY (tenant_id, aggregate_id)
);
CREATE INDEX IF NOT EXISTS idx_snapshots_tenant ON snapshots (tenant_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_type ON snapshots (aggregate_type);
CREATE INDEX IF NOT EXISTS idx_snapshots_wake_at ON snapshots (wake_at);
CREATE INDEX IF NOT EXISTS idx_snapshots_last_event ON snapshots (last_event_at);

CREATE TABLE IF NOT EXISTS push_registrations (
    tenant_id       TEXT NOT NULL,
    agent_id        TEXT NOT NULL,
    transport_type  TEXT NOT NULL,
    config          TEXT NOT NULL,
    PRIMARY KEY (tenant_id, agent_id)
);

CREATE TABLE IF NOT EXISTS projection_checkpoints (
    projection_name TEXT NOT NULL,
    shard_id        INTEGER NOT NULL,
    position        INTEGER NOT NULL,
    version         INTEGER NOT NULL,
    owner_id        TEXT,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (projection_name, shard_id)
);

CREATE TABLE IF NOT EXISTS wake_schedule (
    tenant_id       TEXT NOT NULL,
    aggregate_id    TEXT NOT NULL,
    wake_at         TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (tenant_id, aggregate_id)
);
CREATE INDEX IF NOT EXISTS idx_wake_schedule_wake_at ON wake_schedule (wake_at);

CREATE TABLE IF NOT EXISTS session_index (
    tenant_id       TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    stream_version  INTEGER NOT NULL,
    first_event_at  TEXT,
    last_event_at   TEXT,
    wake_at         TEXT,
    top_level       INTEGER NOT NULL,
    agent_id        TEXT,
    cost            TEXT NOT NULL,
    sub_agent_cost  TEXT NOT NULL,
    status_json     TEXT NOT NULL,
    turn_id         TEXT,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (tenant_id, session_id)
);
CREATE INDEX IF NOT EXISTS idx_session_index_last_event
  ON session_index (tenant_id, last_event_at DESC, session_id DESC);
CREATE INDEX IF NOT EXISTS idx_session_index_top_level_last_event
  ON session_index (tenant_id, top_level, last_event_at DESC, session_id DESC);
CREATE INDEX IF NOT EXISTS idx_session_index_wake_at
  ON session_index (tenant_id, wake_at);
";

#[derive(Iden)]
enum Snapshots {
    Table,
    AggregateId,
    AggregateType,
    TenantId,
    StreamVersion,
    WakeAt,
    FirstEventAt,
    LastEventAt,
}

#[derive(Iden)]
enum Events {
    Table,
    GlobalSequence,
    AggregateId,
    AggregateType,
    TenantId,
    OccurredAt,
    TraceId,
    Data,
}

#[derive(Iden)]
enum SessionIndexRows {
    #[iden = "session_index"]
    Table,
    TenantId,
    SessionId,
    StreamVersion,
    FirstEventAt,
    LastEventAt,
    WakeAt,
    TopLevel,
    AgentId,
    Cost,
    SubAgentCost,
    StatusJson,
    TurnId,
}

pub struct SqliteStore {
    writer: Arc<Mutex<Connection>>,
    path: String,
    tx: broadcast::Sender<StdArc<Vec<Event>>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct StoredEvent {
    id: Uuid,
    tenant_id: String,
    aggregate_type: String,
    aggregate_id: String,
    sequence: u64,
    span: SpanContext,
    occurred_at: DateTime<Utc>,
    payload: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    derived: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    metadata: HashMap<String, String>,
    start_time: DateTime<Utc>,
    end_time: DateTime<Utc>,
}

impl StoredEvent {
    fn from_event(event: &Event) -> Self {
        Self {
            id: event.id,
            tenant_id: event.tenant_id.clone(),
            aggregate_type: event.aggregate_type.clone(),
            aggregate_id: event.aggregate_id.clone(),
            sequence: event.sequence,
            span: event.span.clone(),
            occurred_at: event.occurred_at,
            payload: event.payload.clone(),
            derived: event.derived.clone(),
            metadata: event.metadata.clone(),
            start_time: event.start_time,
            end_time: event.end_time,
        }
    }

    fn into_event(self, position: u64) -> Event {
        Event {
            position,
            id: self.id,
            tenant_id: self.tenant_id,
            aggregate_type: self.aggregate_type,
            aggregate_id: self.aggregate_id,
            sequence: self.sequence,
            span: self.span,
            occurred_at: self.occurred_at,
            payload: self.payload,
            derived: self.derived,
            metadata: self.metadata,
            start_time: self.start_time,
            end_time: self.end_time,
        }
    }
}

impl SqliteStore {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        let writer =
            Connection::open(&path_str).map_err(|e| StoreError::Internal(e.to_string()))?;
        writer
            .execute_batch(SCHEMA)
            .map_err(|e| StoreError::Internal(e.to_string()))?;
        writer
            .pragma_update(None, "journal_mode", "WAL")
            .map_err(|e| StoreError::Internal(e.to_string()))?;

        let (tx, _) = broadcast::channel(1024);

        Ok(Self {
            writer: Arc::new(Mutex::new(writer)),
            path: path_str,
            tx,
        })
    }
}

fn open_conn_flags(path: &str, flags: OpenFlags) -> Result<Connection, StoreError> {
    Connection::open_with_flags(path, flags).map_err(|e| StoreError::Internal(e.to_string()))
}

fn parse_dt(s: &str) -> Option<DateTime<Utc>> {
    s.parse().ok()
}

/// Convert sea-query Values to rusqlite params.
fn sea_params(values: sea_query::Values) -> Vec<Box<dyn rusqlite::types::ToSql>> {
    values
        .0
        .into_iter()
        .map(|v| -> Box<dyn rusqlite::types::ToSql> {
            match v {
                sea_query::Value::String(s) => Box::new(s),
                sea_query::Value::Int(i) => Box::new(i),
                sea_query::Value::BigInt(i) => Box::new(i),
                sea_query::Value::BigUnsigned(u) => Box::new(u.map(|n| n as i64)),
                sea_query::Value::Bool(b) => Box::new(b),
                _ => Box::new(Option::<String>::None),
            }
        })
        .collect()
}

fn do_append(conn: &mut Connection, input: AppendInput) -> Result<Vec<u64>, StoreError> {
    let snap = &input.snapshot;
    let aid = &snap.aggregate_id;
    let mut positions = Vec::with_capacity(input.events.len());
    let tx = conn
        .transaction()
        .map_err(|e| StoreError::Internal(e.to_string()))?;

    let expected_i64 = i64::try_from(input.expected_version)
        .map_err(|_| StoreError::Internal("expected_version exceeds i64".into()))?;

    for event in &input.events {
        let stored = StoredEvent::from_event(event);
        let data =
            serde_json::to_string(&stored).map_err(|e| StoreError::Internal(e.to_string()))?;
        let trace_id = event.span.trace_id.to_string();

        let occurred_at = event.occurred_at.to_rfc3339();

        let rows = tx
            .execute(
                "INSERT INTO events (aggregate_type, aggregate_id, tenant_id, sequence, occurred_at, data, trace_id)
                 SELECT ?1, ?2, ?3, ?4, ?5, ?6, ?7
                 WHERE (
                     (?8 = 0 AND NOT EXISTS (SELECT 1 FROM snapshots WHERE tenant_id = ?3 AND aggregate_id = ?2))
                     OR EXISTS (
                         SELECT 1 FROM snapshots
                         WHERE tenant_id = ?3 AND aggregate_id = ?2 AND stream_version = ?8
                     )
                 )",
                rusqlite::params![
                    snap.aggregate_type,
                    aid,
                    snap.tenant_id,
                    event.sequence,
                    occurred_at,
                    data,
                    trace_id,
                    expected_i64,
                ],
            )
            .map_err(|e| StoreError::Internal(e.to_string()))?;

        if rows == 0 {
            let actual_version: u64 = tx
                .query_row(
                    "SELECT stream_version FROM snapshots WHERE tenant_id = ?1 AND aggregate_id = ?2",
                    rusqlite::params![&snap.tenant_id, &aid],
                    |row| row.get(0),
                )
                .optional()
                .map_err(|e| StoreError::Internal(e.to_string()))?
                .unwrap_or(0);
            return Err(StoreError::VersionConflict {
                expected: Version(input.expected_version),
                actual: Version(actual_version),
            });
        }

        let row_id = tx.last_insert_rowid();
        let position = u64::try_from(row_id)
            .map_err(|_| StoreError::Internal("event position exceeds u64".into()))?;
        positions.push(position);
    }

    let snapshot_data = serde_json::to_string(&snap.data)
        .map_err(|e| StoreError::Internal(e.to_string()))?;

    tx.execute(
        "INSERT INTO snapshots (aggregate_id, aggregate_type, tenant_id, stream_version, data, wake_at, first_event_at, last_event_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
         ON CONFLICT(tenant_id, aggregate_id) DO UPDATE SET
             stream_version = excluded.stream_version,
             data = excluded.data,
             wake_at = excluded.wake_at,
             first_event_at = COALESCE(snapshots.first_event_at, excluded.first_event_at),
             last_event_at = excluded.last_event_at",
        rusqlite::params![
            aid,
            snap.aggregate_type,
            snap.tenant_id,
            snap.stream_version,
            snapshot_data,
            snap.wake_at.map(|t| t.to_rfc3339()),
            snap.first_event_at.map(|t| t.to_rfc3339()),
            snap.last_event_at.map(|t| t.to_rfc3339()),
        ],
    )
    .map_err(|e| StoreError::Internal(e.to_string()))?;

    tx.commit()
        .map_err(|e| StoreError::Internal(e.to_string()))?;

    Ok(positions)
}

fn do_load(conn: &Connection, tenant_id: &str, aggregate_id: &str) -> Result<Snapshot, StoreError> {
    let row = conn
        .query_row(
            "SELECT aggregate_type, tenant_id, data, stream_version, wake_at, first_event_at, last_event_at FROM snapshots WHERE tenant_id = ?1 AND aggregate_id = ?2",
            rusqlite::params![tenant_id, aggregate_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, u64>(3)?,
                    row.get::<_, Option<String>>(4)?,
                    row.get::<_, Option<String>>(5)?,
                    row.get::<_, Option<String>>(6)?,
                ))
            },
        )
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => StoreError::StreamNotFound,
            other => StoreError::Internal(other.to_string()),
        })?;

    let (agg_type, tenant_id, snapshot_data, stream_version, wake_at_str, first_str, last_str) =
        row;

    let data: serde_json::Value =
        serde_json::from_str(&snapshot_data).map_err(|e| StoreError::Internal(e.to_string()))?;

    Ok(Snapshot {
        aggregate_id: aggregate_id.to_string(),
        tenant_id,
        aggregate_type: agg_type,
        data,
        stream_version,
        wake_at: wake_at_str.as_deref().and_then(parse_dt),
        first_event_at: first_str.as_deref().and_then(parse_dt),
        last_event_at: last_str.as_deref().and_then(parse_dt),
    })
}

fn do_list_aggregates(
    conn: &Connection,
    filter: &AggregateFilter,
) -> Result<Vec<AggregateSummary>, StoreError> {
    let mut q = Query::select()
        .columns([
            Snapshots::AggregateId,
            Snapshots::AggregateType,
            Snapshots::TenantId,
            Snapshots::WakeAt,
            Snapshots::StreamVersion,
            Snapshots::FirstEventAt,
            Snapshots::LastEventAt,
        ])
        .from(Snapshots::Table)
        .apply_if(filter.aggregate_type.as_ref(), |q, v| {
            q.and_where(Expr::col(Snapshots::AggregateType).eq(v));
        })
        .apply_if(filter.aggregate_ids.as_ref(), |q, ids| {
            q.and_where(Expr::col(Snapshots::AggregateId).is_in(ids.clone()));
        })
        .apply_if(filter.tenant_id.as_ref(), |q, v| {
            q.and_where(Expr::col(Snapshots::TenantId).eq(v));
        })
        .apply_if(filter.wake_at_before.as_ref(), |q, before| {
            q.and_where(Expr::col(Snapshots::WakeAt).is_not_null());
            q.and_where(Expr::col(Snapshots::WakeAt).lte(before.to_rfc3339()));
        })
        .apply_if(filter.limit, |q, n| {
            q.limit(n as u64);
        })
        .take();

    if filter.wake_at_not_null && filter.wake_at_before.is_none() {
        q.and_where(Expr::col(Snapshots::WakeAt).is_not_null());
    }

    let (order_col, order_dir) = match filter.sort {
        AggregateSort::LastEventDesc => (Snapshots::LastEventAt, Order::Desc),
        AggregateSort::FirstEventAsc => (Snapshots::FirstEventAt, Order::Asc),
        AggregateSort::FirstEventDesc => (Snapshots::FirstEventAt, Order::Desc),
        AggregateSort::WakeAtAsc => (Snapshots::WakeAt, Order::Asc),
    };
    q.order_by(order_col, order_dir);

    let (sql, values) = q.build(SqliteQueryBuilder);
    let params = sea_params(values);
    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();

    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| StoreError::Internal(e.to_string()))?;

    let rows = stmt
        .query_map(param_refs.as_slice(), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, u64>(4)?,
                row.get::<_, Option<String>>(5)?,
                row.get::<_, Option<String>>(6)?,
            ))
        })
        .map_err(|e| StoreError::Internal(e.to_string()))?;

    let mut results = Vec::new();
    for row in rows {
        let (aid, agg_type, tenant, wake_at_str, version, first, last) =
            row.map_err(|e| StoreError::Internal(e.to_string()))?;
        results.push(AggregateSummary {
            aggregate_id: aid,
            aggregate_type: agg_type,
            tenant_id: tenant,
            wake_at: wake_at_str.as_deref().and_then(parse_dt),
            stream_version: version,
            first_event_at: first.as_deref().and_then(parse_dt),
            last_event_at: last.as_deref().and_then(parse_dt),
        });
    }

    Ok(results)
}

fn do_query_events(conn: &Connection, filter: EventFilter) -> Result<Vec<Event>, StoreError> {
    let (sql, values) = Query::select()
        .column(Events::GlobalSequence)
        .column(Events::Data)
        .from(Events::Table)
        .apply_if(filter.after_position, |q, pos| {
            q.and_where(Expr::col(Events::GlobalSequence).gt(pos as i64));
        })
        .apply_if(filter.aggregate_id.as_ref(), |q, id| {
            q.and_where(Expr::col(Events::AggregateId).eq(id));
        })
        .apply_if(filter.aggregate_type.as_ref(), |q, v| {
            q.and_where(Expr::col(Events::AggregateType).eq(v));
        })
        .apply_if(filter.tenant_id.as_ref(), |q, v| {
            q.and_where(Expr::col(Events::TenantId).eq(v));
        })
        .apply_if(filter.trace_id.as_ref(), |q, v| {
            q.and_where(Expr::col(Events::TraceId).eq(v));
        })
        .apply_if(filter.sequence_after, |q, seq| {
            q.and_where(Expr::col(Events::GlobalSequence).gt(seq as i64));
        })
        .apply_if(filter.occurred_after, |q, after| {
            q.and_where(Expr::col(Events::OccurredAt).gt(after.to_rfc3339()));
        })
        .apply_if(filter.occurred_before, |q, before| {
            q.and_where(Expr::col(Events::OccurredAt).lt(before.to_rfc3339()));
        })
        .apply_if(filter.limit, |q, n| {
            q.limit(n as u64);
        })
        .order_by(Events::GlobalSequence, Order::Asc)
        .build(SqliteQueryBuilder);

    let params = sea_params(values);
    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();

    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| StoreError::Internal(e.to_string()))?;

    let rows = stmt
        .query_map(param_refs.as_slice(), |row| {
            Ok((
                row.get::<_, u64>(0)?,
                row.get::<_, String>(1)?,
            ))
        })
        .map_err(|e| StoreError::Internal(e.to_string()))?;

    let mut events = Vec::new();
    for row in rows {
        let (position, data) = row.map_err(|e| StoreError::Internal(e.to_string()))?;
        let stored: StoredEvent =
            serde_json::from_str(&data).map_err(|e| StoreError::Internal(e.to_string()))?;
        let event = stored.into_event(position);
        events.push(event);
    }

    Ok(events)
}

fn do_load_projection_checkpoint(
    conn: &Connection,
    projection: &str,
    shard_id: u32,
) -> Result<ProjectionCheckpoint, CheckpointError> {
    let row = conn
        .query_row(
            "SELECT position, version, updated_at FROM projection_checkpoints WHERE projection_name = ?1 AND shard_id = ?2",
            rusqlite::params![projection, shard_id],
            |row| {
                Ok((
                    row.get::<_, u64>(0)?,
                    row.get::<_, u64>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        )
        .optional()
        .map_err(|e| CheckpointError::Message(e.to_string()))?;

    match row {
        Some((position, version, updated_at)) => Ok(ProjectionCheckpoint {
            position,
            version,
            updated_at: parse_dt(&updated_at).unwrap_or_else(Utc::now),
        }),
        None => Ok(ProjectionCheckpoint {
            position: 0,
            version: 0,
            updated_at: Utc::now(),
        }),
    }
}

fn do_compare_and_set_projection_checkpoint(
    conn: &mut Connection,
    projection: &str,
    shard_id: u32,
    expected_version: u64,
    new_position: u64,
    owner_id: Option<&str>,
) -> Result<bool, CheckpointError> {
    let tx = conn
        .transaction()
        .map_err(|e| CheckpointError::Message(e.to_string()))?;

    if expected_version == 0 {
        tx.execute(
            "INSERT OR IGNORE INTO projection_checkpoints (projection_name, shard_id, position, version, owner_id, updated_at)
             VALUES (?1, ?2, 0, 0, NULL, ?3)",
            rusqlite::params![projection, shard_id, Utc::now().to_rfc3339()],
        )
        .map_err(|e| CheckpointError::Message(e.to_string()))?;
    }

    let updated = tx
        .execute(
            "UPDATE projection_checkpoints
             SET position = ?1,
                 version = version + 1,
                 owner_id = ?2,
                 updated_at = ?3
             WHERE projection_name = ?4
               AND shard_id = ?5
               AND version = ?6",
            rusqlite::params![
                new_position,
                owner_id,
                Utc::now().to_rfc3339(),
                projection,
                shard_id,
                expected_version,
            ],
        )
        .map_err(|e| CheckpointError::Message(e.to_string()))?;

    tx.commit()
        .map_err(|e| CheckpointError::Message(e.to_string()))?;

    Ok(updated == 1)
}

fn do_upsert_wake(
    conn: &Connection,
    tenant_id: &str,
    aggregate_id: &str,
    wake_at: DateTime<Utc>,
) -> Result<(), String> {
    conn.execute(
        "INSERT INTO wake_schedule (tenant_id, aggregate_id, wake_at, updated_at)
         VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(tenant_id, aggregate_id) DO UPDATE SET
             wake_at = excluded.wake_at,
             updated_at = excluded.updated_at",
        rusqlite::params![tenant_id, aggregate_id, wake_at.to_rfc3339(), Utc::now().to_rfc3339()],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

fn do_remove_wake(conn: &Connection, tenant_id: &str, aggregate_id: &str) -> Result<(), String> {
    conn.execute(
        "DELETE FROM wake_schedule WHERE tenant_id = ?1 AND aggregate_id = ?2",
        rusqlite::params![tenant_id, aggregate_id],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

fn do_list_due_wakes(
    conn: &Connection,
    now: DateTime<Utc>,
    limit: usize,
) -> Result<Vec<WakeScheduleItem>, String> {
    let mut stmt = conn
        .prepare(
            "SELECT tenant_id, aggregate_id, wake_at
             FROM wake_schedule
             WHERE wake_at <= ?1
             ORDER BY wake_at ASC
             LIMIT ?2",
        )
        .map_err(|e| e.to_string())?;

    let rows = stmt
        .query_map(rusqlite::params![now.to_rfc3339(), limit as i64], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })
        .map_err(|e| e.to_string())?;

    let mut out = Vec::new();
    for row in rows {
        let (tenant_id, aggregate_id, wake_at_str) = row.map_err(|e| e.to_string())?;
        let Some(wake_at) = parse_dt(&wake_at_str) else {
            continue;
        };
        out.push(WakeScheduleItem {
            tenant_id,
            aggregate_id,
            wake_at,
        });
    }
    Ok(out)
}

fn do_next_wake_at(conn: &Connection) -> Result<Option<DateTime<Utc>>, String> {
    let wake_at = conn
        .query_row(
            "SELECT wake_at FROM wake_schedule ORDER BY wake_at ASC LIMIT 1",
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(|e| e.to_string())?;
    Ok(wake_at.as_deref().and_then(parse_dt))
}

fn spawn_err(e: tokio::task::JoinError) -> StoreError {
    StoreError::Internal(format!("spawn_blocking: {e}"))
}

#[async_trait]
impl EventStore for SqliteStore {
    async fn append(&self, input: AppendInput) -> Result<(), StoreError> {
        let events = input.events.clone();
        let writer = self.writer.clone();
        let positions = tokio::task::spawn_blocking(move || {
            let mut conn = writer
                .lock()
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            do_append(&mut conn, input)
        })
        .await
        .map_err(spawn_err)??;

        if events.len() != positions.len() {
            return Err(StoreError::Internal(
                "inserted event count did not match assigned positions".into(),
            ));
        }

        let mut with_positions = events;
        for (event, position) in with_positions.iter_mut().zip(positions.into_iter()) {
            event.position = position;
        }

        let _ = self.tx.send(StdArc::new(with_positions));
        Ok(())
    }

    async fn load(&self, tenant_id: &str, aggregate_id: &str) -> Result<Snapshot, StoreError> {
        let path = self.path.clone();
        let tenant_id = tenant_id.to_string();
        let aggregate_id = aggregate_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = open_conn_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
            do_load(&conn, &tenant_id, &aggregate_id)
        })
        .await
        .map_err(spawn_err)?
    }

    async fn list_aggregates(
        &self,
        filter: &AggregateFilter,
    ) -> Result<Vec<AggregateSummary>, StoreError> {
        let filter = filter.clone();
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || {
            let conn = open_conn_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
            do_list_aggregates(&conn, &filter)
        })
        .await
        .map_err(spawn_err)?
    }

    async fn query_events(&self, filter: &EventFilter) -> Result<Vec<Event>, StoreError> {
        let filter = filter.clone();
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || {
            let conn = open_conn_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
            do_query_events(&conn, filter)
        })
        .await
        .map_err(spawn_err)?
    }

    fn subscribe(&self) -> broadcast::Receiver<StdArc<Vec<Event>>> {
        self.tx.subscribe()
    }
}

#[async_trait]
impl ProjectionCheckpointStore for SqliteStore {
    async fn load_checkpoint(
        &self,
        projection: &str,
        shard_id: u32,
    ) -> Result<ProjectionCheckpoint, CheckpointError> {
        let projection = projection.to_string();
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || {
            let conn = open_conn_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)
                .map_err(|e| CheckpointError::Message(e.to_string()))?;
            do_load_projection_checkpoint(&conn, &projection, shard_id)
        })
        .await
        .map_err(|e| CheckpointError::Message(e.to_string()))?
    }

    async fn compare_and_set_checkpoint(
        &self,
        projection: &str,
        shard_id: u32,
        expected_version: u64,
        new_position: u64,
        owner_id: Option<&str>,
    ) -> Result<bool, CheckpointError> {
        let projection = projection.to_string();
        let owner_id = owner_id.map(str::to_string);
        let writer = self.writer.clone();
        tokio::task::spawn_blocking(move || {
            let mut conn = writer
                .lock()
                .map_err(|e| CheckpointError::Message(e.to_string()))?;
            do_compare_and_set_projection_checkpoint(
                &mut conn,
                &projection,
                shard_id,
                expected_version,
                new_position,
                owner_id.as_deref(),
            )
        })
        .await
        .map_err(|e| CheckpointError::Message(e.to_string()))?
    }
}

#[async_trait]
impl WakeScheduleStore for SqliteStore {
    async fn upsert_wake(
        &self,
        tenant_id: &str,
        aggregate_id: &str,
        wake_at: DateTime<Utc>,
    ) -> Result<(), String> {
        let tenant_id = tenant_id.to_string();
        let aggregate_id = aggregate_id.to_string();
        let writer = self.writer.clone();
        tokio::task::spawn_blocking(move || {
            let conn = writer.lock().map_err(|e| e.to_string())?;
            do_upsert_wake(&conn, &tenant_id, &aggregate_id, wake_at)
        })
        .await
        .map_err(|e| e.to_string())?
    }

    async fn remove_wake(&self, tenant_id: &str, aggregate_id: &str) -> Result<(), String> {
        let tenant_id = tenant_id.to_string();
        let aggregate_id = aggregate_id.to_string();
        let writer = self.writer.clone();
        tokio::task::spawn_blocking(move || {
            let conn = writer.lock().map_err(|e| e.to_string())?;
            do_remove_wake(&conn, &tenant_id, &aggregate_id)
        })
        .await
        .map_err(|e| e.to_string())?
    }

    async fn list_due_wakes(
        &self,
        now: DateTime<Utc>,
        limit: usize,
    ) -> Result<Vec<WakeScheduleItem>, String> {
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || {
            let conn = open_conn_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)
                .map_err(|e| e.to_string())?;
            do_list_due_wakes(&conn, now, limit)
        })
        .await
        .map_err(|e| e.to_string())?
    }

    async fn next_wake_at(&self) -> Result<Option<DateTime<Utc>>, String> {
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || {
            let conn = open_conn_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)
                .map_err(|e| e.to_string())?;
            do_next_wake_at(&conn)
        })
        .await
        .map_err(|e| e.to_string())?
    }
}

// ---------------------------------------------------------------------------
// SessionIndex
// ---------------------------------------------------------------------------

use substructure_core::session_index::{
    SessionCursor, SessionFilter, SessionIndexRecord, SessionIndexStore,
    SessionItem, SessionPage,
};

fn do_list_sessions(
    conn: &Connection,
    filter: &SessionFilter,
) -> Result<SessionPage, StoreError> {
    let fetch_limit = filter.limit.unwrap_or(50);

    let (order_col, order_dir, cursor_predicate, session_id_order) = match filter.sort {
        AggregateSort::LastEventDesc => (
            SessionIndexRows::LastEventAt,
            Order::Desc,
            "(last_event_at < ? OR (last_event_at = ? AND session_id < ?))",
            Order::Desc,
        ),
        AggregateSort::FirstEventAsc => (
            SessionIndexRows::FirstEventAt,
            Order::Asc,
            "(first_event_at > ? OR (first_event_at = ? AND session_id > ?))",
            Order::Asc,
        ),
        AggregateSort::FirstEventDesc => (
            SessionIndexRows::FirstEventAt,
            Order::Desc,
            "(first_event_at < ? OR (first_event_at = ? AND session_id < ?))",
            Order::Desc,
        ),
        AggregateSort::WakeAtAsc => (
            SessionIndexRows::WakeAt,
            Order::Asc,
            "(wake_at > ? OR (wake_at = ? AND session_id > ?))",
            Order::Asc,
        ),
    };

    let mut q = Query::select()
        .columns([
            SessionIndexRows::SessionId,
            SessionIndexRows::TenantId,
            SessionIndexRows::StreamVersion,
            SessionIndexRows::FirstEventAt,
            SessionIndexRows::LastEventAt,
            SessionIndexRows::WakeAt,
            SessionIndexRows::TopLevel,
            SessionIndexRows::AgentId,
            SessionIndexRows::Cost,
            SessionIndexRows::SubAgentCost,
            SessionIndexRows::StatusJson,
            SessionIndexRows::TurnId,
        ])
        .from(SessionIndexRows::Table)
        .apply_if(filter.tenant_id.as_ref(), |q, v| {
            q.and_where(Expr::col(SessionIndexRows::TenantId).eq(v));
        })
        .take();

    if filter.top_level {
        q.and_where(Expr::col(SessionIndexRows::TopLevel).eq(1));
    }

    if let Some(ref cursor) = filter.cursor {
        let cursor_dt = cursor.at.to_rfc3339();
        let cursor_id = cursor.session_id.clone();
        q.and_where(Expr::cust_with_values(
            cursor_predicate,
            [cursor_dt.clone(), cursor_dt, cursor_id],
        ));
    }

    q.order_by(order_col, order_dir);
    q.order_by(SessionIndexRows::SessionId, session_id_order);
    q.limit((fetch_limit + 1) as u64);

    let (sql, values) = q.build(SqliteQueryBuilder);
    let params = sea_params(values);
    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();

    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| StoreError::Internal(e.to_string()))?;

    let rows = stmt
        .query_map(param_refs.as_slice(), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, u64>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, Option<String>>(5)?,
                row.get::<_, i64>(6)?,
                row.get::<_, Option<String>>(7)?,
                row.get::<_, String>(8)?,
                row.get::<_, String>(9)?,
                row.get::<_, String>(10)?,
                row.get::<_, Option<String>>(11)?,
            ))
        })
        .map_err(|e| StoreError::Internal(e.to_string()))?;

    let mut items = Vec::new();
    for row in rows {
        let (
            session_id,
            tenant_id,
            stream_version,
            first_event_at,
            last_event_at,
            wake_at,
            top_level,
            agent_id,
            cost,
            sub_agent_cost,
            status_json,
            turn_id,
        ) = row.map_err(|e| StoreError::Internal(e.to_string()))?;

        let cost = cost
            .parse()
            .map_err(|e: rust_decimal::Error| StoreError::Internal(e.to_string()))?;
        let sub_agent_cost = sub_agent_cost
            .parse()
            .map_err(|e: rust_decimal::Error| StoreError::Internal(e.to_string()))?;
        let status = serde_json::from_str(&status_json)
            .map_err(|e| StoreError::Internal(e.to_string()))?;

        items.push(SessionItem {
            session_id,
            tenant_id,
            stream_version,
            first_event_at: first_event_at.as_deref().and_then(parse_dt),
            last_event_at: last_event_at.as_deref().and_then(parse_dt),
            wake_at: wake_at.as_deref().and_then(parse_dt),
            top_level: top_level != 0,
            agent_id,
            cost,
            sub_agent_cost,
            status,
            turn_id,
        });
    }

    let next_cursor = if items.len() > fetch_limit {
        items.pop();
        let last = &items[items.len() - 1];
        last.last_event_at.map(|at| SessionCursor {
            at,
            session_id: last.session_id.clone(),
        })
    } else {
        None
    };

    Ok(SessionPage { items, next_cursor })
}

fn do_upsert_session_index(conn: &Connection, record: SessionIndexRecord) -> Result<(), String> {
    let status_json = serde_json::to_string(&record.status).map_err(|e| e.to_string())?;
    conn.execute(
        "INSERT INTO session_index (
            tenant_id,
            session_id,
            stream_version,
            first_event_at,
            last_event_at,
            wake_at,
            top_level,
            agent_id,
            cost,
            sub_agent_cost,
            status_json,
            turn_id,
            updated_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
         ON CONFLICT(tenant_id, session_id) DO UPDATE SET
            stream_version = excluded.stream_version,
            first_event_at = COALESCE(session_index.first_event_at, excluded.first_event_at),
            last_event_at = excluded.last_event_at,
            wake_at = excluded.wake_at,
            top_level = excluded.top_level,
            agent_id = excluded.agent_id,
            cost = excluded.cost,
            sub_agent_cost = excluded.sub_agent_cost,
            status_json = excluded.status_json,
            turn_id = excluded.turn_id,
            updated_at = excluded.updated_at",
        rusqlite::params![
            record.tenant_id,
            record.session_id,
            record.stream_version,
            record.first_event_at.map(|t| t.to_rfc3339()),
            record.last_event_at.map(|t| t.to_rfc3339()),
            record.wake_at.map(|t| t.to_rfc3339()),
            if record.top_level { 1i64 } else { 0i64 },
            record.agent_id,
            record.cost.to_string(),
            record.sub_agent_cost.to_string(),
            status_json,
            record.turn_id,
            Utc::now().to_rfc3339(),
        ],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

#[async_trait]
impl SessionIndexStore for SqliteStore {
    async fn list_sessions(&self, filter: &SessionFilter) -> Result<SessionPage, StoreError> {
        let filter = filter.clone();
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || {
            let conn = open_conn_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
            do_list_sessions(&conn, &filter)
        })
        .await
        .map_err(spawn_err)?
    }
    async fn upsert_session_index(&self, record: SessionIndexRecord) -> Result<(), String> {
        let writer = self.writer.clone();
        tokio::task::spawn_blocking(move || {
            let conn = writer.lock().map_err(|e| e.to_string())?;
            do_upsert_session_index(&conn, record)
        })
        .await
        .map_err(|e| e.to_string())?
    }
}

// ---------------------------------------------------------------------------
// PushRegistrationStore
// ---------------------------------------------------------------------------

use substructure_core::worker::push::{PushRegistrationRecord, PushRegistrationStore};

#[async_trait]
impl PushRegistrationStore for SqliteStore {
    async fn save(&self, record: &PushRegistrationRecord) -> Result<(), String> {
        let record = record.clone();
        let writer = self.writer.clone();
        tokio::task::spawn_blocking(move || {
            let conn = writer.lock().map_err(|e| e.to_string())?;
            let config_str = serde_json::to_string(&record.config).map_err(|e| e.to_string())?;
            conn.execute(
                "INSERT INTO push_registrations (tenant_id, agent_id, transport_type, config)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(tenant_id, agent_id) DO UPDATE SET
                     transport_type = excluded.transport_type,
                     config = excluded.config",
                rusqlite::params![record.tenant_id, record.agent_id, record.transport_type, config_str],
            )
            .map_err(|e| e.to_string())?;
            Ok(())
        })
        .await
        .map_err(|e| e.to_string())?
    }

    async fn remove(&self, tenant_id: &str, agent_id: &str) -> Result<(), String> {
        let tenant_id = tenant_id.to_string();
        let agent_id = agent_id.to_string();
        let writer = self.writer.clone();
        tokio::task::spawn_blocking(move || {
            let conn = writer.lock().map_err(|e| e.to_string())?;
            conn.execute(
                "DELETE FROM push_registrations WHERE tenant_id = ?1 AND agent_id = ?2",
                rusqlite::params![tenant_id, agent_id],
            )
            .map_err(|e| e.to_string())?;
            Ok(())
        })
        .await
        .map_err(|e| e.to_string())?
    }

    async fn get(
        &self,
        tenant_id: &str,
        agent_id: &str,
    ) -> Result<Option<PushRegistrationRecord>, String> {
        let tenant_id = tenant_id.to_string();
        let agent_id = agent_id.to_string();
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || {
            let conn = open_conn_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)
                .map_err(|e| e.to_string())?;
            let result = conn
                .query_row(
                    "SELECT transport_type, config FROM push_registrations
                     WHERE tenant_id = ?1 AND agent_id = ?2",
                    rusqlite::params![tenant_id, agent_id],
                    |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                        ))
                    },
                )
                .optional()
                .map_err(|e| e.to_string())?;
            match result {
                Some((transport_type, config_str)) => {
                    let config = serde_json::from_str(&config_str).map_err(|e| e.to_string())?;
                    Ok(Some(PushRegistrationRecord {
                        tenant_id,
                        agent_id,
                        transport_type,
                        config,
                    }))
                }
                None => Ok(None),
            }
        })
        .await
        .map_err(|e| e.to_string())?
    }

    async fn list_tenants(&self) -> Result<HashMap<String, Vec<String>>, String> {
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || {
            let conn = open_conn_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)
                .map_err(|e| e.to_string())?;
            let mut stmt = conn
                .prepare("SELECT tenant_id, agent_id FROM push_registrations")
                .map_err(|e| e.to_string())?;
            let rows = stmt
                .query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                    ))
                })
                .map_err(|e| e.to_string())?;
            let mut result: HashMap<String, Vec<String>> = HashMap::new();
            for row in rows {
                let (tenant_id, agent_id) = row.map_err(|e| e.to_string())?;
                result.entry(tenant_id).or_default().push(agent_id);
            }
            Ok(result)
        })
        .await
        .map_err(|e| e.to_string())?
    }
}
