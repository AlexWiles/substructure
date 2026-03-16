use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rusqlite::{Connection, OpenFlags, OptionalExtension};
use sea_query::{Expr, ExprTrait, Func, Iden, Order, Query, SqliteQueryBuilder};
use uuid::Uuid;

use std::sync::Arc as StdArc;

use tokio::sync::broadcast;

use crate::runtime::event_store::{
    AggregateFilter, AggregateSort, AggregateSummary, AppendInput, Event, EventFilter, EventStore,
    Snapshot, StoreError, Version,
};

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
CREATE UNIQUE INDEX IF NOT EXISTS idx_events_aggregate_seq ON events (aggregate_id, sequence);
CREATE INDEX IF NOT EXISTS idx_events_aggregate ON events (aggregate_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events (aggregate_type);
CREATE INDEX IF NOT EXISTS idx_events_trace_id ON events (trace_id);

CREATE TABLE IF NOT EXISTS snapshots (
    aggregate_id     TEXT PRIMARY KEY,
    aggregate_type   TEXT NOT NULL,
    tenant_id        TEXT NOT NULL,
    stream_version   INTEGER NOT NULL,
    data             TEXT NOT NULL,
    wake_at          TEXT,
    first_event_at   TEXT,
    last_event_at    TEXT
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

pub struct SqliteStore {
    writer: Arc<Mutex<Connection>>,
    path: String,
    tx: broadcast::Sender<StdArc<Vec<Event>>>,
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

fn do_append(conn: &mut Connection, input: AppendInput) -> Result<(), StoreError> {
    let snap = &input.snapshot;
    let aid = snap.aggregate_id.to_string();
    let tx = conn
        .transaction()
        .map_err(|e| StoreError::Internal(e.to_string()))?;

    let expected_i64 = i64::try_from(input.expected_version)
        .map_err(|_| StoreError::Internal("expected_version exceeds i64".into()))?;

    for event in &input.events {
        let data = serde_json::to_string(event).map_err(|e| StoreError::Internal(e.to_string()))?;
        let trace_id = event.span.trace_id.to_string();

        let occurred_at = event.occurred_at.to_rfc3339();

        let rows = tx
            .execute(
                "INSERT INTO events (aggregate_type, aggregate_id, tenant_id, sequence, occurred_at, data, trace_id)
                 SELECT ?1, ?2, ?3, ?4, ?5, ?6, ?7
                 WHERE (
                     (?8 = 0 AND NOT EXISTS (SELECT 1 FROM snapshots WHERE aggregate_id = ?2))
                     OR EXISTS (
                         SELECT 1 FROM snapshots
                         WHERE aggregate_id = ?2 AND stream_version = ?8
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
                    "SELECT stream_version FROM snapshots WHERE aggregate_id = ?1",
                    [&aid],
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
    }

    let snapshot_data = serde_json::to_string(&snap.data)
        .map_err(|e| StoreError::Internal(e.to_string()))?;

    tx.execute(
        "INSERT INTO snapshots (aggregate_id, aggregate_type, tenant_id, stream_version, data, wake_at, first_event_at, last_event_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
         ON CONFLICT(aggregate_id) DO UPDATE SET
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

    Ok(())
}

fn do_load(conn: &Connection, aggregate_id: Uuid) -> Result<Snapshot, StoreError> {
    let aid = aggregate_id.to_string();

    let row = conn
        .query_row(
            "SELECT aggregate_type, tenant_id, data, stream_version, wake_at, first_event_at, last_event_at FROM snapshots WHERE aggregate_id = ?1",
            [&aid],
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
        aggregate_id,
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
            let id_strs: Vec<String> = ids.iter().map(|id| id.to_string()).collect();
            q.and_where(Expr::col(Snapshots::AggregateId).is_in(id_strs));
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
        let aggregate_id = aid
            .parse()
            .map_err(|e: uuid::Error| StoreError::Internal(format!("bad aggregate_id: {e}")))?;
        results.push(AggregateSummary {
            aggregate_id,
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
        .column(Events::Data)
        .from(Events::Table)
        .apply_if(filter.aggregate_id, |q, id| {
            q.and_where(Expr::col(Events::AggregateId).eq(id.to_string()));
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
        .query_map(param_refs.as_slice(), |row| row.get::<_, String>(0))
        .map_err(|e| StoreError::Internal(e.to_string()))?;

    let mut events = Vec::new();
    for row in rows {
        let data = row.map_err(|e| StoreError::Internal(e.to_string()))?;
        let event: Event =
            serde_json::from_str(&data).map_err(|e| StoreError::Internal(e.to_string()))?;
        events.push(event);
    }

    Ok(events)
}

fn spawn_err(e: tokio::task::JoinError) -> StoreError {
    StoreError::Internal(format!("spawn_blocking: {e}"))
}

#[async_trait]
impl EventStore for SqliteStore {
    async fn append(&self, input: AppendInput) -> Result<(), StoreError> {
        let events = input.events.clone();
        let writer = self.writer.clone();
        tokio::task::spawn_blocking(move || {
            let mut conn = writer
                .lock()
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            do_append(&mut conn, input)
        })
        .await
        .map_err(spawn_err)??;

        let _ = self.tx.send(StdArc::new(events));
        Ok(())
    }

    async fn load(&self, aggregate_id: Uuid) -> Result<Snapshot, StoreError> {
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || {
            let conn = open_conn_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
            do_load(&conn, aggregate_id)
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

// ---------------------------------------------------------------------------
// SessionIndex
// ---------------------------------------------------------------------------

use crate::runtime::aggregate::Aggregate;
use crate::runtime::session::state::SessionState;
use crate::runtime::session_index::{
    SessionCursor, SessionFilter, SessionIndex, SessionItem, SessionPage,
};

#[derive(Iden)]
struct JsonArrayLength;

#[derive(Iden)]
enum SnapshotsData {
    #[iden = "data"]
    Data,
}

fn do_list_sessions(
    conn: &Connection,
    filter: &SessionFilter,
) -> Result<SessionPage, StoreError> {
    let fetch_limit = filter.limit.unwrap_or(50);

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
        .column(SnapshotsData::Data)
        .from(Snapshots::Table)
        .and_where(Expr::col(Snapshots::AggregateType).eq("session"))
        .apply_if(filter.tenant_id.as_ref(), |q, v| {
            q.and_where(Expr::col(Snapshots::TenantId).eq(v));
        })
        .take();

    if filter.top_level {
        let len_expr = Expr::expr(
            Func::cust(JsonArrayLength)
                .arg(Expr::col(SnapshotsData::Data))
                .arg(Expr::val("$.state.ancestry")),
        );
        q.and_where(len_expr.eq(0).or(Expr::expr(
            Func::cust(JsonArrayLength)
                .arg(Expr::col(SnapshotsData::Data))
                .arg(Expr::val("$.state.ancestry")),
        ).is_null()));
    }

    if let Some(ref cursor) = filter.cursor {
        let cursor_dt = cursor.last_event_at.to_rfc3339();
        let cursor_id = cursor.aggregate_id.to_string();
        q.and_where(Expr::cust_with_values(
            "(last_event_at < ? OR (last_event_at = ? AND aggregate_id < ?))",
            [cursor_dt.clone(), cursor_dt, cursor_id],
        ));
    }

    let (order_col, order_dir) = match filter.sort {
        AggregateSort::LastEventDesc => (Snapshots::LastEventAt, Order::Desc),
        AggregateSort::FirstEventAsc => (Snapshots::FirstEventAt, Order::Asc),
        AggregateSort::FirstEventDesc => (Snapshots::FirstEventAt, Order::Desc),
        AggregateSort::WakeAtAsc => (Snapshots::WakeAt, Order::Asc),
    };
    q.order_by(order_col, order_dir);
    q.order_by(Snapshots::AggregateId, Order::Desc);
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
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, u64>(4)?,
                row.get::<_, Option<String>>(5)?,
                row.get::<_, Option<String>>(6)?,
                row.get::<_, String>(7)?,
            ))
        })
        .map_err(|e| StoreError::Internal(e.to_string()))?;

    let mut items = Vec::new();
    for row in rows {
        let (aid, agg_type, tenant, wake_at_str, version, first, last, data_str) =
            row.map_err(|e| StoreError::Internal(e.to_string()))?;
        let aggregate_id: Uuid = aid
            .parse()
            .map_err(|e: uuid::Error| StoreError::Internal(format!("bad aggregate_id: {e}")))?;
        let agg: Aggregate<SessionState> = serde_json::from_str(&data_str)
            .map_err(|e| StoreError::Internal(e.to_string()))?;

        items.push(SessionItem {
            summary: AggregateSummary {
                aggregate_id,
                aggregate_type: agg_type,
                tenant_id: tenant,
                wake_at: wake_at_str.as_deref().and_then(parse_dt),
                stream_version: version,
                first_event_at: first.as_deref().and_then(parse_dt),
                last_event_at: last.as_deref().and_then(parse_dt),
            },
            state: agg.state,
        });
    }

    let next_cursor = if items.len() > fetch_limit {
        items.pop();
        let last = &items[items.len() - 1];
        last.summary.last_event_at.map(|dt| SessionCursor {
            last_event_at: dt,
            aggregate_id: last.summary.aggregate_id,
        })
    } else {
        None
    };

    Ok(SessionPage { items, next_cursor })
}

#[async_trait]
impl SessionIndex for SqliteStore {
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
}

// ---------------------------------------------------------------------------
// PushRegistrationStore
// ---------------------------------------------------------------------------

use crate::runtime::worker::push::{PushRegistrationRecord, PushRegistrationStore};
use std::collections::HashMap;

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
