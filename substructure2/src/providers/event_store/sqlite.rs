use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rusqlite::{Connection, OpenFlags, OptionalExtension};
use sea_query::{Expr, ExprTrait, Iden, Order, Query, SqliteQueryBuilder};
use uuid::Uuid;

use crate::runtime::event_store::{
    AggregateFilter, AggregateSort, AggregateSummary, AppendInput, Event, EventFilter, EventStore,
    StoreError, StreamLoad, Version,
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
CREATE INDEX IF NOT EXISTS idx_events_aggregate ON events (aggregate_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events (aggregate_type);
CREATE INDEX IF NOT EXISTS idx_events_trace_id ON events (trace_id);

CREATE TABLE IF NOT EXISTS snapshots (
    aggregate_id     TEXT PRIMARY KEY,
    aggregate_type   TEXT NOT NULL,
    tenant_id        TEXT NOT NULL,
    stream_version   INTEGER NOT NULL,
    data             TEXT NOT NULL,
    status           TEXT,
    label            TEXT,
    wake_at          TEXT,
    first_event_at   TEXT,
    last_event_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_snapshots_tenant ON snapshots (tenant_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_type ON snapshots (aggregate_type);
CREATE INDEX IF NOT EXISTS idx_snapshots_wake_at ON snapshots (wake_at);
CREATE INDEX IF NOT EXISTS idx_snapshots_last_event ON snapshots (last_event_at);
";

#[derive(Iden)]
enum Snapshots {
    Table,
    AggregateId,
    AggregateType,
    TenantId,
    StreamVersion,
    Status,
    Label,
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

pub struct SqliteEventStore {
    writer: Arc<Mutex<Connection>>,
    path: String,
}

impl SqliteEventStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        let writer =
            Connection::open(&path_str).map_err(|e| StoreError::Internal(e.to_string()))?;
        writer
            .execute_batch(SCHEMA)
            .map_err(|e| StoreError::Internal(e.to_string()))?;
        writer
            .pragma_update(None, "journal_mode", "WAL")
            .map_err(|e| StoreError::Internal(e.to_string()))?;

        Ok(Self {
            writer: Arc::new(Mutex::new(writer)),
            path: path_str,
        })
    }
}

fn open_conn_flags(path: &str, flags: OpenFlags) -> Result<Connection, StoreError> {
    Connection::open_with_flags(path, flags).map_err(|e| StoreError::Internal(e.to_string()))
}

fn parse_dt(s: &str) -> Option<DateTime<Utc>> {
    s.parse().ok()
}

struct IndexFields {
    status: Option<String>,
    label: Option<String>,
    wake_at: Option<String>,
    first_event_at: Option<String>,
    last_event_at: Option<String>,
}

fn extract_index_fields(snapshot: &serde_json::Value) -> IndexFields {
    let str_field =
        |key: &str| -> Option<String> { snapshot.get(key)?.as_str().map(|s| s.to_string()) };
    IndexFields {
        status: str_field("status"),
        label: str_field("label"),
        wake_at: str_field("wake_at"),
        first_event_at: str_field("first_event_at"),
        last_event_at: str_field("last_event_at"),
    }
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

fn do_append(conn: &Connection, input: AppendInput) -> Result<(), StoreError> {
    let aid = input.aggregate_id.to_string();
    let tx = conn
        .unchecked_transaction()
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
                         WHERE aggregate_id = ?2 AND stream_version = ?8 AND tenant_id = ?9
                     )
                 )",
                rusqlite::params![
                    input.aggregate_type,
                    aid,
                    input.tenant_id,
                    event.sequence,
                    occurred_at,
                    data,
                    trace_id,
                    expected_i64,
                    input.tenant_id,
                ],
            )
            .map_err(|e| StoreError::Internal(e.to_string()))?;

        if rows == 0 {
            let existing: Option<(u64, String)> = tx
                .query_row(
                    "SELECT stream_version, tenant_id FROM snapshots WHERE aggregate_id = ?1",
                    [&aid],
                    |row| Ok((row.get::<_, u64>(0)?, row.get::<_, String>(1)?)),
                )
                .optional()
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            return match existing {
                Some((_, ref t)) if t != &input.tenant_id => Err(StoreError::TenantMismatch),
                Some((v, _)) => Err(StoreError::VersionConflict {
                    expected: Version(input.expected_version),
                    actual: Version(v),
                }),
                None => Err(StoreError::VersionConflict {
                    expected: Version(input.expected_version),
                    actual: Version(0),
                }),
            };
        }
    }

    let snapshot_data =
        serde_json::to_string(&input.snapshot).map_err(|e| StoreError::Internal(e.to_string()))?;
    let idx = extract_index_fields(&input.snapshot);

    tx.execute(
        "INSERT INTO snapshots (aggregate_id, aggregate_type, tenant_id, stream_version, data, status, label, wake_at, first_event_at, last_event_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
         ON CONFLICT(aggregate_id) DO UPDATE SET
             stream_version = excluded.stream_version,
             data = excluded.data,
             status = excluded.status,
             label = excluded.label,
             wake_at = excluded.wake_at,
             first_event_at = COALESCE(snapshots.first_event_at, excluded.first_event_at),
             last_event_at = excluded.last_event_at",
        rusqlite::params![
            aid,
            input.aggregate_type,
            input.tenant_id,
            input.new_version,
            snapshot_data,
            idx.status,
            idx.label,
            idx.wake_at,
            idx.first_event_at,
            idx.last_event_at,
        ],
    )
    .map_err(|e| StoreError::Internal(e.to_string()))?;

    tx.commit()
        .map_err(|e| StoreError::Internal(e.to_string()))?;

    Ok(())
}

fn do_load(
    conn: &Connection,
    aggregate_id: Uuid,
    tenant_id: &str,
) -> Result<StreamLoad, StoreError> {
    let aid = aggregate_id.to_string();

    let (stored_tenant, snapshot_data, stream_version) = conn
        .query_row(
            "SELECT tenant_id, data, stream_version FROM snapshots WHERE aggregate_id = ?1",
            [&aid],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, u64>(2)?,
                ))
            },
        )
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => StoreError::StreamNotFound,
            other => StoreError::Internal(other.to_string()),
        })?;

    if stored_tenant != tenant_id {
        return Err(StoreError::TenantMismatch);
    }

    let snapshot: serde_json::Value =
        serde_json::from_str(&snapshot_data).map_err(|e| StoreError::Internal(e.to_string()))?;

    Ok(StreamLoad {
        snapshot,
        stream_version,
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
            Snapshots::Status,
            Snapshots::Label,
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
        .apply_if(filter.status.as_ref(), |q, statuses| {
            q.and_where(Expr::col(Snapshots::Status).is_in(statuses.clone()));
        })
        .apply_if(filter.label.as_ref(), |q, v| {
            q.and_where(Expr::col(Snapshots::Label).eq(v));
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
                row.get::<_, Option<String>>(4)?,
                row.get::<_, Option<String>>(5)?,
                row.get::<_, u64>(6)?,
                row.get::<_, Option<String>>(7)?,
                row.get::<_, Option<String>>(8)?,
            ))
        })
        .map_err(|e| StoreError::Internal(e.to_string()))?;

    let mut results = Vec::new();
    for row in rows {
        let (aid, agg_type, tenant, status, label, wake_at_str, version, first, last) =
            row.map_err(|e| StoreError::Internal(e.to_string()))?;
        let aggregate_id = aid
            .parse()
            .map_err(|e: uuid::Error| StoreError::Internal(format!("bad aggregate_id: {e}")))?;
        results.push(AggregateSummary {
            aggregate_id,
            aggregate_type: agg_type,
            tenant_id: tenant,
            status,
            label,
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
impl EventStore for SqliteEventStore {
    async fn append(&self, input: AppendInput) -> Result<(), StoreError> {
        let writer = self.writer.clone();
        tokio::task::spawn_blocking(move || {
            let conn = writer
                .lock()
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            do_append(&conn, input)
        })
        .await
        .map_err(spawn_err)?
    }

    async fn load(&self, aggregate_id: Uuid, tenant_id: &str) -> Result<StreamLoad, StoreError> {
        let tenant = tenant_id.to_string();
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || {
            let conn = open_conn_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
            do_load(&conn, aggregate_id, &tenant)
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
}
