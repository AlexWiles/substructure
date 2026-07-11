use std::collections::HashMap;
use std::sync::Arc as StdArc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rusqlite::{Connection, OptionalExtension};
use sea_query::{Expr, ExprTrait, Iden, Order, Query, SqliteQueryBuilder};
use tokio::sync::broadcast;
use uuid::Uuid;

use crate::event_store::{
    AggregateFilter, AggregateSort, AggregateSummary, AppendInput, Event, EventFilter, EventStore,
    GlobalPosition, Snapshot, StoreError, StreamVersion, Version,
};
use crate::span::SpanContext;

use super::{parse_dt, sea_params, spawn_err, SqliteDb};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS events (
    global_position  INTEGER PRIMARY KEY AUTOINCREMENT,
    aggregate_type   TEXT NOT NULL,
    aggregate_id     TEXT NOT NULL,
    tenant_id        TEXT NOT NULL,
    stream_version   INTEGER NOT NULL,
    occurred_at      TEXT NOT NULL,
    data             TEXT NOT NULL,
    trace_id         TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_occurred_at ON events (occurred_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_events_aggregate_seq ON events (tenant_id, aggregate_id, stream_version);
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
";

#[derive(Iden)]
enum Snapshots {
    Table,
    AggregateId,
    AggregateType,
    TenantId,
    WakeAt,
    StreamVersion,
    FirstEventAt,
    LastEventAt,
}

#[derive(Iden)]
enum Events {
    Table,
    GlobalPosition,
    AggregateId,
    AggregateType,
    TenantId,
    StreamVersion,
    OccurredAt,
    #[iden = "data"]
    Data,
    TraceId,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct StoredEvent {
    id: Uuid,
    tenant_id: String,
    aggregate_type: String,
    aggregate_id: String,
    stream_version: StreamVersion,
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
            stream_version: event.stream_version,
            span: event.span.clone(),
            occurred_at: event.occurred_at,
            payload: event.payload.clone(),
            derived: event.derived.clone(),
            metadata: event.metadata.clone(),
            start_time: event.start_time,
            end_time: event.end_time,
        }
    }

    fn into_event(self, global_position: u64) -> Event {
        Event {
            global_position: GlobalPosition(global_position),
            id: self.id,
            tenant_id: self.tenant_id,
            aggregate_type: self.aggregate_type,
            aggregate_id: self.aggregate_id,
            stream_version: self.stream_version,
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

pub struct SqliteEventStore {
    db: SqliteDb,
    tx: broadcast::Sender<StdArc<Vec<Event>>>,
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
        let events = input.events.clone();
        let writer = self.db.writer.clone();
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
            event.global_position = GlobalPosition(position);
        }

        let _ = self.tx.send(StdArc::new(with_positions));
        Ok(())
    }

    async fn load(&self, tenant_id: &str, aggregate_id: &str) -> Result<Snapshot, StoreError> {
        let reader = self.db.reader.clone();
        let tenant_id = tenant_id.to_string();
        let aggregate_id = aggregate_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = reader.open()?;
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
        let reader = self.db.reader.clone();
        tokio::task::spawn_blocking(move || {
            let conn = reader.open()?;
            do_list_aggregates(&conn, &filter)
        })
        .await
        .map_err(spawn_err)?
    }

    async fn query_events(&self, filter: &EventFilter) -> Result<Vec<Event>, StoreError> {
        let filter = filter.clone();
        let reader = self.db.reader.clone();
        tokio::task::spawn_blocking(move || {
            let conn = reader.open()?;
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
// Helpers
// ---------------------------------------------------------------------------

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
                "INSERT INTO events (aggregate_type, aggregate_id, tenant_id, stream_version, occurred_at, data, trace_id)
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
                    event.stream_version.0,
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

    let snapshot_data =
        serde_json::to_string(&snap.data).map_err(|e| StoreError::Internal(e.to_string()))?;

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
            snap.stream_version.0,
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
        stream_version: StreamVersion(stream_version),
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
            stream_version: StreamVersion(version),
            first_event_at: first.as_deref().and_then(parse_dt),
            last_event_at: last.as_deref().and_then(parse_dt),
        });
    }

    Ok(results)
}

fn do_query_events(conn: &Connection, filter: EventFilter) -> Result<Vec<Event>, StoreError> {
    let (sql, values) = Query::select()
        .column(Events::GlobalPosition)
        .column(Events::Data)
        .from(Events::Table)
        .apply_if(filter.after_global_position, |q, pos| {
            q.and_where(Expr::col(Events::GlobalPosition).gt(pos.0 as i64));
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
        .apply_if(filter.after_stream_version, |q, v| {
            q.and_where(Expr::col(Events::StreamVersion).gt(v.0 as i64));
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
        .order_by(Events::GlobalPosition, Order::Asc)
        .build(SqliteQueryBuilder);

    let params = sea_params(values);
    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();

    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| StoreError::Internal(e.to_string()))?;

    let rows = stmt
        .query_map(param_refs.as_slice(), |row| {
            Ok((row.get::<_, u64>(0)?, row.get::<_, String>(1)?))
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
