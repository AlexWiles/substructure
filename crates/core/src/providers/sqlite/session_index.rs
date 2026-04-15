use async_trait::async_trait;
use chrono::Utc;
use sea_query::{Expr, ExprTrait, Iden, Order, Query, SqliteQueryBuilder};

use crate::event_store::{AggregateSort, StoreError};
use crate::session::index::{
    SessionCursor, SessionFilter, SessionIndexRecord, SessionIndexStore, SessionItem, SessionPage,
};

use super::{parse_dt, sea_params, spawn_err, SqliteDb};

const SCHEMA: &str = "
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
enum SessionIndexRows {
    Table,
    SessionId,
    TenantId,
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

pub struct SqliteSessionIndexStore {
    db: SqliteDb,
}

impl SqliteSessionIndexStore {
    pub fn new(db: SqliteDb) -> Result<Self, StoreError> {
        db.run_schema(SCHEMA)?;
        Ok(Self { db })
    }
}

#[async_trait]
impl SessionIndexStore for SqliteSessionIndexStore {
    async fn list_sessions(&self, filter: &SessionFilter) -> Result<SessionPage, StoreError> {
        let filter = filter.clone();
        let reader = self.db.reader.clone();
        tokio::task::spawn_blocking(move || {
            let conn = reader.open()?;
            do_list_sessions(&conn, &filter)
        })
        .await
        .map_err(spawn_err)?
    }

    async fn upsert_session_index(&self, record: SessionIndexRecord) -> Result<(), String> {
        let writer = self.db.writer.clone();
        tokio::task::spawn_blocking(move || {
            let conn = writer.lock().map_err(|e| e.to_string())?;
            do_upsert_session_index(&conn, record)
        })
        .await
        .map_err(|e| e.to_string())?
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn do_list_sessions(
    conn: &rusqlite::Connection,
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
        let status =
            serde_json::from_str(&status_json).map_err(|e| StoreError::Internal(e.to_string()))?;

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

fn do_upsert_session_index(
    conn: &rusqlite::Connection,
    record: SessionIndexRecord,
) -> Result<(), String> {
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
