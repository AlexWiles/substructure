use std::collections::VecDeque;
use std::ops::Deref;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use ractor::{call_t, Actor, ActorProcessingErr, ActorRef, OutputPort, RpcReplyPort};
use rusqlite::Connection;
use uuid::Uuid;

use crate::domain::aggregate::AggregateStatus;

use super::store::{
    AggregateFilter, AggregateSort, AggregateSummary, Event, EventBatch, EventStore, StoreError,
    StreamLoad, Version,
};

const SCHEMA_SQL: &str = r#"
    CREATE TABLE IF NOT EXISTS events (
        global_sequence  INTEGER PRIMARY KEY AUTOINCREMENT,
        aggregate_type   TEXT NOT NULL,
        aggregate_id     TEXT NOT NULL,
        event_type       TEXT NOT NULL,
        tenant_id        TEXT NOT NULL,
        sequence         INTEGER NOT NULL,
        data             TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_events_aggregate ON events (aggregate_id);
    CREATE INDEX IF NOT EXISTS idx_events_type ON events (aggregate_type);

    CREATE TABLE IF NOT EXISTS snapshots (
        aggregate_id     TEXT PRIMARY KEY,
        aggregate_type   TEXT NOT NULL,
        tenant_id        TEXT NOT NULL,
        stream_version   INTEGER NOT NULL,
        data             TEXT NOT NULL,
        status           TEXT,
        agent_name       TEXT,
        wake_at          TEXT,
        first_event_at   TEXT,
        last_event_at    TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_snapshots_tenant ON snapshots (tenant_id);
    CREATE INDEX IF NOT EXISTS idx_snapshots_type ON snapshots (aggregate_type);
    CREATE INDEX IF NOT EXISTS idx_snapshots_first_event ON snapshots (first_event_at);
    CREATE INDEX IF NOT EXISTS idx_snapshots_last_event ON snapshots (last_event_at);
    CREATE INDEX IF NOT EXISTS idx_snapshots_wake_at ON snapshots (wake_at);
"#;

// ---------------------------------------------------------------------------
// ConnectionPoolActor — Elixir DBConnection-style checkout/checkin pool
// ---------------------------------------------------------------------------

pub(crate) enum PoolMessage {
    Checkout(RpcReplyPort<Connection>),
    Checkin(Connection),
}

struct ConnectionPoolActor;

struct PoolState {
    available: Vec<Connection>,
    waiting: VecDeque<RpcReplyPort<Connection>>,
}

struct PoolArgs {
    path: String,
    pool_size: usize,
}

/// Open a reader connection with WAL mode enabled.
fn open_reader(path: &str) -> Result<Connection, StoreError> {
    let conn = Connection::open(path).map_err(|e| StoreError::Internal(e.to_string()))?;
    conn.pragma_update(None, "journal_mode", "WAL")
        .map_err(|e| StoreError::Internal(e.to_string()))?;
    Ok(conn)
}

impl Actor for ConnectionPoolActor {
    type Msg = PoolMessage;
    type State = PoolState;
    type Arguments = PoolArgs;

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        let mut available = Vec::with_capacity(args.pool_size);
        for _ in 0..args.pool_size {
            let conn =
                open_reader(&args.path).map_err(|e| format!("reader pool connection: {e}"))?;
            available.push(conn);
        }
        Ok(PoolState {
            available,
            waiting: VecDeque::new(),
        })
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match message {
            PoolMessage::Checkout(reply) => {
                if let Some(conn) = state.available.pop() {
                    let _ = reply.send(conn);
                } else {
                    state.waiting.push_back(reply);
                }
            }
            PoolMessage::Checkin(conn) => {
                if let Some(waiter) = state.waiting.pop_front() {
                    let _ = waiter.send(conn);
                } else {
                    state.available.push(conn);
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PooledConnection — RAII guard that auto-returns connection on drop
// ---------------------------------------------------------------------------

pub(crate) struct PooledConnection {
    conn: Option<Connection>,
    pool: ActorRef<PoolMessage>,
}

impl Drop for PooledConnection {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.take() {
            let _ = self.pool.send_message(PoolMessage::Checkin(conn));
        }
    }
}

impl Deref for PooledConnection {
    type Target = Connection;
    fn deref(&self) -> &Connection {
        self.conn.as_ref().unwrap()
    }
}

// ---------------------------------------------------------------------------
// Free functions — query logic shared by read paths
// ---------------------------------------------------------------------------

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

fn parse_dt(s: &str) -> Option<DateTime<Utc>> {
    chrono::DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.with_timezone(&Utc))
}

fn parse_status(s: &str) -> AggregateStatus {
    match s {
        "active" => AggregateStatus::Active,
        "idle" => AggregateStatus::Idle,
        "done" => AggregateStatus::Done,
        _ => AggregateStatus::Idle,
    }
}

fn status_to_str(s: &AggregateStatus) -> &'static str {
    match s {
        AggregateStatus::Active => "active",
        AggregateStatus::Idle => "idle",
        AggregateStatus::Done => "done",
    }
}

fn do_list_aggregates(
    conn: &Connection,
    filter: &AggregateFilter,
) -> Result<Vec<AggregateSummary>, StoreError> {
    let mut clauses: Vec<String> = Vec::new();
    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

    if let Some(ref agg_type) = filter.aggregate_type {
        clauses.push("aggregate_type = ?".into());
        params.push(Box::new(agg_type.clone()));
    }

    if let Some(ref ids) = filter.aggregate_ids {
        if !ids.is_empty() {
            let placeholders: Vec<&str> = ids
                .iter()
                .map(|id| {
                    params.push(Box::new(id.to_string()));
                    "?"
                })
                .collect();
            clauses.push(format!("aggregate_id IN ({})", placeholders.join(", ")));
        }
    }

    if let Some(ref tenant) = filter.tenant_id {
        clauses.push("tenant_id = ?".into());
        params.push(Box::new(tenant.clone()));
    }

    if let Some(ref statuses) = filter.status {
        if !statuses.is_empty() {
            let placeholders: Vec<&str> = statuses
                .iter()
                .map(|s| {
                    params.push(Box::new(status_to_str(s).to_string()));
                    "?"
                })
                .collect();
            clauses.push(format!("status IN ({})", placeholders.join(", ")));
        }
    }

    if let Some(ref label) = filter.label {
        clauses.push("agent_name = ?".into());
        params.push(Box::new(label.clone()));
    }

    if filter.wake_at_not_null {
        clauses.push("wake_at IS NOT NULL".into());
    }

    if let Some(ref before) = filter.wake_at_before {
        clauses.push("wake_at IS NOT NULL AND wake_at <= ?".into());
        params.push(Box::new(before.to_rfc3339()));
    }

    let where_clause = if clauses.is_empty() {
        String::new()
    } else {
        format!(" WHERE {}", clauses.join(" AND "))
    };

    let order_by = match filter.sort {
        AggregateSort::LastEventDesc => "last_event_at DESC NULLS LAST",
        AggregateSort::FirstEventDesc => "first_event_at DESC NULLS LAST",
        AggregateSort::FirstEventAsc => "first_event_at ASC NULLS LAST",
        AggregateSort::WakeAtAsc => "wake_at ASC NULLS LAST",
    };

    let limit_clause = filter
        .limit
        .map(|n| format!(" LIMIT {n}"))
        .unwrap_or_default();

    let sql = format!(
        "SELECT aggregate_id, aggregate_type, tenant_id, status, agent_name, wake_at, \
         stream_version, first_event_at, last_event_at \
         FROM snapshots{where_clause} ORDER BY {order_by}{limit_clause}"
    );

    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| StoreError::Internal(e.to_string()))?;

    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();

    let rows = stmt
        .query_map(param_refs.as_slice(), |row| {
            Ok((
                row.get::<_, String>(0)?,         // aggregate_id
                row.get::<_, String>(1)?,         // aggregate_type
                row.get::<_, String>(2)?,         // tenant_id
                row.get::<_, Option<String>>(3)?, // status
                row.get::<_, Option<String>>(4)?, // agent_name (label)
                row.get::<_, Option<String>>(5)?, // wake_at
                row.get::<_, u64>(6)?,            // stream_version
                row.get::<_, Option<String>>(7)?, // first_event_at
                row.get::<_, Option<String>>(8)?, // last_event_at
            ))
        })
        .map_err(|e| StoreError::Internal(e.to_string()))?;

    let results = rows
        .filter_map(|r| {
            let (aid, agg_type, tenant, status_str, label, wake_at_str, version, first, last) =
                r.ok()?;
            let aggregate_id: Uuid = aid.parse().ok()?;
            Some(AggregateSummary {
                aggregate_id,
                aggregate_type: agg_type,
                tenant_id: tenant,
                status: status_str.as_deref().map(parse_status).unwrap_or_default(),
                label,
                wake_at: wake_at_str.as_deref().and_then(parse_dt),
                stream_version: version,
                first_event_at: first.as_deref().and_then(parse_dt),
                last_event_at: last.as_deref().and_then(parse_dt),
            })
        })
        .collect();

    Ok(results)
}

// ---------------------------------------------------------------------------
// SqliteEventStore
// ---------------------------------------------------------------------------

pub struct SqliteEventStore {
    writer: Mutex<Connection>,
    reader_pool: ActorRef<PoolMessage>,
    events: OutputPort<EventBatch>,
}

impl SqliteEventStore {
    /// Default number of reader connections in the pool.
    const DEFAULT_POOL_SIZE: usize = 4;

    pub async fn new(path: &str) -> Result<Self, StoreError> {
        // For :memory: databases, use shared cache so all connections see the
        // same data. Generate a unique name to avoid cross-test collisions.
        let effective_path = if path == ":memory:" {
            format!("file:memdb_{}?mode=memory&cache=shared", Uuid::new_v4())
        } else {
            path.to_string()
        };

        let writer = if path == ":memory:" {
            Connection::open_with_flags(
                &effective_path,
                rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE
                    | rusqlite::OpenFlags::SQLITE_OPEN_CREATE
                    | rusqlite::OpenFlags::SQLITE_OPEN_URI,
            )
        } else {
            Connection::open(&effective_path)
        }
        .map_err(|e| StoreError::Internal(e.to_string()))?;

        writer
            .pragma_update(None, "journal_mode", "WAL")
            .map_err(|e| StoreError::Internal(e.to_string()))?;

        writer
            .execute_batch(SCHEMA_SQL)
            .map_err(|e| StoreError::Internal(e.to_string()))?;

        let (pool_actor, _) = Actor::spawn(
            None,
            ConnectionPoolActor,
            PoolArgs {
                path: effective_path,
                pool_size: Self::DEFAULT_POOL_SIZE,
            },
        )
        .await
        .map_err(|e| StoreError::Internal(format!("reader pool: {e}")))?;

        Ok(SqliteEventStore {
            writer: Mutex::new(writer),
            reader_pool: pool_actor,
            events: OutputPort::default(),
        })
    }

    /// Checkout a reader connection from the pool.
    async fn checkout(&self) -> Result<PooledConnection, StoreError> {
        let conn = call_t!(self.reader_pool, PoolMessage::Checkout, 5000)
            .map_err(|e| StoreError::Internal(format!("pool checkout: {e}")))?;
        Ok(PooledConnection {
            conn: Some(conn),
            pool: self.reader_pool.clone(),
        })
    }
}

#[async_trait]
impl EventStore for SqliteEventStore {
    async fn append(
        &self,
        aggregate_id: Uuid,
        tenant_id: &str,
        aggregate_type: &str,
        events: Vec<Event>,
        snapshot: serde_json::Value,
        expected_version: u64,
        new_version: u64,
    ) -> Result<(), StoreError> {
        let batch = {
            let mut conn = self.writer.lock().expect("sqlite writer lock poisoned");
            let tx = conn
                .transaction()
                .map_err(|e| StoreError::Internal(e.to_string()))?;

            let aid = aggregate_id.to_string();

            // Insert events — version + tenant guard embedded in query.
            {
                let mut stmt = tx
                    .prepare(
                        "INSERT INTO events (aggregate_type, aggregate_id, event_type, tenant_id, sequence, data)
                         SELECT ?1, ?2, ?3, ?4, ?5, ?6
                         WHERE (
                             (?7 = 0 AND NOT EXISTS (SELECT 1 FROM snapshots WHERE aggregate_id = ?2))
                             OR EXISTS (
                                 SELECT 1 FROM snapshots
                                 WHERE aggregate_id = ?2 AND stream_version = ?7 AND tenant_id = ?8
                             )
                         )",
                    )
                    .map_err(|e| StoreError::Internal(e.to_string()))?;

                for event in &events {
                    let data = serde_json::to_string(event)
                        .map_err(|e| StoreError::Internal(e.to_string()))?;
                    let rows = stmt
                        .execute(rusqlite::params![
                            aggregate_type,
                            aid,
                            event.event_type,
                            event.tenant_id,
                            event.sequence,
                            data,
                            expected_version as i64,
                            tenant_id
                        ])
                        .map_err(|e| StoreError::Internal(e.to_string()))?;

                    if rows == 0 {
                        return match tx
                            .query_row(
                                "SELECT stream_version, tenant_id FROM snapshots WHERE aggregate_id = ?1",
                                [&aid],
                                |row| Ok((row.get::<_, u64>(0)?, row.get::<_, String>(1)?)),
                            )
                            .optional()
                            .map_err(|e| StoreError::Internal(e.to_string()))?
                        {
                            Some((_, ref t)) if t != tenant_id => Err(StoreError::TenantMismatch),
                            Some((v, _)) => Err(StoreError::VersionConflict {
                                expected: Version(expected_version),
                                actual: Version(v),
                            }),
                            None => Err(StoreError::VersionConflict {
                                expected: Version(expected_version),
                                actual: Version(0),
                            }),
                        };
                    }
                }
            }

            // Upsert snapshot
            let snapshot_data = serde_json::to_string(&snapshot)
                .map_err(|e| StoreError::Internal(e.to_string()))?;

            // Extract indexed fields from the top-level aggregate snapshot.
            let idx = extract_index_fields(&snapshot);

            tx.execute(
                "INSERT INTO snapshots (aggregate_id, aggregate_type, tenant_id, stream_version, data, status, agent_name, wake_at, first_event_at, last_event_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
                 ON CONFLICT(aggregate_id) DO UPDATE SET
                     aggregate_type = excluded.aggregate_type,
                     tenant_id = excluded.tenant_id,
                     stream_version = excluded.stream_version,
                     data = excluded.data,
                     status = excluded.status,
                     agent_name = excluded.agent_name,
                     wake_at = excluded.wake_at,
                     first_event_at = COALESCE(snapshots.first_event_at, excluded.first_event_at),
                     last_event_at = excluded.last_event_at",
                rusqlite::params![aid, aggregate_type, tenant_id, new_version, snapshot_data, idx.status, idx.label, idx.wake_at, idx.first_event_at, idx.last_event_at],
            )
            .map_err(|e| StoreError::Internal(e.to_string()))?;

            tx.commit()
                .map_err(|e| StoreError::Internal(e.to_string()))?;

            // Stamp wake_at from snapshot onto broadcast events.
            let wake_at = idx.wake_at.as_deref().and_then(parse_dt);
            let batch: EventBatch = events
                .into_iter()
                .map(|mut e| {
                    e.wake_at = wake_at;
                    Arc::new(e)
                })
                .collect();

            batch
        };

        self.events.send(batch);
        Ok(())
    }

    async fn load(&self, aggregate_id: Uuid, tenant_id: &str) -> Result<StreamLoad, StoreError> {
        let conn = self.checkout().await?;
        do_load(&conn, aggregate_id, tenant_id)
    }

    fn events(&self) -> &OutputPort<EventBatch> {
        &self.events
    }

    async fn list_aggregates(&self, filter: &AggregateFilter) -> Vec<AggregateSummary> {
        match self.checkout().await {
            Ok(conn) => do_list_aggregates(&conn, filter).unwrap_or_else(|e| {
                tracing::error!(error = %e, "list_aggregates failed");
                Vec::new()
            }),
            Err(e) => {
                tracing::error!(error = %e, "failed to checkout reader for list_aggregates");
                Vec::new()
            }
        }
    }
}

/// Indexed fields extracted from a snapshot for query columns.
struct SnapshotIndexFields {
    status: Option<String>,
    label: Option<String>,
    wake_at: Option<String>,
    first_event_at: Option<String>,
    last_event_at: Option<String>,
}

/// Extract indexed fields from a top-level Aggregate snapshot Value.
/// All fields are read from the Aggregate wrapper (top-level JSON), not from
/// the reducer state. This keeps the store fully generic.
fn extract_index_fields(snapshot: &serde_json::Value) -> SnapshotIndexFields {
    let status = snapshot
        .get("status")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let label = snapshot
        .get("label")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let wake_at = snapshot
        .get("wake_at")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let first_event_at = snapshot
        .get("first_event_at")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let last_event_at = snapshot
        .get("last_event_at")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    SnapshotIndexFields {
        status,
        label,
        wake_at,
        first_event_at,
        last_event_at,
    }
}

/// Extension trait for optional query results.
trait OptionalExt<T> {
    fn optional(self) -> Result<Option<T>, rusqlite::Error>;
}

impl<T> OptionalExt<T> for Result<T, rusqlite::Error> {
    fn optional(self) -> Result<Option<T>, rusqlite::Error> {
        match self {
            Ok(v) => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::agent::{AgentConfig, LlmConfig};
    use crate::domain::aggregate::{Aggregate, DomainEvent};
    use crate::domain::event::{ClientIdentity, EventPayload, SessionCreated, SpanContext};
    use crate::domain::session::{AgentSession, DefaultStrategy};
    use std::sync::Arc;

    fn test_auth(tenant: &str) -> ClientIdentity {
        ClientIdentity {
            tenant_id: tenant.into(),
            sub: None,
            attrs: Default::default(),
        }
    }

    fn test_agent(name: &str) -> AgentConfig {
        AgentConfig {
            id: Uuid::new_v4(),
            name: name.into(),
            description: None,
            llm: LlmConfig {
                client: "mock".into(),
                params: Default::default(),
            },
            system_prompt: "test".into(),
            mcp_servers: vec![],
            strategy: Default::default(),
            retry: Default::default(),
            token_budget: None,
            sub_agents: vec![],
        }
    }

    fn session_created_event(
        session_id: Uuid,
        tenant: &str,
        agent_name: &str,
    ) -> (Vec<Event>, Aggregate<AgentSession>) {
        let auth = test_auth(tenant);
        let agent = test_agent(agent_name);
        let domain_event: DomainEvent<AgentSession> = DomainEvent {
            id: Uuid::new_v4(),
            tenant_id: tenant.into(),
            aggregate_id: session_id,
            sequence: 1,
            span: SpanContext::root(),
            occurred_at: chrono::Utc::now(),
            payload: EventPayload::SessionCreated(SessionCreated {
                agent: agent.clone(),
                auth: auth.clone(),
                on_done: None,
            }),
            derived: None,
        };
        let mut snapshot = Aggregate::new(AgentSession::new(
            session_id,
            Arc::new(DefaultStrategy::default()),
        ));
        snapshot.apply(
            &domain_event.payload,
            domain_event.sequence,
            domain_event.occurred_at,
        );
        (vec![domain_event.into_raw()], snapshot)
    }

    async fn temp_store() -> SqliteEventStore {
        SqliteEventStore::new(":memory:")
            .await
            .expect("open in-memory sqlite")
    }

    /// Smoke test: append events, load them back, list via list_aggregates.
    #[tokio::test]
    async fn append_load_and_list() {
        let store = temp_store().await;
        let id = Uuid::new_v4();
        let (events, snap) = session_created_event(id, "acme", "bot");
        let snap_val = serde_json::to_value(&snap).unwrap();

        store
            .append(
                id,
                "acme",
                "session",
                events,
                snap_val,
                0,
                snap.stream_version,
            )
            .await
            .unwrap();

        // Load returns correct state
        let loaded = store.load(id, "acme").await.unwrap();
        let state: Aggregate<AgentSession> = serde_json::from_value(loaded.snapshot).unwrap();
        assert_eq!(state.state.cloned_state().session_id, id);
        assert_eq!(loaded.stream_version, snap.stream_version);

        // Shows up in list_aggregates
        let results = store
            .list_aggregates(&AggregateFilter {
                aggregate_type: Some("session".into()),
                ..Default::default()
            })
            .await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].aggregate_id, id);
        assert_eq!(results[0].label.as_deref(), Some("bot"));

        // Stream not found for unknown id
        assert!(matches!(
            store.load(Uuid::new_v4(), "acme").await,
            Err(StoreError::StreamNotFound)
        ));
    }

    /// Security boundary: tenant-a data is invisible to tenant-b.
    #[tokio::test]
    async fn tenant_isolation() {
        let store = temp_store().await;
        let id = Uuid::new_v4();
        let (events, snap) = session_created_event(id, "tenant-a", "bot");
        let snap_val = serde_json::to_value(&snap).unwrap();

        store
            .append(
                id,
                "tenant-a",
                "session",
                events,
                snap_val,
                0,
                snap.stream_version,
            )
            .await
            .unwrap();

        // load rejects wrong tenant
        assert!(matches!(
            store.load(id, "tenant-b").await,
            Err(StoreError::TenantMismatch)
        ));

        // list_aggregates filters by tenant
        let filter = AggregateFilter {
            tenant_id: Some("tenant-b".into()),
            ..Default::default()
        };
        assert!(store.list_aggregates(&filter).await.is_empty());
    }

    /// Concurrency boundary: stale expected_version is rejected.
    #[tokio::test]
    async fn version_conflict() {
        let store = temp_store().await;
        let id = Uuid::new_v4();
        let (events, snap) = session_created_event(id, "t", "bot");
        let snap_val = serde_json::to_value(&snap).unwrap();

        store
            .append(id, "t", "session", events, snap_val, 0, snap.stream_version)
            .await
            .unwrap();

        // Same expected_version=0 again → conflict
        let (ev2, snap2) = session_created_event(id, "t", "bot");
        let result = store
            .append(
                id,
                "t",
                "session",
                ev2,
                serde_json::to_value(&snap2).unwrap(),
                0,
                snap2.stream_version,
            )
            .await;
        assert!(matches!(result, Err(StoreError::VersionConflict { .. })));
    }
}
