use std::collections::VecDeque;
use std::ops::Deref;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use ractor::{call_t, Actor, ActorProcessingErr, ActorRef, OutputPort, RpcReplyPort};
use rusqlite::Connection;
use uuid::Uuid;

use crate::domain::session::AgentState;

use super::session_index::{SessionFilter, SessionIndex, SessionSort, SessionSummary};
use super::store::{Event, EventBatch, EventStore, StoreError, StreamLoad, Version};

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
            let conn = open_reader(&args.path)
                .map_err(|e| format!("reader pool connection: {e}"))?;
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

fn do_load(conn: &Connection, aggregate_id: Uuid, tenant_id: &str) -> Result<StreamLoad, StoreError> {
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

    let snapshot: serde_json::Value = serde_json::from_str(&snapshot_data)
        .map_err(|e| StoreError::Internal(e.to_string()))?;

    Ok(StreamLoad {
        snapshot,
        stream_version,
    })
}

fn do_list_sessions(conn: &Connection, filter: &SessionFilter) -> Vec<SessionSummary> {
    let mut clauses: Vec<&str> = vec!["aggregate_type = 'session'"];
    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

    if filter.tenant_id.is_some() {
        clauses.push("tenant_id = ?");
        params.push(Box::new(filter.tenant_id.clone().unwrap()));
    }
    if filter.agent_name.is_some() {
        clauses.push("agent_name = ?");
        params.push(Box::new(filter.agent_name.clone().unwrap()));
    }
    match filter.needs_wake {
        Some(true) => {
            clauses.push("wake_at IS NOT NULL AND wake_at <= ?");
            params.push(Box::new(chrono::Utc::now().to_rfc3339()));
        }
        Some(false) => {
            clauses.push("(wake_at IS NULL OR wake_at > ?)");
            params.push(Box::new(chrono::Utc::now().to_rfc3339()));
        }
        None => {}
    }

    // Status filter uses the indexed column populated by extract_session_index_fields.
    let status_clause = if let Some(ref statuses) = filter.statuses {
        if !statuses.is_empty() {
            let ph: Vec<&str> = statuses.iter().map(|s| {
                let serialized = serde_json::to_value(s)
                    .and_then(|v| serde_json::to_string(&v))
                    .unwrap_or_default();
                params.push(Box::new(serialized));
                "?"
            }).collect();
            Some(format!("status IN ({})", ph.join(", ")))
        } else {
            None
        }
    } else {
        None
    };
    if let Some(ref clause) = status_clause {
        clauses.push(clause);
    }

    let order_by = match filter.sort {
        SessionSort::LastEventDesc => "last_event_at DESC NULLS LAST",
        SessionSort::FirstEventDesc => "first_event_at DESC NULLS LAST",
        SessionSort::FirstEventAsc => "first_event_at ASC NULLS LAST",
    };

    let sql = format!(
        "SELECT data, stream_version, first_event_at, last_event_at FROM snapshots WHERE {} ORDER BY {}",
        clauses.join(" AND "),
        order_by,
    );
    let mut stmt = conn.prepare(&sql).expect("prepare list_sessions");
    let param_refs: Vec<&dyn rusqlite::types::ToSql> =
        params.iter().map(|p| p.as_ref()).collect();

    let rows = stmt
        .query_map(param_refs.as_slice(), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, u64>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, Option<String>>(3)?,
            ))
        })
        .expect("query list_sessions");

    rows.filter_map(|r| {
        let (data, stream_version, first_event_at_str, last_event_at_str) = r.ok()?;
        let snapshot: crate::domain::aggregate::Aggregate<AgentState> =
            serde_json::from_str(&data).ok()?;
        let state = &snapshot.state;

        let auth = state.auth.as_ref()?;
        let agent = state.agent.as_ref()?;

        let first_event_at = first_event_at_str
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));
        let last_event_at = last_event_at_str
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));

        Some(SessionSummary {
            session_id: state.session_id,
            tenant_id: auth.tenant_id.clone(),
            status: state.status.clone(),
            wake_at: state.wake_at(),
            agent_name: agent.name.clone(),
            message_count: state.messages.len(),
            token_usage: state.token_usage.total_tokens,
            stream_version,
            first_event_at,
            last_event_at,
        })
    })
    .collect()
}

fn do_next_wake_at(conn: &Connection) -> Option<chrono::DateTime<chrono::Utc>> {
    conn.query_row(
        "SELECT MIN(wake_at) FROM snapshots WHERE aggregate_type = 'session' AND wake_at IS NOT NULL",
        [],
        |row| row.get::<_, Option<String>>(0),
    )
    .ok()
    .flatten()
    .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
    .map(|dt| dt.with_timezone(&chrono::Utc))
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

            let batch: EventBatch = events.into_iter().map(Arc::new).collect();

            // Upsert snapshot
            let snapshot_data = serde_json::to_string(&snapshot)
                .map_err(|e| StoreError::Internal(e.to_string()))?;

            // Extract indexed fields from the snapshot.
            let idx = extract_session_index_fields(&snapshot);

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
                rusqlite::params![aid, aggregate_type, tenant_id, new_version, snapshot_data, idx.status, idx.agent_name, idx.wake_at, idx.first_event_at, idx.last_event_at],
            )
            .map_err(|e| StoreError::Internal(e.to_string()))?;

            tx.commit()
                .map_err(|e| StoreError::Internal(e.to_string()))?;

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
}

#[async_trait]
impl SessionIndex for SqliteEventStore {
    async fn list_sessions(&self, filter: &SessionFilter) -> Vec<SessionSummary> {
        match self.checkout().await {
            Ok(conn) => do_list_sessions(&conn, filter),
            Err(e) => {
                tracing::error!(error = %e, "failed to checkout reader for list_sessions");
                Vec::new()
            }
        }
    }

    async fn next_wake_at(&self) -> Option<chrono::DateTime<chrono::Utc>> {
        let conn = self.checkout().await.ok()?;
        do_next_wake_at(&conn)
    }
}

/// Indexed fields extracted from a snapshot for the session index.
struct SnapshotIndexFields {
    status: Option<String>,
    agent_name: Option<String>,
    wake_at: Option<String>,
    first_event_at: Option<String>,
    last_event_at: Option<String>,
}

/// Extract indexed fields from a snapshot Value.
fn extract_session_index_fields(snapshot: &serde_json::Value) -> SnapshotIndexFields {
    // The snapshot is Aggregate<AgentState>, so reducer fields live under "state".
    let inner = snapshot.get("state").unwrap_or(snapshot);

    let status = inner.get("status").and_then(|v| {
        // SessionStatus serializes to a JSON value; store as string for index.
        serde_json::to_string(v).ok()
    });
    let agent_name = inner
        .get("agent")
        .and_then(|a| a.get("name"))
        .and_then(|n| n.as_str())
        .map(|s| s.to_string());

    // Compute wake_at from the deserialized AgentState (if possible).
    let wake_at = serde_json::from_value::<AgentState>(inner.clone())
        .ok()
        .and_then(|state| state.wake_at().map(|t| t.to_rfc3339()));

    // Timestamps are on the Aggregate wrapper (top-level).
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
        agent_name,
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
    use crate::domain::event::{EventPayload, ClientIdentity, SessionCreated, SpanContext};

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
    ) -> (Vec<Event>, Aggregate<AgentState>) {
        let auth = test_auth(tenant);
        let agent = test_agent(agent_name);
        let domain_event: DomainEvent<AgentState> = DomainEvent {
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
        let mut snapshot = Aggregate::new(AgentState::new(session_id));
        snapshot.apply(&domain_event.payload, domain_event.sequence, domain_event.occurred_at);
        (vec![domain_event.into_raw()], snapshot)
    }

    async fn temp_store() -> SqliteEventStore {
        SqliteEventStore::new(":memory:").await.expect("open in-memory sqlite")
    }

    /// Smoke test: append events, load them back, list via index.
    #[tokio::test]
    async fn append_load_and_list() {
        let store = temp_store().await;
        let id = Uuid::new_v4();
        let (events, snap) = session_created_event(id, "acme", "bot");
        let snap_val = serde_json::to_value(&snap).unwrap();

        store
            .append(id, "acme", "session", events, snap_val, 0, snap.stream_version)
            .await
            .unwrap();

        // Load returns correct state
        let loaded = store.load(id, "acme").await.unwrap();
        let state: Aggregate<AgentState> = serde_json::from_value(loaded.snapshot).unwrap();
        assert_eq!(state.state.session_id, id);
        assert_eq!(loaded.stream_version, snap.stream_version);

        // Shows up in list_sessions
        let sessions = store.list_sessions(&SessionFilter::default()).await;
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].session_id, id);
        assert_eq!(sessions[0].agent_name, "bot");

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
            .append(id, "tenant-a", "session", events, snap_val, 0, snap.stream_version)
            .await
            .unwrap();

        // load rejects wrong tenant
        assert!(matches!(
            store.load(id, "tenant-b").await,
            Err(StoreError::TenantMismatch)
        ));

        // list_sessions filters by tenant
        let filter = SessionFilter {
            tenant_id: Some("tenant-b".into()),
            ..Default::default()
        };
        assert!(store.list_sessions(&filter).await.is_empty());
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
            .append(id, "t", "session", ev2, serde_json::to_value(&snap2).unwrap(), 0, snap2.stream_version)
            .await;
        assert!(matches!(result, Err(StoreError::VersionConflict { .. })));
    }
}
