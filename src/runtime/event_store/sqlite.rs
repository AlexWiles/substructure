use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use ractor::OutputPort;
use rusqlite::Connection;
use uuid::Uuid;

use crate::domain::event::{Event, SessionAuth};
use crate::domain::session::AgentState;

use super::store::{
    EventBatch, EventStore, SessionFilter, SessionLoad, SessionSummary, StoreError, Version,
};

const SCHEMA_SQL: &str = r#"
    CREATE TABLE IF NOT EXISTS events (
        global_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id      TEXT NOT NULL,
        tenant_id       TEXT NOT NULL,
        sequence        INTEGER NOT NULL,
        data            TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_events_session ON events (session_id);

    CREATE TABLE IF NOT EXISTS snapshots (
        session_id      TEXT PRIMARY KEY,
        tenant_id       TEXT NOT NULL,
        stream_version  INTEGER NOT NULL,
        status          TEXT NOT NULL,
        agent_name      TEXT,
        wake_at         TEXT,
        data            TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_snapshots_tenant ON snapshots (tenant_id);
"#;

pub struct SqliteEventStore {
    conn: Mutex<Connection>,
    events: OutputPort<EventBatch>,
}

impl SqliteEventStore {
    pub fn new(path: &str) -> Result<Self, StoreError> {
        let conn = Connection::open(path).map_err(|e| StoreError::Internal(e.to_string()))?;

        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(|e| StoreError::Internal(e.to_string()))?;

        conn.execute_batch(SCHEMA_SQL)
            .map_err(|e| StoreError::Internal(e.to_string()))?;

        Ok(SqliteEventStore {
            conn: Mutex::new(conn),
            events: OutputPort::default(),
        })
    }
}

#[async_trait]
impl EventStore for SqliteEventStore {
    async fn append(
        &self,
        session_id: Uuid,
        auth: &SessionAuth,
        events: Vec<Event>,
        snapshot: AgentState,
    ) -> Result<(), StoreError> {
        let expected_version = snapshot.stream_version - events.len() as u64;

        let batch = {
            let mut conn = self.conn.lock().expect("sqlite lock poisoned");
            let tx = conn
                .transaction()
                .map_err(|e| StoreError::Internal(e.to_string()))?;

            let sid = session_id.to_string();

            // Insert events — version + tenant guard embedded in query.
            {
                let mut stmt = tx
                    .prepare(
                        "INSERT INTO events (session_id, tenant_id, sequence, data)
                         SELECT ?1, ?2, ?3, ?4
                         WHERE (
                             (?5 = 0 AND NOT EXISTS (SELECT 1 FROM snapshots WHERE session_id = ?1))
                             OR EXISTS (
                                 SELECT 1 FROM snapshots
                                 WHERE session_id = ?1 AND stream_version = ?5 AND tenant_id = ?6
                             )
                         )",
                    )
                    .map_err(|e| StoreError::Internal(e.to_string()))?;

                for event in &events {
                    let data = serde_json::to_string(event)
                        .map_err(|e| StoreError::Internal(e.to_string()))?;
                    let rows = stmt
                        .execute(rusqlite::params![
                            sid,
                            event.tenant_id,
                            event.sequence,
                            data,
                            expected_version as i64,
                            auth.tenant_id
                        ])
                        .map_err(|e| StoreError::Internal(e.to_string()))?;

                    if rows == 0 {
                        return match tx
                            .query_row(
                                "SELECT stream_version, tenant_id FROM snapshots WHERE session_id = ?1",
                                [&sid],
                                |row| Ok((row.get::<_, u64>(0)?, row.get::<_, String>(1)?)),
                            )
                            .optional()
                            .map_err(|e| StoreError::Internal(e.to_string()))?
                        {
                            Some((_, ref t)) if t != &auth.tenant_id => {
                                Err(StoreError::TenantMismatch)
                            }
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
            let status_str = serde_json::to_string(&snapshot.status)
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            let agent_name = snapshot.agent.as_ref().map(|a| a.name.clone());
            let wake_at = snapshot.wake_at().map(|t| t.to_rfc3339());

            tx.execute(
                "INSERT INTO snapshots (session_id, tenant_id, stream_version, status, agent_name, wake_at, data)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                 ON CONFLICT(session_id) DO UPDATE SET
                     tenant_id = excluded.tenant_id,
                     stream_version = excluded.stream_version,
                     status = excluded.status,
                     agent_name = excluded.agent_name,
                     wake_at = excluded.wake_at,
                     data = excluded.data",
                rusqlite::params![sid, auth.tenant_id, snapshot.stream_version, status_str, agent_name, wake_at, snapshot_data],
            )
            .map_err(|e| StoreError::Internal(e.to_string()))?;

            tx.commit()
                .map_err(|e| StoreError::Internal(e.to_string()))?;

            batch
        };

        self.events.send(batch);
        Ok(())
    }

    fn load(&self, session_id: Uuid, auth: &SessionAuth) -> Result<SessionLoad, StoreError> {
        let conn = self.conn.lock().expect("sqlite lock poisoned");
        let sid = session_id.to_string();

        let (tenant_id, snapshot_data) = conn
            .query_row(
                "SELECT tenant_id, data FROM snapshots WHERE session_id = ?1",
                [&sid],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => StoreError::SessionNotFound,
                other => StoreError::Internal(other.to_string()),
            })?;

        if tenant_id != auth.tenant_id {
            return Err(StoreError::TenantMismatch);
        }

        let snapshot: AgentState = serde_json::from_str(&snapshot_data)
            .map_err(|e| StoreError::Internal(e.to_string()))?;

        Ok(SessionLoad { snapshot })
    }

    fn list_sessions(&self, filter: &SessionFilter) -> Vec<SessionSummary> {
        let conn = self.conn.lock().expect("sqlite lock poisoned");
        let now = chrono::Utc::now();

        let mut sql = String::from("SELECT data, stream_version FROM snapshots WHERE 1=1");
        let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        if let Some(ref tid) = filter.tenant_id {
            sql.push_str(&format!(" AND tenant_id = ?{}", params.len() + 1));
            params.push(Box::new(tid.clone()));
        }
        if let Some(ref name) = filter.agent_name {
            sql.push_str(&format!(" AND agent_name = ?{}", params.len() + 1));
            params.push(Box::new(name.clone()));
        }

        let mut stmt = conn.prepare(&sql).expect("prepare list_sessions");
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            params.iter().map(|p| p.as_ref()).collect();

        let rows = stmt
            .query_map(param_refs.as_slice(), |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, u64>(1)?))
            })
            .expect("query list_sessions");

        rows.filter_map(|r| {
            let (data, stream_version) = r.ok()?;
            let state: AgentState = serde_json::from_str(&data).ok()?;

            let auth = state.auth.as_ref()?;
            let agent = state.agent.as_ref()?;

            if let Some(ref statuses) = filter.statuses {
                if !statuses.contains(&state.status) {
                    return None;
                }
            }

            let wake_at = state.wake_at();
            if let Some(needs_wake) = filter.needs_wake {
                let has_wake = wake_at.is_some_and(|t| t <= now);
                if needs_wake != has_wake {
                    return None;
                }
            }

            Some(SessionSummary {
                session_id: state.session_id,
                tenant_id: auth.tenant_id.clone(),
                client_id: auth.client_id.clone(),
                status: state.status.clone(),
                wake_at,
                agent_name: agent.name.clone(),
                message_count: state.messages.len(),
                token_usage: state.token_usage.total_tokens,
                stream_version,
            })
        })
        .collect()
    }

    fn events(&self) -> &OutputPort<EventBatch> {
        &self.events
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
    use crate::domain::event::{EventPayload, SessionCreated, SpanContext};
    use crate::domain::session::SessionStatus;

    fn test_auth(tenant: &str) -> SessionAuth {
        SessionAuth {
            tenant_id: tenant.into(),
            client_id: "client-1".into(),
            sub: None,
        }
    }

    fn test_agent(name: &str) -> AgentConfig {
        AgentConfig {
            id: Uuid::new_v4(),
            name: name.into(),
            llm: LlmConfig {
                client: "mock".into(),
                params: Default::default(),
            },
            system_prompt: "test".into(),
            mcp_servers: vec![],
            strategy: Default::default(),
            retry: Default::default(),
            token_budget: None,
        }
    }

    fn session_created_event(
        session_id: Uuid,
        tenant: &str,
        agent_name: &str,
    ) -> (Event, AgentState) {
        let auth = test_auth(tenant);
        let agent = test_agent(agent_name);
        let event = Event {
            id: Uuid::new_v4(),
            tenant_id: tenant.into(),
            session_id,
            sequence: 0,
            span: SpanContext::root(),
            occurred_at: chrono::Utc::now(),
            payload: EventPayload::SessionCreated(SessionCreated {
                agent: agent.clone(),
                auth: auth.clone(),
            }),
            derived: None,
        };
        let mut state = AgentState::new(session_id);
        state.apply_core(&event);
        (event, state)
    }

    fn temp_store() -> SqliteEventStore {
        SqliteEventStore::new(":memory:").expect("open in-memory sqlite")
    }

    #[tokio::test]
    async fn append_and_load() {
        let store = temp_store();
        let id = Uuid::new_v4();
        let (ev, snap) = session_created_event(id, "acme", "bot");

        store
            .append(id, &test_auth("acme"), vec![ev], snap.clone())
            .await
            .unwrap();

        let loaded = store.load(id, &test_auth("acme")).unwrap();
        assert_eq!(loaded.snapshot.session_id, id);
        assert_eq!(loaded.snapshot.stream_version, snap.stream_version);
    }

    #[tokio::test]
    async fn version_conflict() {
        let store = temp_store();
        let id = Uuid::new_v4();
        let (ev, snap) = session_created_event(id, "t", "bot");

        store
            .append(id, &test_auth("t"), vec![ev], snap)
            .await
            .unwrap();

        let (ev2, snap2) = session_created_event(id, "t", "bot");
        let result = store.append(id, &test_auth("t"), vec![ev2], snap2).await;
        assert!(matches!(result, Err(StoreError::VersionConflict { .. })));
    }

    #[tokio::test]
    async fn tenant_mismatch() {
        let store = temp_store();
        let id = Uuid::new_v4();
        let (ev, snap) = session_created_event(id, "tenant-a", "bot");

        store
            .append(id, &test_auth("tenant-a"), vec![ev], snap)
            .await
            .unwrap();

        let result = store.load(id, &test_auth("tenant-b"));
        assert!(matches!(result, Err(StoreError::TenantMismatch)));
    }

    #[tokio::test]
    async fn session_not_found() {
        let store = temp_store();
        let result = store.load(Uuid::new_v4(), &test_auth("t"));
        assert!(matches!(result, Err(StoreError::SessionNotFound)));
    }

    #[tokio::test]
    async fn list_sessions_empty_store() {
        let store = temp_store();
        let results = store.list_sessions(&SessionFilter::default());
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn list_sessions_returns_all_with_empty_filter() {
        let store = temp_store();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        let (ev1, snap1) = session_created_event(id1, "tenant-a", "bot-1");
        let (ev2, snap2) = session_created_event(id2, "tenant-b", "bot-2");

        store
            .append(id1, &test_auth("tenant-a"), vec![ev1], snap1)
            .await
            .unwrap();
        store
            .append(id2, &test_auth("tenant-b"), vec![ev2], snap2)
            .await
            .unwrap();

        let results = store.list_sessions(&SessionFilter::default());
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn list_sessions_filter_by_tenant() {
        let store = temp_store();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        let (ev1, snap1) = session_created_event(id1, "tenant-a", "bot");
        let (ev2, snap2) = session_created_event(id2, "tenant-b", "bot");

        store
            .append(id1, &test_auth("tenant-a"), vec![ev1], snap1)
            .await
            .unwrap();
        store
            .append(id2, &test_auth("tenant-b"), vec![ev2], snap2)
            .await
            .unwrap();

        let filter = SessionFilter {
            tenant_id: Some("tenant-a".into()),
            ..Default::default()
        };
        let results = store.list_sessions(&filter);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].tenant_id, "tenant-a");
    }

    #[tokio::test]
    async fn list_sessions_filter_by_status() {
        let store = temp_store();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        let (ev1, snap1) = session_created_event(id1, "t", "bot");
        let (ev2, mut snap2) = session_created_event(id2, "t", "bot");
        snap2.status = SessionStatus::Active;

        store
            .append(id1, &test_auth("t"), vec![ev1], snap1)
            .await
            .unwrap();
        store
            .append(id2, &test_auth("t"), vec![ev2], snap2)
            .await
            .unwrap();

        let filter = SessionFilter {
            statuses: Some(vec![SessionStatus::Active]),
            ..Default::default()
        };
        let results = store.list_sessions(&filter);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session_id, id2);
    }

    #[tokio::test]
    async fn list_sessions_filter_by_agent_name() {
        let store = temp_store();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        let (ev1, snap1) = session_created_event(id1, "t", "weather-bot");
        let (ev2, snap2) = session_created_event(id2, "t", "chat-bot");

        store
            .append(id1, &test_auth("t"), vec![ev1], snap1)
            .await
            .unwrap();
        store
            .append(id2, &test_auth("t"), vec![ev2], snap2)
            .await
            .unwrap();

        let filter = SessionFilter {
            agent_name: Some("chat-bot".into()),
            ..Default::default()
        };
        let results = store.list_sessions(&filter);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].agent_name, "chat-bot");
    }

    #[tokio::test]
    async fn list_sessions_summary_fields() {
        let store = temp_store();
        let id = Uuid::new_v4();
        let (ev, snap) = session_created_event(id, "acme", "helper");

        store
            .append(id, &test_auth("acme"), vec![ev], snap)
            .await
            .unwrap();

        let results = store.list_sessions(&SessionFilter::default());
        assert_eq!(results.len(), 1);
        let s = &results[0];
        assert_eq!(s.session_id, id);
        assert_eq!(s.tenant_id, "acme");
        assert_eq!(s.client_id, "client-1");
        assert_eq!(s.agent_name, "helper");
        assert_eq!(s.stream_version, 1);
    }
}
