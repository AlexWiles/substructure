use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use ractor::OutputPort;
use uuid::Uuid;

use crate::domain::event::{Event, SessionAuth};
use crate::domain::session::AgentState;
use chrono::Utc;

use super::store::{EventStore, SessionFilter, SessionLoad, SessionSummary, StoreError, Version};

struct StoreInner {
    streams: HashMap<Uuid, Vec<Event>>,
    snapshots: HashMap<Uuid, (AgentState, u64)>, // (snapshot, version_at_snapshot)
    log: Vec<Arc<Event>>,
}

pub struct InMemoryEventStore {
    inner: Mutex<StoreInner>,
    notify: OutputPort<()>,
}

impl InMemoryEventStore {
    pub fn new() -> Self {
        InMemoryEventStore {
            inner: Mutex::new(StoreInner {
                streams: HashMap::new(),
                snapshots: HashMap::new(),
                log: Vec::new(),
            }),
            notify: OutputPort::default(),
        }
    }
}

#[async_trait]
impl EventStore for InMemoryEventStore {
    async fn append(
        &self,
        session_id: Uuid,
        auth: &SessionAuth,
        expected_version: Version,
        events: Vec<Event>,
        snapshot: AgentState,
    ) -> Result<(), StoreError> {
        {
            let mut inner = self.inner.lock().expect("store lock poisoned");

            let stream_len = inner.streams.entry(session_id).or_default().len();
            let actual = Version(stream_len as u64);
            if expected_version != actual {
                return Err(StoreError::VersionConflict {
                    expected: expected_version,
                    actual,
                });
            }

            // Tenant check on existing events
            if let Some(first) = inner.streams.get(&session_id).and_then(|s| s.first()) {
                if first.tenant_id != auth.tenant_id {
                    return Err(StoreError::TenantMismatch);
                }
            }

            for event in &events {
                inner.log.push(Arc::new(event.clone()));
            }
            let new_version = stream_len as u64 + events.len() as u64;
            inner.streams.entry(session_id).or_default().extend(events);

            // Store snapshot at the new version
            inner.snapshots.insert(session_id, (snapshot, new_version));
        }

        self.notify.send(());
        Ok(())
    }

    fn load(&self, session_id: Uuid, auth: &SessionAuth) -> Result<SessionLoad, StoreError> {
        let inner = self.inner.lock().expect("store lock poisoned");

        // Verify session exists
        let stream = inner
            .streams
            .get(&session_id)
            .ok_or(StoreError::SessionNotFound)?;

        // Verify tenant
        if let Some(first) = stream.first() {
            if first.tenant_id != auth.tenant_id {
                return Err(StoreError::TenantMismatch);
            }
        }

        let (snapshot, _version) = inner
            .snapshots
            .get(&session_id)
            .ok_or(StoreError::SessionNotFound)?;

        Ok(SessionLoad {
            snapshot: snapshot.clone(),
        })
    }

    fn list_sessions(&self, filter: &SessionFilter) -> Vec<SessionSummary> {
        let inner = self.inner.lock().expect("store lock poisoned");
        let now = Utc::now();

        inner
            .snapshots
            .values()
            .filter_map(|(state, _version_at)| {
                let auth = state.auth.as_ref()?;
                let agent = state.agent.as_ref()?;

                // Apply filters
                if let Some(ref tid) = filter.tenant_id {
                    if &auth.tenant_id != tid {
                        return None;
                    }
                }
                if let Some(ref statuses) = filter.statuses {
                    if !statuses.contains(&state.status) {
                        return None;
                    }
                }
                if let Some(ref name) = filter.agent_name {
                    if &agent.name != name {
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
                    token_usage: state.tokens.used,
                    stream_version: state.stream_version,
                })
            })
            .collect()
    }

    fn read_from(&self, offset: u64, limit: usize) -> Vec<Arc<Event>> {
        let inner = self.inner.lock().expect("store lock poisoned");
        let start = offset as usize;
        if start >= inner.log.len() {
            return Vec::new();
        }
        let end = (start + limit).min(inner.log.len());
        inner.log[start..end].to_vec()
    }

    fn notify(&self) -> &OutputPort<()> {
        &self.notify
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
            occurred_at: Utc::now(),
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

    #[tokio::test]
    async fn list_sessions_empty_store() {
        let store = InMemoryEventStore::new();
        let results = store.list_sessions(&SessionFilter::default());
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn list_sessions_returns_all_with_empty_filter() {
        let store = InMemoryEventStore::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        let (ev1, snap1) = session_created_event(id1, "tenant-a", "bot-1");
        let (ev2, snap2) = session_created_event(id2, "tenant-b", "bot-2");

        store
            .append(id1, &test_auth("tenant-a"), Version(0), vec![ev1], snap1)
            .await
            .unwrap();
        store
            .append(id2, &test_auth("tenant-b"), Version(0), vec![ev2], snap2)
            .await
            .unwrap();

        let results = store.list_sessions(&SessionFilter::default());
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn list_sessions_filter_by_tenant() {
        let store = InMemoryEventStore::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        let (ev1, snap1) = session_created_event(id1, "tenant-a", "bot");
        let (ev2, snap2) = session_created_event(id2, "tenant-b", "bot");

        store
            .append(id1, &test_auth("tenant-a"), Version(0), vec![ev1], snap1)
            .await
            .unwrap();
        store
            .append(id2, &test_auth("tenant-b"), Version(0), vec![ev2], snap2)
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
        let store = InMemoryEventStore::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        // session_created sets status to Done via apply_core (SessionCreated doesn't change status,
        // it stays at the initial Done). We'll make one Active by appending an LlmCallRequested.
        let (ev1, snap1) = session_created_event(id1, "t", "bot");
        let (ev2, mut snap2) = session_created_event(id2, "t", "bot");
        snap2.status = SessionStatus::Active;

        store
            .append(id1, &test_auth("t"), Version(0), vec![ev1], snap1)
            .await
            .unwrap();
        store
            .append(id2, &test_auth("t"), Version(0), vec![ev2], snap2)
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
        let store = InMemoryEventStore::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        let (ev1, snap1) = session_created_event(id1, "t", "weather-bot");
        let (ev2, snap2) = session_created_event(id2, "t", "chat-bot");

        store
            .append(id1, &test_auth("t"), Version(0), vec![ev1], snap1)
            .await
            .unwrap();
        store
            .append(id2, &test_auth("t"), Version(0), vec![ev2], snap2)
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
        let store = InMemoryEventStore::new();
        let id = Uuid::new_v4();
        let (ev, snap) = session_created_event(id, "acme", "helper");

        store
            .append(id, &test_auth("acme"), Version(0), vec![ev], snap)
            .await
            .unwrap();

        let results = store.list_sessions(&SessionFilter::default());
        assert_eq!(results.len(), 1);
        let s = &results[0];
        assert_eq!(s.session_id, id);
        assert_eq!(s.tenant_id, "acme");
        assert_eq!(s.client_id, "client-1");
        assert_eq!(s.agent_name, "helper");
        assert_eq!(s.stream_version, 1); // one event applied
    }
}
