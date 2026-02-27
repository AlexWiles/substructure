use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chrono::Utc;
use ractor::OutputPort;
use uuid::Uuid;

use crate::domain::session::AgentState;

use super::session_index::{SessionFilter, SessionIndex, SessionSummary};
use super::store::{Event, EventBatch, EventStore, StoreError, StreamLoad, Version};

struct StreamEntry {
    snapshot: serde_json::Value,
    stream_version: u64,
    tenant_id: String,
    aggregate_type: String,
}

pub struct InMemoryEventStore {
    streams: Mutex<HashMap<Uuid, StreamEntry>>,
    events: OutputPort<EventBatch>,
}

impl Default for InMemoryEventStore {
    fn default() -> Self {
        InMemoryEventStore {
            streams: Mutex::new(HashMap::new()),
            events: OutputPort::default(),
        }
    }
}

impl InMemoryEventStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl EventStore for InMemoryEventStore {
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
            let mut streams = self.streams.lock().expect("store lock poisoned");

            let actual_version = streams.get(&aggregate_id).map_or(0, |e| e.stream_version);
            if expected_version != actual_version {
                return Err(StoreError::VersionConflict {
                    expected: Version(expected_version),
                    actual: Version(actual_version),
                });
            }

            // Tenant check on existing stream
            if let Some(entry) = streams.get(&aggregate_id) {
                if entry.tenant_id != tenant_id {
                    return Err(StoreError::TenantMismatch);
                }
            }

            let batch: EventBatch = events.into_iter().map(Arc::new).collect();
            streams.insert(
                aggregate_id,
                StreamEntry {
                    snapshot,
                    stream_version: new_version,
                    tenant_id: tenant_id.to_string(),
                    aggregate_type: aggregate_type.to_string(),
                },
            );
            batch
        };

        self.events.send(batch);
        Ok(())
    }

    fn load(&self, aggregate_id: Uuid, tenant_id: &str) -> Result<StreamLoad, StoreError> {
        let streams = self.streams.lock().expect("store lock poisoned");

        let entry = streams
            .get(&aggregate_id)
            .ok_or(StoreError::StreamNotFound)?;

        if entry.tenant_id != tenant_id {
            return Err(StoreError::TenantMismatch);
        }

        Ok(StreamLoad {
            snapshot: entry.snapshot.clone(),
            stream_version: entry.stream_version,
        })
    }

    fn events(&self) -> &OutputPort<EventBatch> {
        &self.events
    }
}

impl SessionIndex for InMemoryEventStore {
    fn list_sessions(&self, filter: &SessionFilter) -> Vec<SessionSummary> {
        let streams = self.streams.lock().expect("store lock poisoned");
        let now = Utc::now();

        streams
            .values()
            .filter(|entry| entry.aggregate_type == "session")
            .filter_map(|entry| {
                let state: AgentState = serde_json::from_value(entry.snapshot.clone()).ok()?;

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
                    token_usage: state.token_usage.total_tokens,
                    stream_version: entry.stream_version,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::agent::{AgentConfig, LlmConfig};
    use crate::domain::aggregate::{Aggregate, DomainEvent};
    use crate::domain::event::{EventPayload, SessionAuth, SessionCreated, SpanContext};
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
    ) -> (Vec<Event>, AgentState) {
        let auth = test_auth(tenant);
        let agent = test_agent(agent_name);
        let domain_event: DomainEvent<AgentState> = DomainEvent {
            id: Uuid::new_v4(),
            tenant_id: tenant.into(),
            aggregate_id: session_id,
            sequence: 0,
            span: SpanContext::root(),
            occurred_at: Utc::now(),
            payload: EventPayload::SessionCreated(SessionCreated {
                agent: agent.clone(),
                auth: auth.clone(),
                on_done: None,
            }),
            derived: None,
        };
        let mut state = AgentState::new(session_id);
        state.apply(&domain_event.payload, domain_event.sequence);
        (vec![domain_event.into_raw()], state)
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
            .append(id1, "tenant-a", "session", ev1, serde_json::to_value(&snap1).unwrap(), 0, snap1.stream_version)
            .await
            .unwrap();
        store
            .append(id2, "tenant-b", "session", ev2, serde_json::to_value(&snap2).unwrap(), 0, snap2.stream_version)
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
            .append(id1, "tenant-a", "session", ev1, serde_json::to_value(&snap1).unwrap(), 0, snap1.stream_version)
            .await
            .unwrap();
        store
            .append(id2, "tenant-b", "session", ev2, serde_json::to_value(&snap2).unwrap(), 0, snap2.stream_version)
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

        let (ev1, snap1) = session_created_event(id1, "t", "bot");
        let (ev2, mut snap2) = session_created_event(id2, "t", "bot");
        snap2.status = SessionStatus::Active;

        store
            .append(id1, "t", "session", ev1, serde_json::to_value(&snap1).unwrap(), 0, snap1.stream_version)
            .await
            .unwrap();
        store
            .append(id2, "t", "session", ev2, serde_json::to_value(&snap2).unwrap(), 0, snap2.stream_version)
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
            .append(id1, "t", "session", ev1, serde_json::to_value(&snap1).unwrap(), 0, snap1.stream_version)
            .await
            .unwrap();
        store
            .append(id2, "t", "session", ev2, serde_json::to_value(&snap2).unwrap(), 0, snap2.stream_version)
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
            .append(id, "acme", "session", ev, serde_json::to_value(&snap).unwrap(), 0, snap.stream_version)
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
