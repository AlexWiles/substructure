use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use ractor::OutputPort;
use uuid::Uuid;

use crate::domain::event::{Event, SessionAuth};
use crate::domain::session::SessionSnapshot;
use super::store::{EventStore, SessionLoad, StoreError, Version};

struct StoreInner {
    streams: HashMap<Uuid, Vec<Event>>,
    snapshots: HashMap<Uuid, (SessionSnapshot, u64)>, // (snapshot, version_at_snapshot)
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
        snapshot: SessionSnapshot,
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

        if let Some((snapshot, version_at_snapshot)) = inner.snapshots.get(&session_id) {
            // Return snapshot + any events after it
            let remaining = stream[*version_at_snapshot as usize..].to_vec();
            Ok(SessionLoad {
                snapshot: Some(snapshot.clone()),
                events: remaining,
            })
        } else {
            // No snapshot — cold replay fallback
            Ok(SessionLoad {
                snapshot: None,
                events: stream.clone(),
            })
        }
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
