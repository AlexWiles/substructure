use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chrono::Utc;
use ractor::OutputPort;
use uuid::Uuid;

use crate::domain::event::{Event, EventPayload, SpanContext};
use super::store::{EventStore, StoreError, Version};

struct StoreInner {
    streams: HashMap<Uuid, Vec<Event>>,
    log: Vec<Arc<Event>>,
    next_sequence: u64,
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
                log: Vec::new(),
                next_sequence: 0,
            }),
            notify: OutputPort::default(),
        }
    }

    pub fn read_from(&self, offset: u64, limit: usize) -> Vec<Arc<Event>> {
        let inner = self.inner.lock().expect("store lock poisoned");
        let start = offset as usize;
        if start >= inner.log.len() {
            return Vec::new();
        }
        let end = (start + limit).min(inner.log.len());
        inner.log[start..end].to_vec()
    }

    pub fn notify(&self) -> &OutputPort<()> {
        &self.notify
    }
}

#[async_trait]
impl EventStore for InMemoryEventStore {
    async fn append(
        &self,
        session_id: Uuid,
        expected_version: Version,
        span: SpanContext,
        payloads: Vec<EventPayload>,
    ) -> Result<Vec<Event>, StoreError> {
        let events = {
            let mut inner = self.inner.lock().expect("store lock poisoned");

            let stream_len = inner.streams.entry(session_id).or_default().len();
            let actual = Version(stream_len as u64);
            if expected_version != actual {
                return Err(StoreError::VersionConflict {
                    expected: expected_version,
                    actual,
                });
            }

            let base_seq = inner.next_sequence;
            let events: Vec<Event> = payloads
                .into_iter()
                .enumerate()
                .map(|(i, payload)| Event {
                    id: Uuid::new_v4(),
                    session_id,
                    sequence: base_seq + i as u64,
                    span: span.clone(),
                    occurred_at: Utc::now(),
                    payload,
                })
                .collect();

            inner.next_sequence = base_seq + events.len() as u64;
            for event in &events {
                inner.log.push(Arc::new(event.clone()));
            }
            inner.streams.entry(session_id).or_default().extend(events.clone());

            events
        };

        self.notify.send(());
        Ok(events)
    }

    fn load(&self, session_id: Uuid) -> Result<Vec<Event>, StoreError> {
        let inner = self.inner.lock().expect("store lock poisoned");
        inner
            .streams
            .get(&session_id)
            .cloned()
            .ok_or(StoreError::SessionNotFound)
    }

    fn version(&self, session_id: Uuid) -> Version {
        let inner = self.inner.lock().expect("store lock poisoned");
        inner
            .streams
            .get(&session_id)
            .map(|s| Version(s.len() as u64))
            .unwrap_or(Version::initial())
    }
}
