use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

use crate::runtime::event_store::{AggregateSort, EventStore, StoreError};
use crate::runtime::processor::{
    EventProcessor, EventProcessorRunner, EventProcessorRunnerConfig, ProcessorCheckpointStore,
    ProcessorError,
};
use crate::runtime::session::state::SessionStatus;
use crate::runtime::session::SessionEvent;

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SessionFilter {
    pub tenant_id: Option<String>,
    pub session_id: Option<String>,
    pub agent_id: Option<String>,
    #[serde(default)]
    pub top_level: bool,
    #[serde(default)]
    pub sort: AggregateSort,
    pub limit: Option<usize>,
    pub cursor: Option<SessionCursor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCursor {
    pub at: DateTime<Utc>,
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionPage {
    pub items: Vec<SessionItem>,
    pub next_cursor: Option<SessionCursor>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionItem {
    pub session_id: String,
    pub tenant_id: String,
    pub stream_version: u64,
    pub first_event_at: Option<DateTime<Utc>>,
    pub last_event_at: Option<DateTime<Utc>>,
    pub wake_at: Option<DateTime<Utc>>,
    pub top_level: bool,
    pub agent_id: Option<String>,
    pub cost: Decimal,
    pub sub_agent_cost: Decimal,
    pub status: SessionStatus,
    pub turn_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SessionIndexRecord {
    pub tenant_id: String,
    pub session_id: String,
    pub stream_version: u64,
    pub first_event_at: Option<DateTime<Utc>>,
    pub last_event_at: Option<DateTime<Utc>>,
    pub wake_at: Option<DateTime<Utc>>,
    pub top_level: bool,
    pub agent_id: Option<String>,
    pub cost: Decimal,
    pub sub_agent_cost: Decimal,
    pub status: SessionStatus,
    pub turn_id: Option<String>,
}

#[async_trait]
pub trait SessionIndexStore: Send + Sync {
    async fn list_sessions(&self, filter: &SessionFilter) -> Result<SessionPage, StoreError>;
    async fn count_sessions(&self, filter: &SessionFilter) -> Result<u64, StoreError>;
    async fn upsert_session_index(&self, record: SessionIndexRecord) -> Result<(), String>;
}

struct SessionIndexProjection {
    store: Arc<dyn SessionIndexStore>,
}

impl SessionIndexProjection {
    fn new(store: Arc<dyn SessionIndexStore>) -> Self {
        Self { store }
    }
}

#[async_trait::async_trait]
impl EventProcessor for SessionIndexProjection {
    fn name(&self) -> &'static str {
        "session_index_v1"
    }

    async fn apply(&self, event: SessionEvent) -> Result<(), ProcessorError> {
        let derived = event.derived.ok_or_else(|| {
            ProcessorError::Apply("missing derived state for session event".into())
        })?;

        let record = SessionIndexRecord {
            tenant_id: event.tenant_id,
            session_id: event.aggregate_id,
            stream_version: event.sequence,
            first_event_at: Some(event.occurred_at),
            last_event_at: Some(event.occurred_at),
            wake_at: derived.wake_at,
            top_level: derived.ancestry.is_empty(),
            agent_id: derived.agent_id,
            cost: derived.cost,
            sub_agent_cost: derived.sub_agent_cost,
            status: derived.status,
            turn_id: derived.turn_id,
        };

        self.store
            .upsert_session_index(record)
            .await
            .map_err(ProcessorError::Apply)
    }
}

pub fn spawn_session_index_processor(
    event_store: Arc<dyn EventStore>,
    checkpoint_store: Arc<dyn ProcessorCheckpointStore>,
    session_index_store: Arc<dyn SessionIndexStore>,
    cancel: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    let projection = Arc::new(SessionIndexProjection::new(session_index_store));
    let mut config = EventProcessorRunnerConfig::default();
    config.batch_size = 512;
    config.owner_id = Some("session_index".to_string());
    EventProcessorRunner::new(event_store, checkpoint_store, projection, config, cancel).spawn()
}
