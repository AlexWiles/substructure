use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::event_store::{AggregateSummary, AggregateSort, StoreError};
use super::session::state::SessionState;

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SessionFilter {
    pub tenant_id: Option<String>,
    #[serde(default)]
    pub top_level: bool,
    #[serde(default)]
    pub sort: AggregateSort,
    pub limit: Option<usize>,
    pub cursor: Option<SessionCursor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCursor {
    pub last_event_at: DateTime<Utc>,
    pub aggregate_id: Uuid,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionPage {
    pub items: Vec<SessionItem>,
    pub next_cursor: Option<SessionCursor>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionItem {
    pub summary: AggregateSummary,
    pub state: SessionState,
}

#[async_trait]
pub trait SessionIndex: Send + Sync {
    async fn list_sessions(&self, filter: &SessionFilter) -> Result<SessionPage, StoreError>;
}
