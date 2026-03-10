use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::runtime::session::command::WorkerAction;
use crate::runtime::session::decision::DecisionTrigger;
use crate::runtime::serde_helpers::base64_bytes;
use crate::runtime::span::SpanContext;

#[derive(Debug, Deserialize)]
pub struct PollRequest {
    pub tenant_id: String,
    pub agent_ids: Vec<String>,
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
}

fn default_timeout_ms() -> u64 {
    30_000
}

#[derive(Debug, Serialize)]
pub struct PollResponse {
    pub session_id: Uuid,
    pub tenant_id: String,
    pub decision_id: String,
    pub agent_id: String,
    pub trigger: DecisionTrigger,
    #[serde(with = "base64_bytes")]
    pub worker_state: Vec<u8>,
    pub span: SpanContext,
    pub attempts: u32,
    pub deadline: Option<DateTime<Utc>>,
}

#[derive(Debug, Deserialize)]
pub struct SubmitRequest {
    pub session_id: Uuid,
    pub tenant_id: String,
    pub decision_id: String,
    pub actions: Vec<WorkerAction>,
    #[serde(with = "base64_bytes")]
    pub state: Vec<u8>,
    #[serde(default)]
    pub span: Option<SpanContext>,
}

#[derive(Debug, Serialize)]
pub struct SubmitResponse {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}
