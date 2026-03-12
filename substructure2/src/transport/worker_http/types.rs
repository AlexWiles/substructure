use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::runtime::session::command::WorkerAction;
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

#[derive(Debug, Deserialize)]
pub struct RegisterRequest {
    pub tenant_id: String,
    pub agent_ids: Vec<String>,
    pub transport_type: String,
    pub config: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct RegisterResponse {
    pub ok: bool,
}
