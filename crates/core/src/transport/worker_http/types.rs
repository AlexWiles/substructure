use serde::{Deserialize, Serialize};

use crate::serde_helpers::base64_bytes;
use crate::session::decision::WorkerAction;
use crate::span::SpanContext;

#[derive(Debug, Deserialize)]
pub struct SubmitRequest {
    pub session_id: String,
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
    pub agent_ids: Vec<String>,
    pub transport_type: String,
    pub config: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct RegisterResponse {
    pub ok: bool,
}
