use serde::{Deserialize, Serialize};

use crate::runtime::session::decision::ClientPayload;
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
    pub transport_type: String,
    pub config: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct RegisterResponse {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signing_secret: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MintClientTokenRequest {
    pub tenant_id: String,
    pub sub: String,
    #[serde(default)]
    pub attrs: std::collections::HashMap<String, String>,
    #[serde(default)]
    pub ttl_seconds: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct MintClientTokenResponse {
    pub token: String,
    pub expires_at: i64,
}

#[derive(Debug, Deserialize)]
pub struct SubmitClientPayloadRequest {
    pub agent_id: String,
    pub payload: ClientPayload,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub turn_id: Option<String>,
    pub auth: SubmitClientPayloadAuth,
}

#[derive(Debug, Deserialize)]
pub struct SubmitClientPayloadAuth {
    pub tenant_id: String,
    pub sub: String,
    #[serde(default)]
    pub attrs: std::collections::HashMap<String, String>,
}
