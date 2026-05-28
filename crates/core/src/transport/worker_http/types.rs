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
#[serde(tag = "type")]
pub enum SubmitToolCallResultRequest {
    #[serde(rename = "return.tool.result")]
    Result {
        tool_call_id: String,
        result: String,
        attempt: u32,
    },
    #[serde(rename = "return.tool.error")]
    Error {
        tool_call_id: String,
        error: String,
        retryable: bool,
        attempt: u32,
    },
}

#[derive(Debug, Deserialize)]
pub struct MintClientTokenRequest {
    pub identity: MintClientTokenIdentity,
    #[serde(default)]
    pub ttl_seconds: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct MintClientTokenIdentity {
    pub id: String,
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,
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
    pub identity: SubmitClientPayloadIdentity,
}

#[derive(Debug, Deserialize)]
pub struct SubmitClientPayloadIdentity {
    pub id: String,
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,
}

#[derive(Debug, Serialize)]
pub struct SubmitClientPayloadResponse {
    pub session_id: String,
    pub turn_id: String,
}

#[derive(Debug, Deserialize)]
pub struct StreamSessionEventsParams {
    #[serde(default)]
    pub turn_id: Option<String>,
    #[serde(default)]
    pub sequence_after: Option<u64>,
}
