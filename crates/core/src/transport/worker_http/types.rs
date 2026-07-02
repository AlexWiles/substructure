use serde::{Deserialize, Serialize};

use crate::runtime::session::decision::ClientPayload;
use crate::runtime::session::message::Message;
use crate::session::decision::WorkerAction;
use crate::span::SpanContext;
use crate::worker::WorkerState;

#[derive(Debug, Deserialize)]
pub struct SubmitRequest {
    pub session_id: String,
    pub decision_id: String,
    #[serde(default)]
    pub transcript: Vec<Message>,
    pub actions: Vec<WorkerAction>,
    pub state: WorkerState,
    #[serde(default)]
    pub span: Option<SpanContext>,
}

#[derive(Debug, Serialize)]
pub struct SubmitResponse {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// This endpoint settles tool calls only; typing `kind` as a unit enum makes
/// serde reject any other effect kind.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallKind {
    ToolCall,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum SubmitToolCallResultRequest {
    #[serde(rename = "effect.result")]
    Result {
        #[allow(dead_code)]
        kind: ToolCallKind,
        id: String,
        result: String,
        attempt: u32,
    },
    #[serde(rename = "effect.error")]
    Error {
        #[allow(dead_code)]
        kind: ToolCallKind,
        id: String,
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
