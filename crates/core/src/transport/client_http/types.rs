use serde::{Deserialize, Serialize};

use crate::runtime::session::decision::ClientPayload;

#[derive(Debug, Deserialize)]
pub struct SubmitClientPayloadRequest {
    pub agent_id: String,
    pub payload: ClientPayload,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub turn_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SubmitClientPayloadResponse {
    pub session_id: String,
    pub turn_id: String,
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

#[derive(Debug, Serialize)]
pub struct SubmitToolCallResultResponse {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct InterruptSessionRequest {
    #[serde(default)]
    pub interrupt_id: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub payload: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct InterruptSessionResponse {
    pub ok: bool,
    pub interrupt_id: String,
}

#[derive(Debug, Deserialize)]
pub struct ResumeInterruptRequest {
    pub interrupt_id: String,
    #[serde(default)]
    pub payload: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct ResumeInterruptResponse {
    pub ok: bool,
}

#[derive(Debug, Deserialize)]
pub struct StreamSessionEventsParams {
    #[serde(default)]
    pub turn_id: Option<String>,
    #[serde(default)]
    pub sequence_after: Option<u64>,
}
