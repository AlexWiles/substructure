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
