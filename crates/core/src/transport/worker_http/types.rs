use serde::{Deserialize, Serialize};

use crate::llm::{ErrorCode, LlmResponse};
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
    /// Omitted or `null` keeps the current state; clear with a non-null empty value.
    #[serde(default)]
    pub state: Option<WorkerState>,
    #[serde(default)]
    pub span: Option<SpanContext>,
}

#[derive(Debug, Serialize)]
pub struct SubmitResponse {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Settles a worker-handled tool or llm call.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum SettleEffectRequest {
    #[serde(rename = "tool.result")]
    ToolResult {
        id: String,
        #[serde(default)]
        attempt: Option<u32>,
        result: String,
    },
    #[serde(rename = "llm.result")]
    LlmResult {
        id: String,
        #[serde(default)]
        attempt: Option<u32>,
        response: LlmResponse,
    },
    #[serde(rename = "tool.error")]
    ToolError {
        id: String,
        error: String,
        retryable: bool,
        #[serde(default)]
        attempt: Option<u32>,
        #[serde(default)]
        code: Option<ErrorCode>,
        #[serde(default)]
        detail: Option<serde_json::Value>,
    },
    #[serde(rename = "llm.error")]
    LlmError {
        id: String,
        error: String,
        retryable: bool,
        #[serde(default)]
        attempt: Option<u32>,
        #[serde(default)]
        code: Option<ErrorCode>,
        #[serde(default)]
        detail: Option<serde_json::Value>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_surface_settles_an_llm_result() {
        let body = r#"{"type":"llm.result","id":"llm-1","attempt":0,"response":{"model":"m"}}"#;
        let req: SettleEffectRequest = serde_json::from_str(body).expect("llm.result deserializes");
        match req {
            SettleEffectRequest::LlmResult { id, .. } => assert_eq!(id, "llm-1"),
            other => panic!("expected an llm.result; got {other:?}"),
        }
    }

    #[test]
    fn worker_surface_settles_a_tool_error() {
        let body =
            r#"{"type":"tool.error","id":"tc-1","attempt":1,"error":"boom","retryable":true}"#;
        let req: SettleEffectRequest = serde_json::from_str(body).expect("tool.error deserializes");
        assert!(matches!(req, SettleEffectRequest::ToolError { .. }));
    }
}
