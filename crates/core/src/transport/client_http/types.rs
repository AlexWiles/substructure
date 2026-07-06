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

/// Clients answer client tools only, so only `tool.*` settles deserialize.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum SettleEffectRequest {
    #[serde(rename = "tool.result")]
    Result {
        id: String,
        result: String,
        #[serde(default)]
        attempt: Option<u32>,
    },
    #[serde(rename = "tool.error")]
    Error {
        id: String,
        error: String,
        retryable: bool,
        #[serde(default)]
        attempt: Option<u32>,
    },
}

#[derive(Debug, Serialize)]
pub struct SettleEffectResponse {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_surface_accepts_tool_result() {
        let body = r#"{"type":"tool.result","id":"tc-1","attempt":0,"result":"x"}"#;
        assert!(serde_json::from_str::<SettleEffectRequest>(body).is_ok());
    }

    #[test]
    fn client_surface_rejects_llm_result() {
        let body = r#"{"type":"llm.result","id":"llm-1","attempt":0,"response":{"model":"m"}}"#;
        assert!(
            serde_json::from_str::<SettleEffectRequest>(body).is_err(),
            "the client surface answers client tools only — llm.result must not deserialize"
        );
    }
}
