use serde::{Deserialize, Serialize};

use crate::protocol::ClientInput;

/// The one client input request body. `input` is the tagged union of everything a client
/// can send, carrying its own addressing (a submit's `agent_id`/`turn_id`; a settle's effect
/// id). `session_id` is the one universal address — minted when absent — and rides the envelope.
#[derive(Debug, Deserialize)]
pub struct ClientInputRequest {
    #[serde(default)]
    pub session_id: Option<String>,
    pub input: ClientInput,
}

#[derive(Debug, Serialize)]
pub struct ClientInputResponse {
    pub session_id: String,
    pub turn_id: String,
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
pub struct StreamSessionEventsParams {
    #[serde(default)]
    pub turn_id: Option<String>,
    #[serde(default)]
    pub after_stream_version: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_input_accepts_a_tool_result() {
        let body = r#"{"input":{"type":"tool.result","id":"tc-1","attempt":0,"result":"x"}}"#;
        assert!(serde_json::from_str::<ClientInputRequest>(body).is_ok());
    }

    #[test]
    fn client_input_rejects_llm_result() {
        // The client surface answers client tools only; `llm.result` is not a
        // `ClientInput` variant, so it must not deserialize.
        let body =
            r#"{"input":{"type":"llm.result","id":"llm-1","attempt":0,"response":{"model":"m"}}}"#;
        assert!(
            serde_json::from_str::<ClientInputRequest>(body).is_err(),
            "the client surface answers client tools only — llm.result must not deserialize"
        );
    }
}
