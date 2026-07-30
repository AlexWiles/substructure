use serde::{Deserialize, Serialize};

use crate::protocol::{ClientPayload, DecisionResponse, ErrorCode, LlmResponse};
use crate::span::SpanContext;

/// A decision pushed to the engine out-of-band via the submit route: the
/// `session_id`/`decision_id` that route it, wrapping the worker-authored
/// `decision` payload.
#[derive(Debug, Deserialize)]
pub struct SubmitDecisionRequest {
    pub session_id: String,
    pub decision_id: String,
    #[serde(default)]
    pub span: Option<SpanContext>,
    #[serde(flatten)]
    pub decision: DecisionResponse,
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
        /// Any JSON value; a non-string is canonicalized to its JSON text at the route.
        result: serde_json::Value,
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
        /// Omitted ⇒ terminal.
        #[serde(default)]
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
        /// Omitted ⇒ terminal.
        #[serde(default)]
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
    /// Hold a message-shaped payload for the next turn instead of refusing it
    /// when one is already running.
    #[serde(default)]
    pub queue: bool,
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
    /// The turn was taken but has not started: another turn holds the session.
    pub queued: bool,
}

#[derive(Debug, Deserialize)]
pub struct StreamSessionEventsParams {
    #[serde(default)]
    pub turn_id: Option<String>,
    #[serde(default)]
    pub after_seq: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::DecisionAction;

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

    #[test]
    fn a_call_action_parses_without_an_id() {
        // An SDK-less worker may omit the id and let the engine mint one; the
        // decision.result frame must still parse (it did not before, failing
        // with "missing field `id`").
        let body = r#"{
            "messages":[],
            "actions":[
                {"type":"llm.call","model":"m","messages":[],"handler":"server"},
                {"type":"tool.call","name":"t","arguments":"{}","handler":"worker"}
            ]
        }"#;
        let req: DecisionResponse = serde_json::from_str(body).expect("id-less call actions parse");
        assert!(matches!(
            req.actions.as_slice(),
            [
                DecisionAction::CallLlm { .. },
                DecisionAction::CallTool { .. }
            ]
        ));
        for a in &req.actions {
            let id = match a {
                DecisionAction::CallLlm { id, .. } | DecisionAction::CallTool { id, .. } => id,
                _ => unreachable!(),
            };
            assert!(
                id.is_none(),
                "an omitted id parses as None for the engine to mint"
            );
        }
    }

    #[test]
    fn submit_route_flattens_the_decision_and_requires_ids() {
        let body = r#"{"session_id":"s","decision_id":"d","messages":[],"actions":[]}"#;
        let req: SubmitDecisionRequest = serde_json::from_str(body).expect("submit parses");
        assert_eq!(req.session_id, "s");
        assert_eq!(req.decision_id, "d");
        assert!(req.decision.actions.is_empty());

        // The submit route routes by id, so it requires them.
        assert!(serde_json::from_str::<SubmitDecisionRequest>(r#"{"actions":[]}"#).is_err());
    }
}
