use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Deserialize)]
pub struct SendMessageRequest {
    pub agent_id: String,
    pub message: String,
    #[serde(default)]
    pub tenant_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<Uuid>,
}

#[derive(Debug, Serialize)]
pub struct SendMessageResponse {
    pub session_id: Uuid,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}
