use serde::Deserialize;
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
