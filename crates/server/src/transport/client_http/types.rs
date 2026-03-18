use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct SendMessageRequest {
    pub agent_id: String,
    pub message: String,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub turn_id: Option<String>,
}
