use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::event::McpServerConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub client: String,
    #[serde(flatten, default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub params: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    #[serde(default = "Uuid::new_v4")]
    pub id: Uuid,
    pub name: String,
    pub llm: LlmConfig,
    pub system_prompt: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mcp_servers: Vec<McpServerConfig>,
}
