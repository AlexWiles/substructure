use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::agent::AgentConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmClientConfig {
    #[serde(rename = "type")]
    pub client_type: String,
    #[serde(flatten, default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub settings: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    pub event_store: EventStoreConfig,
    #[serde(default)]
    pub llm_clients: HashMap<String, LlmClientConfig>,
    #[serde(default)]
    pub agents: HashMap<String, AgentConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum EventStoreConfig {
    #[serde(rename = "in_memory")]
    InMemory,
}
