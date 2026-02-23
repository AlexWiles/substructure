use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::event::McpServerConfig;
use super::session::StrategyKind;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub client: String,
    #[serde(flatten, default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub params: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    #[serde(default = "RetryConfig::default_llm_timeout_secs")]
    pub llm_timeout_secs: u64,
    #[serde(default = "RetryConfig::default_tool_timeout_secs")]
    pub tool_timeout_secs: u64,
    #[serde(default = "RetryConfig::default_max_retries")]
    pub max_retries: u32,
    #[serde(default = "RetryConfig::default_backoff_base_secs")]
    pub backoff_base_secs: u64,
    #[serde(default = "RetryConfig::default_backoff_max_secs")]
    pub backoff_max_secs: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        RetryConfig {
            llm_timeout_secs: 60,
            tool_timeout_secs: 120,
            max_retries: 3,
            backoff_base_secs: 2,
            backoff_max_secs: 60,
        }
    }
}

impl RetryConfig {
    fn default_llm_timeout_secs() -> u64 { 60 }
    fn default_tool_timeout_secs() -> u64 { 120 }
    fn default_max_retries() -> u32 { 3 }
    fn default_backoff_base_secs() -> u64 { 2 }
    fn default_backoff_max_secs() -> u64 { 60 }
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
    #[serde(default)]
    pub strategy: StrategyKind,
    #[serde(default)]
    pub retry: RetryConfig,
}
