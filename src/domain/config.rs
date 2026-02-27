use std::collections::HashMap;

use chrono::Duration;
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
pub struct SecretProviderConfig {
    #[serde(rename = "type")]
    pub provider_type: String,
    #[serde(flatten, default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub settings: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    pub event_store: EventStoreConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
    #[serde(default)]
    pub secret_providers: HashMap<String, SecretProviderConfig>,
    #[serde(default)]
    pub llm_clients: HashMap<String, LlmClientConfig>,
    #[serde(default)]
    pub agents: HashMap<String, AgentConfig>,
    #[serde(default)]
    pub budgets: Vec<BudgetPolicyConfig>,
    #[serde(default)]
    pub auth: AuthConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Default log filter directive (e.g. "info", "debug", "substructure=debug,tower_http=info").
    /// Overridden by `RUST_LOG` env var when set.
    #[serde(default = "LoggingConfig::default_level")]
    pub level: String,
    /// Log output format: "compact" (default), "full", "pretty", or "json".
    #[serde(default = "LoggingConfig::default_format")]
    pub format: String,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: Self::default_level(),
            format: Self::default_format(),
        }
    }
}

impl LoggingConfig {
    fn default_level() -> String {
        "info".into()
    }
    fn default_format() -> String {
        "compact".into()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AuthConfig {
    #[serde(rename = "none")]
    None,
    #[serde(rename = "token")]
    Token {
        signing_secret: String,
        #[serde(default = "default_token_ttl")]
        token_ttl: String,
        #[serde(default)]
        tenants: Vec<TenantConfig>,
    },
}

impl Default for AuthConfig {
    fn default() -> Self {
        AuthConfig::None
    }
}

fn default_token_ttl() -> String {
    "1h".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantConfig {
    pub tenant_id: String,
    pub api_key_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum EventStoreConfig {
    #[serde(rename = "sqlite")]
    Sqlite { path: String },
}

// ---------------------------------------------------------------------------
// Budget policy configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetPolicyConfig {
    pub name: String,
    pub group_by: Vec<String>,
    pub dimension: BudgetDimension,
    pub limit: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<String>,
    #[serde(default)]
    pub strategy: ExhaustionStrategy,
    #[serde(default, rename = "match", skip_serializing_if = "Option::is_none")]
    pub match_conditions: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetDimension {
    TotalTokens,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExhaustionStrategy {
    #[default]
    Reject,
    Interrupt,
}

/// Parse a human-readable window string into a `chrono::Duration`.
///
/// Supported suffixes: `m` (minutes), `h` (hours), `d` (days).
/// Examples: "5m", "1h", "30d".
pub fn parse_window(s: &str) -> Option<Duration> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let (digits, suffix) = s.split_at(s.len() - 1);
    let n: i64 = digits.parse().ok()?;
    match suffix {
        "m" => Some(Duration::minutes(n)),
        "h" => Some(Duration::hours(n)),
        "d" => Some(Duration::days(n)),
        _ => None,
    }
}
