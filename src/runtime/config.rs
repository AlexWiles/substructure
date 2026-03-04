use std::collections::HashMap;

use chrono::Duration;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::defaults;

// ---------------------------------------------------------------------------
// Per-call LLM request parameters
// ---------------------------------------------------------------------------

/// Per-call overrides for LLM request parameters.
/// Fields set here take precedence over `LlmConfig` agent defaults.
#[derive(Debug, Clone, Default)]
pub struct LlmRequestParams {
    pub max_completion_tokens: Option<u64>,
    pub temperature: Option<f64>,
}

// ---------------------------------------------------------------------------
// Agent configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub client: String,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "RetryConfig::is_empty")]
    pub retry: RetryConfig,
    #[serde(flatten, default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub params: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    #[serde(default = "StrategyConfig::default_kind")]
    pub kind: String,
    #[serde(flatten, default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub params: serde_json::Map<String, serde_json::Value>,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            kind: Self::default_kind(),
            params: Default::default(),
        }
    }
}

impl StrategyConfig {
    fn default_kind() -> String {
        "default".into()
    }
}

/// Retry/timeout overrides — all fields optional (inherit from parent layer).
///
/// Used on `LlmConfig` and `McpServerConfig`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetryConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_secs: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backoff_base_secs: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backoff_max_secs: Option<u32>,
}

impl RetryConfig {
    /// True when all fields are `None` (for `skip_serializing_if`).
    pub fn is_empty(&self) -> bool {
        self.timeout_secs.is_none()
            && self.max_retries.is_none()
            && self.backoff_base_secs.is_none()
            && self.backoff_max_secs.is_none()
    }

    /// Resolve against a single base layer, filling gaps from `defaults`.
    pub fn resolve(&self, defaults: &RetryPolicy) -> RetryPolicy {
        RetryPolicy {
            timeout_secs: self.timeout_secs.unwrap_or(defaults.timeout_secs),
            max_retries: self.max_retries.unwrap_or(defaults.max_retries),
            backoff_base_secs: self.backoff_base_secs.unwrap_or(defaults.backoff_base_secs),
            backoff_max_secs: self.backoff_max_secs.unwrap_or(defaults.backoff_max_secs),
        }
    }

}

/// Fully-resolved retry policy — no optional fields. Stored on call state and
/// read directly by retry logic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub timeout_secs: u32,
    pub max_retries: u32,
    pub backoff_base_secs: u32,
    pub backoff_max_secs: u32,
}

impl RetryPolicy {
    pub const LLM_DEFAULTS: RetryPolicy = RetryPolicy {
        timeout_secs: defaults::LLM_TIMEOUT_SECS,
        max_retries: defaults::MAX_RETRIES,
        backoff_base_secs: defaults::BACKOFF_BASE_SECS,
        backoff_max_secs: defaults::BACKOFF_MAX_SECS,
    };

    pub const TOOL_DEFAULTS: RetryPolicy = RetryPolicy {
        timeout_secs: defaults::TOOL_TIMEOUT_SECS,
        max_retries: defaults::MAX_RETRIES,
        backoff_base_secs: defaults::BACKOFF_BASE_SECS,
        backoff_max_secs: defaults::BACKOFF_MAX_SECS,
    };
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    #[serde(default = "Uuid::new_v4")]
    pub id: Uuid,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub llm: LlmConfig,
    pub system_prompt: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mcp_servers: Vec<McpServerConfig>,
    #[serde(default)]
    pub strategy: StrategyConfig,
    /// Maximum context window size in tokens. Used for budget reservation
    /// estimates and context utilization tracking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_context_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sub_agents: Vec<String>,
    /// Maximum tool result size in bytes. `None` = inherit, `Some(0)` = no limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_result_max_bytes: Option<usize>,
}

// ---------------------------------------------------------------------------
// System configuration
// ---------------------------------------------------------------------------

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
    #[serde(default, rename = "budget_policies")]
    pub budgets: Vec<BudgetPolicyConfig>,
    #[serde(default)]
    pub auth: AuthConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub otel: Option<OtelConfig>,
    /// Maximum tool result size in bytes. `None` = inherit, `Some(0)` = no limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_result_max_bytes: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtelConfig {
    pub endpoint: String,
    #[serde(default = "OtelConfig::default_service_name")]
    pub service_name: String,
}

impl OtelConfig {
    fn default_service_name() -> String {
        "substructure".into()
    }
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(tag = "type")]
pub enum AuthConfig {
    #[default]
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
    /// The usage key to track. Dot-separated path into the provider's usage JSON,
    /// e.g. "prompt_tokens", "cost", "completion_tokens_details.reasoning_tokens".
    pub dimension: String,
    /// The type of value this dimension represents. Determines how raw JSON values
    /// are interpreted: `tokens` stores integers as-is, `micro_dollars` stores
    /// floats as micro-units (x 1_000_000) so $10 limit = 10_000_000.
    #[serde(default)]
    pub value_type: BudgetValueType,
    pub limit: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<String>,
    #[serde(default)]
    pub strategy: ExhaustionStrategy,
    #[serde(default, rename = "match", skip_serializing_if = "Option::is_none")]
    pub match_conditions: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetValueType {
    /// Integer token counts, stored as-is.
    #[default]
    Tokens,
    /// Dollar amounts stored as micro-units (x 1_000_000).
    /// A raw float 0.95 becomes 950_000. Limit of $10 = 10_000_000.
    MicroDollars,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExhaustionStrategy {
    #[default]
    Reject,
    Interrupt,
}

// ---------------------------------------------------------------------------
// MCP server configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    pub name: String,
    pub transport: McpTransportConfig,
    /// Maximum tool result size in bytes. `None` = inherit, `Some(0)` = no limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_result_max_bytes: Option<usize>,
    /// Per-server retry/timeout overrides (inherits from agent when `None`).
    #[serde(default, skip_serializing_if = "RetryConfig::is_empty")]
    pub retry: RetryConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum McpTransportConfig {
    #[serde(rename = "stdio")]
    Stdio { command: String, args: Vec<String> },
}

// ---------------------------------------------------------------------------
// Client identity
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientIdentity {
    pub tenant_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sub: Option<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub attrs: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
