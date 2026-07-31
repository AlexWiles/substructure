//! `/api/v1` wire types. Field names and casing are the contract: the local
//! server serializes these, the CLI deserializes them, and they must match
//! what the hosted cloud sends.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Org {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub role: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct App {
    pub id: String,
    pub organization_id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub balance_usd: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_count: Option<i64>,
}

/// What a deployment says about itself, so the CLI can degrade with a real
/// message rather than a 404.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Meta {
    /// One org and one app, advertised on every response. `subs link` adopts
    /// them and pickers are skipped.
    #[serde(default)]
    pub single_tenant: bool,
    #[serde(default)]
    pub features: Vec<String>,
}

impl Meta {
    pub fn has(&self, feature: &str) -> bool {
        self.features.iter().any(|f| f == feature)
    }
}

/// An MCP connection an org has authorized.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpConnection {
    pub id: String,
    /// The id an agent config names.
    pub connection_id: String,
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub scopes: String,
    #[serde(default)]
    pub granted_apps: Vec<String>,
}

/// Declare a connection: the id an agent config names and the URL it points
/// at. Inert until a human consents; whether this deployment will send a
/// credential to that URL at all is the deployment's policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpDeclareRequest {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connection_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpAuthorizeResponse {
    pub authorize_url: String,
    #[serde(default)]
    pub scope: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpGrantRequest {
    pub app_id: String,
}

/// An app's configuration as the deployment holds it: what a manifest says,
/// plus the state only the deployment knows.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppConfig {
    pub name: String,
    #[serde(default)]
    pub worker: ConfigWorker,
    #[serde(default)]
    pub mcp: Vec<ConfigConnection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_applied: Option<ConfigApplied>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConfigWorker {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default)]
    pub state: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConfigConnection {
    pub id: String,
    pub url: String,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub granted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConfigApplied {
    pub seq: i64,
    pub created_at: String,
    #[serde(default)]
    pub actor_email: Option<String>,
}

/// The manifest `subs apply` pushes. An absent field says nothing about that
/// setting rather than unsetting it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConfigUpdate {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker: Option<ConfigWorkerUpdate>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp: Option<Vec<ConfigConnectionRef>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConfigWorkerUpdate {
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConfigConnectionRef {
    pub id: String,
    pub url: String,
}

/// One entry in an app's configuration history.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConfigEvent {
    pub seq: i64,
    pub kind: String,
    #[serde(default)]
    pub data: serde_json::Value,
    #[serde(default)]
    pub actor_email: Option<String>,
    #[serde(default)]
    pub source: String,
    pub created_at: String,
}

/// What an apply did. Empty `changes` means the document already held.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApplyResponse {
    pub app_id: String,
    #[serde(default)]
    pub changes: Vec<ConfigEvent>,
}

/// A cursor-paged slice. The wire shape is snake_case, unlike the camelCase
/// bodies around it, because it is the store's `Page<T>` verbatim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Page<T> {
    pub items: Vec<T>,
    #[serde(default)]
    pub next_cursor: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiError {
    pub error: ErrorDetail,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ErrorDetail {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

impl ApiError {
    pub fn new(code: &str, message: &str) -> Self {
        Self {
            error: ErrorDetail {
                code: Some(code.to_string()),
                message: Some(message.to_string()),
            },
        }
    }

    pub fn message(message: impl Into<String>) -> Self {
        Self {
            error: ErrorDetail {
                code: None,
                message: Some(message.into()),
            },
        }
    }
}
