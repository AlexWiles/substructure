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

/// Current webhook (worker) config. Snake-case on the wire, unlike the
/// camel-case [`WorkerUpsert`] request body.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkerConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signing_secret: Option<String>,
}

/// Request body for setting/disabling the webhook. Omitted fields are left
/// unchanged; `state` is "enabled" or "disabled".
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerUpsert {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
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

/// Start an authorization. The URL is the manifest's; whether this deployment
/// will send a credential to it is the deployment's policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpAuthorizeRequest {
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
