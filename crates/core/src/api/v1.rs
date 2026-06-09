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
