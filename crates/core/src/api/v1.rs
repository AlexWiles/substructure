use std::collections::BTreeMap;

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
pub struct Project {
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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Meta {
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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpConnection {
    pub id: String,

    pub connection_id: String,
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub scopes: String,

    #[serde(default)]
    pub auth: Option<String>,
    #[serde(default)]
    pub granted_projects: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpDeclareRequest {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connection_id: Option<String>,

    pub project_id: String,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub header: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpTokenRequest {
    pub token: String,
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
    pub project_id: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SlackApp {
    pub agent_id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub answers: crate::manifest::SlackAudience,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub installed: Option<SlackInstall>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SlackInstall {
    #[serde(default)]
    pub team_id: String,
    #[serde(default)]
    pub team_name: String,
}

impl SlackApp {
    pub fn label(&self) -> String {
        match &self.installed {
            None => "not set".to_string(),
            Some(install) if install.team_name.is_empty() => install.team_id.clone(),
            Some(install) => format!("{} in {}", self.name, install.team_name),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SlackManifest {
    pub agent_id: String,
    pub manifest: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SlackCredentials {
    pub bot_token: String,
    pub signing_secret: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectConfigView {
    pub manifest: crate::manifest::Manifest,
    #[serde(default)]
    pub llm: BTreeMap<String, LlmState>,
    #[serde(default)]
    pub mcp: Vec<ConfigConnection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_applied: Option<ConfigApplied>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmState {
    #[serde(default)]
    pub key_bound: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Agent {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<crate::protocol::AgentConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmBlockView {
    pub name: String,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(default)]
    pub key_bound: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RunRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<crate::protocol::AgentConfig>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker: Option<crate::protocol::WorkerRef>,
    pub input: crate::protocol::ClientInput,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RunFormat {
    #[default]
    AgUi,

    Events,
}

pub const RUN_DONE_EVENT: &str = "done";

impl RunFormat {
    pub fn as_query(self) -> &'static str {
        match self {
            RunFormat::AgUi => "ag-ui",
            RunFormat::Events => "events",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmKeyRequest {
    pub key: String,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApplyResponse {
    pub project_id: String,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_url: Option<String>,
    #[serde(default)]
    pub changes: Vec<ConfigEvent>,

    #[serde(default)]
    pub notices: Vec<Notice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PluginHead {
    pub id: String,
    pub hash: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PluginHeads {
    #[serde(default)]
    pub plugins: Vec<PluginHead>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PluginPush {
    pub hash: String,

    pub bundle: crate::plugins::PluginBundle,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub binaries: Vec<PluginBinary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PluginBinary {
    pub skill: String,
    pub path: String,
    pub mime: String,
    pub bytes: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PluginPushed {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub hash: String,
    #[serde(default)]
    pub binaries: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NoticesResponse {
    #[serde(default)]
    pub notices: Vec<Notice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Notice {
    #[serde(default)]
    pub level: NoticeLevel,
    pub message: String,
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
}

impl Notice {
    pub fn action(message: impl Into<String>) -> Self {
        Self {
            level: NoticeLevel::Action,
            message: message.into(),
            command: None,
            url: None,
        }
    }

    pub fn with_command(mut self, command: impl Into<String>) -> Self {
        self.command = Some(command.into());
        self
    }

    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum NoticeLevel {
    #[default]
    Action,

    Warn,

    Info,
}

impl<'de> Deserialize<'de> for NoticeLevel {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(match String::deserialize(deserializer)?.as_str() {
            "warn" => NoticeLevel::Warn,
            "info" => NoticeLevel::Info,
            _ => NoticeLevel::Action,
        })
    }
}

impl NoticeLevel {
    pub const ORDER: [NoticeLevel; 3] = [NoticeLevel::Action, NoticeLevel::Warn, NoticeLevel::Info];

    pub fn heading(self) -> &'static str {
        match self {
            NoticeLevel::Action => "Action required:",
            NoticeLevel::Warn => "Warnings:",
            NoticeLevel::Info => "Notes:",
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::SlackAudience;

    #[test]
    fn a_slack_app_reads_back_from_what_a_deployment_sends() {
        let waiting: SlackApp =
            serde_json::from_str(r#"{ "agentId": "support", "name": "Support", "answers": "dm" }"#)
                .unwrap();
        assert_eq!(waiting.answers, SlackAudience::Dm);
        assert!(waiting.installed.is_none());
        assert_eq!(waiting.label(), "not set");

        let installed: SlackApp = serde_json::from_str(
            r#"{
                "agentId": "support",
                "name": "Support",
                "answers": "both",
                "installed": { "teamId": "T1", "teamName": "Acme" }
            }"#,
        )
        .unwrap();
        assert_eq!(installed.answers, SlackAudience::Both);
        assert_eq!(installed.label(), "Support in Acme");

        let null: SlackApp =
            serde_json::from_str(r#"{ "agentId": "support", "installed": null }"#).unwrap();
        assert!(null.installed.is_none());
        assert_eq!(
            null.answers,
            SlackAudience::Both,
            "the widest is the default"
        );
    }
}
