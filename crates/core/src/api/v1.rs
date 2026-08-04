//! `/api/v1` wire types. Field names and casing are the contract: the local
//! server serializes these, the CLI deserializes them, and they must match
//! what the hosted cloud sends.

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

/// What a deployment says about itself, so the CLI can degrade with a real
/// message rather than a 404.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Meta {
    /// One org and one project, advertised on every response. `subs link` adopts
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
    pub granted_projects: Vec<String>,
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
    /// The project this connection is for. A connection belongs to one project,
    /// so the deployment answers with that project's own — two projects naming
    /// one id hold two credentials, and neither login touches the other.
    pub project_id: String,
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

/// The Slack workspaces an org has connected, and whether the deployment
/// connects any at all. A deployment holding no Slack app answers
/// `configured: false` rather than an empty list, which is a different thing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SlackStatus {
    #[serde(default)]
    pub configured: bool,
    #[serde(default)]
    pub connections: Vec<SlackConnection>,
}

/// One connected workspace. Only what identifies it: the routing a deployment
/// keeps beside it is applied from the file, not read from here.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SlackConnection {
    pub id: String,
    #[serde(default)]
    pub team_id: String,
    #[serde(default)]
    pub team_name: String,
    #[serde(default)]
    pub status: String,
}

impl SlackConnection {
    /// The workspace as a reader knows it. A deployment that sends no name
    /// leaves the id, which is still the workspace.
    pub fn label(&self) -> String {
        match self.team_name.trim() {
            "" => self.team_id.clone(),
            name => format!("{name} ({})", self.team_id),
        }
    }
}

/// Slack's own consent page, built by the deployment that holds the app
/// credentials and the redirect.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SlackInstallUrl {
    pub url: String,
}

/// A project's configuration as the deployment holds it: the manifest it was
/// last applied, plus the state only the deployment knows — which agents have a
/// signing secret, which blocks have a key. Status is not config, so it is a
/// view rather than something `subs apply` could send back.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectConfigView {
    pub manifest: crate::manifest::Manifest,
    #[serde(default)]
    pub agents: BTreeMap<String, AgentState>,
    #[serde(default)]
    pub llm: BTreeMap<String, LlmState>,
    #[serde(default)]
    pub mcp: Vec<ConfigConnection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_applied: Option<ConfigApplied>,
}

/// What the deployment knows about one declared agent that the manifest cannot
/// say. The secret itself is never in a list.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentState {
    #[serde(default)]
    pub secret_set: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmState {
    #[serde(default)]
    pub key_bound: bool,
}

/// One declared agent, as `subs agents` reads it. `signing_secret` is present
/// only on the single-agent GET.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Agent {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<crate::protocol::AgentConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker_url: Option<String>,
    #[serde(default)]
    pub secret_set: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signing_secret: Option<String>,
}

/// One declared `[llm.*]` block, as `subs llm list` reads it.
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

/// The customer key for one block. Write-only: no read ever returns it.
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

/// One entry in a project's configuration history.
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
    pub project_id: String,
    /// Where to see this project, as the deployment itself names it. Absent
    /// from one that has no page to send a reader to, and from one that
    /// predates the field — the CLI cannot derive it, since the server the API
    /// answers on is not always the one the browser uses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_url: Option<String>,
    #[serde(default)]
    pub changes: Vec<ConfigEvent>,
    /// What the deployment has to say about the document it just took, with its
    /// own links. Recomputed server-side on every apply rather than logged, so
    /// an unchanged re-apply still reports it.
    #[serde(default)]
    pub notices: Vec<Notice>,
}

/// One thing the deployment wants the reader to know, and the ways to act on
/// it. Both routes are optional: an OAuth consent has no CLI command, and a
/// purely local fix has no page.
///
/// The deployment decides what it says and how loudly, so a deployment that
/// learns to report something new needs no CLI release to say it.
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

/// How much of the reader's attention a notice is asking for.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum NoticeLevel {
    /// The document does not fully work until a human does this.
    #[default]
    Action,
    /// Something is wrong that the deployment took anyway.
    Warn,
    /// Worth knowing, and nothing to do.
    Info,
}

/// A level this build has no name for reads as the loudest one rather than
/// failing the response: the deployment decided it was worth saying, and a CLI
/// older than the deployment must still say it.
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
    /// The order levels are printed in, loudest first.
    pub const ORDER: [NoticeLevel; 3] = [NoticeLevel::Action, NoticeLevel::Warn, NoticeLevel::Info];

    /// The heading notices of this level are printed under.
    pub fn heading(self) -> &'static str {
        match self {
            NoticeLevel::Action => "Action required:",
            NoticeLevel::Warn => "Warnings:",
            NoticeLevel::Info => "Notes:",
        }
    }
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
