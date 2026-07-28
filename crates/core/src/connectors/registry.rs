//! Where a connector id turns into something the engine can call.
//!
//! Three steps, and all three exist in every deployment even when one of them
//! is trivial locally: **look up** the connection, **check the grant** that lets
//! this tenant use it, **resolve the credential**. Keeping the shape constant is
//! the point — the cloud swaps the implementations for a database, a grant
//! table, and a vault without moving a step into or out of the pipeline.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use async_trait::async_trait;
use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::Mutex;

use super::mcp::{auth_headers, McpClient};
use super::{ConnectorError, RemoteTool, ToolOutcome};
use crate::protocol::ConnectorProtocol;

/// A connection as configured: where it is and how to authenticate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConnectionSpec {
    pub url: String,
    /// Which protocol reaches this connection. Not written in the config: it
    /// follows from the section the connection was declared under, and a second
    /// copy would only be a way for the two to disagree. The loader stamps it.
    #[serde(skip)]
    pub protocol: ConnectorProtocol,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth: Option<AuthSpec>,
    /// Whether the model sees `<id>__<tool>` rather than the connection's own
    /// tool names. On by default, and the operator's call rather than the agent
    /// author's: only whoever configured the connections knows whether their
    /// names collide.
    ///
    /// Turning it off is safe — a name that collides with another connection, a
    /// declared tool, or a sub-agent is dropped and reported rather than
    /// silently shadowing anything.
    #[serde(default = "yes")]
    pub prefix_tools: bool,
}

fn yes() -> bool {
    true
}

impl ConnectionSpec {
    /// The prefix to expand `id`'s tools under, or `None` when this connection
    /// offers its tools under their own names.
    pub fn prefix_for<'a>(&self, id: &'a str) -> Option<&'a str> {
        self.prefix_tools.then_some(id)
    }
}

/// How a connection is authenticated. The token is always *named*, never
/// written: a file meant to be committed must not be able to hold a secret, so
/// there is deliberately no inline `token` field and `deny_unknown_fields`
/// turns an attempt to add one into a loud parse error.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthSpec {
    /// Header carrying the credential. Absent ⇒ `Authorization: Bearer <token>`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub header: Option<String>,
    /// Environment variable holding the token.
    pub token_env: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    /// No connection by that id.
    Unknown(String),
    /// The connection exists but this tenant has no grant for it.
    NotGranted(String),
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistryError::Unknown(id) => write!(f, "no connection `{id}` is configured"),
            RegistryError::NotGranted(id) => {
                write!(f, "connection `{id}` is not granted to this app")
            }
        }
    }
}

impl From<RegistryError> for ConnectorError {
    fn from(err: RegistryError) -> Self {
        ConnectorError::permanent(err.to_string())
    }
}

#[async_trait]
pub trait ConnectionRegistry: Send + Sync {
    async fn resolve(&self, tenant_id: &str, id: &str) -> Result<ConnectionSpec, RegistryError>;
}

/// Every connection declared in `substructure.toml`, granted to the single
/// tenant a local engine serves. The grant check is a constant `true` here; it
/// is a real lookup in the cloud.
pub struct LocalRegistry {
    connections: BTreeMap<String, ConnectionSpec>,
}

impl LocalRegistry {
    pub fn new(connections: BTreeMap<String, ConnectionSpec>) -> Self {
        Self { connections }
    }

    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.connections.keys().map(String::as_str)
    }
}

#[async_trait]
impl ConnectionRegistry for LocalRegistry {
    async fn resolve(&self, _tenant_id: &str, id: &str) -> Result<ConnectionSpec, RegistryError> {
        self.connections
            .get(id)
            .cloned()
            .ok_or_else(|| RegistryError::Unknown(id.to_string()))
    }
}

#[async_trait]
pub trait CredentialResolver: Send + Sync {
    /// The credential headers for one connection, resolved at call time. The
    /// cloud implementation reads a vault and refreshes an expired token here;
    /// this is the seam that keeps tokens out of the event store.
    async fn resolve(
        &self,
        tenant_id: &str,
        id: &str,
        spec: &ConnectionSpec,
    ) -> Result<HeaderMap, ConnectorError>;
}

/// Reads the token from the environment variable the connection names.
///
/// A missing or empty token is reported as needing re-auth rather than as a
/// config error, so an unset credential locally takes the same path a revoked
/// one will take in the cloud.
pub struct EnvCredentials;

#[async_trait]
impl CredentialResolver for EnvCredentials {
    async fn resolve(
        &self,
        _tenant_id: &str,
        id: &str,
        spec: &ConnectionSpec,
    ) -> Result<HeaderMap, ConnectorError> {
        let Some(auth) = &spec.auth else {
            return Ok(HeaderMap::new());
        };
        match std::env::var(&auth.token_env) {
            Ok(token) if !token.trim().is_empty() => {
                auth_headers(auth.header.as_deref(), token.trim())
            }
            _ => Err(ConnectorError::unauthorized(format!(
                "connection `{id}` has no credential: ${} is unset",
                auth.token_env
            ))),
        }
    }
}

/// One connection's tool list as fetched, with the prefix it expands under.
#[derive(Debug, Clone, PartialEq)]
pub struct Offer {
    pub prefix: Option<String>,
    pub tools: Vec<RemoteTool>,
}

/// The engine's handle on its connections. Holds one live client per resolved
/// connection so an agent's calls reuse a single connector session; a rejected
/// credential drops the client so the next call resolves a fresh one.
pub struct Connections {
    registry: Arc<dyn ConnectionRegistry>,
    credentials: Arc<dyn CredentialResolver>,
    http: reqwest::Client,
    clients: Mutex<HashMap<(String, String), Arc<McpClient>>>,
}

impl Connections {
    pub fn new(
        registry: Arc<dyn ConnectionRegistry>,
        credentials: Arc<dyn CredentialResolver>,
    ) -> Self {
        Self {
            registry,
            credentials,
            http: reqwest::Client::new(),
            clients: Mutex::new(HashMap::new()),
        }
    }

    /// What a connection offers, with the prefix its tools expand under. The
    /// prefix rides along so the caller can record it with the offer rather than
    /// reading config again at prompt time, where a since-edited file would
    /// rename tools underneath a live session.
    pub async fn list_tools(&self, tenant_id: &str, id: &str) -> Result<Offer, ConnectorError> {
        let spec = self.registry.resolve(tenant_id, id).await?;
        let client = self.client(tenant_id, id).await?;
        let tools = self.guard(tenant_id, id, client.list_tools().await).await?;
        Ok(Offer {
            prefix: spec.prefix_for(id).map(str::to_string),
            tools,
        })
    }

    pub async fn call_tool(
        &self,
        tenant_id: &str,
        id: &str,
        name: &str,
        arguments: &Value,
    ) -> Result<ToolOutcome, ConnectorError> {
        let client = self.client(tenant_id, id).await?;
        self.guard(tenant_id, id, client.call_tool(name, arguments).await)
            .await
    }

    async fn client(&self, tenant_id: &str, id: &str) -> Result<Arc<McpClient>, ConnectorError> {
        let key = (tenant_id.to_string(), id.to_string());
        if let Some(client) = self.clients.lock().await.get(&key) {
            return Ok(client.clone());
        }

        let spec = self.registry.resolve(tenant_id, id).await?;
        let headers = self.credentials.resolve(tenant_id, id, &spec).await?;
        let client = match spec.protocol {
            ConnectorProtocol::Mcp => {
                Arc::new(McpClient::new(self.http.clone(), spec.url.clone(), headers))
            }
        };

        let mut clients = self.clients.lock().await;
        // Another caller may have won the race; keep whichever landed first so
        // one connection means one session.
        Ok(clients.entry(key).or_insert(client).clone())
    }

    /// Drop the cached client when the connection rejected its credential, so a
    /// refreshed token is picked up without restarting the engine.
    async fn guard<T>(
        &self,
        tenant_id: &str,
        id: &str,
        result: Result<T, ConnectorError>,
    ) -> Result<T, ConnectorError> {
        if let Err(err) = &result {
            if err.needs_reauth {
                self.clients
                    .lock()
                    .await
                    .remove(&(tenant_id.to_string(), id.to_string()));
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(token_env: &str) -> ConnectionSpec {
        ConnectionSpec {
            url: "https://example.test/mcp".to_string(),
            protocol: ConnectorProtocol::Mcp,
            auth: Some(AuthSpec {
                header: None,
                token_env: token_env.to_string(),
            }),
            prefix_tools: true,
        }
    }

    #[test]
    fn an_inline_token_is_a_parse_error_not_a_silent_secret() {
        let err = toml::from_str::<ConnectionSpec>(
            r#"
            url = "https://example.test/mcp"
            auth = { token = "sk-live-oops" }
        "#,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("token"),
            "the file must not be able to hold a secret; got {err}"
        );
    }

    #[test]
    fn a_connection_defaults_to_mcp_and_needs_no_auth() {
        let spec: ConnectionSpec = toml::from_str(r#"url = "https://example.test/mcp""#).unwrap();
        assert_eq!(spec.protocol, ConnectorProtocol::Mcp);
        assert_eq!(spec.auth, None);
    }

    #[tokio::test]
    async fn an_unknown_id_is_reported_as_unconfigured() {
        let registry = LocalRegistry::new(BTreeMap::new());
        let err = registry.resolve("t", "sentry").await.unwrap_err();
        assert_eq!(err, RegistryError::Unknown("sentry".to_string()));
    }

    #[tokio::test]
    async fn a_missing_token_asks_for_re_auth() {
        let err = EnvCredentials
            .resolve("t", "sentry", &spec("SUBS_TEST_DEFINITELY_UNSET"))
            .await
            .unwrap_err();
        assert!(
            err.needs_reauth,
            "an unset credential takes the re-auth path, not a config-error path"
        );
    }

    #[tokio::test]
    async fn a_connection_without_auth_sends_no_credential_header() {
        let spec = ConnectionSpec {
            url: "https://example.test/mcp".to_string(),
            protocol: ConnectorProtocol::Mcp,
            auth: None,
            prefix_tools: true,
        };
        let headers = EnvCredentials.resolve("t", "open", &spec).await.unwrap();
        assert!(headers.is_empty());
    }
}
