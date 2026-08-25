use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use async_trait::async_trait;
use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::Mutex;

use super::mcp::McpClient;
use super::oauth::{ClientId, Probed};
use super::{AuthNeed, ConnectorError, CredentialSource, RemoteTool, Requester, Slot, Visibility};
pub use crate::protocol::ConnectionPath;
use crate::protocol::ConnectorProtocol;
use crate::protocol::StoredResult;
use crate::runtime::blob::BlobStore;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConnectionDecl {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth: Option<AuthKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub header: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credential: Option<CredentialScope>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scopes: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_id_env: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_secret_env: Option<String>,
    #[serde(default = "yes", skip_serializing_if = "is_yes")]
    pub prefix_tools: bool,
}

fn yes() -> bool {
    true
}

fn is_yes(v: &bool) -> bool {
    *v
}

impl ConnectionDecl {
    pub fn at(self, path: ConnectionPath, protocol: ConnectorProtocol) -> ConnectionSpec {
        ConnectionSpec {
            path,
            protocol,
            decl: self,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConnectionSpec {
    pub path: ConnectionPath,
    pub protocol: ConnectorProtocol,
    pub decl: ConnectionDecl,
}

impl ConnectionSpec {
    pub fn prefix(&self) -> Option<String> {
        self.decl.prefix_tools.then(|| self.path.tool_prefix())
    }

    pub fn effective_scope(&self) -> CredentialScope {
        self.decl.credential.unwrap_or(CredentialScope::Shared)
    }

    pub fn configured_client(&self) -> Option<ClientId> {
        Some(ClientId::Registered {
            client_id: env_value(self.decl.client_id_env.as_deref()?)?,
            client_secret: self.decl.client_secret_env.as_deref().and_then(env_value),
        })
    }
}

fn env_value(var: &str) -> Option<String> {
    std::env::var(var)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CredentialScope {
    Shared,
    User,
}

impl CredentialScope {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Shared => "shared",
            Self::User => "user",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AuthKind {
    Oauth,
    Token,
    None,
}

impl AuthKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Oauth => "oauth",
            Self::Token => "token",
            Self::None => "none",
        }
    }

    pub fn parse(v: &str) -> Option<Self> {
        match v {
            "oauth" => Some(Self::Oauth),
            "token" => Some(Self::Token),
            "none" => Some(Self::None),
            _ => None,
        }
    }
}

impl<'de> Deserialize<'de> for AuthKind {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct Kind;

        impl<'de> serde::de::Visitor<'de> for Kind {
            type Value = AuthKind;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str(r#""oauth", "token", or "none""#)
            }

            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<AuthKind, E> {
                AuthKind::parse(v).ok_or_else(|| {
                    E::custom(format!(
                        r#"unknown auth `{v}`; use "oauth", "token", or "none""#
                    ))
                })
            }
        }

        d.deserialize_any(Kind)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    Unknown(String),
    NotGranted(String),
    NotAuthorized { id: String, need: AuthNeed },
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistryError::Unknown(id) => write!(f, "no connection `{id}` is configured"),
            RegistryError::NotGranted(id) => {
                write!(f, "connection `{id}` is not granted to this app")
            }
            RegistryError::NotAuthorized {
                id,
                need: AuthNeed::Reauthorize,
            } => write!(f, "connection `{id}` needs authorizing again"),
            RegistryError::NotAuthorized { id, .. } => {
                write!(f, "connection `{id}` is not authorized")
            }
        }
    }
}

impl From<RegistryError> for ConnectorError {
    fn from(err: RegistryError) -> Self {
        let message = err.to_string();
        match err {
            RegistryError::NotAuthorized { need, .. } => {
                ConnectorError::unauthorized(need, message)
            }
            _ => ConnectorError::permanent(message),
        }
    }
}

#[async_trait]
pub trait ConnectionRegistry: Send + Sync {
    async fn resolve(
        &self,
        tenant_id: &str,
        path: &ConnectionPath,
    ) -> Result<ConnectionSpec, RegistryError>;
}

pub struct LocalRegistry {
    connections: BTreeMap<ConnectionPath, ConnectionSpec>,
}

impl LocalRegistry {
    pub fn new(connections: BTreeMap<ConnectionPath, ConnectionSpec>) -> Self {
        Self { connections }
    }

    pub fn paths(&self) -> impl Iterator<Item = &ConnectionPath> {
        self.connections.keys()
    }
}

#[async_trait]
impl ConnectionRegistry for LocalRegistry {
    async fn resolve(
        &self,
        _tenant_id: &str,
        path: &ConnectionPath,
    ) -> Result<ConnectionSpec, RegistryError> {
        self.connections
            .get(path)
            .cloned()
            .ok_or_else(|| RegistryError::Unknown(path.to_string()))
    }
}

#[async_trait]
pub trait CredentialResolver: Send + Sync {
    async fn resolve(
        &self,
        tenant_id: &str,
        id: &str,
        subject: &Slot,
        spec: &ConnectionSpec,
    ) -> Result<HeaderMap, ConnectorError>;

    async fn refresh(
        &self,
        _tenant_id: &str,
        _id: &str,
        _subject: &Slot,
        _spec: &ConnectionSpec,
    ) -> Result<bool, ConnectorError> {
        Ok(false)
    }
}

struct Resolved {
    credentials: Arc<dyn CredentialResolver>,
    tenant_id: String,
    id: String,
    subject: Slot,
    spec: ConnectionSpec,
}

#[async_trait]
impl CredentialSource for Resolved {
    async fn headers(&self) -> Result<HeaderMap, ConnectorError> {
        self.credentials
            .resolve(&self.tenant_id, &self.id, &self.subject, &self.spec)
            .await
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Offer {
    pub prefix: Option<String>,
    pub server: Option<String>,
    pub tools: Vec<RemoteTool>,
    pub instructions: Option<String>,
}

struct CachedClient {
    client: Arc<McpClient>,
    last_used: std::time::Instant,
}

const MAX_CLIENTS: usize = 64;
const CLIENT_IDLE: std::time::Duration = std::time::Duration::from_secs(30 * 60);

pub struct Connections {
    registry: Arc<dyn ConnectionRegistry>,
    credentials: Arc<dyn CredentialResolver>,
    blobs: Arc<dyn BlobStore>,
    http: reqwest::Client,
    clients: Mutex<HashMap<(String, String, Slot), CachedClient>>,
    probed: Mutex<HashMap<String, Probed>>,
}

impl Connections {
    pub fn new(
        registry: Arc<dyn ConnectionRegistry>,
        credentials: Arc<dyn CredentialResolver>,
        blobs: Arc<dyn BlobStore>,
    ) -> Self {
        Self {
            registry,
            credentials,
            blobs,
            http: reqwest::Client::new(),
            clients: Mutex::new(HashMap::new()),
            probed: Mutex::new(HashMap::new()),
        }
    }

    pub fn slot_for(
        id: &str,
        spec: &ConnectionSpec,
        requester: &Requester,
    ) -> Result<Slot, ConnectorError> {
        if spec.effective_scope() == CredentialScope::Shared {
            return Ok(Slot::Shared);
        }
        let Some(subject) = &requester.subject else {
            return Err(ConnectorError::permanent(format!(
                "connection `{id}` uses a personal credential, and no person is behind \
                 this session"
            )));
        };
        match requester.visibility {
            Visibility::Private => Ok(Slot::Of(subject.clone())),
            Visibility::Shared => Err(ConnectorError::permanent(format!(
                "connection `{id}` uses your personal credential, and other people can \
                 read this conversation. Continue in a direct message."
            ))),
        }
    }

    pub async fn list_tools(
        &self,
        tenant_id: &str,
        path: &ConnectionPath,
        requester: &Requester,
    ) -> Result<Offer, ConnectorError> {
        let spec = self.registry.resolve(tenant_id, path).await?;
        let id = &path.to_string();
        let subject = Self::slot_for(id, &spec, requester)?;
        let (tools, instructions, server) = self
            .attempt(tenant_id, id, &subject, &spec, |client| async move {
                let tools = client.list_tools().await?;
                Ok((
                    tools,
                    client.instructions().await,
                    client.server_title().await,
                ))
            })
            .await?;
        Ok(Offer {
            prefix: spec.prefix(),
            server,
            tools,
            instructions,
        })
    }

    pub async fn call_tool(
        &self,
        tenant_id: &str,
        path: &ConnectionPath,
        requester: &Requester,
        name: &str,
        arguments: &Value,
    ) -> Result<StoredResult, ConnectorError> {
        let spec = self.registry.resolve(tenant_id, path).await?;
        let id = &path.to_string();
        let subject = Self::slot_for(id, &spec, requester)?;
        self.attempt(tenant_id, id, &subject, &spec, |client| async move {
            client.call_tool(name, arguments).await
        })
        .await
    }

    async fn attempt<T, F, Fut>(
        &self,
        tenant_id: &str,
        id: &str,
        subject: &Slot,
        spec: &ConnectionSpec,
        op: F,
    ) -> Result<T, ConnectorError>
    where
        F: Fn(Arc<McpClient>) -> Fut,
        Fut: std::future::Future<Output = Result<T, ConnectorError>>,
    {
        let first = op(self.client(tenant_id, id, subject, spec).await?).await;
        let Err(err) = first else {
            return first;
        };
        if err.auth.is_none() {
            return Err(err);
        }
        if self
            .credentials
            .refresh(tenant_id, id, subject, spec)
            .await?
        {
            let retried = op(self.client(tenant_id, id, subject, spec).await?).await;
            if !matches!(&retried, Err(again) if again.auth.is_some()) {
                return retried;
            }
        }
        Err(self.explain_refusal(spec, err).await)
    }

    async fn client(
        &self,
        tenant_id: &str,
        id: &str,
        subject: &Slot,
        spec: &ConnectionSpec,
    ) -> Result<Arc<McpClient>, ConnectorError> {
        let key = (tenant_id.to_string(), id.to_string(), subject.clone());
        if let Some(cached) = self.clients.lock().await.get_mut(&key) {
            cached.last_used = std::time::Instant::now();
            return Ok(cached.client.clone());
        }

        let source = Arc::new(Resolved {
            credentials: self.credentials.clone(),
            tenant_id: tenant_id.to_string(),
            id: id.to_string(),
            subject: subject.clone(),
            spec: spec.clone(),
        });
        let client = match spec.protocol {
            ConnectorProtocol::Mcp => Arc::new(McpClient::new(
                self.http.clone(),
                spec.decl.url.clone(),
                source,
                self.blobs.clone(),
                tenant_id,
            )),
        };

        let mut clients = self.clients.lock().await;
        Self::evict(&mut clients);
        Ok(clients
            .entry(key)
            .or_insert(CachedClient {
                client,
                last_used: std::time::Instant::now(),
            })
            .client
            .clone())
    }

    fn evict(clients: &mut HashMap<(String, String, Slot), CachedClient>) {
        let now = std::time::Instant::now();
        clients.retain(|_, c| now.duration_since(c.last_used) < CLIENT_IDLE);
        while clients.len() >= MAX_CLIENTS {
            let oldest = clients
                .iter()
                .min_by_key(|(_, c)| c.last_used)
                .map(|(k, _)| k.clone());
            match oldest {
                Some(key) => clients.remove(&key),
                None => break,
            };
        }
    }

    async fn explain_refusal(&self, spec: &ConnectionSpec, err: ConnectorError) -> ConnectorError {
        match spec.decl.auth {
            Some(AuthKind::Token) => ConnectorError::unauthorized(
                AuthNeed::TokenRejected,
                format!(
                    "connection `{}` rejected its token: run `subs auth {}` ({err})",
                    spec.path, spec.path
                ),
            ),
            Some(_) => err,
            None => self.explain(spec).await.unwrap_or(err),
        }
    }

    async fn explain(&self, spec: &ConnectionSpec) -> Option<ConnectorError> {
        let cached = self.probed.lock().await.get(&spec.decl.url).copied();
        let probed = match cached {
            Some(probed) => probed,
            None => {
                let probed = crate::connectors::oauth::sniff(&self.http, &spec.decl.url)
                    .await
                    .ok()?;
                self.probed
                    .lock()
                    .await
                    .insert(spec.decl.url.clone(), probed);
                probed
            }
        };
        match probed {
            Probed::Oauth => Some(ConnectorError::unauthorized(
                AuthNeed::NeverAuthorized,
                format!(
                    "connection `{}` is not authorized: run `subs auth {}`",
                    spec.path, spec.path
                ),
            )),
            Probed::Protected => Some(ConnectorError::unauthorized(
                AuthNeed::NeverAuthorized,
                format!(
                    "connection `{path}` wants a credential and publishes no way to obtain one. \
                     If it accepts a static token, declare `auth = \"token\"` on [{path}] \
                     and run `subs auth {path}`",
                    path = spec.path
                ),
            )),
            Probed::NoChallenge => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{Issuer, Subject};
    use std::sync::atomic::{AtomicUsize, Ordering};

    async fn serve_accepting(accepted: &'static str) -> String {
        use axum::response::IntoResponse;
        use axum::routing::post;

        let app = post(
            move |headers: axum::http::HeaderMap, body: String| async move {
                let sent = headers
                    .get("authorization")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or_default()
                    .to_string();
                if sent != format!("Bearer {accepted}") {
                    return (reqwest::StatusCode::UNAUTHORIZED, "no").into_response();
                }
                let request: Value = serde_json::from_str(&body).unwrap_or_default();
                let json = [(reqwest::header::CONTENT_TYPE, "application/json")];
                let result = match request["method"].as_str().unwrap_or_default() {
                    "server/discover" => {
                        return (
                            reqwest::StatusCode::NOT_FOUND,
                            json,
                            serde_json::json!({
                                "jsonrpc": "2.0", "id": request["id"],
                                "error": { "code": -32601, "message": "no such method" }
                            })
                            .to_string(),
                        )
                            .into_response()
                    }
                    "initialize" => serde_json::json!({
                        "protocolVersion": "2025-11-25",
                        "capabilities": {},
                        "serverInfo": { "name": "mock", "version": "0" },
                    }),
                    "notifications/initialized" => {
                        return reqwest::StatusCode::ACCEPTED.into_response()
                    }
                    _ => {
                        serde_json::json!({ "tools": [ { "name": "search", "inputSchema": {} } ] })
                    }
                };
                (
                    json,
                    serde_json::json!({ "jsonrpc": "2.0", "id": request["id"], "result": result })
                        .to_string(),
                )
                    .into_response()
            },
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, axum::Router::new().route("/mcp", app)).await;
        });
        format!("http://{addr}/mcp")
    }

    struct Rotating {
        current: Mutex<String>,
        next: Option<&'static str>,
        refreshes: AtomicUsize,
    }

    impl Rotating {
        fn new(current: &str, next: Option<&'static str>) -> Self {
            Self {
                current: Mutex::new(current.to_string()),
                next,
                refreshes: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl CredentialResolver for Rotating {
        async fn resolve(
            &self,
            _: &str,
            _: &str,
            _: &Slot,
            _: &ConnectionSpec,
        ) -> Result<HeaderMap, ConnectorError> {
            crate::connectors::mcp::auth_headers(None, &self.current.lock().await.clone())
        }

        async fn refresh(
            &self,
            _: &str,
            _: &str,
            _: &Slot,
            _: &ConnectionSpec,
        ) -> Result<bool, ConnectorError> {
            self.refreshes.fetch_add(1, Ordering::SeqCst);
            match self.next {
                Some(next) => {
                    *self.current.lock().await = next.to_string();
                    Ok(true)
                }
                None => Ok(false),
            }
        }
    }

    async fn connections(url: String, credentials: Arc<Rotating>) -> Connections {
        let spec = ConnectionDecl {
            url,
            auth: Some(AuthKind::Oauth),
            header: None,
            credential: None,
            scopes: Vec::new(),
            client_id_env: None,
            client_secret_env: None,
            prefix_tools: true,
        }
        .at(ConnectionPath::Mcp("x".into()), ConnectorProtocol::Mcp);
        Connections::new(
            Arc::new(LocalRegistry::new(BTreeMap::from([(
                ConnectionPath::Mcp("sentry".into()),
                spec,
            )]))),
            credentials,
            Arc::new(crate::runtime::blob::MemoryBlobStore::new()),
        )
    }

    #[tokio::test]
    async fn a_refused_credential_is_refreshed_and_the_call_tried_again() {
        let url = serve_accepting("fresh").await;
        let credentials = Arc::new(Rotating::new("stale", Some("fresh")));
        let offer = connections(url, credentials.clone())
            .await
            .list_tools(
                "t",
                &ConnectionPath::Mcp("sentry".into()),
                &Requester::machine(),
            )
            .await
            .expect("the retry carries the refreshed token");

        assert_eq!(offer.tools[0].name, "search");
        assert_eq!(credentials.refreshes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_credential_replaced_in_the_store_reaches_the_next_request() {
        let url = serve_accepting("fresh").await;
        let credentials = Arc::new(Rotating::new("stale", None));
        let connections = connections(url, credentials.clone()).await;

        connections
            .list_tools(
                "t",
                &ConnectionPath::Mcp("sentry".into()),
                &Requester::machine(),
            )
            .await
            .expect_err("the stored credential is refused and cannot be refreshed");

        *credentials.current.lock().await = "fresh".to_string();

        let offer = connections
            .list_tools(
                "t",
                &ConnectionPath::Mcp("sentry".into()),
                &Requester::machine(),
            )
            .await
            .expect("the credential written elsewhere goes out on the next request");

        assert_eq!(offer.tools[0].name, "search");
        assert_eq!(
            credentials.refreshes.load(Ordering::SeqCst),
            1,
            "the first refusal asked once; the new credential needed no refresh"
        );
    }

    #[tokio::test]
    async fn a_credential_refused_after_refresh_asks_for_authorization() {
        let url = serve_accepting("never-issued").await;
        let credentials = Arc::new(Rotating::new("stale", Some("also-stale")));
        let err = connections(url, credentials.clone())
            .await
            .list_tools(
                "t",
                &ConnectionPath::Mcp("sentry".into()),
                &Requester::machine(),
            )
            .await
            .expect_err("the second refusal stands");

        assert_eq!(err.auth, Some(AuthNeed::Reauthorize));
        assert_eq!(credentials.refreshes.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_configured_client_comes_from_the_environment_or_not_at_all() {
        let mut spec = ConnectionDecl {
            url: "https://gmailmcp.googleapis.com/mcp/v1".into(),
            auth: None,
            header: None,
            credential: None,
            scopes: Vec::new(),
            client_id_env: None,
            client_secret_env: None,
            prefix_tools: true,
        }
        .at(ConnectionPath::Mcp("x".into()), ConnectorProtocol::Mcp);
        assert!(spec.configured_client().is_none(), "named none");

        spec.decl.client_id_env = Some("SUBS_TEST_CORE_CLIENT_ID".into());
        spec.decl.client_secret_env = Some("SUBS_TEST_CORE_CLIENT_SECRET".into());
        assert!(spec.configured_client().is_none(), "named but unset");

        unsafe {
            std::env::set_var("SUBS_TEST_CORE_CLIENT_ID", "client-1");
            std::env::set_var("SUBS_TEST_CORE_CLIENT_SECRET", "secret-1");
        }
        assert_eq!(
            spec.configured_client(),
            Some(ClientId::Registered {
                client_id: "client-1".into(),
                client_secret: Some("secret-1".into()),
            })
        );

        spec.decl.client_secret_env = None;
        assert_eq!(
            spec.configured_client(),
            Some(ClientId::Registered {
                client_id: "client-1".into(),
                client_secret: None,
            })
        );
        unsafe {
            std::env::remove_var("SUBS_TEST_CORE_CLIENT_ID");
            std::env::remove_var("SUBS_TEST_CORE_CLIENT_SECRET");
        }
    }

    #[test]
    fn an_unauthorized_connection_asks_for_a_person_and_a_misconfigured_one_does_not() {
        let asks = ConnectorError::from(RegistryError::NotAuthorized {
            id: "linear".to_string(),
            need: AuthNeed::NeverAuthorized,
        });
        assert_eq!(asks.auth, Some(AuthNeed::NeverAuthorized));
        assert!(!asks.retryable, "an attempt does not authorize anything");

        for quiet in [
            RegistryError::NotGranted("linear".to_string()),
            RegistryError::Unknown("linear".to_string()),
        ] {
            assert_eq!(ConnectorError::from(quiet).auth, None);
        }
    }

    #[tokio::test]
    async fn a_credential_that_cannot_be_refreshed_is_not_retried() {
        let url = serve_accepting("never-issued").await;
        let credentials = Arc::new(Rotating::new("stale", None));
        let err = connections(url, credentials.clone())
            .await
            .list_tools(
                "t",
                &ConnectionPath::Mcp("sentry".into()),
                &Requester::machine(),
            )
            .await
            .expect_err("a refusal it cannot correct");

        assert_eq!(err.auth, Some(AuthNeed::Reauthorize));
        assert_eq!(credentials.refreshes.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn an_inline_token_is_a_parse_error_not_a_silent_secret() {
        let err = toml::from_str::<ConnectionDecl>(
            r#"
            url = "https://example.test/mcp"
            token = "sk-live-oops"
        "#,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("token"),
            "the file must not be able to hold a secret; got {err}"
        );
    }

    #[test]
    fn a_connection_defaults_to_mcp_and_to_discovery() {
        let spec: ConnectionDecl = toml::from_str(r#"url = "https://example.test/mcp""#).unwrap();
        assert_eq!(
            spec.auth, None,
            "a bare connection is discovered, not assumed"
        );
    }

    #[test]
    fn the_scope_and_requester_decide_whose_slot_a_call_reads() {
        use crate::protocol::Visibility;
        let shared = ConnectionDecl {
            url: "https://mcp.sentry.dev/mcp".into(),
            auth: None,
            header: None,
            credential: None,
            scopes: Vec::new(),
            client_id_env: None,
            client_secret_env: None,
            prefix_tools: true,
        }
        .at(ConnectionPath::Mcp("x".into()), ConnectorProtocol::Mcp);
        let personal = ConnectionDecl {
            credential: Some(CredentialScope::User),
            ..shared.decl.clone()
        }
        .at(shared.path.clone(), shared.protocol);
        let person =
            |visibility| Requester::new(Subject::new(Issuer::slack(), "T1:U1"), visibility);

        for requester in [
            Requester::machine(),
            person(Visibility::Private),
            person(Visibility::Shared),
        ] {
            assert_eq!(
                Connections::slot_for("sentry", &shared, &requester).unwrap(),
                Slot::Shared
            );
        }

        assert_eq!(
            Connections::slot_for("gmail", &personal, &person(Visibility::Private)).unwrap(),
            Slot::Of(Subject::new(Issuer::slack(), "T1:U1"))
        );

        let refused =
            Connections::slot_for("gmail", &personal, &person(Visibility::Shared)).unwrap_err();
        assert!(refused.to_string().contains("direct message"), "{refused}");
        assert!(!refused.retryable);

        let machine = Connections::slot_for("gmail", &personal, &Requester::machine()).unwrap_err();
        assert!(machine.to_string().contains("no person"), "{machine}");
        assert!(machine.to_string().contains("gmail"), "names it: {machine}");
    }

    #[test]
    fn the_client_map_stays_bounded_and_evicts_the_least_recent() {
        let mut clients: HashMap<(String, String, Slot), CachedClient> = HashMap::new();
        let now = std::time::Instant::now();
        for i in 0..MAX_CLIENTS + 5 {
            clients.insert(
                ("t".into(), format!("c{i}"), Slot::Shared),
                CachedClient {
                    client: Arc::new(McpClient::new(
                        reqwest::Client::new(),
                        "https://example.test/mcp".to_string(),
                        Arc::new(Resolved {
                            credentials: Arc::new(Rotating::new("x", None)),
                            tenant_id: "t".into(),
                            id: format!("c{i}"),
                            subject: Slot::Shared,
                            spec: ConnectionDecl {
                                url: "https://example.test/mcp".into(),
                                auth: None,
                                header: None,
                                credential: None,
                                scopes: Vec::new(),
                                client_id_env: None,
                                client_secret_env: None,
                                prefix_tools: true,
                            }
                            .at(ConnectionPath::Mcp("x".into()), ConnectorProtocol::Mcp),
                        }),
                        Arc::new(crate::runtime::blob::MemoryBlobStore::new()),
                        "t",
                    )),
                    last_used: now - std::time::Duration::from_secs(i as u64),
                },
            );
        }
        Connections::evict(&mut clients);
        assert!(clients.len() < MAX_CLIENTS, "bounded: {}", clients.len());
        assert!(
            clients.contains_key(&("t".to_string(), "c0".to_string(), Slot::Shared)),
            "the most recently used stays"
        );
    }

    #[test]
    fn the_scope_is_read_from_the_file_and_defaults_to_shared() {
        let bare: ConnectionDecl = toml::from_str(r#"url = "https://mcp.sentry.dev/mcp""#).unwrap();
        assert_eq!(
            bare.clone()
                .at(ConnectionPath::Mcp("x".into()), ConnectorProtocol::Mcp)
                .effective_scope(),
            CredentialScope::Shared
        );
        assert!(
            !toml::to_string(&bare).unwrap().contains("credential"),
            "the default is not written back"
        );

        let personal: ConnectionDecl =
            toml::from_str("url = \"https://mcp.sentry.dev/mcp\"\ncredential = \"user\"").unwrap();
        assert_eq!(
            personal
                .clone()
                .at(ConnectionPath::Mcp("x".into()), ConnectorProtocol::Mcp)
                .effective_scope(),
            CredentialScope::User
        );
        assert!(toml::to_string(&personal)
            .unwrap()
            .contains("credential = \"user\""));
    }

    #[test]
    fn each_auth_kind_round_trips() {
        for (written, kind) in [
            ("oauth", AuthKind::Oauth),
            ("token", AuthKind::Token),
            ("none", AuthKind::None),
        ] {
            let spec: ConnectionDecl = toml::from_str(&format!(
                "url = \"https://example.test/mcp\"\nauth = \"{written}\""
            ))
            .unwrap();
            assert_eq!(spec.auth, Some(kind));
        }
    }

    #[test]
    fn an_unknown_auth_kind_names_the_ones_that_exist() {
        let err = toml::from_str::<ConnectionDecl>(
            r#"
            url = "https://example.test/mcp"
            auth = "bearer"
        "#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("\"token\""), "got {err}");
    }

    #[tokio::test]
    async fn a_server_is_asked_once_however_often_it_refuses() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let probes = Arc::new(AtomicUsize::new(0));
        let counted = probes.clone();
        let app = axum::Router::new().route(
            "/mcp",
            axum::routing::post(move || {
                let counted = counted.clone();
                async move {
                    counted.fetch_add(1, Ordering::SeqCst);
                    (
                        axum::http::StatusCode::UNAUTHORIZED,
                        [(
                            "www-authenticate",
                            "Bearer resource_metadata=\"http://127.0.0.1/.well-known/x\"",
                        )],
                    )
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/mcp", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        struct NoCredential;
        #[async_trait]
        impl CredentialResolver for NoCredential {
            async fn resolve(
                &self,
                _: &str,
                _: &str,
                _: &Slot,
                _: &ConnectionSpec,
            ) -> Result<HeaderMap, ConnectorError> {
                Ok(HeaderMap::new())
            }
        }

        let spec = ConnectionDecl {
            url: url.clone(),
            auth: None,
            header: None,
            credential: None,
            scopes: Vec::new(),
            client_id_env: None,
            client_secret_env: None,
            prefix_tools: true,
        }
        .at(ConnectionPath::Mcp("x".into()), ConnectorProtocol::Mcp);
        let connections = Connections::new(
            Arc::new(LocalRegistry::new(BTreeMap::from([(
                ConnectionPath::Mcp("x".into()),
                spec.clone(),
            )]))),
            Arc::new(NoCredential),
            Arc::new(crate::runtime::blob::MemoryBlobStore::new()),
        );

        for _ in 0..3 {
            let explained = connections
                .explain(&spec)
                .await
                .expect("a 401 explains itself");
            assert!(explained.to_string().contains("subs auth mcp.x"));
        }
        assert_eq!(
            probes.load(Ordering::SeqCst),
            1,
            "the server is asked once, not once per refusal"
        );
    }

    #[tokio::test]
    async fn an_unknown_id_is_reported_as_unconfigured() {
        let registry = LocalRegistry::new(BTreeMap::new());
        let err = registry
            .resolve("t", &ConnectionPath::Mcp("sentry".into()))
            .await
            .unwrap_err();
        assert_eq!(err, RegistryError::Unknown("mcp.sentry".to_string()));
    }
}
