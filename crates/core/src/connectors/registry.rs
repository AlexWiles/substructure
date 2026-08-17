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

use super::mcp::McpClient;
use super::oauth::Probed;
use super::{AuthNeed, ConnectorError, CredentialSource, RemoteTool, Requester, Slot, Visibility};
use crate::protocol::ConnectorProtocol;
use crate::protocol::StoredResult;
use crate::runtime::blob::BlobStore;

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
    /// How this connection authenticates, when the file says. Absent is the
    /// usual case: a server announces OAuth or answers without a challenge, so
    /// asking it beats writing down what it would have said.
    ///
    /// Written for the one thing no request can discover — a static token — and
    /// for a server whose discovery is broken enough to need overriding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth: Option<AuthKind>,
    /// Header carrying a static token. Absent ⇒ `Authorization: Bearer`.
    /// Only under `auth = "token"`: OAuth binds its own header.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub header: Option<String>,
    /// Whose credential this connection dials with. Absent is `shared`, and
    /// stays absent so a rewrite does not stamp the default into every
    /// section.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credential: Option<CredentialScope>,
    /// Whether the model sees `<id>__<tool>` rather than the connection's own
    /// tool names. On by default, and the operator's call rather than the agent
    /// author's: only whoever configured the connections knows whether their
    /// names collide.
    ///
    /// Turning it off is safe — a name that collides with another connection, a
    /// declared tool, or a sub-agent is dropped and reported rather than
    /// silently shadowing anything.
    /// Written only when turned off, so a config `subs mcp add` rewrites does
    /// not gain a line per connection saying what the default already says.
    #[serde(default = "yes", skip_serializing_if = "is_yes")]
    pub prefix_tools: bool,
}

fn yes() -> bool {
    true
}

fn is_yes(v: &bool) -> bool {
    *v
}

impl ConnectionSpec {
    /// The prefix to expand `id`'s tools under, or `None` when this connection
    /// offers its tools under their own names.
    pub fn prefix_for<'a>(&self, id: &'a str) -> Option<&'a str> {
        self.prefix_tools.then_some(id)
    }

    pub fn effective_scope(&self) -> CredentialScope {
        self.credential.unwrap_or(CredentialScope::Shared)
    }
}

/// Whose credential a connection dials with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CredentialScope {
    /// One credential for the whole deployment.
    Shared,
    /// One credential per person, usable only in a private conversation.
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

/// How a connection is authenticated, where the file overrides what asking the
/// server would answer. Never *what with*: a file meant to be committed must
/// not be able to hold a secret, so the credential itself is always set out of
/// band — `subs mcp login` or `subs mcp set-token` — and an attempt to write
/// one here is a loud parse error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AuthKind {
    /// The MCP authorization spec's own scheme: discovery, consent, refresh.
    Oauth,
    /// A static token this deployment holds, sent on every call.
    Token,
    /// The server wants no credential, and saying so keeps it out of every
    /// report of what is left to authorize.
    None,
}

impl AuthKind {
    /// The spelling the file uses.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Oauth => "oauth",
            Self::Token => "token",
            Self::None => "none",
        }
    }

    /// The reverse of `as_str`, for a store that keeps the word rather than
    /// the enum.
    pub fn parse(v: &str) -> Option<Self> {
        match v {
            "oauth" => Some(Self::Oauth),
            "token" => Some(Self::Token),
            "none" => Some(Self::None),
            _ => None,
        }
    }
}

/// Hand-written so the shape this key used to take answers with the migration
/// rather than with `invalid type: map`.
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

            fn visit_map<A: serde::de::MapAccess<'de>>(self, _: A) -> Result<AuthKind, A::Error> {
                Err(serde::de::Error::custom(
                    "`auth` is a string now: write `auth = \"token\"` and set the credential \
                     with `subs mcp set-token <id>`",
                ))
            }
        }

        d.deserialize_any(Kind)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    /// No connection by that id.
    Unknown(String),
    /// The connection exists but this tenant has no grant for it.
    NotGranted(String),
    /// The app may use the connection, and nobody has authorized it. Carries
    /// the need because no credential is ever reached to be refused.
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

/// Only an unauthorized connection asks for a person. A misconfigured one is
/// caught in the config, not at runtime.
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
    ///
    async fn resolve(
        &self,
        tenant_id: &str,
        id: &str,
        subject: &Slot,
        spec: &ConnectionSpec,
    ) -> Result<HeaderMap, ConnectorError>;

    /// Replace the credential this connection dials with, after the server
    /// refused it. `Ok(false)` if nothing can be replaced without a person.
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

/// One connection's tool list as fetched, with the prefix it expands under.
#[derive(Debug, Clone, PartialEq)]
pub struct Offer {
    pub prefix: Option<String>,
    pub tools: Vec<RemoteTool>,
    /// What the server said it is for at the handshake, if it said anything.
    pub instructions: Option<String>,
}

/// One live client and when it last carried a call, for the idle sweep.
struct CachedClient {
    client: Arc<McpClient>,
    last_used: std::time::Instant,
}

/// One client per person per connection grows without limit, so the map is
/// bounded. Eviction only drops the map's `Arc`; a call in flight holds its
/// own and the transport closes on the last drop.
const MAX_CLIENTS: usize = 64;
const CLIENT_IDLE: std::time::Duration = std::time::Duration::from_secs(30 * 60);

/// The engine's handle on its connections. Holds one live client per resolved
/// credential so an agent's calls reuse a single connector session; a rejected
/// credential drops the client so the next call resolves a fresh one.
pub struct Connections {
    registry: Arc<dyn ConnectionRegistry>,
    credentials: Arc<dyn CredentialResolver>,
    blobs: Arc<dyn BlobStore>,
    http: reqwest::Client,
    /// Keyed by the credential's identity, not the connection's: two persons
    /// on one connection are two credentials, and one client would send the
    /// first person's to the second.
    clients: Mutex<HashMap<(String, String, Slot), CachedClient>>,
    /// What each server answered an unauthenticated call with, keyed by URL:
    /// one server gives one answer however many connections name it, and the
    /// answer is the same for every person, so sharing it is correct.
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

    /// Which credential this call reads, or why it must read none. The one
    /// decision point: every call and every fetch passes here, and an outside
    /// implementation applies it rather than restating it.
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

    /// What a connection offers, with the prefix its tools expand under. The
    /// prefix rides along so the caller can record it with the offer rather than
    /// reading config again at prompt time, where a since-edited file would
    /// rename tools underneath a live session.
    pub async fn list_tools(
        &self,
        tenant_id: &str,
        id: &str,
        principal: &Requester,
    ) -> Result<Offer, ConnectorError> {
        let spec = self.registry.resolve(tenant_id, id).await?;
        let subject = Self::slot_for(id, &spec, principal)?;
        let (tools, instructions) = self
            .attempt(tenant_id, id, &subject, &spec, |client| async move {
                let tools = client.list_tools().await?;
                Ok((tools, client.instructions().await))
            })
            .await?;
        Ok(Offer {
            prefix: spec.prefix_for(id).map(str::to_string),
            tools,
            instructions,
        })
    }

    pub async fn call_tool(
        &self,
        tenant_id: &str,
        id: &str,
        principal: &Requester,
        name: &str,
        arguments: &Value,
    ) -> Result<StoredResult, ConnectorError> {
        let spec = self.registry.resolve(tenant_id, id).await?;
        let subject = Self::slot_for(id, &spec, principal)?;
        self.attempt(tenant_id, id, &subject, &spec, |client| async move {
            client.call_tool(name, arguments).await
        })
        .await
    }

    /// Run one operation, and once more with a fresh credential if the first
    /// attempt was refused. Without this a refused token that has not expired
    /// goes out again on every attempt, and the connection never recovers.
    ///
    /// The connector session is kept: only the credential was refused.
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
        Err(self.explain_refusal(id, spec, err).await)
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
                spec.url.clone(),
                source,
                self.blobs.clone(),
                tenant_id,
            )),
        };

        let mut clients = self.clients.lock().await;
        Self::evict(&mut clients);
        // Another caller may have won the race; keep whichever landed first so
        // one credential means one session.
        Ok(clients
            .entry(key)
            .or_insert(CachedClient {
                client,
                last_used: std::time::Instant::now(),
            })
            .client
            .clone())
    }

    /// Drop idle clients, then the least recently used past the cap. Dropping
    /// the map's `Arc` is the whole eviction: a call in flight holds its own.
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

    ///
    /// A refusal is also the answer to how the connection authenticates, where
    /// the file did not say — so this is where a declaration-free connection
    /// finds out, at the one moment it matters and at no cost until then.
    async fn explain_refusal(
        &self,
        id: &str,
        spec: &ConnectionSpec,
        err: ConnectorError,
    ) -> ConnectorError {
        match spec.auth {
            // The file already said how. A refusal means the credential is
            // wrong or spent, and the command that replaces it is the answer.
            Some(AuthKind::Token) => ConnectorError::unauthorized(
                AuthNeed::TokenRejected,
                format!(
                    "connection `{id}` rejected its token: run `subs mcp set-token {id}` ({err})"
                ),
            ),
            Some(_) => err,
            None => self.explain(id, spec).await.unwrap_or(err),
        }
    }

    /// What a connection the file says nothing about wants, asked of the server
    /// that just refused. Nothing to say is not an error: the original refusal
    /// stands.
    ///
    /// Asked once per server per process. A refused connection is retried, and
    /// probing on every attempt would double the traffic we send somewhere that
    /// is already turning us away.
    async fn explain(&self, id: &str, spec: &ConnectionSpec) -> Option<ConnectorError> {
        // Dropped before the miss branch runs: a guard held across the `await`
        // below would wait on a lock this task is the one holding.
        let cached = self.probed.lock().await.get(&spec.url).copied();
        let probed = match cached {
            Some(probed) => probed,
            None => {
                let probed = crate::connectors::oauth::sniff(&self.http, &spec.url)
                    .await
                    .ok()?;
                self.probed.lock().await.insert(spec.url.clone(), probed);
                probed
            }
        };
        match probed {
            Probed::Oauth => Some(ConnectorError::unauthorized(
                AuthNeed::NeverAuthorized,
                format!("connection `{id}` is not authorized: run `subs mcp login {id}`"),
            )),
            Probed::Protected => Some(ConnectorError::unauthorized(
                AuthNeed::NeverAuthorized,
                format!(
                    "connection `{id}` wants a credential and publishes no way to obtain one. \
                     If it accepts a static token, declare `auth = \"token\"` on [mcp.{id}] \
                     and run `subs mcp set-token {id}`"
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
        let spec = ConnectionSpec {
            url,
            protocol: ConnectorProtocol::Mcp,
            auth: Some(AuthKind::Oauth),
            header: None,
            credential: None,
            prefix_tools: true,
        };
        Connections::new(
            Arc::new(LocalRegistry::new(BTreeMap::from([(
                "sentry".to_string(),
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
            .list_tools("t", "sentry", &Requester::machine())
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
            .list_tools("t", "sentry", &Requester::machine())
            .await
            .expect_err("the stored credential is refused and cannot be refreshed");

        // What `subs mcp login` does, from another process.
        *credentials.current.lock().await = "fresh".to_string();

        let offer = connections
            .list_tools("t", "sentry", &Requester::machine())
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
            .list_tools("t", "sentry", &Requester::machine())
            .await
            .expect_err("the second refusal stands");

        assert_eq!(err.auth, Some(AuthNeed::Reauthorize));
        assert_eq!(credentials.refreshes.load(Ordering::SeqCst), 1);
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
            .list_tools("t", "sentry", &Requester::machine())
            .await
            .expect_err("a refusal it cannot correct");

        assert_eq!(err.auth, Some(AuthNeed::Reauthorize));
        assert_eq!(credentials.refreshes.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn an_inline_token_is_a_parse_error_not_a_silent_secret() {
        let err = toml::from_str::<ConnectionSpec>(
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
    fn the_old_auth_table_answers_with_the_migration() {
        let err = toml::from_str::<ConnectionSpec>(
            r#"
            url = "https://example.test/mcp"
            auth = { token_env = "GITHUB_TOKEN" }
        "#,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("subs mcp set-token"),
            "a file written against the old shape is told what to write instead; got {err}"
        );
    }

    #[test]
    fn a_connection_defaults_to_mcp_and_to_discovery() {
        let spec: ConnectionSpec = toml::from_str(r#"url = "https://example.test/mcp""#).unwrap();
        assert_eq!(spec.protocol, ConnectorProtocol::Mcp);
        assert_eq!(
            spec.auth, None,
            "a bare connection is discovered, not assumed"
        );
    }

    /// The one decision point, row by row: a shared connection answers with
    /// the company slot for anybody; a personal one answers only for a person
    /// in a private conversation, and refuses the rest with the reason.
    #[test]
    fn the_scope_and_principal_decide_whose_slot_a_call_reads() {
        use crate::protocol::Visibility;
        let shared = ConnectionSpec {
            url: "https://mcp.sentry.dev/mcp".into(),
            protocol: ConnectorProtocol::Mcp,
            auth: None,
            header: None,
            credential: None,
            prefix_tools: true,
        };
        let personal = ConnectionSpec {
            credential: Some(CredentialScope::User),
            ..shared.clone()
        };
        let person =
            |visibility| Requester::new(Subject::new(Issuer::slack(), "T1:U1"), visibility);

        for principal in [
            Requester::machine(),
            person(Visibility::Private),
            person(Visibility::Shared),
        ] {
            assert_eq!(
                Connections::slot_for("sentry", &shared, &principal).unwrap(),
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

    /// Eviction drops the map's `Arc` only: a bounded map, and never a client
    /// pulled out from under a call.
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
                            spec: ConnectionSpec {
                                url: "https://example.test/mcp".into(),
                                protocol: ConnectorProtocol::Mcp,
                                auth: None,
                                header: None,
                                credential: None,
                                prefix_tools: true,
                            },
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
        let bare: ConnectionSpec = toml::from_str(r#"url = "https://mcp.sentry.dev/mcp""#).unwrap();
        assert_eq!(bare.effective_scope(), CredentialScope::Shared);
        assert!(
            !toml::to_string(&bare).unwrap().contains("credential"),
            "the default is not written back"
        );

        let personal: ConnectionSpec =
            toml::from_str("url = \"https://mcp.sentry.dev/mcp\"\ncredential = \"user\"").unwrap();
        assert_eq!(personal.effective_scope(), CredentialScope::User);
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
            let spec: ConnectionSpec = toml::from_str(&format!(
                "url = \"https://example.test/mcp\"\nauth = \"{written}\""
            ))
            .unwrap();
            assert_eq!(spec.auth, Some(kind));
        }
    }

    #[test]
    fn an_unknown_auth_kind_names_the_ones_that_exist() {
        let err = toml::from_str::<ConnectionSpec>(
            r#"
            url = "https://example.test/mcp"
            auth = "bearer"
        "#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("\"token\""), "got {err}");
    }

    /// The probe is what makes a refusal explain itself, and a refused
    /// connection is retried. Asking the server once per process is the
    /// difference between one extra request and one per attempt.
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

        let spec = ConnectionSpec {
            url: url.clone(),
            protocol: ConnectorProtocol::Mcp,
            auth: None,
            header: None,
            credential: None,
            prefix_tools: true,
        };
        let connections = Connections::new(
            Arc::new(LocalRegistry::new(BTreeMap::from([(
                "x".to_string(),
                spec.clone(),
            )]))),
            Arc::new(NoCredential),
            Arc::new(crate::runtime::blob::MemoryBlobStore::new()),
        );

        for _ in 0..3 {
            let explained = connections
                .explain("x", &spec)
                .await
                .expect("a 401 explains itself");
            assert!(explained.to_string().contains("subs mcp login x"));
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
        let err = registry.resolve("t", "sentry").await.unwrap_err();
        assert_eq!(err, RegistryError::Unknown("sentry".to_string()));
    }
}
