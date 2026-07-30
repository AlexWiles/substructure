//! OAuth for MCP connections: discovery, client registration, and the
//! authorization code flow.
//!
//! Storage-free and transport-free. The caller owns the browser, the redirect
//! target, and wherever tokens land, so the same flow drives the CLI's loopback
//! login and a deployed server's callback route.
//!
//! Implements the pieces the MCP authorization spec makes mandatory:
//! protected resource metadata (RFC 9728), authorization server metadata
//! (RFC 8414 and OpenID Discovery), PKCE with `S256`, the `resource` indicator
//! (RFC 8707) on both requests, and issuer validation (RFC 9207).

use super::ConnectorError;
use base64::engine::general_purpose::URL_SAFE_NO_PAD as B64URL;
use base64::Engine;
use chrono::{DateTime, Duration, Utc};
use rand::Rng;
use reqwest::header::WWW_AUTHENTICATE;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Refresh this far ahead of expiry so a call never races the clock.
pub const REFRESH_SKEW: i64 = 120;

#[derive(Debug, thiserror::Error)]
pub enum OauthError {
    #[error("{0}")]
    Discovery(String),
    #[error("{0}")]
    Registration(String),
    #[error("{0}")]
    Token(String),
    /// The authorization response named an issuer other than the one whose
    /// metadata we validated. The code is not redeemed and the error the
    /// response carried is not reported: neither can be trusted.
    #[error("authorization server mismatch: expected `{expected}`, got `{got}`")]
    IssuerMismatch { expected: String, got: String },
}

/// What a protected MCP server says about itself (RFC 9728).
#[derive(Debug, Clone, Deserialize)]
pub struct ProtectedResource {
    /// The canonical URI the `resource` indicator must carry. Used verbatim —
    /// servers disagree on trailing slashes and the token is bound to whichever
    /// form the server published.
    pub resource: String,
    #[serde(default)]
    pub authorization_servers: Vec<String>,
    #[serde(default)]
    pub scopes_supported: Vec<String>,
}

/// The authorization server's own metadata (RFC 8414 / OpenID Discovery).
#[derive(Debug, Clone, Deserialize)]
pub struct AuthServer {
    pub issuer: String,
    pub authorization_endpoint: String,
    pub token_endpoint: String,
    #[serde(default)]
    pub registration_endpoint: Option<String>,
    #[serde(default)]
    pub scopes_supported: Vec<String>,
    #[serde(default)]
    pub code_challenge_methods_supported: Vec<String>,
    #[serde(default)]
    pub client_id_metadata_document_supported: bool,
    #[serde(default)]
    pub authorization_response_iss_parameter_supported: bool,
}

/// How this client identifies itself. Selection order follows the spec:
/// pre-registered, then a metadata document, then dynamic registration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ClientId {
    /// An HTTPS URL that resolves to this client's metadata document. Portable
    /// across authorization servers, so it is never bound to an issuer.
    Metadata { url: String },
    /// Credentials issued to us, either by hand or by dynamic registration.
    /// Only valid at the issuer that minted them.
    Registered {
        client_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        client_secret: Option<String>,
    },
}

impl ClientId {
    fn as_param(&self) -> &str {
        match self {
            ClientId::Metadata { url } => url,
            ClientId::Registered { client_id, .. } => client_id,
        }
    }

    fn secret(&self) -> Option<&str> {
        match self {
            ClientId::Metadata { .. } => None,
            ClientId::Registered { client_secret, .. } => client_secret.as_deref(),
        }
    }
}

/// A discovered connection, ready to authorize against.
#[derive(Debug, Clone)]
pub struct Discovered {
    pub resource: ProtectedResource,
    pub server: AuthServer,
}

/// Everything the caller must hold between opening the browser and the
/// redirect coming back. Short-lived and single-use.
#[derive(Debug, Clone)]
pub struct Pending {
    pub url: String,
    pub state: String,
    pub verifier: String,
    /// Recorded from validated metadata, for the RFC 9207 check on return.
    pub issuer: String,
    pub redirect_uri: String,
    pub resource: String,
    pub scope: Option<String>,
    pub iss_expected: bool,
}

/// A credential as stored. `client` rides along because dynamic registration
/// mints an id that refresh needs back, and the spec binds it to its issuer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tokens {
    pub access_token: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    pub issuer: String,
    pub token_endpoint: String,
    /// Sent again on every refresh: a refresh is a token request, and RFC 8707
    /// binds the new token to the same audience as the old one.
    pub resource: String,
    pub client: ClientId,
}

impl Tokens {
    /// Whether the access token is close enough to expiry to refresh. An
    /// absent expiry is treated as live: the server never said otherwise, and
    /// a 401 is the authoritative signal.
    pub fn stale(&self) -> bool {
        self.expires_at
            .is_some_and(|at| Utc::now() + Duration::seconds(REFRESH_SKEW) >= at)
    }

    pub fn refreshable(&self) -> bool {
        self.refresh_token.is_some()
    }
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default)]
    expires_in: Option<i64>,
    #[serde(default)]
    scope: Option<String>,
}

// ── Discovery ────────────────────────────────────────────────────────────

/// Find the authorization server protecting an MCP endpoint.
///
/// The unauthenticated probe is what the spec intends: the 401 carries a
/// `resource_metadata` pointer. A server that omits it still resolves, via the
/// well-known path built from the endpoint URL.
pub async fn discover(http: &Client, mcp_url: &str) -> Result<Discovered, OauthError> {
    require_secure(mcp_url)?;
    let metadata_url = match probe(http, mcp_url).await {
        Some(url) => url,
        None => prm_url(mcp_url)?,
    };
    let resource: ProtectedResource = fetch_json(http, &metadata_url)
        .await
        .map_err(|e| OauthError::Discovery(format!("reading {metadata_url}: {e}")))?;

    // RFC 9728: the metadata a server points us at must describe that server.
    // Without this a compromised endpoint could aim the flow at an
    // authorization server of its choosing.
    if !same_origin(&resource.resource, mcp_url) {
        return Err(OauthError::Discovery(format!(
            "{metadata_url} describes `{}`, not `{mcp_url}`",
            resource.resource
        )));
    }

    let issuer = resource
        .authorization_servers
        .first()
        .cloned()
        .ok_or_else(|| {
            OauthError::Discovery(format!("{metadata_url} names no authorization server"))
        })?;
    let server = fetch_auth_server(http, &issuer).await?;
    Ok(Discovered { resource, server })
}

/// The `resource_metadata` URL a 401 challenge points at, if there is one.
async fn probe(http: &Client, mcp_url: &str) -> Option<String> {
    let response = http
        .post(mcp_url)
        .header(
            reqwest::header::ACCEPT,
            "application/json, text/event-stream",
        )
        .json(&serde_json::json!({ "jsonrpc": "2.0", "id": 0, "method": "tools/list" }))
        .send()
        .await
        .ok()?;
    let challenge = response.headers().get(WWW_AUTHENTICATE)?.to_str().ok()?;
    challenge_param(challenge, "resource_metadata")
}

/// One parameter out of a `WWW-Authenticate` header. Values may be quoted or
/// bare; both appear in the wild.
pub fn challenge_param(challenge: &str, name: &str) -> Option<String> {
    for part in challenge.split(',') {
        let (key, value) = part.split_once('=')?;
        if key.trim().trim_start_matches("Bearer").trim() == name {
            return Some(value.trim().trim_matches('"').to_string());
        }
    }
    None
}

/// A bearer credential must not cross the network in the clear. Loopback is
/// exempt: a server on this machine has no certificate and nothing off-host to
/// intercept it.
fn require_secure(url: &str) -> Result<(), OauthError> {
    let parsed = reqwest::Url::parse(url)
        .map_err(|e| OauthError::Discovery(format!("`{url}` is not a URL: {e}")))?;
    if parsed.scheme() == "https" || is_loopback(url) {
        return Ok(());
    }
    Err(OauthError::Discovery(format!(
        "`{url}` is not https: a credential would cross the network in the clear"
    )))
}

fn same_origin(a: &str, b: &str) -> bool {
    match (reqwest::Url::parse(a), reqwest::Url::parse(b)) {
        (Ok(a), Ok(b)) => a.origin() == b.origin(),
        _ => false,
    }
}

/// RFC 9728 well-known location: the path is inserted after the well-known
/// segment, not appended to the origin.
fn prm_url(mcp_url: &str) -> Result<String, OauthError> {
    let url = reqwest::Url::parse(mcp_url)
        .map_err(|e| OauthError::Discovery(format!("`{mcp_url}` is not a URL: {e}")))?;
    let origin = url.origin().ascii_serialization();
    let path = url.path().trim_end_matches('/');
    Ok(format!(
        "{origin}/.well-known/oauth-protected-resource{path}"
    ))
}

/// Authorization server metadata, trying every location the spec requires a
/// client to support. The path-insertion forms come first: an issuer with a
/// path (GitHub's `https://github.com/login/oauth`) only answers there.
async fn fetch_auth_server(http: &Client, issuer: &str) -> Result<AuthServer, OauthError> {
    let mut last = String::new();
    for candidate in metadata_urls(issuer)? {
        match fetch_json::<AuthServer>(http, &candidate).await {
            Ok(server) => {
                // RFC 8414: the document must claim the issuer we asked about,
                // or it is somebody else's metadata.
                if server.issuer != issuer {
                    return Err(OauthError::Discovery(format!(
                        "{candidate} claims issuer `{}`, expected `{issuer}`",
                        server.issuer
                    )));
                }
                return Ok(server);
            }
            Err(e) => last = format!("{candidate}: {e}"),
        }
    }
    Err(OauthError::Discovery(format!(
        "no authorization server metadata for `{issuer}` ({last})"
    )))
}

fn metadata_urls(issuer: &str) -> Result<Vec<String>, OauthError> {
    let url = reqwest::Url::parse(issuer)
        .map_err(|e| OauthError::Discovery(format!("`{issuer}` is not a URL: {e}")))?;
    let origin = url.origin().ascii_serialization();
    let path = url.path().trim_end_matches('/');
    let base = issuer.trim_end_matches('/');
    Ok(vec![
        format!("{origin}/.well-known/oauth-authorization-server{path}"),
        format!("{origin}/.well-known/openid-configuration{path}"),
        format!("{base}/.well-known/openid-configuration"),
        format!("{base}/.well-known/oauth-authorization-server"),
    ])
}

async fn fetch_json<T: serde::de::DeserializeOwned>(http: &Client, url: &str) -> Result<T, String> {
    let response = http.get(url).send().await.map_err(|e| e.to_string())?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!("HTTP {}", status.as_u16()));
    }
    response.json().await.map_err(|e| e.to_string())
}

// ── Client registration ──────────────────────────────────────────────────

/// Pick how to identify this client, preferring what the spec prefers.
///
/// `metadata_document` is this deployment's own document URL, present only
/// where something serves one at a stable HTTPS address; a laptop has none and
/// falls through to dynamic registration.
pub async fn client_id(
    http: &Client,
    server: &AuthServer,
    metadata_document: Option<&str>,
    redirect_uri: &str,
    client_name: &str,
) -> Result<ClientId, OauthError> {
    if let Some(url) = metadata_document {
        if server.client_id_metadata_document_supported {
            return Ok(ClientId::Metadata {
                url: url.to_string(),
            });
        }
    }
    let endpoint = server.registration_endpoint.as_deref().ok_or_else(|| {
        OauthError::Registration(format!(
            "`{}` supports neither client metadata documents nor dynamic registration; \
             register a client by hand and configure its id",
            server.issuer
        ))
    })?;
    register(http, endpoint, redirect_uri, client_name).await
}

/// Dynamic client registration (RFC 7591). Deprecated in favour of metadata
/// documents, and still the only mechanism several servers offer.
async fn register(
    http: &Client,
    endpoint: &str,
    redirect_uri: &str,
    client_name: &str,
) -> Result<ClientId, OauthError> {
    // `application_type` is mandatory: omitted, OIDC servers default to `web`
    // and reject a loopback redirect.
    let application_type = if is_loopback(redirect_uri) {
        "native"
    } else {
        "web"
    };
    let body = serde_json::json!({
        "client_name": client_name,
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
        "application_type": application_type,
    });
    let response = http
        .post(endpoint)
        .json(&body)
        .send()
        .await
        .map_err(|e| OauthError::Registration(format!("registering at {endpoint}: {e}")))?;
    let status = response.status();
    let payload: serde_json::Value = response
        .json()
        .await
        .map_err(|e| OauthError::Registration(format!("registration response: {e}")))?;
    if !status.is_success() {
        return Err(OauthError::Registration(format!(
            "registration refused (HTTP {}): {}",
            status.as_u16(),
            payload
        )));
    }
    let client_id = payload["client_id"]
        .as_str()
        .ok_or_else(|| OauthError::Registration("registration returned no client_id".into()))?
        .to_string();
    Ok(ClientId::Registered {
        client_id,
        client_secret: payload["client_secret"].as_str().map(str::to_string),
    })
}

/// One definition of the exemption, shared with the CLI's declaration check so
/// the two cannot drift.
pub(crate) fn is_loopback(redirect_uri: &str) -> bool {
    reqwest::Url::parse(redirect_uri).is_ok_and(|u| {
        matches!(
            u.host_str(),
            Some("127.0.0.1") | Some("localhost") | Some("[::1]")
        )
    })
}

// ── Authorization ────────────────────────────────────────────────────────

/// Build the URL to send the user to, and the secrets that must survive until
/// the redirect returns.
pub fn authorize(
    discovered: &Discovered,
    client: &ClientId,
    redirect_uri: &str,
    extra_scopes: &[String],
) -> Result<Pending, OauthError> {
    let verifier = random_token(64);
    let challenge = B64URL.encode(Sha256::digest(verifier.as_bytes()));
    let state = random_token(32);

    // The challenge's scopes are authoritative when a server sends them; none
    // of the servers seen in practice do, so fall back to what the resource
    // advertises. `scopes_supported` is the minimal useful set by convention.
    let mut scopes: Vec<String> = discovered.resource.scopes_supported.clone();
    for scope in extra_scopes {
        if !scopes.contains(scope) {
            scopes.push(scope.clone());
        }
    }
    let scope = (!scopes.is_empty()).then(|| scopes.join(" "));

    let mut params = vec![
        ("response_type", "code".to_string()),
        ("client_id", client.as_param().to_string()),
        ("redirect_uri", redirect_uri.to_string()),
        ("state", state.clone()),
        ("code_challenge", challenge),
        ("code_challenge_method", "S256".to_string()),
        ("resource", discovered.resource.resource.clone()),
    ];
    if let Some(scope) = &scope {
        params.push(("scope", scope.clone()));
    }

    let url = reqwest::Url::parse_with_params(&discovered.server.authorization_endpoint, &params)
        .map_err(|e| OauthError::Discovery(format!("building authorize URL: {e}")))?
        .to_string();

    Ok(Pending {
        url,
        state,
        verifier,
        issuer: discovered.server.issuer.clone(),
        redirect_uri: redirect_uri.to_string(),
        resource: discovered.resource.resource.clone(),
        scope,
        iss_expected: discovered
            .server
            .authorization_response_iss_parameter_supported,
    })
}

/// Check the `iss` an authorization response carried, per RFC 9207 §2.4.
///
/// A present issuer is always compared, even where metadata did not promise
/// one — servers emit `iss` before advertising it. Comparison is byte-for-byte:
/// normalizing here would defeat the check.
pub fn check_issuer(pending: &Pending, iss: Option<&str>) -> Result<(), OauthError> {
    match iss {
        Some(iss) if iss != pending.issuer => Err(OauthError::IssuerMismatch {
            expected: pending.issuer.clone(),
            got: iss.to_string(),
        }),
        Some(_) => Ok(()),
        None if pending.iss_expected => Err(OauthError::IssuerMismatch {
            expected: pending.issuer.clone(),
            got: String::new(),
        }),
        None => Ok(()),
    }
}

/// Exchange an authorization code. Validates `iss` first: a mismatched
/// response must not reach the token endpoint.
pub async fn redeem(
    http: &Client,
    pending: &Pending,
    client: &ClientId,
    token_endpoint: &str,
    code: &str,
    iss: Option<&str>,
) -> Result<Tokens, OauthError> {
    check_issuer(pending, iss)?;
    let form = vec![
        ("grant_type", "authorization_code".to_string()),
        ("code", code.to_string()),
        ("redirect_uri", pending.redirect_uri.clone()),
        ("client_id", client.as_param().to_string()),
        ("code_verifier", pending.verifier.clone()),
        ("resource", pending.resource.clone()),
    ];
    let tokens = post_token(http, token_endpoint, form, client).await?;
    Ok(build(
        tokens,
        pending.issuer.clone(),
        token_endpoint,
        &pending.resource,
        client,
    ))
}

/// Exchange a refresh token. The response may rotate the refresh token, and
/// may omit it — the previous one stays valid only in the latter case.
pub async fn refresh(http: &Client, tokens: &Tokens) -> Result<Tokens, OauthError> {
    let refresh_token = tokens
        .refresh_token
        .clone()
        .ok_or_else(|| OauthError::Token("no refresh token; authorize again".into()))?;
    let form = vec![
        ("grant_type", "refresh_token".to_string()),
        ("refresh_token", refresh_token.clone()),
        ("client_id", tokens.client.as_param().to_string()),
        ("resource", tokens.resource.clone()),
    ];
    let response = post_token(http, &tokens.token_endpoint, form, &tokens.client).await?;
    let mut next = build(
        response,
        tokens.issuer.clone(),
        &tokens.token_endpoint,
        &tokens.resource,
        &tokens.client,
    );
    if next.refresh_token.is_none() {
        next.refresh_token = Some(refresh_token);
    }
    Ok(next)
}

async fn post_token(
    http: &Client,
    endpoint: &str,
    form: Vec<(&str, String)>,
    client: &ClientId,
) -> Result<TokenResponse, OauthError> {
    let mut request = http.post(endpoint).form(&form);
    if let Some(secret) = client.secret() {
        request = request.basic_auth(client.as_param(), Some(secret));
    }
    let response = request
        .send()
        .await
        .map_err(|e| OauthError::Token(format!("token request to {endpoint}: {e}")))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|e| OauthError::Token(format!("reading token response: {e}")))?;
    if !status.is_success() {
        return Err(OauthError::Token(format!(
            "token request refused (HTTP {}): {body}",
            status.as_u16()
        )));
    }
    serde_json::from_str(&body)
        .map_err(|e| OauthError::Token(format!("decoding token response: {e}")))
}

fn build(
    response: TokenResponse,
    issuer: String,
    token_endpoint: &str,
    resource: &str,
    client: &ClientId,
) -> Tokens {
    Tokens {
        access_token: response.access_token,
        refresh_token: response.refresh_token,
        expires_at: response
            .expires_in
            .map(|secs| Utc::now() + Duration::seconds(secs)),
        scope: response.scope,
        issuer,
        token_endpoint: token_endpoint.to_string(),
        resource: resource.to_string(),
        client: client.clone(),
    }
}

fn random_token(bytes: usize) -> String {
    let mut buf = vec![0u8; bytes];
    rand::rng().fill(&mut buf[..]);
    B64URL.encode(buf)
}

// ── Resolving a stored credential ────────────────────────────────────────

/// Where authorized credentials live. Keyed by the connection's URL: that is
/// what `substructure.toml` names and what the token is bound to, so two ids
/// pointing at one server share a login.
#[async_trait::async_trait]
pub trait TokenStore: Send + Sync {
    async fn get(&self, tenant_id: &str, url: &str) -> Option<Tokens>;
    /// Called on every refresh, because a rotated refresh token invalidates
    /// the one it replaced. A failure to persist is reported, not swallowed:
    /// silently dropping a rotated token locks the connection out.
    async fn put(&self, tenant_id: &str, url: &str, tokens: Tokens) -> Result<(), String>;
}

/// Credentials for connections authorized through OAuth.
///
/// Refresh happens here rather than at login: a long-running agent outlives
/// its access token, and the resolver is the only place that sees every call.
pub struct StoredCredentials {
    store: std::sync::Arc<dyn TokenStore>,
    http: Client,
    /// One refresh at a time per connection. Rotation makes concurrent
    /// refreshes a correctness problem, not just wasted work: the loser
    /// persists a refresh token the server has already retired.
    refreshing: tokio::sync::Mutex<
        std::collections::HashMap<String, std::sync::Arc<tokio::sync::Mutex<()>>>,
    >,
}

impl StoredCredentials {
    pub fn new(store: std::sync::Arc<dyn TokenStore>) -> Self {
        Self {
            store,
            http: Client::new(),
            refreshing: tokio::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// A live access token, refreshed if it is close to expiry.
    pub async fn access_token(
        &self,
        tenant_id: &str,
        url: &str,
    ) -> Result<Option<String>, OauthError> {
        let Some(tokens) = self.store.get(tenant_id, url).await else {
            return Ok(None);
        };
        if !tokens.stale() {
            return Ok(Some(tokens.access_token));
        }
        if !tokens.refreshable() {
            return Err(OauthError::Token(
                "the access token expired and the server issued no refresh token".into(),
            ));
        }

        let gate = {
            let mut held = self.refreshing.lock().await;
            held.entry(format!("{tenant_id}\u{0}{url}"))
                .or_default()
                .clone()
        };
        let _guard = gate.lock().await;

        // Another caller may have refreshed while we waited for the gate.
        if let Some(current) = self.store.get(tenant_id, url).await {
            if !current.stale() {
                return Ok(Some(current.access_token));
            }
        }

        let next = refresh(&self.http, &tokens).await?;
        self.store
            .put(tenant_id, url, next.clone())
            .await
            .map_err(OauthError::Token)?;
        Ok(Some(next.access_token))
    }
}

/// Connections keep resolving through `token_env` where the project file names
/// one; only a connection that names no variable consults the store. A server
/// needing no credential still gets none, so declaring nothing keeps working.
#[async_trait::async_trait]
impl crate::connectors::registry::CredentialResolver for StoredCredentials {
    async fn resolve(
        &self,
        tenant_id: &str,
        id: &str,
        spec: &crate::connectors::registry::ConnectionSpec,
    ) -> Result<reqwest::header::HeaderMap, ConnectorError> {
        if spec.auth.is_some() {
            return crate::connectors::registry::EnvCredentials
                .resolve(tenant_id, id, spec)
                .await;
        }
        match self.access_token(tenant_id, &spec.url).await {
            // Checked again here, not only at login: the project file can name
            // a different URL than the one authorized.
            Ok(Some(token)) => {
                require_secure(&spec.url).map_err(|e| ConnectorError::permanent(e.to_string()))?;
                crate::connectors::mcp::auth_headers(None, &token)
            }
            // Nothing stored: either a server wanting no credential, or one
            // never logged in to. Its 401 tells them apart.
            Ok(None) => Ok(reqwest::header::HeaderMap::new()),
            Err(e) => Err(ConnectorError::unauthorized(format!(
                "connection `{id}` needs authorizing again: run `subs mcp login {id}` ({e})"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn discovered() -> Discovered {
        Discovered {
            resource: ProtectedResource {
                resource: "https://mcp.example.test/mcp".into(),
                authorization_servers: vec!["https://mcp.example.test".into()],
                scopes_supported: vec!["org:read".into()],
            },
            server: AuthServer {
                issuer: "https://mcp.example.test".into(),
                authorization_endpoint: "https://mcp.example.test/oauth/authorize".into(),
                token_endpoint: "https://mcp.example.test/oauth/token".into(),
                registration_endpoint: Some("https://mcp.example.test/register".into()),
                scopes_supported: vec![],
                code_challenge_methods_supported: vec!["S256".into()],
                client_id_metadata_document_supported: true,
                authorization_response_iss_parameter_supported: true,
            },
        }
    }

    #[test]
    fn a_challenge_yields_the_metadata_pointer() {
        let challenge = r#"Bearer realm="OAuth", error="invalid_token", resource_metadata="https://mcp.sentry.dev/.well-known/oauth-protected-resource/mcp""#;
        assert_eq!(
            challenge_param(challenge, "resource_metadata").as_deref(),
            Some("https://mcp.sentry.dev/.well-known/oauth-protected-resource/mcp")
        );
        // Stripe sends this one unquoted.
        assert_eq!(
            challenge_param(
                "Bearer resource_metadata=https://mcp.stripe.com/.well-known/oauth-protected-resource",
                "resource_metadata"
            )
            .as_deref(),
            Some("https://mcp.stripe.com/.well-known/oauth-protected-resource")
        );
        assert_eq!(challenge_param("Bearer realm=\"OAuth\"", "scope"), None);
    }

    #[test]
    fn well_known_paths_insert_rather_than_append() {
        assert_eq!(
            prm_url("https://mcp.sentry.dev/mcp").unwrap(),
            "https://mcp.sentry.dev/.well-known/oauth-protected-resource/mcp"
        );
        assert_eq!(
            prm_url("https://mcp.example.test/").unwrap(),
            "https://mcp.example.test/.well-known/oauth-protected-resource"
        );

        // GitHub's issuer has a path and answers only at the inserted form.
        let urls = metadata_urls("https://github.com/login/oauth").unwrap();
        assert_eq!(
            urls[0],
            "https://github.com/.well-known/oauth-authorization-server/login/oauth"
        );
        assert!(urls.contains(
            &"https://github.com/login/oauth/.well-known/openid-configuration".to_string()
        ));
    }

    #[test]
    fn an_authorize_url_carries_pkce_the_resource_and_scopes() {
        let pending = authorize(
            &discovered(),
            &ClientId::Metadata {
                url: "https://app.test/client.json".into(),
            },
            "http://127.0.0.1:7777/callback",
            &["project:write".into()],
        )
        .unwrap();

        let url = reqwest::Url::parse(&pending.url).unwrap();
        let params: std::collections::HashMap<_, _> = url.query_pairs().collect();
        assert_eq!(params["code_challenge_method"], "S256");
        assert_eq!(params["resource"], "https://mcp.example.test/mcp");
        assert_eq!(params["client_id"], "https://app.test/client.json");
        assert_eq!(params["scope"], "org:read project:write");
        assert_eq!(
            params["code_challenge"],
            B64URL.encode(Sha256::digest(pending.verifier.as_bytes())),
            "the challenge must verify against the verifier we kept"
        );
        assert_ne!(pending.state, pending.verifier);
    }

    #[test]
    fn issuer_validation_follows_the_rfc_9207_table() {
        let mut pending = authorize(
            &discovered(),
            &ClientId::Metadata {
                url: "https://app.test/client.json".into(),
            },
            "http://127.0.0.1:7777/callback",
            &[],
        )
        .unwrap();

        assert!(check_issuer(&pending, Some("https://mcp.example.test")).is_ok());
        assert!(matches!(
            check_issuer(&pending, Some("https://evil.test")),
            Err(OauthError::IssuerMismatch { .. })
        ));
        // Advertised but absent: reject.
        assert!(check_issuer(&pending, None).is_err());

        // Not advertised: a present iss is still compared, an absent one passes.
        pending.iss_expected = false;
        assert!(check_issuer(&pending, None).is_ok());
        assert!(check_issuer(&pending, Some("https://evil.test")).is_err());
    }

    #[test]
    fn no_normalization_creeps_into_the_issuer_comparison() {
        let mut pending = authorize(
            &discovered(),
            &ClientId::Metadata {
                url: "https://app.test/client.json".into(),
            },
            "http://127.0.0.1:7777/callback",
            &[],
        )
        .unwrap();
        pending.issuer = "https://mcp.example.test".into();
        for near_miss in [
            "https://mcp.example.test/",
            "https://MCP.example.test",
            "https://mcp.example.test:443",
        ] {
            assert!(
                check_issuer(&pending, Some(near_miss)).is_err(),
                "{near_miss} must not compare equal"
            );
        }
    }

    #[test]
    fn staleness_uses_the_skew_and_treats_no_expiry_as_live() {
        let base = Tokens {
            access_token: "a".into(),
            refresh_token: None,
            expires_at: None,
            scope: None,
            issuer: "https://mcp.example.test".into(),
            token_endpoint: "https://mcp.example.test/oauth/token".into(),
            resource: "https://mcp.example.test/mcp".into(),
            client: ClientId::Metadata {
                url: "https://app.test/client.json".into(),
            },
        };
        assert!(!base.stale());
        assert!(!base.refreshable());

        let fresh = Tokens {
            expires_at: Some(Utc::now() + Duration::seconds(REFRESH_SKEW + 60)),
            ..base.clone()
        };
        assert!(!fresh.stale());

        let expiring = Tokens {
            expires_at: Some(Utc::now() + Duration::seconds(REFRESH_SKEW - 10)),
            ..base.clone()
        };
        assert!(expiring.stale(), "refresh before the token actually dies");
    }

    /// Discovery against the real servers. Ignored by default: it needs the
    /// network, and it is the only check that the metadata locations and
    /// challenge parsing match what is actually deployed.
    #[tokio::test]
    #[ignore = "network"]
    async fn discovery_resolves_the_servers_people_actually_connect() {
        let http = Client::new();
        for (url, issuer) in [
            ("https://mcp.sentry.dev/mcp", "https://mcp.sentry.dev"),
            ("https://mcp.linear.app/mcp", "https://mcp.linear.app"),
            ("https://mcp.notion.com/mcp", "https://mcp.notion.com"),
            (
                "https://api.githubcopilot.com/mcp/",
                "https://github.com/login/oauth",
            ),
        ] {
            let found = discover(&http, url)
                .await
                .unwrap_or_else(|e| panic!("{url}: {e}"));
            assert_eq!(found.server.issuer, issuer, "{url}");
            assert!(!found.server.authorization_endpoint.is_empty(), "{url}");
            assert!(!found.server.token_endpoint.is_empty(), "{url}");
            assert!(
                found.server.client_id_metadata_document_supported
                    || found.server.registration_endpoint.is_some()
                    || found.server.issuer == "https://github.com/login/oauth",
                "{url} offers no registration mechanism we support"
            );
        }
    }

    #[test]
    fn a_credential_never_goes_out_over_cleartext() {
        assert!(require_secure("https://mcp.linear.app/mcp").is_ok());
        // A local server has no certificate and nothing off-host to intercept.
        assert!(require_secure("http://127.0.0.1:8080/mcp").is_ok());
        assert!(require_secure("http://localhost:8080/mcp").is_ok());

        let err = require_secure("http://mcp.linear.app/mcp").unwrap_err();
        assert!(err.to_string().contains("not https"), "{err}");
    }

    #[test]
    fn metadata_must_describe_the_server_that_pointed_at_it() {
        assert!(same_origin(
            "https://mcp.linear.app/mcp",
            "https://mcp.linear.app/mcp"
        ));
        // GitHub publishes its resource with a trailing slash; still its own.
        assert!(same_origin(
            "https://api.githubcopilot.com/mcp/",
            "https://api.githubcopilot.com/mcp"
        ));
        assert!(!same_origin(
            "https://evil.test/mcp",
            "https://mcp.linear.app/mcp"
        ));
        assert!(!same_origin(
            "https://mcp.linear.app.evil.test/mcp",
            "https://mcp.linear.app/mcp"
        ));
    }

    #[test]
    fn loopback_registers_as_native() {
        assert!(is_loopback("http://127.0.0.1:7777/callback"));
        assert!(is_loopback("http://localhost:7777/callback"));
        assert!(!is_loopback(
            "https://subs.acme.test/api/mcp/oauth/callback"
        ));
    }
}
