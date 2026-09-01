use base64::engine::general_purpose::URL_SAFE_NO_PAD as B64URL;
use base64::Engine;
use chrono::{DateTime, Duration, Utc};
use reqwest::header::WWW_AUTHENTICATE;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const REFRESH_SKEW: i64 = 120;

#[derive(Debug, thiserror::Error)]
pub enum OauthError {
    #[error("{0}")]
    Discovery(String),
    #[error("{0}")]
    Registration(String),
    #[error("{0}")]
    Token(String),
    #[error("{code}{}", .description.as_deref().map(|d| format!(": {d}")).unwrap_or_default())]
    Refused {
        code: String,
        description: Option<String>,
    },
    #[error("{0}")]
    Unrenewable(String),
    #[error("authorization server mismatch: expected `{expected}`, got `{got}`")]
    IssuerMismatch { expected: String, got: String },
}

impl OauthError {
    pub fn is_spent(&self) -> bool {
        match self {
            Self::Unrenewable(_) => true,
            Self::Refused { code, .. } => matches!(
                code.as_str(),
                "invalid_grant" | "invalid_client" | "unauthorized_client"
            ),
            _ => false,
        }
    }
}

#[derive(Debug, Deserialize)]
struct ErrorResponse {
    error: String,
    #[serde(default)]
    error_description: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProtectedResource {
    pub resource: String,
    #[serde(default)]
    pub authorization_servers: Vec<String>,
    #[serde(default)]
    pub scopes_supported: Vec<String>,
}

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ClientId {
    Metadata {
        url: String,
    },
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

#[derive(Debug, Clone)]
pub struct Discovered {
    pub resource: ProtectedResource,
    pub server: AuthServer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pending {
    pub url: String,
    pub state: String,
    pub verifier: String,
    pub issuer: String,
    pub redirect_uri: String,
    pub resource: String,
    pub scope: Option<String>,
    pub iss_expected: bool,
}

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
    pub resource: String,
    pub client: ClientId,
}

impl Tokens {
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

pub async fn discover(http: &Client, mcp_url: &str) -> Result<Discovered, OauthError> {
    let metadata_url = match probe(http, mcp_url).await {
        Some(url) => url,
        None => prm_url(mcp_url)?,
    };
    let resource: ProtectedResource = fetch_json(http, &metadata_url)
        .await
        .map_err(|e| OauthError::Discovery(format!("reading {metadata_url}: {e}")))?;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Probed {
    NoChallenge,
    Oauth,
    Protected,
}

pub async fn sniff(http: &Client, mcp_url: &str) -> Result<Probed, OauthError> {
    let answered = unauthenticated(http, mcp_url).await?;
    if !answered.challenged {
        return Ok(Probed::NoChallenge);
    }
    if answered.metadata.is_some() {
        return Ok(Probed::Oauth);
    }
    match fetch_json::<ProtectedResource>(http, &prm_url(mcp_url)?).await {
        Ok(_) => Ok(Probed::Oauth),
        Err(_) => Ok(Probed::Protected),
    }
}

struct Answered {
    challenged: bool,
    metadata: Option<String>,
}

async fn unauthenticated(http: &Client, mcp_url: &str) -> Result<Answered, OauthError> {
    let response = http
        .post(mcp_url)
        .header(
            reqwest::header::ACCEPT,
            "application/json, text/event-stream",
        )
        .json(&serde_json::json!({ "jsonrpc": "2.0", "id": 0, "method": "tools/list" }))
        .send()
        .await
        .map_err(|e| OauthError::Discovery(format!("could not reach {mcp_url}: {e}")))?;
    let challenged = matches!(response.status().as_u16(), 401 | 403);
    let metadata = response
        .headers()
        .get(WWW_AUTHENTICATE)
        .and_then(|v| v.to_str().ok())
        .and_then(|challenge| challenge_param(challenge, "resource_metadata"));
    Ok(Answered {
        challenged: challenged || metadata.is_some(),
        metadata,
    })
}

async fn probe(http: &Client, mcp_url: &str) -> Option<String> {
    unauthenticated(http, mcp_url).await.ok()?.metadata
}

pub fn challenge_param(challenge: &str, name: &str) -> Option<String> {
    for part in challenge.split(',') {
        let (key, value) = part.split_once('=')?;
        if key.trim().trim_start_matches("Bearer").trim() == name {
            return Some(value.trim().trim_matches('"').to_string());
        }
    }
    None
}

pub(crate) fn same_origin(a: &str, b: &str) -> bool {
    match (reqwest::Url::parse(a), reqwest::Url::parse(b)) {
        (Ok(a), Ok(b)) => a.origin() == b.origin(),
        _ => false,
    }
}

fn prm_url(mcp_url: &str) -> Result<String, OauthError> {
    let url = reqwest::Url::parse(mcp_url)
        .map_err(|e| OauthError::Discovery(format!("`{mcp_url}` is not a URL: {e}")))?;
    let origin = url.origin().ascii_serialization();
    let path = url.path().trim_end_matches('/');
    Ok(format!(
        "{origin}/.well-known/oauth-protected-resource{path}"
    ))
}

async fn fetch_auth_server(http: &Client, issuer: &str) -> Result<AuthServer, OauthError> {
    let mut last = String::new();
    for candidate in metadata_urls(issuer)? {
        match fetch_json::<AuthServer>(http, &candidate).await {
            Ok(server) => {
                if !same_issuer(&server.issuer, issuer) {
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

fn same_issuer(document: &str, asked: &str) -> bool {
    document.trim_end_matches('/') == asked.trim_end_matches('/')
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
            "`{}` does not support dynamic client registration",
            server.issuer
        ))
    })?;
    register(http, endpoint, redirect_uri, client_name).await
}

async fn register(
    http: &Client,
    endpoint: &str,
    redirect_uri: &str,
    client_name: &str,
) -> Result<ClientId, OauthError> {
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

pub(crate) fn is_loopback(redirect_uri: &str) -> bool {
    reqwest::Url::parse(redirect_uri).is_ok_and(|u| {
        matches!(
            u.host_str(),
            Some("127.0.0.1") | Some("localhost") | Some("[::1]")
        )
    })
}

pub fn authorize(
    discovered: &Discovered,
    client: &ClientId,
    redirect_uri: &str,
    scopes: &[String],
) -> Result<Pending, OauthError> {
    let verifier = crate::runtime::secret::random_token(64);
    let challenge = B64URL.encode(Sha256::digest(verifier.as_bytes()));
    let state = crate::runtime::secret::random_token(32);

    let scopes: &[String] = if scopes.is_empty() {
        &discovered.resource.scopes_supported
    } else {
        scopes
    };
    let scope = (!scopes.is_empty()).then(|| scopes.join(" "));

    let mut params = vec![
        ("response_type", "code".to_string()),
        ("client_id", client.as_param().to_string()),
        ("redirect_uri", redirect_uri.to_string()),
        ("state", state.clone()),
        ("code_challenge", challenge),
        ("code_challenge_method", "S256".to_string()),
        ("resource", discovered.resource.resource.clone()),
        ("access_type", "offline".to_string()),
        ("prompt", "consent".to_string()),
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

pub async fn refresh(http: &Client, tokens: &Tokens) -> Result<Tokens, OauthError> {
    let refresh_token = tokens
        .refresh_token
        .clone()
        .ok_or_else(|| OauthError::Unrenewable("no refresh token; authorize again".into()))?;
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
        return Err(match serde_json::from_str::<ErrorResponse>(&body) {
            Ok(refusal) => OauthError::Refused {
                code: refusal.error,
                description: refusal.error_description,
            },
            Err(_) => OauthError::Token(format!(
                "token request refused (HTTP {}): {body}",
                status.as_u16()
            )),
        });
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
        assert_eq!(params["scope"], "project:write");
        assert_eq!(params["access_type"], "offline");
        assert_eq!(params["prompt"], "consent");
        assert_eq!(
            params["code_challenge"],
            B64URL.encode(Sha256::digest(pending.verifier.as_bytes())),
            "the challenge must verify against the verifier we kept"
        );
        assert_ne!(pending.state, pending.verifier);
    }

    #[test]
    fn given_scopes_replace_what_the_resource_advertises() {
        let ask = |scopes: &[String]| {
            let pending = authorize(
                &discovered(),
                &ClientId::Metadata {
                    url: "https://app.test/client.json".into(),
                },
                "http://127.0.0.1:7777/callback",
                scopes,
            )
            .unwrap();
            pending.scope
        };

        assert_eq!(
            ask(&[]).as_deref(),
            Some("org:read"),
            "none given: fall back"
        );
        assert_eq!(
            ask(&["gmail.readonly".into()]).as_deref(),
            Some("gmail.readonly"),
            "the advertised `org:read` must not survive"
        );

        let mut narrower = discovered();
        narrower.resource.scopes_supported = vec![];
        let pending = authorize(
            &narrower,
            &ClientId::Metadata {
                url: "https://app.test/client.json".into(),
            },
            "http://127.0.0.1:7777/callback",
            &[],
        )
        .unwrap();
        assert_eq!(pending.scope, None, "nothing to ask for is no scope param");
    }

    #[test]
    fn an_issuer_may_differ_by_a_trailing_slash_at_discovery() {
        assert!(same_issuer(
            "https://accounts.google.com",
            "https://accounts.google.com/"
        ));
        assert!(same_issuer("https://a.test/oauth", "https://a.test/oauth/"));
        assert!(!same_issuer("https://evil.test", "https://a.test"));
        assert!(!same_issuer("https://a.test/oauth", "https://a.test/other"));
        assert!(!same_issuer("https://A.test", "https://a.test"));
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
        assert!(check_issuer(&pending, None).is_err());

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

    #[tokio::test]
    #[ignore = "network"]
    async fn discovery_resolves_the_servers_people_actually_connect() {
        let http = Client::new();
        for (url, issuer, self_registers) in [
            ("https://mcp.sentry.dev/mcp", "https://mcp.sentry.dev", true),
            ("https://mcp.linear.app/mcp", "https://mcp.linear.app", true),
            ("https://mcp.notion.com/mcp", "https://mcp.notion.com", true),
            (
                "https://api.githubcopilot.com/mcp/",
                "https://github.com/login/oauth",
                false,
            ),
            (
                "https://gmailmcp.googleapis.com/mcp/v1",
                "https://accounts.google.com",
                false,
            ),
            (
                "https://drivemcp.googleapis.com/mcp/v1",
                "https://accounts.google.com",
                false,
            ),
        ] {
            let found = discover(&http, url)
                .await
                .unwrap_or_else(|e| panic!("{url}: {e}"));
            assert_eq!(found.server.issuer, issuer, "{url}");
            assert!(!found.server.authorization_endpoint.is_empty(), "{url}");
            assert!(!found.server.token_endpoint.is_empty(), "{url}");
            assert_eq!(
                found.server.client_id_metadata_document_supported
                    || found.server.registration_endpoint.is_some(),
                self_registers,
                "{url}"
            );
        }
    }

    #[test]
    fn metadata_must_describe_the_server_that_pointed_at_it() {
        assert!(same_origin(
            "https://mcp.linear.app/mcp",
            "https://mcp.linear.app/mcp"
        ));
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
