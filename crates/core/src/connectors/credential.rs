//! The credential a connection dials with, whichever way it was obtained.
//!
//! One slot per connection: `subs mcp login` fills it with an OAuth grant and
//! `subs mcp set-token` fills it with a static token, and the resolver below
//! turns whichever landed there into headers. Keeping them in one slot is what
//! lets a deployment hold either — the store is the seam, and neither kind ever
//! reaches the event log or the file.

use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};

use super::mcp::auth_headers;
use super::oauth::{refresh, require_secure, same_origin, OauthError, Tokens};
use super::registry::{AuthKind, ConnectionSpec, CredentialResolver};
use super::{AuthNeed, ConnectorError, Subject};

/// What the store holds for one connection.
///
/// Untagged, and `Static` first: OAuth grants were written here before static
/// tokens existed, as a bare `Tokens` object, and a stored credential outlives
/// the release that wrote it. A `Tokens` has no `token` field, so the two
/// shapes cannot be read for each other.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Credential {
    Static { token: String },
    Oauth(Box<Tokens>),
}

impl Credential {
    /// What `auth` this credential answers to, for reporting a slot holding the
    /// other kind.
    pub fn kind(&self) -> AuthKind {
        match self {
            Self::Static { .. } => AuthKind::Token,
            Self::Oauth(_) => AuthKind::Oauth,
        }
    }
}

/// Where credentials live. Keyed by the connection's id, so two ids naming one
/// server hold two of them: `[mcp.sentry]` and `[mcp.sentry2]` at the same URL
/// are two accounts.
#[async_trait::async_trait]
pub trait CredentialStore: Send + Sync {
    async fn get(
        &self,
        tenant_id: &str,
        connection_id: &str,
        subject: &Subject,
    ) -> Option<Credential>;
    /// Called on every refresh too, because a rotated refresh token invalidates
    /// the one it replaced. A failure to persist is reported, not swallowed:
    /// silently dropping a rotated token locks the connection out.
    async fn put(
        &self,
        tenant_id: &str,
        connection_id: &str,
        subject: &Subject,
        credential: Credential,
    ) -> Result<(), String>;
}

/// What the slot holds. An empty slot is a failure to one caller below and not
/// to the other.
enum Held {
    Grant(Tokens),
    Empty,
    Wrong(&'static str),
}

/// Resolves a connection's stored credential to headers.
///
/// Refresh happens here rather than at login: a long-running agent outlives its
/// access token, and the resolver is the only place that sees every call.
pub struct StoredCredentials {
    store: std::sync::Arc<dyn CredentialStore>,
    http: reqwest::Client,
    /// One refresh at a time per connection. Rotation makes concurrent
    /// refreshes a correctness problem, not just wasted work: the loser
    /// persists a refresh token the server has already retired.
    refreshing: tokio::sync::Mutex<
        std::collections::HashMap<String, std::sync::Arc<tokio::sync::Mutex<()>>>,
    >,
}

impl StoredCredentials {
    pub fn new(store: std::sync::Arc<dyn CredentialStore>) -> Self {
        Self {
            store,
            http: reqwest::Client::new(),
            refreshing: tokio::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    async fn access_token(
        &self,
        tenant_id: &str,
        connection_id: &str,
        subject: &Subject,
        url: &str,
    ) -> Result<Option<String>, OauthError> {
        let tokens = match self.grant(tenant_id, connection_id, subject, url).await? {
            Held::Grant(tokens) => tokens,
            Held::Empty => return Ok(None),
            Held::Wrong(why) => return Err(OauthError::Unrenewable(why.into())),
        };
        if !tokens.stale() {
            return Ok(Some(tokens.access_token));
        }
        if !tokens.refreshable() {
            return Err(OauthError::Unrenewable(
                "the access token expired and the server issued no refresh token".into(),
            ));
        }
        self.renew(tenant_id, connection_id, subject, &tokens)
            .await
            .map(Some)
    }

    /// Renew the grant whether or not it looks expired, because the server
    /// refused what we sent. A server can revoke a token early, thus a refusal
    /// is a better signal than the expiry we recorded.
    ///
    /// `Ok(false)` if there is nothing to renew.
    async fn refresh_now(
        &self,
        tenant_id: &str,
        connection_id: &str,
        subject: &Subject,
        url: &str,
    ) -> Result<bool, OauthError> {
        let Held::Grant(tokens) = self.grant(tenant_id, connection_id, subject, url).await? else {
            return Ok(false);
        };
        if !tokens.refreshable() {
            return Ok(false);
        }
        self.renew(tenant_id, connection_id, subject, &tokens)
            .await?;
        Ok(true)
    }

    /// What the slot holds, checked against where the connection now points.
    /// The id is the key, so an edited `url` would otherwise send one server's
    /// token to another.
    async fn grant(
        &self,
        tenant_id: &str,
        connection_id: &str,
        subject: &Subject,
        url: &str,
    ) -> Result<Held, OauthError> {
        let tokens = match self.store.get(tenant_id, connection_id, subject).await {
            Some(Credential::Oauth(tokens)) => *tokens,
            Some(Credential::Static { .. }) => {
                return Ok(Held::Wrong(
                    "a static token is stored for this connection, not an OAuth grant",
                ))
            }
            None => return Ok(Held::Empty),
        };
        if !same_origin(&tokens.resource, url) {
            return Err(OauthError::Unrenewable(format!(
                "the stored credential was issued for `{}`, not `{url}`",
                tokens.resource
            )));
        }
        Ok(Held::Grant(tokens))
    }

    /// Exchange `held` for a new access token, one caller at a time.
    async fn renew(
        &self,
        tenant_id: &str,
        connection_id: &str,
        subject: &Subject,
        held: &Tokens,
    ) -> Result<String, OauthError> {
        let gate = {
            let mut gates = self.refreshing.lock().await;
            gates
                .entry(format!("{tenant_id}\u{0}{connection_id}\u{0}{subject}"))
                .or_default()
                .clone()
        };
        let _guard = gate.lock().await;

        // Adopt a token another caller landed. To refresh again would retire
        // the one they just wrote.
        if let Some(Credential::Oauth(current)) =
            self.store.get(tenant_id, connection_id, subject).await
        {
            if current.access_token != held.access_token && !current.stale() {
                return Ok(current.access_token);
            }
        }

        let next = refresh(&self.http, held).await?;
        self.store
            .put(
                tenant_id,
                connection_id,
                subject,
                Credential::Oauth(Box::new(next.clone())),
            )
            .await
            .map_err(OauthError::Token)?;
        Ok(next.access_token)
    }

    /// The grant a connection holds, as headers, or nothing where the slot is
    /// empty.
    async fn oauth_headers(
        &self,
        tenant_id: &str,
        id: &str,
        subject: &Subject,
        spec: &ConnectionSpec,
    ) -> Result<Option<HeaderMap>, ConnectorError> {
        match self.access_token(tenant_id, id, subject, &spec.url).await {
            // Checked again here, not only at login: the project file can name
            // a different URL than the one authorized.
            Ok(Some(token)) => {
                secure(&spec.url)?;
                auth_headers(None, &token).map(Some)
            }
            Ok(None) => Ok(None),
            // A grant that cannot be renewed is reported even where the file
            // declared nothing: the connection was authorized once, so falling
            // back to sending nothing would lose that.
            Err(e) => Err(refresh_failed(id, &e)),
        }
    }

    /// The static token for one connection, sent under whichever header the
    /// connection declares.
    async fn static_headers(
        &self,
        tenant_id: &str,
        id: &str,
        subject: &Subject,
        spec: &ConnectionSpec,
    ) -> Result<HeaderMap, ConnectorError> {
        match self.store.get(tenant_id, id, subject).await {
            Some(Credential::Static { token }) => {
                secure(&spec.url)?;
                auth_headers(spec.header.as_deref(), &token)
            }
            Some(Credential::Oauth(_)) => Err(mismatch(id, AuthKind::Token)),
            None => Err(ConnectorError::unauthorized(
                AuthNeed::NeverAuthorized,
                format!("connection `{id}` has no token: run `subs mcp set-token {id}`"),
            )),
        }
    }
}

/// Where the file declares a method, that is what this sends. Where it does
/// not — the usual case — the slot decides: a grant is sent, and an empty slot
/// sends nothing and lets the server say whether it minded. Asking the server
/// is what discovery is, and its refusal is the answer.
#[async_trait::async_trait]
impl CredentialResolver for StoredCredentials {
    async fn resolve(
        &self,
        tenant_id: &str,
        id: &str,
        subject: &Subject,
        spec: &ConnectionSpec,
    ) -> Result<HeaderMap, ConnectorError> {
        match spec.auth {
            Some(AuthKind::None) => Ok(HeaderMap::new()),
            Some(AuthKind::Token) => self.static_headers(tenant_id, id, subject, spec).await,
            Some(AuthKind::Oauth) => {
                match self.oauth_headers(tenant_id, id, subject, spec).await? {
                    Some(headers) => Ok(headers),
                    None => Err(ConnectorError::unauthorized(
                        AuthNeed::NeverAuthorized,
                        format!("connection `{id}` is not authorized: run `subs mcp login {id}`"),
                    )),
                }
            }
            None => Ok(self
                .oauth_headers(tenant_id, id, subject, spec)
                .await?
                .unwrap_or_default()),
        }
    }

    async fn refresh(
        &self,
        tenant_id: &str,
        id: &str,
        subject: &Subject,
        spec: &ConnectionSpec,
    ) -> Result<bool, ConnectorError> {
        match spec.auth {
            Some(AuthKind::None) | Some(AuthKind::Token) => Ok(false),
            _ => self
                .refresh_now(tenant_id, id, subject, &spec.url)
                .await
                .map_err(|e| refresh_failed(id, &e)),
        }
    }
}

/// A refresh that failed, as either a spent grant or a passing fault. Logging
/// in corrects every spent case.
fn refresh_failed(id: &str, e: &OauthError) -> ConnectorError {
    if !e.is_spent() {
        return ConnectorError::retryable(format!("connection `{id}`: token refresh failed ({e})"));
    }
    let what = match e {
        OauthError::Refused { code, .. } if code == "invalid_grant" => {
            "the server retired its refresh token"
        }
        OauthError::Refused { .. } => "the server no longer knows this client",
        _ => "its grant cannot be renewed",
    };
    ConnectorError::unauthorized(
        AuthNeed::Reauthorize,
        format!("connection `{id}`: {what}. Run `subs mcp login {id}` ({e})"),
    )
}

fn secure(url: &str) -> Result<(), ConnectorError> {
    require_secure(url).map_err(|e| ConnectorError::permanent(e.to_string()))
}

fn mismatch(id: &str, declared: AuthKind) -> ConnectorError {
    let (holds, fix) = match declared {
        AuthKind::Token => ("an OAuth grant", format!("subs mcp set-token {id}")),
        _ => ("a static token", format!("subs mcp login {id}")),
    };
    ConnectorError::unauthorized(
        AuthNeed::NeverAuthorized,
        format!(
            "connection `{id}` declares `auth = \"{}\"` but holds {holds}: run `{fix}`",
            declared.as_str()
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::ConnectorProtocol;
    use std::sync::Arc;

    struct Slot(Option<Credential>);

    #[async_trait::async_trait]
    impl CredentialStore for Slot {
        async fn get(&self, _: &str, _: &str, _: &Subject) -> Option<Credential> {
            self.0.clone()
        }
        async fn put(&self, _: &str, _: &str, _: &Subject, _: Credential) -> Result<(), String> {
            Ok(())
        }
    }

    fn spec(auth: Option<AuthKind>, header: Option<&str>) -> ConnectionSpec {
        ConnectionSpec {
            url: "https://example.test/mcp".to_string(),
            protocol: ConnectorProtocol::Mcp,
            auth,
            header: header.map(str::to_string),
            credential: None,
            prefix_tools: true,
        }
    }

    fn resolver(held: Option<Credential>) -> StoredCredentials {
        StoredCredentials::new(Arc::new(Slot(held)))
    }

    fn stored_token() -> Credential {
        Credential::Static {
            token: "tok".into(),
        }
    }

    fn granted() -> Credential {
        Credential::Oauth(Box::new(Tokens {
            access_token: "at".into(),
            refresh_token: None,
            expires_at: None,
            scope: None,
            issuer: "https://example.test".into(),
            token_endpoint: "https://example.test/token".into(),
            resource: "https://example.test/mcp".into(),
            client: crate::connectors::oauth::ClientId::Registered {
                client_id: "c".into(),
                client_secret: None,
            },
        }))
    }

    #[tokio::test]
    async fn a_static_token_defaults_to_bearer() {
        let headers = resolver(Some(stored_token()))
            .resolve(
                "t",
                "github",
                &Subject::Shared,
                &spec(Some(AuthKind::Token), None),
            )
            .await
            .unwrap();
        assert_eq!(headers.get("authorization").unwrap(), "Bearer tok");
    }

    #[tokio::test]
    async fn a_declared_header_carries_the_token_raw() {
        let headers = resolver(Some(stored_token()))
            .resolve(
                "t",
                "sentry",
                &Subject::Shared,
                &spec(Some(AuthKind::Token), Some("sentry-bearer")),
            )
            .await
            .unwrap();
        assert_eq!(headers.get("sentry-bearer").unwrap(), "tok");
        assert!(headers.get("authorization").is_none());
    }

    #[tokio::test]
    async fn a_token_connection_with_an_empty_slot_asks_for_one() {
        let err = resolver(None)
            .resolve(
                "t",
                "github",
                &Subject::Shared,
                &spec(Some(AuthKind::Token), None),
            )
            .await
            .unwrap_err();
        assert_eq!(
            err.auth,
            Some(AuthNeed::NeverAuthorized),
            "an empty slot was never authorized; nothing is spent and nothing needs replacing"
        );
        assert!(err.to_string().contains("subs mcp set-token github"));
    }

    #[tokio::test]
    async fn a_slot_holding_the_other_kind_is_reported_not_sent() {
        let err = resolver(Some(stored_token()))
            .resolve(
                "t",
                "sentry",
                &Subject::Shared,
                &spec(Some(AuthKind::Oauth), None),
            )
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("subs mcp login sentry"),
            "got {err}"
        );
    }

    #[test]
    fn a_passing_fault_at_the_token_endpoint_is_not_a_dead_grant() {
        let transient = refresh_failed("sentry", &OauthError::Token("connection reset".into()));
        assert_eq!(transient.auth, None);
        assert!(transient.retryable);

        let dead = refresh_failed(
            "sentry",
            &OauthError::Refused {
                code: "invalid_grant".into(),
                description: None,
            },
        );
        assert_eq!(dead.auth, Some(AuthNeed::Reauthorize));
        assert!(!dead.retryable);
        assert!(dead.to_string().contains("retired its refresh token"));
    }

    #[test]
    fn a_forgotten_client_registration_asks_for_the_same_correction() {
        let err = refresh_failed(
            "sentry",
            &OauthError::Refused {
                code: "invalid_client".into(),
                description: None,
            },
        );
        assert_eq!(err.auth, Some(AuthNeed::Reauthorize));
        assert!(
            err.to_string().contains("subs mcp login sentry"),
            "got {err}"
        );
    }

    /// The default. Nothing declared and nothing held sends nothing, so an open
    /// server works with a bare URL and a protected one answers with its own
    /// 401 — which is the discovery.
    #[tokio::test]
    async fn an_undeclared_connection_sends_what_it_holds_or_nothing() {
        let headers = resolver(None)
            .resolve("t", "open", &Subject::Shared, &spec(None, None))
            .await
            .unwrap();
        assert!(headers.is_empty());

        let headers = resolver(Some(granted()))
            .resolve("t", "linear", &Subject::Shared, &spec(None, None))
            .await
            .unwrap();
        assert_eq!(headers.get("authorization").unwrap(), "Bearer at");
    }

    #[tokio::test]
    async fn auth_none_sends_nothing_and_asks_for_nothing() {
        let headers = resolver(None)
            .resolve(
                "t",
                "open",
                &Subject::Shared,
                &spec(Some(AuthKind::None), None),
            )
            .await
            .unwrap();
        assert!(headers.is_empty());
    }

    #[tokio::test]
    async fn a_token_never_crosses_plaintext_http() {
        let mut spec = spec(Some(AuthKind::Token), None);
        spec.url = "http://example.test/mcp".to_string();
        let err = resolver(Some(stored_token()))
            .resolve("t", "github", &Subject::Shared, &spec)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not https"), "got {err}");
    }

    #[test]
    fn an_oauth_grant_written_before_this_enum_still_reads() {
        let stored = r#"{
            "access_token": "at",
            "issuer": "https://mcp.linear.app",
            "token_endpoint": "https://mcp.linear.app/token",
            "resource": "https://mcp.linear.app/mcp",
            "client": {"kind": "registered", "client_id": "c"}
        }"#;
        let credential: Credential = serde_json::from_str(stored).unwrap();
        assert_eq!(credential.kind(), AuthKind::Oauth);
    }
}
