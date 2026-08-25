use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};

use super::mcp::auth_headers;
use super::oauth::{refresh, same_origin, OauthError, Tokens};
use super::registry::{AuthKind, ConnectionPath, ConnectionSpec, CredentialResolver};
use super::{AuthNeed, ConnectorError, Slot};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Credential {
    Static { token: String },
    Oauth(Box<Tokens>),
}

impl Credential {
    pub fn kind(&self) -> AuthKind {
        match self {
            Self::Static { .. } => AuthKind::Token,
            Self::Oauth(_) => AuthKind::Oauth,
        }
    }
}

#[async_trait::async_trait]
pub trait CredentialStore: Send + Sync {
    async fn get(&self, tenant_id: &str, connection_id: &str, subject: &Slot)
        -> Option<Credential>;
    async fn put(
        &self,
        tenant_id: &str,
        connection_id: &str,
        subject: &Slot,
        credential: Credential,
    ) -> Result<(), String>;
}

enum Held {
    Grant(Tokens),
    Empty,
    Wrong(&'static str),
}

pub struct StoredCredentials {
    store: std::sync::Arc<dyn CredentialStore>,
    http: reqwest::Client,
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
        subject: &Slot,
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

    async fn refresh_now(
        &self,
        tenant_id: &str,
        connection_id: &str,
        subject: &Slot,
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

    async fn grant(
        &self,
        tenant_id: &str,
        connection_id: &str,
        subject: &Slot,
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

    async fn renew(
        &self,
        tenant_id: &str,
        connection_id: &str,
        subject: &Slot,
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

    async fn oauth_headers(
        &self,
        tenant_id: &str,
        id: &str,
        subject: &Slot,
        spec: &ConnectionSpec,
    ) -> Result<Option<HeaderMap>, ConnectorError> {
        match self
            .access_token(tenant_id, id, subject, &spec.decl.url)
            .await
        {
            Ok(Some(token)) => auth_headers(None, &token).map(Some),
            Ok(None) => Ok(None),
            Err(e) => Err(refresh_failed(&spec.path, &e)),
        }
    }

    async fn static_headers(
        &self,
        tenant_id: &str,
        id: &str,
        subject: &Slot,
        spec: &ConnectionSpec,
    ) -> Result<HeaderMap, ConnectorError> {
        match self.store.get(tenant_id, id, subject).await {
            Some(Credential::Static { token }) => auth_headers(spec.decl.header.as_deref(), &token),
            Some(Credential::Oauth(_)) => Err(mismatch(spec, AuthKind::Token)),
            None => Err(ConnectorError::unauthorized(
                AuthNeed::NeverAuthorized,
                format!(
                    "connection `{}` has no token: run `subs auth {}`",
                    spec.path, spec.path
                ),
            )),
        }
    }
}

#[async_trait::async_trait]
impl CredentialResolver for StoredCredentials {
    async fn resolve(
        &self,
        tenant_id: &str,
        id: &str,
        subject: &Slot,
        spec: &ConnectionSpec,
    ) -> Result<HeaderMap, ConnectorError> {
        match spec.decl.auth {
            Some(AuthKind::None) => Ok(HeaderMap::new()),
            Some(AuthKind::Token) => self.static_headers(tenant_id, id, subject, spec).await,
            Some(AuthKind::Oauth) => {
                match self.oauth_headers(tenant_id, id, subject, spec).await? {
                    Some(headers) => Ok(headers),
                    None => Err(ConnectorError::unauthorized(
                        AuthNeed::NeverAuthorized,
                        format!(
                            "connection `{}` is not authorized: run `subs auth {}`",
                            spec.path, spec.path
                        ),
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
        subject: &Slot,
        spec: &ConnectionSpec,
    ) -> Result<bool, ConnectorError> {
        match spec.decl.auth {
            Some(AuthKind::None) | Some(AuthKind::Token) => Ok(false),
            _ => self
                .refresh_now(tenant_id, id, subject, &spec.decl.url)
                .await
                .map_err(|e| refresh_failed(&spec.path, &e)),
        }
    }
}

fn refresh_failed(path: &ConnectionPath, e: &OauthError) -> ConnectorError {
    if !e.is_spent() {
        return ConnectorError::retryable(format!(
            "connection `{path}`: token refresh failed ({e})"
        ));
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
        format!("connection `{path}`: {what}. Run `subs auth {path}` ({e})"),
    )
}

fn mismatch(spec: &ConnectionSpec, declared: AuthKind) -> ConnectorError {
    let holds = match declared {
        AuthKind::Token => "an OAuth grant",
        _ => "a static token",
    };
    let path = &spec.path;
    ConnectorError::unauthorized(
        AuthNeed::NeverAuthorized,
        format!(
            "connection `{path}` declares `auth = \"{}\"` but holds {holds}: run `subs auth {path}`",
            declared.as_str()
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::super::registry::ConnectionDecl;
    use super::*;
    use crate::protocol::ConnectorProtocol;
    use std::sync::Arc;

    struct Vault(Option<Credential>);

    #[async_trait::async_trait]
    impl CredentialStore for Vault {
        async fn get(&self, _: &str, _: &str, _: &Slot) -> Option<Credential> {
            self.0.clone()
        }
        async fn put(&self, _: &str, _: &str, _: &Slot, _: Credential) -> Result<(), String> {
            Ok(())
        }
    }

    fn spec(auth: Option<AuthKind>, header: Option<&str>) -> ConnectionSpec {
        spec_at(ConnectionPath::Mcp("sentry".into()), auth, header)
    }

    fn spec_at(
        path: ConnectionPath,
        auth: Option<AuthKind>,
        header: Option<&str>,
    ) -> ConnectionSpec {
        ConnectionDecl {
            url: "https://example.test/mcp".to_string(),
            auth,
            header: header.map(str::to_string),
            credential: None,
            scopes: Vec::new(),
            client_id_env: None,
            client_secret_env: None,
            prefix_tools: true,
        }
        .at(path, ConnectorProtocol::Mcp)
    }

    fn resolver(held: Option<Credential>) -> StoredCredentials {
        StoredCredentials::new(Arc::new(Vault(held)))
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
                &Slot::Shared,
                &spec_at(
                    ConnectionPath::Mcp("github".into()),
                    Some(AuthKind::Token),
                    None,
                ),
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
                &Slot::Shared,
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
                &Slot::Shared,
                &spec_at(
                    ConnectionPath::Mcp("github".into()),
                    Some(AuthKind::Token),
                    None,
                ),
            )
            .await
            .unwrap_err();
        assert_eq!(
            err.auth,
            Some(AuthNeed::NeverAuthorized),
            "an empty slot was never authorized; nothing is spent and nothing needs replacing"
        );
        assert!(err.to_string().contains("subs auth mcp.github"));
    }

    #[tokio::test]
    async fn a_slot_holding_the_other_kind_is_reported_not_sent() {
        let err = resolver(Some(stored_token()))
            .resolve(
                "t",
                "sentry",
                &Slot::Shared,
                &spec(Some(AuthKind::Oauth), None),
            )
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("subs auth mcp.sentry"),
            "got {err}"
        );
    }

    #[test]
    fn a_passing_fault_at_the_token_endpoint_is_not_a_dead_grant() {
        let transient = refresh_failed(
            &ConnectionPath::Mcp("sentry".into()),
            &OauthError::Token("connection reset".into()),
        );
        assert_eq!(transient.auth, None);
        assert!(transient.retryable);

        let dead = refresh_failed(
            &ConnectionPath::Mcp("sentry".into()),
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
            &ConnectionPath::Mcp("sentry".into()),
            &OauthError::Refused {
                code: "invalid_client".into(),
                description: None,
            },
        );
        assert_eq!(err.auth, Some(AuthNeed::Reauthorize));
        assert!(
            err.to_string().contains("subs auth mcp.sentry"),
            "got {err}"
        );
    }

    #[tokio::test]
    async fn an_undeclared_connection_sends_what_it_holds_or_nothing() {
        let headers = resolver(None)
            .resolve("t", "open", &Slot::Shared, &spec(None, None))
            .await
            .unwrap();
        assert!(headers.is_empty());

        let headers = resolver(Some(granted()))
            .resolve("t", "linear", &Slot::Shared, &spec(None, None))
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
                &Slot::Shared,
                &spec(Some(AuthKind::None), None),
            )
            .await
            .unwrap();
        assert!(headers.is_empty());
    }

    #[tokio::test]
    async fn a_token_goes_to_a_plaintext_url_the_file_named() {
        let mut spec = spec(Some(AuthKind::Token), None);
        spec.decl.url = "http://mcp.internal:8080/mcp".to_string();
        let headers = resolver(Some(stored_token()))
            .resolve("t", "github", &Slot::Shared, &spec)
            .await
            .unwrap();
        assert!(headers.contains_key("authorization"));
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
