use std::collections::HashMap;

use async_trait::async_trait;
use axum::http::HeaderMap;

use crate::protocol::{OwnerKind, SessionOwner};
use crate::Caller;

mod resolvers;
pub use resolvers::{
    ApiKeyBinding, BearerHashedApiKeyAuthResolver, ClientTokenClaims, ClientTokenIssuerConfig,
    ClientTokenIssuerError, JwtHs256ClientTokenAuthResolver, NoopAuthResolver,
};

pub const SOURCE_API_KEY: &str = "api_key";
/// A login, rather than a key.
pub const SOURCE_USER: &str = "user";

#[derive(Debug, Clone)]
pub struct AuthPrincipal {
    pub tenant_id: String,
    pub source: &'static str,
    pub subject: Option<String>,
    /// Additional claims surfaced by the resolver (e.g. JWT `attrs`).
    /// Empty for credential types that don't carry extra data.
    pub attrs: HashMap<String, String>,
}

impl AuthPrincipal {
    /// A worker is a program, so this is always a key.
    pub fn api_key_caller(&self) -> Option<Caller> {
        self.named_subject().map(|key_id| Caller::ApiKey {
            tenant_id: self.tenant_id.clone(),
            key_id,
        })
    }

    pub fn operator_caller(&self) -> Option<Caller> {
        let subject = self.named_subject()?;
        Some(match self.source {
            SOURCE_USER => Caller::Admin {
                tenant_id: self.tenant_id.clone(),
                user_id: subject,
            },
            _ => Caller::ApiKey {
                tenant_id: self.tenant_id.clone(),
                key_id: subject,
            },
        })
    }

    fn named_subject(&self) -> Option<String> {
        self.subject
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    }

    pub fn frontend_caller(&self) -> Option<Caller> {
        self.subject.clone().map(|user_id| Caller::Frontend {
            tenant_id: self.tenant_id.clone(),
            user_id,
            attrs: self.attrs.clone(),
        })
    }

    pub fn session_owner(&self) -> Option<SessionOwner> {
        self.subject.clone().map(|id| SessionOwner {
            tenant_id: self.tenant_id.clone(),
            id: Some(id),
            kind: OwnerKind::Frontend,
            metadata: HashMap::new(),
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("missing credentials")]
    MissingCredentials,
    #[error("invalid credentials")]
    InvalidCredentials,
    #[error("auth resolver unavailable: {0}")]
    Internal(String),
}

#[async_trait]
pub trait AuthResolver: Send + Sync {
    async fn resolve(&self, headers: &HeaderMap) -> Result<AuthPrincipal, AuthError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn principal(source: &'static str, subject: Option<&str>) -> AuthPrincipal {
        AuthPrincipal {
            tenant_id: "tenant-a".to_string(),
            source,
            subject: subject.map(str::to_string),
            attrs: HashMap::new(),
        }
    }

    #[test]
    fn the_credential_decides_which_operator_it_is() {
        let user = principal(SOURCE_USER, Some("alex@example.test"));
        assert!(matches!(
            user.operator_caller(),
            Some(Caller::Admin { user_id, .. }) if user_id == "alex@example.test"
        ));

        let key = principal(SOURCE_API_KEY, Some("key-1"));
        assert!(matches!(
            key.operator_caller(),
            Some(Caller::ApiKey { key_id, .. }) if key_id == "key-1"
        ));
    }

    #[test]
    fn the_worker_surface_is_always_a_key() {
        let user = principal(SOURCE_USER, Some("alex@example.test"));
        assert!(matches!(user.api_key_caller(), Some(Caller::ApiKey { .. })));
    }

    #[test]
    fn a_machine_subject_must_name_something() {
        assert!(principal(SOURCE_API_KEY, None).operator_caller().is_none());
        assert!(principal(SOURCE_API_KEY, Some(""))
            .operator_caller()
            .is_none());
        assert!(principal(SOURCE_USER, Some("")).api_key_caller().is_none());
    }

    #[test]
    fn an_unknown_source_is_a_key() {
        assert!(matches!(
            principal("something-new", Some("s")).operator_caller(),
            Some(Caller::ApiKey { .. })
        ));
    }
}
