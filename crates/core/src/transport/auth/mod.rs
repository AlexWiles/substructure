use std::collections::HashMap;

use async_trait::async_trait;
use axum::http::HeaderMap;

use crate::protocol::SessionOwner;
use crate::Caller;

mod resolvers;
pub use resolvers::{
    ApiKeyBinding, BearerHashedApiKeyAuthResolver, ClientTokenClaims, ClientTokenIssuerConfig,
    ClientTokenIssuerError, JwtHs256ClientTokenAuthResolver, NoopAuthResolver,
};

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
    pub fn machine_caller(&self) -> Option<Caller> {
        self.subject
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(|key_id| Caller::Machine {
                tenant_id: self.tenant_id.clone(),
                key_id: key_id.to_string(),
            })
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
