use async_trait::async_trait;
use axum::http::HeaderMap;

mod resolvers;
pub use resolvers::{
    ApiKeyBinding, BearerHashedApiKeyAuthResolver, ClientTokenClaims, ClientTokenIssuerConfig,
    ClientTokenIssuerError, JwtHs256ClientTokenAuthResolver,
};

#[derive(Debug, Clone)]
pub struct AuthPrincipal {
    pub tenant_id: String,
    pub source: &'static str,
    pub subject: Option<String>,
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
