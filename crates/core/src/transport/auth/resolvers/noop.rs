use async_trait::async_trait;
use axum::http::HeaderMap;

use crate::transport::auth::{AuthError, AuthPrincipal, AuthResolver};

pub struct NoopAuthResolver;

#[async_trait]
impl AuthResolver for NoopAuthResolver {
    async fn resolve(&self, _headers: &HeaderMap) -> Result<AuthPrincipal, AuthError> {
        Ok(AuthPrincipal {
            tenant_id: "default".to_string(),
            source: "noop",
            subject: None,
        })
    }
}
