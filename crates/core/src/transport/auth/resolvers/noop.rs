use async_trait::async_trait;
use axum::http::HeaderMap;

use crate::transport::auth::{AuthError, AuthPrincipal, AuthResolver};

pub struct NoopAuthResolver {
    tenant_id: String,
}

impl NoopAuthResolver {
    pub fn new(tenant_id: impl Into<String>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
        }
    }
}

#[async_trait]
impl AuthResolver for NoopAuthResolver {
    async fn resolve(&self, _headers: &HeaderMap) -> Result<AuthPrincipal, AuthError> {
        Ok(AuthPrincipal {
            tenant_id: self.tenant_id.clone(),
            source: "noop",
            subject: None,
        })
    }
}
