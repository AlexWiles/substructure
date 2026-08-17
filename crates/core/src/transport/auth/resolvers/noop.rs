use async_trait::async_trait;
use axum::http::HeaderMap;

use crate::transport::auth::{AuthError, AuthRequester, AuthResolver};

pub struct NoopAuthResolver {
    tenant_id: String,
    source: &'static str,
    subject: Option<String>,
}

impl NoopAuthResolver {
    pub fn new(tenant_id: impl Into<String>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            source: "noop",
            subject: None,
        }
    }

    /// Override the principal `source` and `subject` returned for every
    /// request. Used by dev-mode wiring to make machine endpoints (which
    /// require an api_key principal) pass without real auth.
    pub fn with_principal(mut self, source: &'static str, subject: impl Into<String>) -> Self {
        self.source = source;
        self.subject = Some(subject.into());
        self
    }
}

#[async_trait]
impl AuthResolver for NoopAuthResolver {
    async fn resolve(&self, _headers: &HeaderMap) -> Result<AuthRequester, AuthError> {
        Ok(AuthRequester {
            tenant_id: self.tenant_id.clone(),
            source: self.source,
            subject: self.subject.clone(),
            attrs: std::collections::HashMap::new(),
        })
    }
}
