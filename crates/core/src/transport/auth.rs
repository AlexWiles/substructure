use std::sync::Arc;

use async_trait::async_trait;
use axum::http::HeaderMap;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

#[derive(Debug, Clone, Copy)]
pub enum AuthCapability {
    WorkerApi,
    ClientApi,
}

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
    async fn resolve(
        &self,
        headers: &HeaderMap,
        capability: AuthCapability,
    ) -> Result<AuthPrincipal, AuthError>;
}

#[derive(Debug, Clone)]
pub struct ApiKeyBinding {
    pub tenant_id: String,
    pub key_sha256_hex: String,
    pub key_id: String,
}

impl ApiKeyBinding {
    pub fn new(
        tenant_id: impl Into<String>,
        key_sha256_hex: impl Into<String>,
        key_id: impl Into<String>,
    ) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            key_sha256_hex: key_sha256_hex.into(),
            key_id: key_id.into(),
        }
    }
}

#[derive(Debug, Clone)]
struct CompiledKeyBinding {
    tenant_id: String,
    key_hash: [u8; 32],
    key_id: String,
}

pub struct HashedApiKeyAuthResolver {
    bindings: Vec<CompiledKeyBinding>,
}

impl HashedApiKeyAuthResolver {
    pub fn new(bindings: Vec<ApiKeyBinding>) -> Result<Self, String> {
        if bindings.is_empty() {
            return Err("at least one API key binding is required".to_string());
        }

        let mut compiled = Vec::with_capacity(bindings.len());
        for binding in bindings {
            if binding.key_id.trim().is_empty() {
                return Err(format!("key_id is required for tenant {}", binding.tenant_id));
            }
            let raw = hex::decode(&binding.key_sha256_hex)
                .map_err(|e| format!("invalid SHA-256 hex for tenant {}: {e}", binding.tenant_id))?;
            let key_hash: [u8; 32] = raw.try_into().map_err(|_| {
                format!(
                    "invalid SHA-256 length for tenant {}: expected 32 bytes",
                    binding.tenant_id
                )
            })?;

            compiled.push(CompiledKeyBinding {
                tenant_id: binding.tenant_id,
                key_hash,
                key_id: binding.key_id,
            });
        }

        Ok(Self { bindings: compiled })
    }

    pub fn into_dyn(self) -> Arc<dyn AuthResolver> {
        Arc::new(self)
    }
}

#[async_trait]
impl AuthResolver for HashedApiKeyAuthResolver {
    async fn resolve(
        &self,
        headers: &HeaderMap,
        _capability: AuthCapability,
    ) -> Result<AuthPrincipal, AuthError> {
        let key = extract_api_key(headers).ok_or(AuthError::MissingCredentials)?;
        let key_hash = Sha256::digest(key.as_bytes());

        for binding in &self.bindings {
            if key_hash.ct_eq(&binding.key_hash).unwrap_u8() == 1 {
                return Ok(AuthPrincipal {
                    tenant_id: binding.tenant_id.clone(),
                    source: "api_key",
                    subject: Some(binding.key_id.clone()),
                });
            }
        }

        Err(AuthError::InvalidCredentials)
    }
}

fn extract_api_key(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|raw| raw.strip_prefix("Bearer "))
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn resolves_valid_bearer_key() {
        let resolver = HashedApiKeyAuthResolver::new(vec![ApiKeyBinding::new(
            "tenant-a",
            "0b42357e3654716d9915e42b3b44d9c762169d7c4c972906b45a1d8b28dbad2e",
            "tenant-a-k1",
        )])
        .unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::AUTHORIZATION,
            "Bearer dev-worker-key".parse().unwrap(),
        );

        let principal = resolver
            .resolve(&headers, AuthCapability::WorkerApi)
            .await
            .unwrap();
        assert_eq!(principal.tenant_id, "tenant-a");
    }

    #[tokio::test]
    async fn rejects_missing_key() {
        let resolver = HashedApiKeyAuthResolver::new(vec![ApiKeyBinding::new(
            "tenant-a",
            "0b42357e3654716d9915e42b3b44d9c762169d7c4c972906b45a1d8b28dbad2e",
            "tenant-a-k1",
        )])
        .unwrap();
        let headers = HeaderMap::new();

        let err = resolver
            .resolve(&headers, AuthCapability::WorkerApi)
            .await
            .unwrap_err();
        assert!(matches!(err, AuthError::MissingCredentials));
    }
}
