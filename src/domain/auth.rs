use std::sync::Arc;

use async_trait::async_trait;
use thiserror::Error;

use super::config::AuthConfig;
use super::event::ClientIdentity;

#[derive(Debug, Error)]
pub enum AuthError {
    #[error("missing authorization")]
    MissingCredential,
    #[error("invalid token: {0}")]
    InvalidToken(String),
    #[error("invalid api key")]
    InvalidApiKey,
    #[error("auth configuration error: {0}")]
    Config(String),
}

#[derive(Debug, Clone)]
pub struct AdminContext {
    pub tenant_id: String,
}

#[async_trait]
pub trait AuthResolver: Send + Sync {
    /// Resolve a Bearer token into a ClientIdentity (client routes).
    async fn resolve(&self, token: Option<&str>) -> Result<ClientIdentity, AuthError>;
    /// Resolve an API key into an AdminContext (admin routes).
    fn resolve_admin(&self, api_key: Option<&str>) -> Result<AdminContext, AuthError>;
    /// Issue a new client token for the given tenant and subject.
    fn issue_token(&self, tenant_id: &str, sub: Option<&str>) -> Result<String, AuthError>;
}

// ---------------------------------------------------------------------------
// NoneAuthResolver — backwards-compatible default (no auth)
// ---------------------------------------------------------------------------

pub struct NoneAuthResolver;

#[async_trait]
impl AuthResolver for NoneAuthResolver {
    async fn resolve(&self, _token: Option<&str>) -> Result<ClientIdentity, AuthError> {
        Ok(ClientIdentity {
            tenant_id: "http".into(),
            sub: None,
            attrs: Default::default(),
        })
    }

    fn resolve_admin(&self, _api_key: Option<&str>) -> Result<AdminContext, AuthError> {
        Ok(AdminContext {
            tenant_id: "http".into(),
        })
    }

    fn issue_token(&self, _tenant_id: &str, _sub: Option<&str>) -> Result<String, AuthError> {
        Err(AuthError::Config("token auth is not configured".into()))
    }
}

// ---------------------------------------------------------------------------
// TokenAuthResolver — validates substructure-issued JWTs
// ---------------------------------------------------------------------------

#[cfg(feature = "http")]
mod token {
    use std::collections::HashMap;

    use super::*;
    use crate::domain::config::TenantConfig;
    use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, TokenData, Validation};
    use sha2::{Digest, Sha256};

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    pub struct Claims {
        pub sub: Option<String>,
        pub tenant_id: String,
        pub exp: usize,
        pub iat: usize,
    }

    pub struct TokenAuthResolver {
        decoding_key: DecodingKey,
        encoding_key: EncodingKey,
        /// Maps hex-encoded SHA-256 hash of API key → TenantConfig
        tenants: HashMap<String, TenantConfig>,
        token_ttl: chrono::Duration,
    }

    impl TokenAuthResolver {
        pub fn new(
            signing_secret: &str,
            token_ttl_str: &str,
            tenants: &[TenantConfig],
        ) -> Result<Self, AuthError> {
            let token_ttl = crate::domain::config::parse_window(token_ttl_str)
                .ok_or_else(|| AuthError::Config(format!("invalid token_ttl: {token_ttl_str}")))?;

            let mut tenant_map = HashMap::new();
            for t in tenants {
                // Expected format: "sha256:hexdigest"
                let hash = t
                    .api_key_hash
                    .strip_prefix("sha256:")
                    .ok_or_else(|| {
                        AuthError::Config(format!(
                            "api_key_hash must start with 'sha256:', got: {}",
                            t.api_key_hash
                        ))
                    })?
                    .to_lowercase();
                tenant_map.insert(hash, t.clone());
            }

            Ok(Self {
                decoding_key: DecodingKey::from_secret(signing_secret.as_bytes()),
                encoding_key: EncodingKey::from_secret(signing_secret.as_bytes()),
                tenants: tenant_map,
                token_ttl,
            })
        }

        /// Hash a raw API key with SHA-256 and return the hex-encoded digest.
        fn hash_api_key(raw_key: &str) -> String {
            let mut hasher = Sha256::new();
            hasher.update(raw_key.as_bytes());
            hex::encode(hasher.finalize())
        }

        /// Validate a raw API key and return the matching tenant config.
        pub fn validate_api_key(&self, raw_key: &str) -> Result<&TenantConfig, AuthError> {
            let hash = Self::hash_api_key(raw_key);
            self.tenants.get(&hash).ok_or(AuthError::InvalidApiKey)
        }

        /// Issue a new JWT for the given tenant and subject.
        pub fn issue_token(&self, tenant_id: &str, sub: Option<&str>) -> Result<String, AuthError> {
            let now = chrono::Utc::now();
            let exp = now + self.token_ttl;

            let claims = Claims {
                sub: sub.map(|s| s.to_string()),
                tenant_id: tenant_id.to_string(),
                exp: exp.timestamp() as usize,
                iat: now.timestamp() as usize,
            };

            encode(&Header::default(), &claims, &self.encoding_key)
                .map_err(|e| AuthError::InvalidToken(format!("failed to encode: {e}")))
        }

        /// Decode and validate a JWT, returning its claims.
        fn decode_token(&self, token: &str) -> Result<TokenData<Claims>, AuthError> {
            let mut validation = Validation::default();
            validation.set_required_spec_claims(&["exp", "iat"]);

            decode::<Claims>(token, &self.decoding_key, &validation)
                .map_err(|e| AuthError::InvalidToken(e.to_string()))
        }
    }

    #[async_trait]
    impl AuthResolver for TokenAuthResolver {
        async fn resolve(&self, token: Option<&str>) -> Result<ClientIdentity, AuthError> {
            let token = token.ok_or(AuthError::MissingCredential)?;
            let data = self.decode_token(token)?;

            Ok(ClientIdentity {
                tenant_id: data.claims.tenant_id,
                sub: data.claims.sub,
                attrs: Default::default(),
            })
        }

        fn resolve_admin(&self, api_key: Option<&str>) -> Result<AdminContext, AuthError> {
            let api_key = api_key.ok_or(AuthError::MissingCredential)?;
            let tenant = self.validate_api_key(api_key)?;
            Ok(AdminContext {
                tenant_id: tenant.tenant_id.clone(),
            })
        }

        fn issue_token(&self, tenant_id: &str, sub: Option<&str>) -> Result<String, AuthError> {
            TokenAuthResolver::issue_token(self, tenant_id, sub)
        }
    }
}

#[cfg(feature = "http")]
pub use token::TokenAuthResolver;

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

pub fn build_auth_resolver(config: &AuthConfig) -> Result<Arc<dyn AuthResolver>, AuthError> {
    match config {
        AuthConfig::None => Ok(Arc::new(NoneAuthResolver)),
        #[cfg(feature = "http")]
        AuthConfig::Token {
            signing_secret,
            token_ttl,
            tenants,
        } => {
            let resolver = TokenAuthResolver::new(signing_secret, token_ttl, tenants)?;
            Ok(Arc::new(resolver))
        }
        #[cfg(not(feature = "http"))]
        AuthConfig::Token { .. } => Err(AuthError::Config(
            "token auth requires the 'http' feature".into(),
        )),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn none_resolver_always_succeeds() {
        let resolver = NoneAuthResolver;
        let auth = resolver.resolve(None).await.unwrap();
        assert_eq!(auth.tenant_id, "http");
        assert!(auth.sub.is_none());
    }

    #[test]
    fn none_resolver_admin_always_succeeds() {
        let resolver = NoneAuthResolver;
        let ctx = resolver.resolve_admin(None).unwrap();
        assert_eq!(ctx.tenant_id, "http");
    }

    #[test]
    fn none_resolver_issue_token_errors() {
        let resolver = NoneAuthResolver;
        let err = resolver.issue_token("acme", None).unwrap_err();
        assert!(matches!(err, AuthError::Config(_)));
    }

    #[test]
    fn auth_config_defaults_to_none() {
        let config: AuthConfig = serde_json::from_str(r#"{"type":"none"}"#).unwrap();
        assert!(matches!(config, AuthConfig::None));
    }

    #[test]
    fn auth_config_deserializes_token() {
        let json = r#"{
            "type": "token",
            "signing_secret": "test-secret",
            "tenants": [
                { "tenant_id": "acme", "api_key_hash": "sha256:abc123" }
            ]
        }"#;
        let config: AuthConfig = serde_json::from_str(json).unwrap();
        match config {
            AuthConfig::Token {
                signing_secret,
                token_ttl,
                tenants,
            } => {
                assert_eq!(signing_secret, "test-secret");
                assert_eq!(token_ttl, "1h");
                assert_eq!(tenants.len(), 1);
                assert_eq!(tenants[0].tenant_id, "acme");
            }
            _ => panic!("expected Token variant"),
        }
    }

    #[cfg(feature = "http")]
    mod token_tests {
        use super::*;
        use crate::domain::config::TenantConfig;

        fn test_resolver() -> TokenAuthResolver {
            let hash = sha2_hex("test-api-key");
            TokenAuthResolver::new(
                "my-signing-secret",
                "1h",
                &[TenantConfig {
                    tenant_id: "acme".into(),
                    api_key_hash: format!("sha256:{hash}"),
                }],
            )
            .unwrap()
        }

        fn sha2_hex(input: &str) -> String {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(input.as_bytes());
            hex::encode(hasher.finalize())
        }

        #[tokio::test]
        async fn rejects_missing_token() {
            let resolver = test_resolver();
            let err = resolver.resolve(None).await.unwrap_err();
            assert!(matches!(err, AuthError::MissingCredential));
        }

        #[tokio::test]
        async fn rejects_bad_signature() {
            let resolver = test_resolver();
            // Token signed with a different secret
            let other = TokenAuthResolver::new("other-secret", "1h", &[]).unwrap();
            let token = other.issue_token("acme", Some("user_1")).unwrap();
            let err = resolver.resolve(Some(&token)).await.unwrap_err();
            assert!(matches!(err, AuthError::InvalidToken(_)));
        }

        #[tokio::test]
        async fn rejects_expired_token() {
            let resolver = test_resolver();

            // Manually build an expired token
            use jsonwebtoken::{encode, EncodingKey, Header};
            let claims = token::Claims {
                sub: Some("user_1".into()),
                tenant_id: "acme".into(),
                exp: 0, // epoch = expired
                iat: 0,
            };
            let token = encode(
                &Header::default(),
                &claims,
                &EncodingKey::from_secret(b"my-signing-secret"),
            )
            .unwrap();

            let err = resolver.resolve(Some(&token)).await.unwrap_err();
            assert!(matches!(err, AuthError::InvalidToken(_)));
        }

        #[tokio::test]
        async fn issue_then_resolve_happy_path() {
            let resolver = test_resolver();
            let token = resolver.issue_token("acme", Some("user_123")).unwrap();
            let auth = resolver.resolve(Some(&token)).await.unwrap();
            assert_eq!(auth.tenant_id, "acme");
            assert_eq!(auth.sub.as_deref(), Some("user_123"));
        }

        #[tokio::test]
        async fn issue_then_resolve_no_sub() {
            let resolver = test_resolver();
            let token = resolver.issue_token("acme", None).unwrap();
            let auth = resolver.resolve(Some(&token)).await.unwrap();
            assert_eq!(auth.tenant_id, "acme");
            assert!(auth.sub.is_none());
        }

        #[test]
        fn api_key_validation_correct_key() {
            let resolver = test_resolver();
            let tenant = resolver.validate_api_key("test-api-key").unwrap();
            assert_eq!(tenant.tenant_id, "acme");
        }

        #[test]
        fn api_key_validation_wrong_key() {
            let resolver = test_resolver();
            let err = resolver.validate_api_key("wrong-key").unwrap_err();
            assert!(matches!(err, AuthError::InvalidApiKey));
        }

        #[test]
        fn resolve_admin_with_valid_key() {
            let resolver = test_resolver();
            let ctx = resolver.resolve_admin(Some("test-api-key")).unwrap();
            assert_eq!(ctx.tenant_id, "acme");
        }

        #[test]
        fn resolve_admin_missing_key() {
            let resolver = test_resolver();
            let err = resolver.resolve_admin(None).unwrap_err();
            assert!(matches!(err, AuthError::MissingCredential));
        }

        #[test]
        fn resolve_admin_wrong_key() {
            let resolver = test_resolver();
            let err = resolver.resolve_admin(Some("wrong-key")).unwrap_err();
            assert!(matches!(err, AuthError::InvalidApiKey));
        }

        #[test]
        fn trait_issue_token_works() {
            let resolver = test_resolver();
            let token = AuthResolver::issue_token(&resolver, "acme", Some("user_1")).unwrap();
            assert!(!token.is_empty());
        }
    }
}
