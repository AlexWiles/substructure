use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use axum::http::HeaderMap;
use chrono::Utc;
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::transport::auth::{AuthError, AuthPrincipal, AuthResolver};

#[derive(Debug, Clone)]
pub struct ClientTokenIssuerConfig {
    pub issuer: String,
    pub audience: String,
    pub default_ttl: Duration,
}

#[derive(Debug, thiserror::Error)]
pub enum ClientTokenIssuerError {
    #[error("client subject is required")]
    MissingSubject,
    #[error("client token ttl must be positive")]
    InvalidTtl,
    #[error("client token signing failed: {0}")]
    Signing(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientTokenClaims {
    pub iss: String,
    pub aud: String,
    pub exp: i64,
    pub iat: i64,
    pub jti: String,
    pub tenant_id: String,
    pub sub: String,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub attrs: HashMap<String, String>,
}

pub struct JwtHs256ClientTokenAuthResolver {
    issuer: String,
    audience: String,
    signing_key: EncodingKey,
    decoding_key: DecodingKey,
}

impl JwtHs256ClientTokenAuthResolver {
    pub fn new(
        issuer: impl Into<String>,
        audience: impl Into<String>,
        secret: impl AsRef<[u8]>,
    ) -> Self {
        let secret = secret.as_ref();
        Self {
            issuer: issuer.into(),
            audience: audience.into(),
            signing_key: EncodingKey::from_secret(secret),
            decoding_key: DecodingKey::from_secret(secret),
        }
    }

    pub fn issue_token(
        &self,
        tenant_id: String,
        subject: String,
        attrs: HashMap<String, String>,
        ttl: Duration,
    ) -> Result<(String, i64), ClientTokenIssuerError> {
        if subject.trim().is_empty() {
            return Err(ClientTokenIssuerError::MissingSubject);
        }

        if ttl.as_secs() == 0 {
            return Err(ClientTokenIssuerError::InvalidTtl);
        }

        let now = Utc::now().timestamp();

        let exp = now + i64::try_from(ttl.as_secs()).unwrap_or(i64::MAX);

        let claims = ClientTokenClaims {
            iss: self.issuer.clone(),
            aud: self.audience.clone(),
            exp,
            iat: now,
            jti: Uuid::now_v7().to_string(),
            tenant_id,
            sub: subject,
            attrs,
        };

        let header = Header::new(Algorithm::HS256);
        let token = encode(&header, &claims, &self.signing_key)
            .map_err(|e| ClientTokenIssuerError::Signing(e.to_string()))?;

        Ok((token, exp))
    }
}

#[async_trait]
impl AuthResolver for JwtHs256ClientTokenAuthResolver {
    async fn resolve(&self, headers: &HeaderMap) -> Result<AuthPrincipal, AuthError> {
        let token = extract_bearer_token(headers).ok_or(AuthError::MissingCredentials)?;

        let mut validation = Validation::new(Algorithm::HS256);

        validation.validate_exp = true;
        validation.set_audience(std::slice::from_ref(&self.audience));
        validation.set_issuer(std::slice::from_ref(&self.issuer));

        let decoded = decode::<ClientTokenClaims>(token, &self.decoding_key, &validation)
            .map_err(|_| AuthError::InvalidCredentials)?;

        if decoded.claims.sub.trim().is_empty() {
            return Err(AuthError::InvalidCredentials);
        }

        Ok(AuthPrincipal {
            tenant_id: decoded.claims.tenant_id,
            source: "client_token",
            subject: Some(decoded.claims.sub),
        })
    }
}

fn extract_bearer_token(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|raw| raw.strip_prefix("Bearer "))
        .map(str::trim)
        .filter(|s| !s.is_empty())
}
