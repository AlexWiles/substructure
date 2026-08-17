use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use axum::http::HeaderMap;
use chrono::Utc;
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::protocol::Visibility;
use crate::transport::auth::{AuthError, AuthResolver, Authenticated};

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
    /// Whether anyone but `sub` can read this session. A token minted before
    /// this said nothing, and one person's own UI is what these are for.
    #[serde(default = "private")]
    pub visibility: Visibility,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub attrs: HashMap<String, String>,
}

fn private() -> Visibility {
    Visibility::Private
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

    /// `visibility` is the minter's to assert: only they know whether the
    /// surface this token is for is one person's. It decides whether a
    /// personal credential may answer there, so the restrictive value is what
    /// a caller that says nothing gets.
    pub fn issue_token(
        &self,
        tenant_id: &str,
        subject: String,
        visibility: Visibility,
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
            tenant_id: tenant_id.to_string(),
            sub: subject,
            visibility,
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
    async fn resolve(&self, headers: &HeaderMap) -> Result<Authenticated, AuthError> {
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

        Ok(Authenticated {
            tenant_id: decoded.claims.tenant_id,
            source: "client_token",
            subject: Some(decoded.claims.sub),
            visibility: decoded.claims.visibility,
            attrs: decoded.claims.attrs,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::auth::AuthResolver;

    fn issuer() -> JwtHs256ClientTokenAuthResolver {
        JwtHs256ClientTokenAuthResolver::new("iss", "aud", "s3cret")
    }

    async fn owner_of(token: &str) -> crate::protocol::SessionOwner {
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::AUTHORIZATION,
            format!("Bearer {token}").parse().unwrap(),
        );
        issuer()
            .resolve(&headers)
            .await
            .unwrap()
            .session_owner()
            .expect("a token names a person")
    }

    /// The minter says who else can read, because only they know what surface
    /// the token is for. It is what decides whether this person's own
    /// credentials may answer there.
    #[tokio::test]
    async fn the_minter_says_who_else_can_read() {
        for visibility in [Visibility::Private, Visibility::Shared] {
            let (token, _) = issuer()
                .issue_token(
                    "t",
                    "bob".into(),
                    visibility,
                    HashMap::new(),
                    Duration::from_secs(60),
                )
                .unwrap();
            assert_eq!(owner_of(&token).await.requester.visibility, visibility);
        }
    }

    /// A token minted before the claim existed was one person's own UI, which
    /// is what these are for.
    #[tokio::test]
    async fn a_token_from_before_the_claim_is_one_persons() {
        let claims = serde_json::json!({
            "iss": "iss", "aud": "aud", "sub": "bob", "tenant_id": "t",
            "jti": "j", "iat": 0, "exp": Utc::now().timestamp() + 60,
        });
        let token = jsonwebtoken::encode(
            &Header::new(Algorithm::HS256),
            &claims,
            &EncodingKey::from_secret(b"s3cret"),
        )
        .unwrap();
        assert_eq!(
            owner_of(&token).await.requester.visibility,
            Visibility::Private
        );
    }
}
