use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use axum::http::StatusCode;

/// Extracts tenant ID from the `X-Tenant-Id` request header.
///
/// In a cloud deployment this would be derived from auth (JWT, API key).
/// For local use, clients send the header directly.
/// Defaults to `"default"` when the header is absent.
#[derive(Debug, Clone)]
pub struct TenantId(pub String);

impl<S: Send + Sync> FromRequestParts<S> for TenantId {
    type Rejection = (StatusCode, &'static str);

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        match parts.headers.get("x-tenant-id") {
            Some(value) => {
                let s = value
                    .to_str()
                    .map_err(|_| (StatusCode::BAD_REQUEST, "invalid X-Tenant-Id header"))?;
                Ok(TenantId(s.to_string()))
            }
            None => Ok(TenantId("default".to_string())),
        }
    }
}
