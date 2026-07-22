use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use tower_http::cors::{AllowHeaders, Any, CorsLayer};

use crate::session::command::SessionError;
use crate::transport::auth::{AuthError, AuthResolver};
use crate::RuntimeError;

/// `allow_headers` mirrors the request rather than sending `*`: per the Fetch
/// spec, `*` does not cover `Authorization`, so browsers would strip the bearer
/// token on cross-origin requests.
pub fn client_cors() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(AllowHeaders::mirror_request())
        .expose_headers(Any)
}

/// Resolves a frontend `Caller` and `SessionOwner` and injects them as request
/// extensions.
pub async fn client_auth_middleware(
    State(auth): State<Arc<dyn AuthResolver>>,
    mut request: axum::http::Request<axum::body::Body>,
    next: Next,
) -> Response {
    match auth.resolve(request.headers()).await {
        Ok(principal) => {
            let (Some(caller), Some(owner)) =
                (principal.frontend_caller(), principal.session_owner())
            else {
                return (
                    StatusCode::FORBIDDEN,
                    Json(serde_json::json!({"error": "client subject is required"})),
                )
                    .into_response();
            };
            request.extensions_mut().insert(caller);
            request.extensions_mut().insert(owner);
            next.run(request).await
        }
        Err(AuthError::MissingCredentials | AuthError::InvalidCredentials) => (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response(),
        Err(AuthError::Internal(e)) => {
            tracing::error!(error = %e, "client auth resolver unavailable");
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "auth_unavailable"})),
            )
                .into_response()
        }
    }
}

/// Map RuntimeError variants to HTTP status codes for the client API.
pub fn runtime_error_status(err: &RuntimeError) -> (StatusCode, String) {
    let status = match err {
        RuntimeError::Session(
            SessionError::MissingSubject
            | SessionError::SessionAccessDenied
            | SessionError::EffectWrongHandler,
        ) => StatusCode::FORBIDDEN,
        RuntimeError::Session(SessionError::EffectNotFound | SessionError::SessionNotCreated) => {
            StatusCode::NOT_FOUND
        }
        RuntimeError::Session(
            SessionError::EffectNotPending
            | SessionError::EffectAttemptMismatch
            | SessionError::SessionInterrupted
            | SessionError::TurnAlreadyActive { .. }
            | SessionError::TurnAlreadyCompleted { .. }
            | SessionError::NoActiveTurn,
        ) => StatusCode::CONFLICT,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (status, err.to_string())
}

pub fn runtime_error_response(err: RuntimeError) -> Response {
    let (status, message) = runtime_error_status(&err);
    (status, Json(serde_json::json!({"error": message}))).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn status_of(err: SessionError) -> StatusCode {
        runtime_error_status(&RuntimeError::Session(err)).0
    }

    #[test]
    fn effect_settle_errors_map_to_expected_statuses() {
        assert_eq!(
            status_of(SessionError::EffectNotFound),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            status_of(SessionError::EffectWrongHandler),
            StatusCode::FORBIDDEN
        );
        assert_eq!(
            status_of(SessionError::EffectNotPending),
            StatusCode::CONFLICT
        );
        assert_eq!(
            status_of(SessionError::EffectAttemptMismatch),
            StatusCode::CONFLICT
        );
        assert_eq!(status_of(SessionError::NoActiveTurn), StatusCode::CONFLICT);
    }
}
