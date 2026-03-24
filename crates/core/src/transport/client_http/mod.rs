mod routes;
mod types;

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};

use crate::Runtime;
use crate::transport::auth::{AuthError, AuthResolver};

#[derive(Clone)]
pub struct ClientHttpState {
    pub runtime: Arc<Runtime>,
    pub auth: Arc<dyn AuthResolver>,
}

pub fn router(state: ClientHttpState) -> Router {
    Router::new()
        .route("/api/client/sessions/submit", post(routes::submit_client_payload))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            client_auth_middleware,
        ))
        .with_state(state)
}

async fn client_auth_middleware(
    State(state): State<ClientHttpState>,
    mut request: axum::http::Request<axum::body::Body>,
    next: Next,
) -> Response {
    match state
        .auth
        .resolve(request.headers())
        .await
    {
        Ok(principal) => {
            request.extensions_mut().insert(principal);
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
