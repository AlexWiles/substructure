mod routes;
pub mod types;

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};

use tokio_util::sync::CancellationToken;

use crate::transport::auth::{AuthError, AuthResolver, JwtHs256ClientTokenAuthResolver};
use crate::Runtime;

#[derive(Clone)]
pub struct WorkerHttpState {
    pub runtime: Arc<Runtime>,
    pub auth: Arc<dyn AuthResolver>,
    pub client_token_issuer: Arc<JwtHs256ClientTokenAuthResolver>,
    pub shutdown: CancellationToken,
}

pub fn router(state: WorkerHttpState) -> Router {
    Router::new()
        .route("/api/machine/workers/submit", post(routes::submit))
        .route(
            "/api/machine/client-tokens",
            post(routes::mint_client_token),
        )
        .route(
            "/api/machine/sessions/submit",
            post(routes::submit_client_payload),
        )
        .route(
            "/api/machine/sessions/{session_id}/calls/settle",
            post(routes::settle_effect),
        )
        .route(
            "/api/machine/sessions/{session_id}/events/stream",
            get(routes::stream_session_events),
        )
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            worker_auth_middleware,
        ))
        .with_state(state)
}

async fn worker_auth_middleware(
    State(state): State<WorkerHttpState>,
    mut request: axum::http::Request<axum::body::Body>,
    next: Next,
) -> Response {
    match state.auth.resolve(request.headers()).await {
        Ok(principal) => {
            let Some(caller) = principal.machine_caller() else {
                return (
                    StatusCode::FORBIDDEN,
                    Json(serde_json::json!({"error": "machine subject is required"})),
                )
                    .into_response();
            };
            request.extensions_mut().insert(caller);
            next.run(request).await
        }
        Err(AuthError::MissingCredentials | AuthError::InvalidCredentials) => (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response(),
        Err(AuthError::Internal(e)) => {
            tracing::error!(error = %e, "worker auth resolver unavailable");
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "auth_unavailable"})),
            )
                .into_response()
        }
    }
}
