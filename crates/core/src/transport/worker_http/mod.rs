mod routes;
pub mod types;

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};

use crate::transport::auth::{AuthCapability, AuthError, AuthResolver};
use crate::transport::push::PushAdapter;

#[derive(Clone)]
pub struct WorkerHttpState {
    pub adapter: Arc<PushAdapter>,
    pub auth: Arc<dyn AuthResolver>,
}

pub fn router(state: WorkerHttpState) -> Router {
    Router::new()
        .route("/workers/submit", post(routes::submit))
        .route("/workers/register", post(routes::register))
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
    match state
        .auth
        .resolve(request.headers(), AuthCapability::WorkerApi)
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
            tracing::error!(error = %e, "worker auth resolver unavailable");
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "auth_unavailable"})),
            )
                .into_response()
        }
    }
}
