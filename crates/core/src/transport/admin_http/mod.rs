mod routes;

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};

use tokio_util::sync::CancellationToken;

use crate::transport::auth::{AuthError, AuthResolver};
use crate::Runtime;

#[derive(Clone)]
pub struct AdminHttpState {
    pub runtime: Arc<Runtime>,
    pub auth: Arc<dyn AuthResolver>,
    pub shutdown: CancellationToken,
}

pub fn router(state: AdminHttpState) -> Router {
    Router::new()
        .route("/api/admin/sessions", get(routes::list_sessions))
        .route("/api/admin/sessions/{session_id}", get(routes::get_session))
        .route(
            "/api/admin/sessions/{session_id}/events",
            get(routes::get_session_events),
        )
        .route(
            "/api/admin/sessions/{session_id}/events/stream",
            get(routes::stream_session_events),
        )
        .route(
            "/api/admin/sessions/{session_id}/ag-ui/connect",
            post(routes::connect_session_ag_ui),
        )
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            admin_auth_middleware,
        ))
        .with_state(state)
}

async fn admin_auth_middleware(
    State(state): State<AdminHttpState>,
    mut request: axum::http::Request<axum::body::Body>,
    next: Next,
) -> Response {
    match state.auth.resolve(request.headers()).await {
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
            tracing::error!(error = %e, "admin auth resolver unavailable");
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "auth_unavailable"})),
            )
                .into_response()
        }
    }
}
