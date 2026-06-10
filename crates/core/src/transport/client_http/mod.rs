mod routes;
mod types;

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
pub struct ClientHttpState {
    pub runtime: Arc<Runtime>,
    pub auth: Arc<dyn AuthResolver>,
    pub shutdown: CancellationToken,
}

pub fn router(state: ClientHttpState) -> Router {
    // `allow_headers` mirrors the request rather than sending `*`: per the Fetch
    // spec, `*` does not cover `Authorization`, so browsers would strip the bearer
    // token on cross-origin requests.
    let cors = {
        use tower_http::cors::{AllowHeaders, Any, CorsLayer};
        CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(AllowHeaders::mirror_request())
            .expose_headers(Any)
    };
    Router::new()
        .route(
            "/api/client/sessions/submit",
            post(routes::submit_client_payload),
        )
        .route(
            "/api/client/sessions/{session_id}/tool-call-results",
            post(routes::submit_tool_call_result),
        )
        .route(
            "/api/client/sessions/{session_id}/events/stream",
            get(routes::stream_session_events),
        )
        .route(
            "/api/client/sessions/{session_id}/interrupt",
            post(routes::interrupt_session),
        )
        .route(
            "/api/client/sessions/{session_id}/interrupt/resume",
            post(routes::resume_interrupt),
        )
        .route(
            "/api/client/ag-ui/agents/{agent_id}/run",
            post(routes::ag_ui_run),
        )
        .route(
            "/api/client/ag-ui/agents/{agent_id}/connect",
            post(routes::ag_ui_connect),
        )
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            client_auth_middleware,
        ))
        // Outermost so CORS preflight (OPTIONS) is answered before auth runs.
        .layer(cors)
        .with_state(state)
}

async fn client_auth_middleware(
    State(state): State<ClientHttpState>,
    mut request: axum::http::Request<axum::body::Body>,
    next: Next,
) -> Response {
    match state.auth.resolve(request.headers()).await {
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
