//! The hosted cloud's `/api/v1` surface, served by a local server so the same
//! commands work against both. A local server is single-tenant, so `{org}`/
//! `{app}` segments are accepted and ignored; control-plane mutations
//! (create/rename/delete app, API keys) are rejected.

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::middleware;
use axum::response::IntoResponse;
use axum::routing::{delete, get, patch, post};
use axum::{Extension, Json, Router};

use crate::Caller;

use super::routes::{self, SessionEventsParams};
use super::{machine_auth_middleware, AdminHttpState};

const LOCAL_ORG: &str = "local";
const LOCAL_APP: &str = "local";

pub fn router(state: AdminHttpState) -> Router {
    Router::new()
        .route("/api/v1/apps/{app}/sessions", get(routes::list_sessions))
        .route("/api/v1/apps/{app}/sessions/{session_id}", get(get_session))
        .route(
            "/api/v1/apps/{app}/sessions/{session_id}/events",
            get(get_session_events),
        )
        .route(
            "/api/v1/apps/{app}/sessions/{session_id}/events/stream",
            get(stream_session_events),
        )
        .route("/api/v1/orgs", get(list_orgs))
        .route("/api/v1/orgs/{org}/apps", get(list_apps))
        .route("/api/v1/apps/{app}", get(get_app))
        .route("/api/v1/orgs/{org}/apps", post(unsupported))
        .route("/api/v1/apps/{app}", patch(unsupported).delete(unsupported))
        .route(
            "/api/v1/apps/{app}/api-keys",
            get(unsupported).post(unsupported),
        )
        .route("/api/v1/apps/{app}/api-keys/{key_id}", delete(unsupported))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            machine_auth_middleware,
        ))
        .with_state(state)
}

async fn get_session(
    state: State<AdminHttpState>,
    caller: Extension<Caller>,
    Path((_app, session_id)): Path<(String, String)>,
) -> impl IntoResponse {
    routes::get_session(state, caller, Path(session_id)).await
}

async fn get_session_events(
    state: State<AdminHttpState>,
    caller: Extension<Caller>,
    Path((_app, session_id)): Path<(String, String)>,
    params: Query<SessionEventsParams>,
) -> impl IntoResponse {
    routes::get_session_events(state, caller, Path(session_id), params).await
}

async fn stream_session_events(
    state: State<AdminHttpState>,
    caller: Extension<Caller>,
    Path((_app, session_id)): Path<(String, String)>,
    params: Query<SessionEventsParams>,
) -> impl IntoResponse {
    routes::stream_session_events(state, caller, Path(session_id), params).await
}

async fn list_orgs() -> impl IntoResponse {
    Json(serde_json::json!([
        { "id": LOCAL_ORG, "name": LOCAL_ORG, "role": "owner" }
    ]))
}

async fn list_apps() -> impl IntoResponse {
    Json(serde_json::json!([local_app()]))
}

async fn get_app() -> impl IntoResponse {
    Json(local_app())
}

// `balanceUsd` is omitted so the CLI's zero-balance warning stays cloud-only.
fn local_app() -> serde_json::Value {
    serde_json::json!({
        "id": LOCAL_APP,
        "organizationId": LOCAL_ORG,
        "name": LOCAL_APP,
    })
}

async fn unsupported() -> impl IntoResponse {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({
            "error": {
                "code": "unsupported_on_local",
                "message": "this operation is not supported on a local server; it only exists in the hosted cloud"
            }
        })),
    )
}
