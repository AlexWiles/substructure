//! The hosted cloud's `/api/v1` surface, served by a local server so the same
//! commands work against both. A local server is single-tenant, so `{org}`/
//! `{app}` segments are accepted and ignored; control-plane mutations
//! (create/rename/delete app, API keys) are rejected.

use axum::extract::{FromRef, Path, Query, State};
use axum::http::header::{HeaderName, HeaderValue};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, patch, post};
use axum::{Extension, Json, Router};

use crate::api::v1::{ApiError, App, Meta, Org};
use crate::Caller;

use super::routes::{self, SessionEventsParams};
use super::{machine_auth_middleware, AdminHttpState};

const LOCAL_ORG: &str = "local";
const LOCAL_APP: &str = "local";

#[derive(Clone)]
pub struct V1State {
    admin: AdminHttpState,
}

impl FromRef<V1State> for AdminHttpState {
    fn from_ref(state: &V1State) -> Self {
        state.admin.clone()
    }
}

pub fn router(admin: AdminHttpState) -> Router {
    let state = V1State {
        admin: admin.clone(),
    };
    Router::new()
        .route("/api/v1/meta", get(meta))
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
            admin,
            machine_auth_middleware,
        ))
        .layer(middleware::from_fn(advertise_defaults))
        .with_state(state)
}

// Tells the CLI it's a single-tenant server: it adopts these as the org/app
// and skips the interactive picker.
async fn advertise_defaults(
    request: axum::http::Request<axum::body::Body>,
    next: Next,
) -> Response {
    let mut res = next.run(request).await;
    let headers = res.headers_mut();
    headers.insert(
        HeaderName::from_static("x-substructure-org"),
        HeaderValue::from_static(LOCAL_ORG),
    );
    headers.insert(
        HeaderName::from_static("x-substructure-app"),
        HeaderValue::from_static(LOCAL_APP),
    );
    res
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
    headers: axum::http::HeaderMap,
) -> impl IntoResponse {
    routes::stream_session_events(state, caller, Path(session_id), params, headers).await
}

/// What this server offers. `single_tenant` is what lets the CLI adopt the
/// advertised org/app instead of asking which one.
async fn meta() -> impl IntoResponse {
    Json(Meta {
        single_tenant: true,
        features: vec!["sessions".into()],
    })
}

async fn list_orgs() -> impl IntoResponse {
    Json(vec![local_org()])
}

async fn list_apps() -> impl IntoResponse {
    Json(vec![local_app()])
}

async fn get_app() -> impl IntoResponse {
    Json(local_app())
}

fn local_org() -> Org {
    Org {
        id: LOCAL_ORG.into(),
        name: LOCAL_ORG.into(),
        role: "owner".into(),
    }
}

// balance_usd is left None: local servers don't track a balance.
fn local_app() -> App {
    App {
        id: LOCAL_APP.into(),
        organization_id: LOCAL_ORG.into(),
        name: LOCAL_APP.into(),
        created_at: None,
        balance_usd: None,
        session_count: None,
    }
}

async fn unsupported() -> impl IntoResponse {
    (
        StatusCode::BAD_REQUEST,
        Json(ApiError::new(
            "unsupported_on_local",
            "this operation is not supported on a local server; it only exists in the hosted cloud",
        )),
    )
}
