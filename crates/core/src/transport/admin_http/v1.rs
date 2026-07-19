//! The hosted cloud's `/api/v1` surface, served by a local server so the same
//! commands work against both. A local server is single-tenant, so `{org}`/
//! `{app}` segments are accepted and ignored; control-plane mutations
//! (create/rename/delete app, API keys) are rejected.

use std::sync::Arc;

use axum::extract::{FromRef, Path, Query, State};
use axum::http::header::{HeaderName, HeaderValue};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, patch, post};
use axum::{Extension, Json, Router};

use crate::api::v1::{ApiError, App, Org, WorkerConfig, WorkerUpsert};
use crate::cli::DEFAULT_TENANT;
use crate::transport::push::PushAdapter;
use crate::worker::push::PushRegistrationRecord;
use crate::Caller;

use super::routes::{self, SessionEventsParams};
use super::{machine_auth_middleware, AdminHttpState};

const LOCAL_ORG: &str = "local";
const LOCAL_APP: &str = "local";

#[derive(Clone)]
pub struct V1State {
    admin: AdminHttpState,
    push: Arc<PushAdapter>,
}

impl FromRef<V1State> for AdminHttpState {
    fn from_ref(state: &V1State) -> Self {
        state.admin.clone()
    }
}

pub fn router(admin: AdminHttpState, push: Arc<PushAdapter>) -> Router {
    let state = V1State {
        admin: admin.clone(),
        push,
    };
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
        .route("/api/v1/apps/{app}/worker", get(get_worker).put(put_worker))
        .route(
            "/api/v1/apps/{app}/worker/rotate-secret",
            post(rotate_secret),
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

async fn get_worker(State(state): State<V1State>) -> impl IntoResponse {
    Json(worker_config(state.push.registration(DEFAULT_TENANT).await))
}

async fn put_worker(
    State(state): State<V1State>,
    Json(body): Json<WorkerUpsert>,
) -> impl IntoResponse {
    let existing = state.push.registration(DEFAULT_TENANT).await;

    if body.state.as_deref() == Some("disabled") {
        if let Err(e) = state.push.unregister(DEFAULT_TENANT).await {
            return registry_error(e);
        }
        return Json(worker_config(None)).into_response();
    }

    let endpoint_url = body
        .endpoint_url
        .or_else(|| existing.as_ref().and_then(config_endpoint));
    let Some(endpoint_url) = endpoint_url else {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiError::message(
                "endpointUrl is required to enable the worker",
            )),
        )
            .into_response();
    };
    let signing_secret = existing
        .as_ref()
        .and_then(config_secret)
        .unwrap_or_else(new_secret);

    match register(&state, endpoint_url, signing_secret).await {
        Ok(record) => Json(worker_config(Some(record))).into_response(),
        Err(e) => registry_error(e),
    }
}

async fn rotate_secret(State(state): State<V1State>) -> impl IntoResponse {
    let Some(existing) = state.push.registration(DEFAULT_TENANT).await else {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiError::message("no worker configured")),
        )
            .into_response();
    };
    let Some(endpoint_url) = config_endpoint(&existing) else {
        return registry_error("stored worker has no endpoint_url".into());
    };
    match register(&state, endpoint_url, new_secret()).await {
        Ok(record) => Json(worker_config(Some(record))).into_response(),
        Err(e) => registry_error(e),
    }
}

async fn register(
    state: &V1State,
    endpoint_url: String,
    signing_secret: String,
) -> Result<PushRegistrationRecord, String> {
    let record = PushRegistrationRecord {
        tenant_id: DEFAULT_TENANT.into(),
        transport_type: "http".into(),
        config: serde_json::json!({
            "endpoint_url": endpoint_url,
            "signing_secret": signing_secret,
        }),
    };
    state.push.register(record.clone()).await?;
    Ok(record)
}

fn worker_config(record: Option<PushRegistrationRecord>) -> WorkerConfig {
    match record {
        Some(r) => WorkerConfig {
            endpoint_url: config_endpoint(&r),
            state: Some("enabled".into()),
            signing_secret: config_secret(&r),
        },
        None => WorkerConfig {
            endpoint_url: None,
            state: Some("disabled".into()),
            signing_secret: None,
        },
    }
}

fn config_endpoint(r: &PushRegistrationRecord) -> Option<String> {
    r.config.get("endpoint_url")?.as_str().map(String::from)
}

fn config_secret(r: &PushRegistrationRecord) -> Option<String> {
    r.config.get("signing_secret")?.as_str().map(String::from)
}

fn new_secret() -> String {
    hex::encode(rand::random::<[u8; 32]>())
}

fn registry_error(message: String) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ApiError::message(message)),
    )
        .into_response()
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
