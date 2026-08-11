//! The hosted cloud's `/api/v1` surface, served by a local server so the same
//! commands work against both. Scope comes from the caller's tenant, so `{org}`
//! and `{project}` are accepted and ignored: a local server has one of each.
//! Control-plane mutations (create/rename/delete project, API keys) are
//! rejected.

use axum::extract::{FromRef, Path, Query, State};
use axum::http::header::{HeaderName, HeaderValue};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, patch, post};
use axum::{Extension, Json, Router};
use futures_util::StreamExt;
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;

use crate::api::v1::{ApiError, Meta, Org, Project, RunFormat, RunRequest, RUN_DONE_EVENT};
use crate::session::subscriptions::{SessionSubscriptionSpec, SubscriptionScope};
use crate::session::SessionEvent;
use crate::transport::ag_ui::translator::run_ag_ui_translation;
use crate::transport::http::runtime_error_response;
use crate::{Caller, HandleClientInput};

use super::routes::{self, SessionEventsParams};
use super::{machine_auth_middleware, AdminHttpState};

const LOCAL_ORG: &str = "local";
const LOCAL_PROJECT: &str = "local";

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
        .route(
            "/api/v1/projects/{project}/sessions",
            get(routes::list_sessions),
        )
        .route(
            "/api/v1/projects/{project}/sessions/{session_id}",
            get(get_session),
        )
        .route(
            "/api/v1/projects/{project}/sessions/{session_id}/events",
            get(get_session_events),
        )
        .route(
            "/api/v1/projects/{project}/sessions/{session_id}/events/stream",
            get(stream_session_events),
        )
        .route("/api/v1/projects/{project}/run", post(run))
        .route("/api/v1/orgs", get(list_orgs))
        .route("/api/v1/orgs/{org}/projects", get(list_projects))
        .route("/api/v1/projects/{project}", get(get_project))
        .route("/api/v1/orgs/{org}/projects", post(unsupported))
        .route(
            "/api/v1/projects/{project}",
            patch(unsupported).delete(unsupported),
        )
        .route(
            "/api/v1/projects/{project}/api-keys",
            get(unsupported).post(unsupported),
        )
        .route(
            "/api/v1/projects/{project}/api-keys/{key_id}",
            delete(unsupported),
        )
        .route("/api/v1/orgs/{org}/slack", get(no_slack_install))
        .route(
            "/api/v1/orgs/{org}/slack/install-url",
            get(no_slack_install),
        )
        .route_layer(middleware::from_fn_with_state(
            admin,
            machine_auth_middleware,
        ))
        .layer(middleware::from_fn(advertise_defaults))
        .with_state(state)
}

// Tells the CLI it's a single-tenant server: it adopts these as the org/project
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
        HeaderName::from_static("x-substructure-project"),
        HeaderValue::from_static(LOCAL_PROJECT),
    );
    res
}

/// Runs one turn and streams it back, for an operator credential.
///
/// The caller is a machine, so the session has no end user. To run as a user,
/// mint a client token and use the client surface.
///
/// One call submits and streams. A caller that submits first and subscribes
/// second loses the start of its turn.
async fn run(
    State(state): State<V1State>,
    Extension(caller): Extension<Caller>,
    Path(_project): Path<String>,
    Query(params): Query<RunParams>,
    Json(req): Json<RunRequest>,
) -> Response {
    let state = state.admin;
    let session_id = req
        .session_id
        .unwrap_or_else(|| uuid::Uuid::now_v7().to_string());
    let owner = caller.as_owner();

    // Subscribe before the input, or the first events have no listener.
    let spec = SessionSubscriptionSpec {
        scope: SubscriptionScope::All,
        caller: caller.clone(),
        session_id: session_id.clone(),
    };
    let event_rx = match state.runtime.stream(spec, None).await {
        Ok(rx) => rx,
        Err(e) => return runtime_error_response(e),
    };
    // A token delta is a fragment of a translated message, so only the
    // translated stream subscribes. Before the input, like the events: both are
    // transient.
    let delta_rx = match params.format {
        RunFormat::AgUi => Some(
            state
                .runtime
                .subscribe_token_deltas(&caller, &session_id)
                .await,
        ),
        RunFormat::Events => None,
    };

    let turn = state
        .runtime
        .handle_client_input(HandleClientInput {
            session_id: session_id.clone(),
            caller,
            owner,
            // The tagged union carries its own addressing.
            input: req.input,
            span: crate::span::SpanContext::root().child("v1_run"),
        })
        .await;
    let turn_id = match turn {
        Ok(output) => output.turn_id,
        Err(e) => return runtime_error_response(e),
    };

    let shutdown = state.shutdown.clone();
    let out = match delta_rx {
        Some(delta_rx) => run_ag_ui_translation(event_rx, delta_rx, session_id, turn_id, shutdown),
        None => run_raw_events(event_rx, shutdown),
    };
    Sse::new(ReceiverStream::new(out).map(Ok::<_, std::convert::Infallible>))
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Stored engine events, as `subs run -o jsonl` prints them. Ends with
/// [`RUN_DONE_EVENT`] when the turn does, and with nothing when the server
/// stops first.
fn run_raw_events(
    mut event_rx: mpsc::Receiver<SessionEvent>,
    shutdown: CancellationToken,
) -> mpsc::Receiver<SseEvent> {
    let (tx, rx) = mpsc::channel(64);
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = shutdown.cancelled() => return,
                event = event_rx.recv() => {
                    let Some(event) = event else { return };
                    let sse = SseEvent::default()
                        .id(event.seq.to_string())
                        .event(event.payload_type())
                        .data(serde_json::to_string(&event).unwrap_or_default());
                    if tx.send(sse).await.is_err() {
                        return;
                    }
                    if event.ends_run() {
                        let _ = tx.send(SseEvent::default().event(RUN_DONE_EVENT).data("")).await;
                        return;
                    }
                }
            }
        }
    });
    rx
}

#[derive(Debug, Default, Deserialize)]
struct RunParams {
    #[serde(default)]
    format: RunFormat,
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
/// advertised org/project instead of asking which one.
async fn meta() -> impl IntoResponse {
    Json(Meta {
        single_tenant: true,
        features: vec!["sessions".into(), "projects".into()],
    })
}

async fn list_orgs() -> impl IntoResponse {
    Json(vec![local_org()])
}

async fn list_projects() -> impl IntoResponse {
    Json(vec![local_project()])
}

async fn get_project() -> impl IntoResponse {
    Json(local_project())
}

fn local_org() -> Org {
    Org {
        id: LOCAL_ORG.into(),
        name: LOCAL_ORG.into(),
        role: "owner".into(),
    }
}

// balance_usd is left None: local servers don't track a balance.
fn local_project() -> Project {
    Project {
        id: LOCAL_PROJECT.into(),
        organization_id: LOCAL_ORG.into(),
        name: LOCAL_PROJECT.into(),
        created_at: None,
        balance_usd: None,
        session_count: None,
    }
}

/// A local server holds no Slack app, so there is no workspace for it to
/// install one into. `meta` leaves the feature out, so the CLI says this
/// before asking; the route answers the same for anything that asks directly.
async fn no_slack_install() -> impl IntoResponse {
    (
        StatusCode::BAD_REQUEST,
        Json(ApiError::new(
            "unsupported_on_local",
            "a local server takes its Slack credential from SLACK_APP_TOKEN and SLACK_BOT_TOKEN \
             and answers over Socket Mode, so there is no workspace to connect",
        )),
    )
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
