use axum::extract::{FromRef, Path, Query, State};
use axum::http::header::{HeaderName, HeaderValue};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, patch, post};
use axum::{Extension, Json, Router};
use futures_util::stream::BoxStream;
use futures_util::StreamExt;
use serde::Deserialize;
use tokio_stream::wrappers::ReceiverStream;

use crate::api::v1::{ApiError, Meta, Org, Project, RunFormat, RunRequest};
use crate::transport::ag_ui::run as ag_ui_run;
use crate::transport::ag_ui::translator::run_ag_ui_translation;
use crate::transport::channel::ChannelContext;
use crate::transport::http::runtime_error_response;
use crate::transport::session_sse::run_event_stream;
use crate::Caller;

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
        .route(
            "/api/v1/projects/{project}/sessions/{session_id}/ag-ui/connect",
            post(connect_session_ag_ui),
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
    let owner = crate::protocol::SessionOwner {
        tenant_id: caller.tenant_id().to_string(),
        requester: crate::protocol::Requester::machine(),
        metadata: Default::default(),
    };

    if (req.agent.is_some() || req.worker.is_some()) && req.input.agent_id().is_none() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiError::new(
                "agent_without_submit",
                "`agent` and `worker` open a session, so they ride only on an input \
                 that names an `agent_id`",
            )),
        )
            .into_response();
    }

    let shutdown = state.shutdown.clone();
    let ctx = ChannelContext::new(state.runtime.clone(), shutdown.clone());
    let turn = ag_ui_run::start(
        &ctx,
        crate::HandleClientInput {
            session_id: session_id.clone(),
            caller,
            owner,
            input: req.input,
            agent: req.agent,
            worker: req.worker,
            span: crate::span::SpanContext::root().child("v1_run"),
        },
        matches!(params.format, RunFormat::AgUi),
    )
    .await;
    let turn = match turn {
        Ok(turn) => turn,
        Err(e) => return runtime_error_response(e),
    };

    let out: BoxStream<'static, SseEvent> = match turn.deltas {
        Some(deltas) => Box::pin(
            ReceiverStream::new(run_ag_ui_translation(
                turn.events,
                deltas,
                session_id,
                turn.turn_id,
                shutdown,
            ))
            .map(|e| e.to_sse()),
        ),
        None => Box::pin(ReceiverStream::new(run_event_stream(turn.events, shutdown))),
    };
    Sse::new(out.map(Ok::<_, std::convert::Infallible>))
        .keep_alive(KeepAlive::default())
        .into_response()
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

async fn connect_session_ag_ui(
    state: State<AdminHttpState>,
    caller: Extension<Caller>,
    Path((_app, session_id)): Path<(String, String)>,
    input: Json<crate::transport::ag_ui::types::RunAgentInput>,
) -> impl IntoResponse {
    routes::connect_session_ag_ui(state, caller, Path(session_id), input).await
}

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
