use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use tokio_stream::{Stream, StreamExt};
use tower_http::cors::{Any, CorsLayer};
use uuid::Uuid;

use crate::ag_ui::{observe_session, resume_run, run_existing_session, AgUiError, AgUiEvent};
use crate::domain::config::SystemConfig;
use crate::domain::event::SessionAuth;
use crate::runtime::{Runtime, RuntimeError, SessionFilter, SessionSummary};

#[derive(Clone)]
pub struct HttpState {
    runtime: Arc<Runtime>,
}

#[derive(serde::Deserialize)]
pub struct CreateSessionRequest {
    pub agent: String,
}

#[derive(serde::Serialize)]
pub struct CreateSessionResponse {
    pub session_id: String,
}

#[derive(serde::Deserialize)]
pub struct ObserveQuery {
    pub run_id: Option<String>,
}

pub async fn start_server(config: SystemConfig, addr: std::net::SocketAddr) -> anyhow::Result<()> {
    let runtime = Runtime::start(&config).await?;
    let state = HttpState {
        runtime: Arc::new(runtime),
    };

    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/sessions", post(create_session))
        .route("/sessions/:session_id/ag-ui", post(run_ag_ui))
        .route("/sessions/:session_id/ag-ui/observe", get(observe_ag_ui))
        .layer(
            CorsLayer::new()
                .allow_methods(Any)
                .allow_headers(Any)
                .allow_origin(Any),
        )
        .with_state(state);

    axum::serve(
        tokio::net::TcpListener::bind(addr).await?,
        app.into_make_service(),
    )
    .await?;

    Ok(())
}

async fn healthz() -> StatusCode {
    StatusCode::OK
}

async fn create_session(
    State(state): State<HttpState>,
    Json(payload): Json<CreateSessionRequest>,
) -> Result<Json<CreateSessionResponse>, AppError> {
    let session_id = Uuid::new_v4();
    let auth = SessionAuth {
        tenant_id: "http".into(),
        client_id: "substructure-http".into(),
        sub: None,
    };

    let _session = state
        .runtime
        .start_session(session_id, &payload.agent, auth)
        .await?;

    Ok(Json(CreateSessionResponse {
        session_id: session_id.to_string(),
    }))
}

async fn run_ag_ui(
    State(state): State<HttpState>,
    Path(session_id): Path<Uuid>,
    Json(input): Json<crate::ag_ui::RunAgentInput>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, AppError> {
    let auth = SessionAuth {
        tenant_id: "http".into(),
        client_id: "substructure-http".into(),
        sub: None,
    };

    ensure_session_running(&state.runtime, session_id, &auth).await?;

    let stream: Pin<Box<dyn Stream<Item = AgUiEvent> + Send>> = if input.resume.is_some() {
        Box::pin(resume_run(&state.runtime, session_id, auth, input).await?)
    } else {
        match run_existing_session(&state.runtime, session_id, auth.clone(), input).await {
            Ok(stream) => Box::pin(stream),
            Err(AgUiError::NoUserMessage) => {
                let run_id = Uuid::new_v4().to_string();
                Box::pin(observe_session(&state.runtime, session_id, auth, run_id).await?)
            }
            Err(err) => return Err(err.into()),
        }
    };

    Ok(sse_from_ag_ui(stream))
}

async fn observe_ag_ui(
    State(state): State<HttpState>,
    Path(session_id): Path<Uuid>,
    Query(query): Query<ObserveQuery>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, AppError> {
    let auth = SessionAuth {
        tenant_id: "http".into(),
        client_id: "substructure-http".into(),
        sub: None,
    };

    ensure_session_running(&state.runtime, session_id, &auth).await?;

    let run_id = query.run_id.unwrap_or_else(|| Uuid::new_v4().to_string());
    let stream = observe_session(&state.runtime, session_id, auth, run_id).await?;

    Ok(sse_from_ag_ui(Box::pin(stream)))
}

fn sse_from_ag_ui(
    stream: Pin<Box<dyn Stream<Item = AgUiEvent> + Send>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let mapped = stream.map(|event| {
        let json = serde_json::to_string(&event).unwrap_or_else(|_| "{}".to_string());
        Ok(Event::default().data(json))
    });

    Sse::new(mapped).keep_alive(KeepAlive::new().interval(Duration::from_secs(20)))
}

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("runtime error: {0}")]
    Runtime(#[from] RuntimeError),
    #[error("ag-ui error: {0}")]
    AgUi(#[from] AgUiError),
}

impl axum::response::IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let status = match self {
            AppError::Runtime(RuntimeError::SessionNotFound) => StatusCode::NOT_FOUND,
            AppError::Runtime(RuntimeError::UnknownAgent(_)) => StatusCode::BAD_REQUEST,
            _ => StatusCode::BAD_REQUEST,
        };
        let body = Json(serde_json::json!({
            "error": self.to_string(),
        }));
        (status, body).into_response()
    }
}

/// Ensure the session actor is running (start from store if needed).
async fn ensure_session_running(
    runtime: &Runtime,
    session_id: Uuid,
    auth: &SessionAuth,
) -> Result<(), AppError> {
    if runtime.session_is_running(session_id) {
        return Ok(());
    }

    let summary = find_session_summary(runtime, session_id)?;
    let _handle = runtime
        .start_session(session_id, &summary.agent_name, auth.clone())
        .await?;
    Ok(())
}

fn find_session_summary(runtime: &Runtime, session_id: Uuid) -> Result<SessionSummary, AppError> {
    let filter = SessionFilter {
        tenant_id: Some("http".into()),
        ..Default::default()
    };
    runtime
        .session_index()
        .list_sessions(&filter)
        .into_iter()
        .find(|summary| summary.session_id == session_id)
        .ok_or_else(|| AppError::Runtime(RuntimeError::SessionNotFound))
}
