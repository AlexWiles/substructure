use std::convert::Infallible;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::connect_info::ConnectInfo;
use axum::extract::{FromRequestParts, Path, Query, State};
use axum::http::request::Parts;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use chrono::{DateTime, Utc};
use tokio_stream::{Stream, StreamExt};
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use uuid::Uuid;

use crate::ag_ui::{observe_session, resume_run, run_existing_session, AgUiError, AgUiEvent};
use crate::domain::aggregate::AggregateStatus;
use crate::domain::auth::{build_auth_resolver, AdminContext, AuthError, AuthResolver};
use crate::domain::config::SystemConfig;
use crate::domain::event::ClientIdentity;
use crate::runtime::{
    AggregateFilter, AggregateSort, AggregateSummary, EventFilter, Runtime, RuntimeError,
};

#[derive(Clone)]
pub struct HttpState {
    runtime: Arc<Runtime>,
    auth: Arc<dyn AuthResolver>,
}

// ---------------------------------------------------------------------------
// Extractors
// ---------------------------------------------------------------------------

/// Extractor for admin routes — validates `X-API-Key` header.
pub struct AdminAuth(pub AdminContext);

impl FromRequestParts<HttpState> for AdminAuth {
    type Rejection = AppError;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &HttpState,
    ) -> Result<Self, Self::Rejection> {
        let api_key = parts.headers.get("x-api-key").and_then(|v| v.to_str().ok());
        let ctx = state.auth.resolve_admin(api_key)?;
        Ok(AdminAuth(ctx))
    }
}

/// Extractor for client routes — validates `Authorization: Bearer <token>` header
/// and merges `client_ip` into attrs.
pub struct ClientAuth(pub ClientIdentity);

impl FromRequestParts<HttpState> for ClientAuth {
    type Rejection = AppError;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &HttpState,
    ) -> Result<Self, Self::Rejection> {
        let token = parts
            .headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));
        let mut session_auth = state.auth.resolve(token).await?;
        if let Some(ConnectInfo(addr)) = parts.extensions.get::<ConnectInfo<SocketAddr>>() {
            session_auth
                .attrs
                .insert("client_ip".into(), addr.ip().to_string());
        }
        Ok(ClientAuth(session_auth))
    }
}

// ---------------------------------------------------------------------------
// Request / response types
// ---------------------------------------------------------------------------

#[derive(serde::Serialize)]
pub struct AgentInfoResponse {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
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

#[derive(serde::Deserialize, Default)]
pub struct ListSessionsQuery {
    /// Repeated query param, e.g. ?status=active&status=idle
    #[serde(default)]
    pub status: Vec<AggregateStatus>,
    pub agent: Option<String>,
    #[serde(default)]
    pub sort: AggregateSort,
}

#[derive(serde::Deserialize)]
pub struct TokenRequest {
    pub sub: Option<String>,
}

#[derive(serde::Serialize)]
pub struct TokenResponse {
    pub token: String,
}

/// Session-specific HTTP response type, derived from the generic AggregateSummary.
#[derive(serde::Serialize)]
pub struct SessionSummary {
    pub session_id: Uuid,
    pub tenant_id: String,
    pub status: AggregateStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wake_at: Option<DateTime<Utc>>,
    pub stream_version: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_event_at: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_event_at: Option<DateTime<Utc>>,
}

impl From<AggregateSummary> for SessionSummary {
    fn from(s: AggregateSummary) -> Self {
        SessionSummary {
            session_id: s.aggregate_id,
            tenant_id: s.tenant_id,
            status: s.status,
            agent_name: s.label,
            wake_at: s.wake_at,
            stream_version: s.stream_version,
            first_event_at: s.first_event_at,
            last_event_at: s.last_event_at,
        }
    }
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

pub async fn start_server(config: SystemConfig, addr: std::net::SocketAddr) -> anyhow::Result<()> {
    let runtime = Runtime::start(&config).await?;

    let auth: Arc<dyn AuthResolver> = build_auth_resolver(&config.auth)
        .map_err(|e| anyhow::anyhow!("failed to build auth resolver: {e}"))?;

    let state = HttpState {
        runtime: Arc::new(runtime),
        auth,
    };

    let admin_routes = Router::new()
        .route("/auth/token", post(issue_token))
        .route("/sessions/{session_id}/events", get(session_events))
        .route("/traces/{trace_id}", get(trace_events));

    let client_routes = Router::new()
        .route("/agents", get(list_agents))
        .route("/sessions", get(list_sessions).post(create_session))
        .route("/sessions/{session_id}", get(get_session))
        .route("/sessions/{session_id}/ag-ui", post(run_ag_ui))
        .route("/sessions/{session_id}/ag-ui/observe", get(observe_ag_ui));

    let app = Router::new()
        .route("/healthz", get(healthz))
        .nest("/admin", admin_routes)
        .nest("/client", client_routes)
        .layer(
            CorsLayer::new()
                .allow_methods(Any)
                .allow_headers(Any)
                .allow_origin(Any),
        )
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "listening");

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;

    Ok(())
}

async fn healthz() -> StatusCode {
    StatusCode::OK
}

// ---------------------------------------------------------------------------
// Admin handlers
// ---------------------------------------------------------------------------

async fn issue_token(
    AdminAuth(admin): AdminAuth,
    State(state): State<HttpState>,
    Json(payload): Json<TokenRequest>,
) -> Result<Json<TokenResponse>, AppError> {
    let token = state
        .auth
        .issue_token(&admin.tenant_id, payload.sub.as_deref())?;

    Ok(Json(TokenResponse { token }))
}

// ---------------------------------------------------------------------------
// Admin: introspection
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize, Default)]
pub struct EventsQuery {
    pub event_type: Option<String>,
    pub sequence_after: Option<u64>,
    pub limit: Option<usize>,
}

async fn session_events(
    AdminAuth(_admin): AdminAuth,
    State(state): State<HttpState>,
    Path(session_id): Path<Uuid>,
    axum_extra::extract::Query(query): axum_extra::extract::Query<EventsQuery>,
) -> Result<Json<Vec<crate::runtime::event_store::Event>>, AppError> {
    let filter = EventFilter {
        aggregate_id: Some(session_id),
        event_type: query.event_type,
        sequence_after: query.sequence_after,
        limit: query.limit,
        ..Default::default()
    };
    let events = state
        .runtime
        .store()
        .query_events(&filter)
        .await
        .map_err(|e| AppError::Runtime(RuntimeError::from(e)))?;
    Ok(Json(events))
}

async fn trace_events(
    AdminAuth(_admin): AdminAuth,
    State(state): State<HttpState>,
    Path(trace_id): Path<String>,
    axum_extra::extract::Query(query): axum_extra::extract::Query<EventsQuery>,
) -> Result<Json<Vec<crate::runtime::event_store::Event>>, AppError> {
    let filter = EventFilter {
        trace_id: Some(trace_id),
        event_type: query.event_type,
        limit: query.limit,
        ..Default::default()
    };
    let events = state
        .runtime
        .store()
        .query_events(&filter)
        .await
        .map_err(|e| AppError::Runtime(RuntimeError::from(e)))?;
    Ok(Json(events))
}

// ---------------------------------------------------------------------------
// Client handlers
// ---------------------------------------------------------------------------

async fn list_agents(
    ClientAuth(_auth): ClientAuth,
    State(state): State<HttpState>,
) -> Json<Vec<AgentInfoResponse>> {
    let agents: Vec<AgentInfoResponse> = state
        .runtime
        .agent_names()
        .into_iter()
        .map(|name| {
            let description = state
                .runtime
                .agent(name)
                .and_then(|a| a.description.clone());
            AgentInfoResponse {
                name: name.to_string(),
                description,
            }
        })
        .collect();
    Json(agents)
}

async fn create_session(
    ClientAuth(auth): ClientAuth,
    State(state): State<HttpState>,
    Json(payload): Json<CreateSessionRequest>,
) -> Result<Json<CreateSessionResponse>, AppError> {
    let session_id = Uuid::new_v4();
    let _session = state
        .runtime
        .start_session(session_id, &payload.agent, auth)
        .await?;

    Ok(Json(CreateSessionResponse {
        session_id: session_id.to_string(),
    }))
}

async fn list_sessions(
    ClientAuth(auth): ClientAuth,
    State(state): State<HttpState>,
    axum_extra::extract::Query(query): axum_extra::extract::Query<ListSessionsQuery>,
) -> Result<Json<Vec<SessionSummary>>, AppError> {
    let statuses = if query.status.is_empty() {
        None
    } else {
        Some(query.status)
    };

    let filter = AggregateFilter {
        aggregate_type: Some("session".into()),
        tenant_id: Some(auth.tenant_id),
        status: statuses,
        label: query.agent,
        sort: query.sort,
        ..Default::default()
    };

    let summaries: Vec<SessionSummary> = state
        .runtime
        .store()
        .list_aggregates(&filter)
        .await
        .into_iter()
        .map(SessionSummary::from)
        .collect();
    Ok(Json(summaries))
}

async fn get_session(
    ClientAuth(auth): ClientAuth,
    State(state): State<HttpState>,
    Path(session_id): Path<Uuid>,
) -> Result<Json<SessionSummary>, AppError> {
    let summary = find_aggregate_summary(&state.runtime, session_id, &auth.tenant_id).await?;
    Ok(Json(SessionSummary::from(summary)))
}

#[tracing::instrument(skip(auth, state, input), fields(%session_id))]
async fn run_ag_ui(
    ClientAuth(auth): ClientAuth,
    State(state): State<HttpState>,
    Path(session_id): Path<Uuid>,
    Json(input): Json<crate::ag_ui::RunAgentInput>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, AppError> {
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
    ClientAuth(auth): ClientAuth,
    State(state): State<HttpState>,
    Path(session_id): Path<Uuid>,
    Query(query): Query<ObserveQuery>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, AppError> {
    ensure_session_running(&state.runtime, session_id, &auth).await?;

    let run_id = query.run_id.unwrap_or_else(|| Uuid::new_v4().to_string());
    let stream = observe_session(&state.runtime, session_id, auth, run_id).await?;

    Ok(sse_from_ag_ui(Box::pin(stream)))
}

// ---------------------------------------------------------------------------
// Shared
// ---------------------------------------------------------------------------

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
    #[error("auth error: {0}")]
    Auth(#[from] AuthError),
}

impl axum::response::IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let status = match &self {
            AppError::Runtime(RuntimeError::SessionNotFound) => StatusCode::NOT_FOUND,
            AppError::Runtime(RuntimeError::UnknownAgent(_)) => StatusCode::BAD_REQUEST,
            AppError::Auth(_) => StatusCode::UNAUTHORIZED,
            _ => StatusCode::BAD_REQUEST,
        };
        let body = Json(serde_json::json!({
            "error": self.to_string(),
        }));
        (status, body).into_response()
    }
}

/// Ensure the session actor is running (start from store if needed).
#[tracing::instrument(skip(runtime, auth), fields(%session_id))]
async fn ensure_session_running(
    runtime: &Runtime,
    session_id: Uuid,
    auth: &ClientIdentity,
) -> Result<(), AppError> {
    if runtime.session_is_running(session_id) {
        return Ok(());
    }

    let summary = find_aggregate_summary(runtime, session_id, &auth.tenant_id).await?;
    let agent_name = summary
        .label
        .as_deref()
        .ok_or(AppError::Runtime(RuntimeError::SessionNotFound))?;
    let _handle = runtime
        .start_session(session_id, agent_name, auth.clone())
        .await?;
    Ok(())
}

async fn find_aggregate_summary(
    runtime: &Runtime,
    session_id: Uuid,
    tenant_id: &str,
) -> Result<AggregateSummary, AppError> {
    let filter = AggregateFilter {
        aggregate_ids: Some(vec![session_id]),
        tenant_id: Some(tenant_id.into()),
        ..Default::default()
    };
    runtime
        .store()
        .list_aggregates(&filter)
        .await
        .into_iter()
        .next()
        .ok_or_else(|| AppError::Runtime(RuntimeError::SessionNotFound))
}
