use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use base64::Engine;
use futures_util::StreamExt;
use serde::Deserialize;
use tokio_stream::wrappers::ReceiverStream;

use uuid::Uuid;

use crate::event_store::{AggregateSort, EventFilter};
use crate::session::index::{SessionCursor, SessionFilter};
use crate::session::subscriptions::SessionSubscriptionSpec;
use crate::transport::ag_ui::snapshot::{message_history, snapshot_events};
use crate::transport::auth::AuthPrincipal;

use super::AdminHttpState;

#[derive(Debug, Deserialize)]
pub struct ListSessionsParams {
    #[serde(default = "default_true")]
    pub top_level: bool,
    #[serde(default)]
    pub sort: AggregateSort,
    pub limit: Option<usize>,
    pub cursor: Option<String>,
    pub session_id: Option<String>,
    pub agent_id: Option<String>,
}

fn default_true() -> bool {
    true
}

pub async fn list_sessions(
    State(state): State<AdminHttpState>,
    Extension(principal): Extension<AuthPrincipal>,
    Query(params): Query<ListSessionsParams>,
) -> impl IntoResponse {
    let tenant_id = principal.tenant_id;
    let cursor = match params.cursor {
        Some(ref encoded) => match decode_cursor(encoded) {
            Ok(c) => Some(c),
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({"error": e})),
                )
                    .into_response();
            }
        },
        None => None,
    };

    let filter = SessionFilter {
        tenant_id: Some(tenant_id),
        session_id: params.session_id,
        agent_id: params.agent_id,
        top_level: params.top_level,
        sort: params.sort,
        limit: params.limit,
        cursor,
    };

    match state.runtime.list_sessions(&filter).await {
        Ok(page) => {
            let next_cursor = page
                .next_cursor
                .as_ref()
                .and_then(|c| encode_cursor(c).ok());
            Json(serde_json::json!({
                "items": page.items,
                "next_cursor": next_cursor,
            }))
            .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

fn encode_cursor(cursor: &SessionCursor) -> Result<String, String> {
    let json = serde_json::to_string(cursor).map_err(|e| e.to_string())?;
    Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(json.as_bytes()))
}

fn decode_cursor(encoded: &str) -> Result<SessionCursor, String> {
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(encoded)
        .map_err(|e| format!("invalid cursor encoding: {e}"))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("invalid cursor: {e}"))
}

pub async fn get_session(
    State(state): State<AdminHttpState>,
    Extension(principal): Extension<AuthPrincipal>,
    Path(session_id): Path<String>,
) -> impl IntoResponse {
    match state
        .runtime
        .get_session(&principal.tenant_id, &session_id)
        .await
    {
        Ok((snapshot, state)) => Json(serde_json::json!({
            "stream_version": snapshot.stream_version,
            "first_event_at": snapshot.first_event_at,
            "last_event_at": snapshot.last_event_at,
            "state": state,
        }))
        .into_response(),
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct SessionEventsParams {
    pub sequence_after: Option<u64>,
    pub limit: Option<usize>,
}

pub async fn get_session_events(
    State(state): State<AdminHttpState>,
    Path(session_id): Path<String>,
    Query(params): Query<SessionEventsParams>,
) -> impl IntoResponse {
    let filter = EventFilter {
        aggregate_id: Some(session_id.clone()),
        sequence_after: params.sequence_after,
        limit: params.limit,
        ..Default::default()
    };
    match state.runtime.get_session_events(&filter).await {
        Ok(events) => Json(events).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

pub async fn stream_session_events(
    State(state): State<AdminHttpState>,
    Extension(principal): Extension<AuthPrincipal>,
    Path(session_id): Path<String>,
    Query(params): Query<SessionEventsParams>,
) -> Response {
    let spec = SessionSubscriptionSpec::All {
        tenant_id: principal.tenant_id,
        root_session_id: session_id,
    };
    // Admin endpoint defaults to full-history replay (sequence_after defaults to 0).
    let sequence_after = Some(params.sequence_after.unwrap_or(0));
    let rx = state.runtime.stream(spec, sequence_after).await;

    let stream = ReceiverStream::new(rx)
        .take_until(state.shutdown.clone().cancelled_owned())
        .map(|event| {
            let event_type = event.payload_type().to_owned();
            let data = serde_json::to_string(&event).unwrap_or_default();
            Ok::<_, std::convert::Infallible>(
                SseEvent::default()
                    .id(event.sequence.to_string())
                    .event(event_type)
                    .data(data),
            )
        });

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// `GET /api/admin/sessions/{session_id}/ag-ui/messages` — the session's history
/// as an AG-UI message list, for hydrating an admin console (e.g. as `HttpAgent`'s
/// `initialMessages`). Tenant-scoped to any session in the tenant. Read-only.
pub async fn session_ag_ui_messages(
    State(state): State<AdminHttpState>,
    Extension(principal): Extension<AuthPrincipal>,
    Path(session_id): Path<String>,
) -> Response {
    let events = match load_session_events(&state, &principal, session_id).await {
        Ok(events) => events,
        Err(response) => return response,
    };
    Json(message_history(&events)).into_response()
}

/// `GET /api/admin/sessions/{session_id}/ag-ui/snapshot` — the same history as an
/// SSE `RUN_STARTED → MESSAGES_SNAPSHOT → RUN_FINISHED` sequence, for replaying a
/// session into a stock AG-UI client. Read-only.
pub async fn stream_session_ag_ui(
    State(state): State<AdminHttpState>,
    Extension(principal): Extension<AuthPrincipal>,
    Path(session_id): Path<String>,
) -> Response {
    let events = match load_session_events(&state, &principal, session_id.clone()).await {
        Ok(events) => events,
        Err(response) => return response,
    };
    let run_id = Uuid::now_v7().to_string();
    let frames = snapshot_events(session_id, run_id, &events)
        .into_iter()
        .map(|e| Ok::<_, std::convert::Infallible>(e.to_sse()));
    Sse::new(futures_util::stream::iter(frames)).into_response()
}

/// Load a session's events, tenant-scoped. On error, returns the response to send.
async fn load_session_events(
    state: &AdminHttpState,
    principal: &AuthPrincipal,
    session_id: String,
) -> Result<Vec<crate::event_store::Event>, Response> {
    let filter = EventFilter {
        aggregate_id: Some(session_id),
        tenant_id: Some(principal.tenant_id.clone()),
        ..Default::default()
    };
    state
        .runtime
        .get_session_events(&filter)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response()
        })
}
