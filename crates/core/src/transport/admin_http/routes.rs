use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use base64::Engine;
use futures_util::StreamExt;
use serde::Deserialize;
use tokio_stream::wrappers::ReceiverStream;

use crate::event_store::Seq;
use crate::session::index::SessionSort;
use crate::session::index::{SessionCursor, SessionFilter};
use crate::session::subscriptions::{SessionSubscriptionSpec, SubscriptionScope};
use crate::transport::ag_ui::snapshot::snapshot_events;
use crate::transport::ag_ui::types::RunAgentInput;
use crate::Caller;

use super::AdminHttpState;

#[derive(Debug, Deserialize)]
pub struct ListSessionsParams {
    #[serde(default = "default_true")]
    pub top_level: bool,
    #[serde(default)]
    pub sort: SessionSort,
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
    Extension(caller): Extension<Caller>,
    Query(params): Query<ListSessionsParams>,
) -> impl IntoResponse {
    let tenant_id = caller.tenant_id().to_string();
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
    Extension(caller): Extension<Caller>,
    Path(session_id): Path<String>,
) -> impl IntoResponse {
    match state
        .runtime
        .get_session(caller.tenant_id(), &session_id)
        .await
    {
        Ok(session) => Json(serde_json::json!({
            "seq": session.seq,
            "first_event_at": session.first_event_at,
            "last_event_at": session.last_event_at,
            "state": session.state,
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
    pub after_seq: Option<u64>,
    pub limit: Option<usize>,
}

pub async fn get_session_events(
    State(state): State<AdminHttpState>,
    Extension(caller): Extension<Caller>,
    Path(session_id): Path<String>,
    Query(params): Query<SessionEventsParams>,
) -> Response {
    match state
        .runtime
        .read_session_events(
            &caller,
            &session_id,
            params.after_seq.map(Seq),
            params.limit,
        )
        .await
    {
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
    Extension(caller): Extension<Caller>,
    Path(session_id): Path<String>,
    Query(params): Query<SessionEventsParams>,
) -> Response {
    let spec = SessionSubscriptionSpec {
        root_session_id: session_id,
        caller,
        scope: SubscriptionScope::All,
    };
    // Admin endpoint defaults to full-history replay (cursor defaults to 0).
    let after = Some(Seq(params.after_seq.unwrap_or(0)));
    let rx = match state.runtime.stream(spec, after).await {
        Ok(rx) => rx,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response()
        }
    };

    let stream = ReceiverStream::new(rx)
        .take_until(state.shutdown.clone().cancelled_owned())
        .map(|event| {
            let event_type = event.payload_type();
            let data = serde_json::to_string(&event).unwrap_or_default();
            Ok::<_, std::convert::Infallible>(
                SseEvent::default()
                    .id(event.seq.to_string())
                    .event(event_type)
                    .data(data),
            )
        });

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

pub async fn connect_session_ag_ui(
    State(state): State<AdminHttpState>,
    Extension(caller): Extension<Caller>,
    Path(session_id): Path<String>,
    Json(input): Json<RunAgentInput>,
) -> Response {
    let events = match state
        .runtime
        .read_session_events(&caller, &session_id, None, Some(1))
        .await
    {
        Ok(events) => events,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response()
        }
    };
    // No events ⇒ session not yet created (fresh thread); empty snapshot.
    let session = if events.is_empty() {
        None
    } else {
        match state
            .runtime
            .get_session(caller.tenant_id(), &session_id)
            .await
        {
            Ok(session) => Some(session.state),
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": e.to_string()})),
                )
                    .into_response()
            }
        }
    };
    let frames = snapshot_events(session_id, input.run_id, session.as_ref())
        .into_iter()
        .map(|e| Ok::<_, std::convert::Infallible>(e.to_sse()));
    Sse::new(futures_util::stream::iter(frames))
        .keep_alive(KeepAlive::default())
        .into_response()
}
