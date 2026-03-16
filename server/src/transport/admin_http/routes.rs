use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use base64::Engine;
use serde::Deserialize;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use uuid::Uuid;

use substructure_core::event_store::{AggregateSort, EventFilter};
use substructure_core::session_index::{SessionCursor, SessionFilter};
use substructure_core::Runtime;

#[derive(Debug, Deserialize)]
pub struct ListSessionsParams {
    pub tenant_id: Option<String>,
    #[serde(default = "default_true")]
    pub top_level: bool,
    #[serde(default)]
    pub sort: AggregateSort,
    pub limit: Option<usize>,
    pub cursor: Option<String>,
}

fn default_true() -> bool {
    true
}

pub async fn list_sessions(
    State(runtime): State<Arc<Runtime>>,
    Query(params): Query<ListSessionsParams>,
) -> impl IntoResponse {
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
        tenant_id: params.tenant_id,
        top_level: params.top_level,
        sort: params.sort,
        limit: params.limit,
        cursor,
    };

    match runtime.list_sessions(&filter).await {
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
    State(runtime): State<Arc<Runtime>>,
    Path(session_id): Path<Uuid>,
) -> impl IntoResponse {
    match runtime.get_session(session_id).await {
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
    State(runtime): State<Arc<Runtime>>,
    Path(session_id): Path<Uuid>,
    Query(params): Query<SessionEventsParams>,
) -> impl IntoResponse {
    let filter = EventFilter {
        aggregate_id: Some(session_id),
        sequence_after: params.sequence_after,
        limit: params.limit,
        ..Default::default()
    };
    match runtime.get_session_events(&filter).await {
        Ok(events) => Json(events).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

pub async fn stream_session_events(
    State(runtime): State<Arc<Runtime>>,
    Path(session_id): Path<Uuid>,
    Query(params): Query<SessionEventsParams>,
) -> Response {
    // Subscribe BEFORE querying historical events so we don't miss anything
    let live_rx = runtime.subscribe_session_admin(session_id);

    // Query historical events
    let filter = EventFilter {
        aggregate_id: Some(session_id),
        sequence_after: params.sequence_after,
        limit: None,
        ..Default::default()
    };
    let historical = runtime.get_session_events(&filter).await.unwrap_or_default();

    // Track last sequence sent to deduplicate overlap between historical and live
    let last_historical_seq = historical.last().map(|e| e.sequence).unwrap_or(0);

    // Create a channel that combines historical + live
    let (tx, rx) = tokio::sync::mpsc::channel(64);

    tokio::spawn(async move {
        // Send historical events first
        for event in historical {
            if tx.send(event).await.is_err() {
                return;
            }
        }

        // Then stream live events, deduplicating
        let mut live_stream = ReceiverStream::new(live_rx);
        while let Some(event) = live_stream.next().await {
            if event.sequence <= last_historical_seq {
                continue;
            }
            if tx.send(event).await.is_err() {
                return;
            }
        }
    });

    let stream = ReceiverStream::new(rx).map(|event| {
        let mut line = serde_json::to_string(&event).unwrap_or_default();
        line.push('\n');
        Ok::<_, std::convert::Infallible>(line)
    });

    Response::builder()
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .body(Body::from_stream(stream))
        .unwrap()
        .into_response()
}
