use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;
use uuid::Uuid;

use crate::runtime::event_store::{AggregateFilter, AggregateSort, EventFilter};
use crate::runtime::Runtime;

#[derive(Debug, Deserialize)]
pub struct ListSessionsParams {
    pub tenant_id: Option<String>,
    #[serde(default)]
    pub sort: AggregateSort,
    pub limit: Option<usize>,
}

pub async fn list_sessions(
    State(runtime): State<Arc<Runtime>>,
    Query(params): Query<ListSessionsParams>,
) -> impl IntoResponse {
    let filter = AggregateFilter {
        aggregate_type: Some("session".to_string()),
        tenant_id: params.tenant_id,
        sort: params.sort,
        limit: params.limit,
        ..Default::default()
    };
    match runtime.list_sessions(&filter).await {
        Ok(results) => {
            let body: Vec<serde_json::Value> = results
                .into_iter()
                .map(|(summary, state)| {
                    serde_json::json!({
                        "summary": summary,
                        "state": state,
                    })
                })
                .collect();
            Json(body).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
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
