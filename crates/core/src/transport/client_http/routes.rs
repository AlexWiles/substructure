use axum::extract::{Extension, Path, Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::identity::ClientIdentity;
use crate::session::subscriptions::SessionSubscriptionSpec;
use crate::transport::auth::AuthPrincipal;
use crate::SubmitClientPayload;

use super::types::{
    StreamSessionEventsParams, SubmitClientPayloadRequest, SubmitClientPayloadResponse,
};
use super::ClientHttpState;

pub async fn submit_client_payload(
    State(state): State<ClientHttpState>,
    Extension(principal): Extension<AuthPrincipal>,
    Json(req): Json<SubmitClientPayloadRequest>,
) -> Response {
    let session_id = req.session_id.unwrap_or_else(|| Uuid::now_v7().to_string());
    let Some(subject) = principal.subject.clone() else {
        let body = serde_json::json!({"error": "client subject is required"});
        return (StatusCode::FORBIDDEN, Json(body)).into_response();
    };
    let identity = ClientIdentity {
        tenant_id: principal.tenant_id.clone(),
        id: Some(subject),
        metadata: std::collections::HashMap::new(),
    };

    let result = state
        .runtime
        .submit_client_payload(SubmitClientPayload {
            session_id,
            tenant_id: principal.tenant_id,
            identity,
            agent_id: req.agent_id,
            payload: req.payload,
            turn_id: req.turn_id,
        })
        .await;

    match result {
        Ok(output) => (
            StatusCode::ACCEPTED,
            Json(SubmitClientPayloadResponse {
                session_id: output.session_id,
                turn_id: output.turn_id,
            }),
        )
            .into_response(),
        Err(e) => {
            let message = e.to_string();
            if message.contains("client subject is required")
                || message.contains("session access denied")
            {
                let body = serde_json::json!({"error": message});
                return (StatusCode::FORBIDDEN, Json(body)).into_response();
            }
            let body = serde_json::json!({"error": message});
            (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
        }
    }
}

pub async fn stream_session_events(
    State(state): State<ClientHttpState>,
    Extension(_principal): Extension<AuthPrincipal>,
    Path(session_id): Path<String>,
    Query(params): Query<StreamSessionEventsParams>,
) -> Response {
    let spec = match params.turn_id {
        Some(turn_id) => SessionSubscriptionSpec::Turn {
            root_session_id: session_id,
            turn_id,
        },
        None => SessionSubscriptionSpec::All {
            root_session_id: session_id,
        },
    };

    let rx = state.runtime.stream(spec, params.sequence_after).await;
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
