use axum::extract::{Extension, State};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use uuid::Uuid;

use crate::identity::ClientIdentity;
use crate::transport::auth::AuthPrincipal;
use crate::SubmitClientPayload;

use super::types::SubmitClientPayloadRequest;
use super::ClientHttpState;

pub async fn submit_client_payload(
    State(state): State<ClientHttpState>,
    Extension(principal): Extension<AuthPrincipal>,
    Json(req): Json<SubmitClientPayloadRequest>,
) -> Response {
    let session_id = req.session_id.unwrap_or_else(|| Uuid::now_v7().to_string());
    let Some(subject) = principal.subject.clone() else {
        let body = serde_json::json!({"error": "client subject is required"});
        return (axum::http::StatusCode::FORBIDDEN, Json(body)).into_response();
    };
    let mut attrs = std::collections::HashMap::new();
    attrs.insert("source".to_string(), principal.source.to_string());
    let auth = ClientIdentity {
        tenant_id: principal.tenant_id.clone(),
        sub: Some(subject),
        attrs,
    };

    let result = state
        .runtime
        .submit_client_payload(SubmitClientPayload {
            session_id,
            tenant_id: principal.tenant_id,
            auth,
            agent_id: req.agent_id,
            payload: req.payload,
            turn_id: req.turn_id,
        })
        .await;

    match result {
        Ok((_session_id, rx)) => {
            let stream = ReceiverStream::new(rx).map(|event| {
                let event_type = event.payload_type().to_owned();
                let data = serde_json::to_string(&event).unwrap_or_default();
                Ok::<_, std::convert::Infallible>(SseEvent::default()
                    .id(event.sequence.to_string())
                    .event(event_type)
                    .data(data))
            });

            Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
        }
        Err(e) => {
            let message = e.to_string();
            if message.contains("client subject is required") || message.contains("session access denied") {
                let body = serde_json::json!({"error": message});
                return (axum::http::StatusCode::FORBIDDEN, Json(body)).into_response();
            }
            let body = serde_json::json!({"error": e.to_string()});
            (axum::http::StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
        }
    }
}
