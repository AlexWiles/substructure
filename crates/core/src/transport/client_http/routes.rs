use axum::body::Body;
use axum::extract::{Extension, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::Json;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use uuid::Uuid;

use crate::transport::auth::AuthPrincipal;
use crate::SendMessage;

use super::types::SendMessageRequest;
use super::ClientHttpState;

pub async fn send_message(
    State(state): State<ClientHttpState>,
    Extension(principal): Extension<AuthPrincipal>,
    Json(req): Json<SendMessageRequest>,
) -> Response {
    let session_id = req.session_id.unwrap_or_else(|| Uuid::now_v7().to_string());

    let result = state
        .runtime
        .send_message(SendMessage {
            session_id,
            tenant_id: principal.tenant_id,
            agent_id: req.agent_id,
            content: req.message,
            turn_id: req.turn_id,
        })
        .await;

    match result {
        Ok((_session_id, rx)) => {
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
        Err(e) => {
            let body = serde_json::json!({"error": e.to_string()});
            (axum::http::StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
        }
    }
}
