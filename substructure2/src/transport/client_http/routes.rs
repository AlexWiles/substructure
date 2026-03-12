use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::Json;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use uuid::Uuid;

use crate::runtime::{Runtime, SendMessage};

use super::types::SendMessageRequest;

pub async fn send_message(
    State(runtime): State<Arc<Runtime>>,
    Json(req): Json<SendMessageRequest>,
) -> Response {
    let session_id = req.session_id.unwrap_or_else(Uuid::now_v7);
    let tenant_id = req.tenant_id.unwrap_or_else(|| "default".to_string());

    let result = runtime
        .send_message(SendMessage {
            session_id,
            tenant_id,
            agent_id: req.agent_id,
            content: req.message,
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
