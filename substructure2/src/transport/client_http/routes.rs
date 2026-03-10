use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use uuid::Uuid;

use crate::runtime::{Runtime, SendMessage};

use super::types::{SendMessageRequest, SendMessageResponse};

pub async fn send_message(
    State(runtime): State<Arc<Runtime>>,
    Json(req): Json<SendMessageRequest>,
) -> Json<SendMessageResponse> {
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
        Ok(()) => Json(SendMessageResponse {
            session_id,
            ok: true,
            error: None,
        }),
        Err(e) => Json(SendMessageResponse {
            session_id,
            ok: false,
            error: Some(e.to_string()),
        }),
    }
}
