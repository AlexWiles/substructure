use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;

use crate::push::PushAdapter;
use crate::runtime::span::SpanContext;
use crate::runtime::worker::push::PushRegistrationRecord;
use crate::runtime::worker::SubmitDecision;

use super::types::{RegisterRequest, RegisterResponse, SubmitRequest, SubmitResponse};

pub async fn submit(
    State(adapter): State<Arc<PushAdapter>>,
    Json(req): Json<SubmitRequest>,
) -> Json<SubmitResponse> {
    let span = req
        .span
        .unwrap_or_else(SpanContext::root)
        .child("worker_submit");

    let result = adapter
        .runtime
        .submit_decision(SubmitDecision {
            session_id: req.session_id,
            tenant_id: req.tenant_id,
            decision_id: req.decision_id,
            actions: req.actions,
            state: req.state,
            span,
        })
        .await;

    match result {
        Ok(()) => Json(SubmitResponse {
            ok: true,
            error: None,
        }),
        Err(e) => Json(SubmitResponse {
            ok: false,
            error: Some(e.to_string()),
        }),
    }
}

pub async fn register(
    State(adapter): State<Arc<PushAdapter>>,
    Json(req): Json<RegisterRequest>,
) -> impl IntoResponse {
    for agent_id in req.agent_ids {
        let record = PushRegistrationRecord {
            tenant_id: req.tenant_id.clone(),
            agent_id,
            transport_type: req.transport_type.clone(),
            config: req.config.clone(),
        };
        if let Err(e) = adapter.register(record).await {
            return (
                StatusCode::BAD_REQUEST,
                Json(SubmitResponse {
                    ok: false,
                    error: Some(e),
                }),
            )
                .into_response();
        }
    }

    Json(RegisterResponse { ok: true }).into_response()
}
