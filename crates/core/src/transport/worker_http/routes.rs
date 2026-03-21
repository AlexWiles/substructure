use axum::extract::{Extension, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;

use crate::transport::auth::AuthPrincipal;
use crate::span::SpanContext;
use crate::worker::push::PushRegistrationRecord;
use crate::worker::SubmitDecision;

use super::WorkerHttpState;
use super::types::{RegisterRequest, RegisterResponse, SubmitRequest, SubmitResponse};

pub async fn submit(
    State(state): State<WorkerHttpState>,
    Extension(principal): Extension<AuthPrincipal>,
    Json(req): Json<SubmitRequest>,
) -> impl IntoResponse {
    let span = req
        .span
        .unwrap_or_else(SpanContext::root)
        .child("worker_submit");

    let result = state
        .adapter
        .runtime
        .submit_decision(SubmitDecision {
            session_id: req.session_id,
            tenant_id: principal.tenant_id,
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
        })
        .into_response(),
        Err(e) => Json(SubmitResponse {
            ok: false,
            error: Some(e.to_string()),
        })
        .into_response(),
    }
}

pub async fn register(
    State(state): State<WorkerHttpState>,
    Extension(principal): Extension<AuthPrincipal>,
    Json(req): Json<RegisterRequest>,
) -> impl IntoResponse {
    for agent_id in req.agent_ids {
        let record = PushRegistrationRecord {
            tenant_id: principal.tenant_id.clone(),
            agent_id,
            transport_type: req.transport_type.clone(),
            config: req.config.clone(),
        };
        if let Err(e) = state.adapter.register(record).await {
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
