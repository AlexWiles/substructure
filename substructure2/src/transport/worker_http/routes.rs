use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;

use crate::runtime::span::SpanContext;
use crate::runtime::worker::{DequeueFilter, SubmitDecision};
use crate::runtime::Runtime;

use super::types::{PollRequest, PollResponse, SubmitRequest, SubmitResponse};

const MAX_POLL_TIMEOUT: Duration = Duration::from_secs(60);

pub async fn poll(
    State(runtime): State<Arc<Runtime>>,
    Json(req): Json<PollRequest>,
) -> impl IntoResponse {
    let timeout = Duration::from_millis(req.timeout_ms).min(MAX_POLL_TIMEOUT);

    let filter = DequeueFilter {
        tenant_id: req.tenant_id,
        agent_ids: req.agent_ids,
    };

    let result = tokio::time::timeout(timeout, runtime.dequeue_decision(&filter)).await;

    match result {
        Ok(Some(decision)) => {
            let response = PollResponse {
                session_id: decision.session_id,
                tenant_id: decision.tenant_id,
                decision_id: decision.decision_id,
                agent_id: decision.agent_id,
                trigger: decision.trigger,
                worker_state: decision.worker_state,
                span: decision.span,
                attempts: decision.attempts,
                deadline: decision.deadline,
            };
            (StatusCode::OK, Json(response)).into_response()
        }
        _ => StatusCode::NO_CONTENT.into_response(),
    }
}

pub async fn submit(
    State(runtime): State<Arc<Runtime>>,
    Json(req): Json<SubmitRequest>,
) -> Json<SubmitResponse> {
    let span = req
        .span
        .unwrap_or_else(SpanContext::root)
        .child("worker_submit");

    let result = runtime
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
