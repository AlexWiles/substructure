use axum::extract::{Extension, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use std::time::Duration;

use crate::transport::auth::AuthPrincipal;
use crate::span::SpanContext;
use crate::worker::push::PushRegistrationRecord;
use crate::worker::SubmitDecision;

use super::WorkerHttpState;
use super::types::{
    MintClientTokenRequest, MintClientTokenResponse, RegisterRequest, RegisterResponse,
    SubmitRequest, SubmitResponse,
};

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

pub async fn mint_client_token(
    State(state): State<WorkerHttpState>,
    Extension(principal): Extension<AuthPrincipal>,
    Json(req): Json<MintClientTokenRequest>,
) -> impl IntoResponse {
    if principal.tenant_id != req.tenant_id {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error": "tenant mismatch"})),
        )
            .into_response();
    }
    if principal.source != "api_key" {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error": "machine auth required"})),
        )
            .into_response();
    }
    if req.sub.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "sub is required"})),
        )
            .into_response();
    }
    if principal.subject.as_deref().is_none_or(str::is_empty) {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error": "machine subject is required"})),
        )
            .into_response();
    }

    let ttl_secs = req.ttl_seconds.unwrap_or(600);
    if ttl_secs == 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "ttl_seconds must be positive"})),
        )
            .into_response();
    }

    match state.client_token_issuer.issue_token(
        req.tenant_id,
        req.sub,
        req.attrs,
        Duration::from_secs(ttl_secs),
    ) {
        Ok((token, expires_at)) => Json(MintClientTokenResponse { token, expires_at }).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}
