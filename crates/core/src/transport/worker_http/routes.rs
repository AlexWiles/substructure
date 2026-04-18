use axum::extract::{Extension, State};
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use std::time::Duration;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use uuid::Uuid;

use crate::identity::ClientIdentity;
use crate::span::SpanContext;
use crate::transport::auth::AuthPrincipal;
use crate::worker::SubmitDecision;
use crate::SubmitClientPayload;

use super::types::{
    MintClientTokenRequest, MintClientTokenResponse, SubmitClientPayloadRequest, SubmitRequest,
    SubmitResponse,
};
use super::WorkerHttpState;

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

pub async fn mint_client_token(
    State(state): State<WorkerHttpState>,
    Extension(principal): Extension<AuthPrincipal>,
    Json(req): Json<MintClientTokenRequest>,
) -> impl IntoResponse {
    if principal.source != "api_key" {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error": "machine auth required"})),
        )
            .into_response();
    }
    if req.identity.id.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "identity.id is required"})),
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
        principal.tenant_id,
        req.identity.id,
        req.identity.metadata,
        Duration::from_secs(ttl_secs),
    ) {
        Ok((token, expires_at)) => {
            Json(MintClientTokenResponse { token, expires_at }).into_response()
        }
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

pub async fn submit_client_payload(
    State(state): State<WorkerHttpState>,
    Extension(principal): Extension<AuthPrincipal>,
    Json(req): Json<SubmitClientPayloadRequest>,
) -> Response {
    if req.identity.id.trim().is_empty() {
        let body = serde_json::json!({"error": "id is required"});
        return (StatusCode::BAD_REQUEST, Json(body)).into_response();
    }

    let session_id = req.session_id.unwrap_or_else(|| Uuid::now_v7().to_string());
    let identity = ClientIdentity {
        tenant_id: principal.tenant_id.clone(),
        id: Some(req.identity.id),
        metadata: req.identity.metadata,
    };

    let result = state
        .adapter
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
        Ok((_session_id, rx)) => {
            let stream = ReceiverStream::new(rx).map(|event| {
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
