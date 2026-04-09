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
use crate::worker::push::PushRegistrationRecord;
use crate::worker::SubmitDecision;
use crate::SubmitClientPayload;

use super::types::{
    MintClientTokenRequest, MintClientTokenResponse, RegisterRequest, RegisterResponse,
    SubmitClientPayloadRequest, SubmitRequest, SubmitResponse,
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

pub async fn register(
    State(state): State<WorkerHttpState>,
    Extension(principal): Extension<AuthPrincipal>,
    Json(req): Json<RegisterRequest>,
) -> impl IntoResponse {
    let mut config = req.config.clone();
    let mut signing_secret: Option<String> = None;

    if req.transport_type == "http" {
        if let Some(obj) = config.as_object_mut() {
            let has_secret = obj
                .get("signing_secret")
                .is_some_and(|v| v.is_string() && !v.as_str().unwrap_or("").is_empty());
            if has_secret {
                signing_secret = obj
                    .get("signing_secret")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
            } else {
                let secret = hex::encode(rand::random::<[u8; 32]>());
                obj.insert(
                    "signing_secret".to_string(),
                    serde_json::Value::String(secret.clone()),
                );
                signing_secret = Some(secret);
            }
        }
    }

    let record = PushRegistrationRecord {
        tenant_id: principal.tenant_id.clone(),
        transport_type: req.transport_type.clone(),
        config: config.clone(),
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

    Json(RegisterResponse {
        ok: true,
        signing_secret,
    })
    .into_response()
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
    if principal.tenant_id != req.auth.tenant_id {
        let body = serde_json::json!({"error": "tenant mismatch"});
        return (StatusCode::FORBIDDEN, Json(body)).into_response();
    }
    if req.auth.sub.trim().is_empty() {
        let body = serde_json::json!({"error": "sub is required"});
        return (StatusCode::BAD_REQUEST, Json(body)).into_response();
    }

    let session_id = req.session_id.unwrap_or_else(|| Uuid::now_v7().to_string());
    let auth = ClientIdentity {
        tenant_id: req.auth.tenant_id.clone(),
        sub: Some(req.auth.sub),
        attrs: req.auth.attrs,
    };

    let result = state
        .adapter
        .runtime
        .submit_client_payload(SubmitClientPayload {
            session_id,
            tenant_id: req.auth.tenant_id,
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
