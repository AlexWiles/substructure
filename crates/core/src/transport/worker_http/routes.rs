use axum::extract::{Extension, Path, Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::StreamExt;
use std::time::Duration;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::identity::ClientIdentity;
use crate::session::command::SessionError;
use crate::session::subscriptions::SessionSubscriptionSpec;
use crate::span::SpanContext;
use crate::transport::auth::AuthPrincipal;
use crate::worker::SubmitDecision;
use crate::{
    Caller, RuntimeError, SubmitClientPayload, SubmitToolCallResult, SubmitToolCallResultInput,
};

use super::types::{
    MintClientTokenRequest, MintClientTokenResponse, StreamSessionEventsParams,
    SubmitClientPayloadRequest, SubmitClientPayloadResponse, SubmitRequest, SubmitResponse,
    SubmitToolCallResultRequest,
};
use super::WorkerHttpState;

pub async fn submit(
    State(state): State<WorkerHttpState>,
    Extension(principal): Extension<AuthPrincipal>,
    Json(req): Json<SubmitRequest>,
) -> Response {
    let span = req
        .span
        .unwrap_or_else(SpanContext::root)
        .child("worker_submit");

    let Some(caller) = machine_caller(&principal) else {
        return machine_subject_required();
    };

    let result = state
        .runtime
        .submit_decision(SubmitDecision {
            session_id: req.session_id,
            tenant_id: principal.tenant_id,
            caller,
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
        Err(e) => {
            let (status, message) = runtime_error_status(&e);
            (
                status,
                Json(SubmitResponse {
                    ok: false,
                    error: Some(message),
                }),
            )
                .into_response()
        }
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

pub async fn submit_tool_call_result(
    State(state): State<WorkerHttpState>,
    Extension(principal): Extension<AuthPrincipal>,
    Path(session_id): Path<String>,
    Json(req): Json<SubmitToolCallResultRequest>,
) -> Response {
    let Some(caller) = machine_caller(&principal) else {
        return machine_subject_required();
    };

    let (tool_call_id, attempt, result) = match req {
        SubmitToolCallResultRequest::Result {
            tool_call_id,
            result,
            attempt,
        } => (
            tool_call_id,
            attempt,
            SubmitToolCallResult::Result { result },
        ),
        SubmitToolCallResultRequest::Error {
            tool_call_id,
            error,
            retryable,
            attempt,
        } => (
            tool_call_id,
            attempt,
            SubmitToolCallResult::Error { error, retryable },
        ),
    };

    let result = state
        .runtime
        .submit_tool_call_result(SubmitToolCallResultInput {
            session_id,
            tenant_id: principal.tenant_id,
            tool_call_id,
            attempt,
            result,
            caller,
            span: SpanContext::root().child("machine_tool_call_result"),
        })
        .await;

    match result {
        Ok(()) => Json(SubmitResponse {
            ok: true,
            error: None,
        })
        .into_response(),
        Err(e) => {
            let (status, message) = runtime_error_status(&e);
            (
                status,
                Json(SubmitResponse {
                    ok: false,
                    error: Some(message),
                }),
            )
                .into_response()
        }
    }
}

fn machine_caller(principal: &AuthPrincipal) -> Option<Caller> {
    principal
        .subject
        .as_deref()
        .filter(|s| !s.is_empty())
        .map(|s| Caller::Machine {
            tenant_id: principal.tenant_id.clone(),
            key_id: s.to_string(),
        })
}

fn machine_subject_required() -> Response {
    (
        StatusCode::FORBIDDEN,
        Json(serde_json::json!({"error": "machine subject is required"})),
    )
        .into_response()
}

/// Map RuntimeError variants to HTTP status codes for the worker API.
pub(crate) fn runtime_error_status(err: &RuntimeError) -> (StatusCode, String) {
    let status = match err {
        RuntimeError::Session(
            SessionError::MissingSubject
            | SessionError::SessionAccessDenied
            | SessionError::ToolCallWrongHandler,
        ) => StatusCode::FORBIDDEN,
        RuntimeError::Session(SessionError::ToolCallNotFound) => StatusCode::NOT_FOUND,
        RuntimeError::Session(
            SessionError::ToolCallNotPending
            | SessionError::ToolCallAttemptMismatch
            | SessionError::SessionInterrupted
            | SessionError::TurnAlreadyActive { .. }
            | SessionError::TurnAlreadyCompleted { .. },
        ) => StatusCode::CONFLICT,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (status, err.to_string())
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

    let Some(caller) = machine_caller(&principal) else {
        return machine_subject_required();
    };

    let session_id = req.session_id.unwrap_or_else(|| Uuid::now_v7().to_string());
    let identity = ClientIdentity {
        tenant_id: principal.tenant_id.clone(),
        id: Some(req.identity.id),
        metadata: req.identity.metadata,
    };

    let result = state
        .runtime
        .submit_client_payload(SubmitClientPayload {
            session_id,
            tenant_id: principal.tenant_id,
            caller,
            identity,
            agent_id: req.agent_id,
            payload: req.payload,
            turn_id: req.turn_id,
        })
        .await;

    match result {
        Ok(output) => (
            StatusCode::ACCEPTED,
            Json(SubmitClientPayloadResponse {
                session_id: output.session_id,
                turn_id: output.turn_id,
            }),
        )
            .into_response(),
        Err(e) => {
            let (status, message) = runtime_error_status(&e);
            (status, Json(serde_json::json!({"error": message}))).into_response()
        }
    }
}

pub async fn stream_session_events(
    State(state): State<WorkerHttpState>,
    Extension(_principal): Extension<AuthPrincipal>,
    Path(session_id): Path<String>,
    Query(params): Query<StreamSessionEventsParams>,
) -> Response {
    let spec = match params.turn_id {
        Some(turn_id) => SessionSubscriptionSpec::Turn {
            root_session_id: session_id,
            turn_id,
        },
        None => SessionSubscriptionSpec::All {
            root_session_id: session_id,
        },
    };

    let rx = state.runtime.stream(spec, params.sequence_after).await;
    let stream = ReceiverStream::new(rx)
        .take_until(state.shutdown.clone().cancelled_owned())
        .map(|event| {
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
