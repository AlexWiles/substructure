use axum::extract::{Extension, Path, Query, State};
use axum::http::StatusCode;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::StreamExt;
use std::time::Duration;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::identity::ClientIdentity;
use crate::session::command::SessionError;
use crate::session::subscriptions::{SessionRequester, SessionSubscriptionSpec};
use crate::span::SpanContext;
use crate::transport::auth::AuthPrincipal;
use crate::transport::session_sse::merge_session_stream;
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
        RuntimeError::Session(SessionError::ToolCallNotFound | SessionError::SessionNotCreated) => {
            StatusCode::NOT_FOUND
        }
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
    Extension(principal): Extension<AuthPrincipal>,
    Path(session_id): Path<String>,
    Query(params): Query<StreamSessionEventsParams>,
) -> Response {
    let root_session_id = session_id.clone();
    let scope_turn_id = params.turn_id.clone();
    // The worker transport is a privileged machine caller: unrestricted.
    let spec = match params.turn_id {
        Some(turn_id) => SessionSubscriptionSpec::Turn {
            tenant_id: principal.tenant_id.clone(),
            root_session_id: session_id,
            turn_id,
            requester: SessionRequester::Privileged,
        },
        None => SessionSubscriptionSpec::All {
            tenant_id: principal.tenant_id.clone(),
            root_session_id: session_id,
            requester: SessionRequester::Privileged,
        },
    };

    let event_rx = match state.runtime.stream(spec, params.sequence_after).await {
        Ok(rx) => rx,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response()
        }
    };
    let delta_rx = state
        .runtime
        .subscribe_token_deltas(&principal.tenant_id, &root_session_id)
        .await;
    let out_rx = merge_session_stream(event_rx, delta_rx, scope_turn_id, state.shutdown.clone());
    let stream = ReceiverStream::new(out_rx).map(Ok::<_, std::convert::Infallible>);

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn status_of(err: SessionError) -> StatusCode {
        runtime_error_status(&RuntimeError::Session(err)).0
    }

    #[test]
    fn missing_session_is_not_found_not_server_error() {
        // A worker only ever holds a decision a session emitted, so a
        // submission against a never-created session is a stale/garbage
        // reference (client error), never a server fault.
        assert_eq!(
            status_of(SessionError::SessionNotCreated),
            StatusCode::NOT_FOUND
        );
    }

    #[test]
    fn access_denied_is_forbidden() {
        assert_eq!(
            status_of(SessionError::SessionAccessDenied),
            StatusCode::FORBIDDEN
        );
    }
}
