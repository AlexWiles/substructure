use axum::extract::{Extension, Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::StreamExt;
use std::time::Duration;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::protocol::{OwnerKind, SessionOwner};
use crate::session::command::SessionError;
use crate::session::decision::{EffectResultPayload, WorkKind};
use crate::session::subscriptions::{SessionSubscriptionSpec, SubscriptionScope};
use crate::session::wire::{resolve_response, result_to_string};
use crate::span::SpanContext;
use crate::transport::http::Body;
use crate::transport::session_sse::{merge_session_stream, resume_cursor};
use crate::worker::SubmitDecision;
use crate::{Caller, EffectSettlement, RuntimeError, SettleEffectInput, SubmitClientPayload};

use super::types::{
    MintClientTokenRequest, MintClientTokenResponse, SettleEffectRequest,
    StreamSessionEventsParams, SubmitClientPayloadRequest, SubmitClientPayloadResponse,
    SubmitDecisionRequest, SubmitResponse,
};
use super::WorkerHttpState;

pub async fn submit(
    State(state): State<WorkerHttpState>,
    Extension(caller): Extension<Caller>,
    Body(req): Body<SubmitDecisionRequest>,
) -> Response {
    let span = req
        .span
        .unwrap_or_else(SpanContext::root)
        .child("worker_submit");

    // Out-of-band submit: no answered trigger and no resolved config in scope, so
    // every settle must name its own id and every llm.call must name its model
    // and its llm block. An unresolvable one can't be filled here — reject it.
    let blocks = state.runtime.llm_blocks(caller.tenant_id());
    let resolved = match resolve_response(req.decision, None, None, &blocks) {
        Ok(resolved) => resolved,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(SubmitResponse {
                    ok: false,
                    error: Some(e.to_string()),
                }),
            )
                .into_response();
        }
    };

    let result = state
        .runtime
        .submit_decision(SubmitDecision {
            session_id: req.session_id,
            caller,
            decision_id: req.decision_id,
            transcript: resolved.messages,
            actions: resolved.actions,
            state: resolved.state,
            agent: resolved.agent,
            channels: resolved.channels,
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
    Extension(caller): Extension<Caller>,
    Body(req): Body<MintClientTokenRequest>,
) -> impl IntoResponse {
    if req.identity.id.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "identity.id is required"})),
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
        caller.tenant_id(),
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

pub async fn settle_effect(
    State(state): State<WorkerHttpState>,
    Extension(caller): Extension<Caller>,
    Path(session_id): Path<String>,
    Body(req): Body<SettleEffectRequest>,
) -> Response {
    let (kind, id, attempt, settlement) = match req {
        SettleEffectRequest::ToolResult {
            id,
            attempt,
            result,
        } => (
            WorkKind::ToolCall,
            id,
            attempt,
            EffectSettlement::Result(EffectResultPayload::ToolCall {
                result: result_to_string(result),
            }),
        ),
        SettleEffectRequest::LlmResult {
            id,
            attempt,
            response,
        } => (
            WorkKind::LlmCall,
            id,
            attempt,
            EffectSettlement::Result(EffectResultPayload::LlmCall {
                response: Box::new(response),
            }),
        ),
        SettleEffectRequest::ToolError {
            id,
            error,
            retryable,
            attempt,
            code,
            detail,
        } => (
            WorkKind::ToolCall,
            id,
            attempt,
            EffectSettlement::Error {
                error,
                retryable,
                code,
                detail,
            },
        ),
        SettleEffectRequest::LlmError {
            id,
            error,
            retryable,
            attempt,
            code,
            detail,
        } => (
            WorkKind::LlmCall,
            id,
            attempt,
            EffectSettlement::Error {
                error,
                retryable,
                code,
                detail,
            },
        ),
    };

    let result = state
        .runtime
        .settle_effect(SettleEffectInput {
            session_id,
            kind,
            id,
            attempt,
            settlement,
            caller,
            span: SpanContext::root().child("machine_effect_settle"),
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

/// Map RuntimeError variants to HTTP status codes for the worker API.
pub(crate) fn runtime_error_status(err: &RuntimeError) -> (StatusCode, String) {
    let status = match err {
        RuntimeError::Session(
            SessionError::MissingSubject
            | SessionError::SessionAccessDenied
            | SessionError::EffectWrongHandler,
        ) => StatusCode::FORBIDDEN,
        RuntimeError::Session(SessionError::EffectNotFound | SessionError::SessionNotCreated) => {
            StatusCode::NOT_FOUND
        }
        RuntimeError::Session(
            SessionError::EffectNotPending
            | SessionError::EffectAttemptMismatch
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
    Extension(caller): Extension<Caller>,
    Body(req): Body<SubmitClientPayloadRequest>,
) -> Response {
    if req.identity.id.trim().is_empty() {
        let body = serde_json::json!({"error": "id is required"});
        return (StatusCode::BAD_REQUEST, Json(body)).into_response();
    }

    let session_id = req.session_id.unwrap_or_else(|| Uuid::now_v7().to_string());
    let owner = SessionOwner {
        tenant_id: caller.tenant_id().to_string(),
        id: Some(req.identity.id),
        kind: OwnerKind::Frontend,
        metadata: req.identity.metadata,
    };

    let result = state
        .runtime
        .submit_client_payload(SubmitClientPayload {
            session_id,
            caller,
            owner,
            agent_id: req.agent_id,
            payload: req.payload,
            turn_id: req.turn_id,
            continue_turn: false,
            queue: req.queue,
        })
        .await;

    match result {
        Ok(output) => (
            StatusCode::ACCEPTED,
            Json(SubmitClientPayloadResponse {
                session_id: output.session_id,
                turn_id: output.turn_id,
                queued: output.queued,
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
    Extension(caller): Extension<Caller>,
    Path(session_id): Path<String>,
    Query(params): Query<StreamSessionEventsParams>,
    headers: HeaderMap,
) -> Response {
    let root_session_id = session_id.clone();
    let scope_turn_id = params.turn_id.clone();
    let delta_rx = state
        .runtime
        .subscribe_token_deltas(&caller, &root_session_id)
        .await;
    let spec = SessionSubscriptionSpec {
        session_id,
        caller,
        scope: match params.turn_id {
            Some(turn_id) => SubscriptionScope::Turn { turn_id },
            None => SubscriptionScope::All,
        },
    };

    let after = resume_cursor(&headers, params.after_seq);
    let event_rx = match state.runtime.stream(spec, after).await {
        Ok(rx) => rx,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response()
        }
    };
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
        // A never-created session is a stale client reference, not a server fault.
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

    #[test]
    fn effect_settle_errors_map_to_expected_statuses() {
        assert_eq!(
            status_of(SessionError::EffectNotFound),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            status_of(SessionError::EffectWrongHandler),
            StatusCode::FORBIDDEN
        );
        assert_eq!(
            status_of(SessionError::EffectNotPending),
            StatusCode::CONFLICT
        );
        assert_eq!(
            status_of(SessionError::EffectAttemptMismatch),
            StatusCode::CONFLICT
        );
    }
}
