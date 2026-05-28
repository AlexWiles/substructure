use axum::extract::{Extension, Path, Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::identity::ClientIdentity;
use crate::llm::TokenDelta;
use crate::session::command::SessionError;
use crate::session::subscriptions::SessionSubscriptionSpec;
use crate::transport::auth::AuthPrincipal;
use crate::{
    Caller, RuntimeError, SubmitClientPayload, SubmitToolCallResult, SubmitToolCallResultInput,
};

use super::types::{
    StreamSessionEventsParams, SubmitClientPayloadRequest, SubmitClientPayloadResponse,
    SubmitToolCallResultRequest, SubmitToolCallResultResponse,
};
use super::ClientHttpState;

pub async fn submit_client_payload(
    State(state): State<ClientHttpState>,
    Extension(principal): Extension<AuthPrincipal>,
    Json(req): Json<SubmitClientPayloadRequest>,
) -> Response {
    let Some(user_id) = principal.subject.clone() else {
        let body = serde_json::json!({"error": "client subject is required"});
        return (StatusCode::FORBIDDEN, Json(body)).into_response();
    };
    let session_id = req.session_id.unwrap_or_else(|| Uuid::now_v7().to_string());
    let identity = ClientIdentity {
        tenant_id: principal.tenant_id.clone(),
        id: Some(user_id.clone()),
        metadata: std::collections::HashMap::new(),
    };
    let caller = Caller::Frontend {
        tenant_id: principal.tenant_id.clone(),
        user_id,
        attrs: principal.attrs.clone(),
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
        Err(e) => runtime_error_response(e),
    }
}

pub async fn submit_tool_call_result(
    State(state): State<ClientHttpState>,
    Extension(principal): Extension<AuthPrincipal>,
    Path(session_id): Path<String>,
    Json(req): Json<SubmitToolCallResultRequest>,
) -> Response {
    let Some(user_id) = principal.subject.clone() else {
        let body = serde_json::json!({"error": "client subject is required"});
        return (StatusCode::FORBIDDEN, Json(body)).into_response();
    };
    let caller = Caller::Frontend {
        tenant_id: principal.tenant_id.clone(),
        user_id,
        attrs: principal.attrs.clone(),
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
            span: crate::span::SpanContext::root().child("client_tool_call_result"),
        })
        .await;

    match result {
        Ok(()) => Json(SubmitToolCallResultResponse {
            ok: true,
            error: None,
        })
        .into_response(),
        Err(e) => {
            let (status, message) = runtime_error_status(&e);
            (
                status,
                Json(SubmitToolCallResultResponse {
                    ok: false,
                    error: Some(message),
                }),
            )
                .into_response()
        }
    }
}

/// Map RuntimeError variants to HTTP status codes for the client API.
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

fn runtime_error_response(err: RuntimeError) -> Response {
    let (status, message) = runtime_error_status(&err);
    (status, Json(serde_json::json!({"error": message}))).into_response()
}

pub async fn stream_session_events(
    State(state): State<ClientHttpState>,
    Extension(_principal): Extension<AuthPrincipal>,
    Path(session_id): Path<String>,
    Query(params): Query<StreamSessionEventsParams>,
) -> Response {
    let root_session_id = session_id.clone();
    let scope_turn_id = params.turn_id.clone();
    let spec = match params.turn_id {
        Some(turn_id) => SessionSubscriptionSpec::Turn {
            root_session_id: session_id,
            turn_id,
        },
        None => SessionSubscriptionSpec::All {
            root_session_id: session_id,
        },
    };

    let mut event_rx = state.runtime.stream(spec, params.sequence_after).await;
    let mut delta_rx = state.runtime.subscribe_token_deltas(&root_session_id).await;

    // Fan in persisted events + transient deltas. The session subscription's
    // sender drops when the scope completes (for Turn scope, on
    // `turn.completed`); we mirror that here by terminating the merged
    // stream as soon as the event side closes, so the SSE response ends
    // and the client's for-await loop exits.
    let (out_tx, out_rx) = tokio::sync::mpsc::channel::<SseEvent>(64);
    let shutdown = state.shutdown.clone();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => return,
                ev = event_rx.recv() => match ev {
                    Some(event) => {
                        let event_type = event.payload_type().to_owned();
                        let data = serde_json::to_string(&event).unwrap_or_default();
                        let sse = SseEvent::default()
                            .id(event.sequence.to_string())
                            .event(event_type)
                            .data(data);
                        if out_tx.send(sse).await.is_err() {
                            return;
                        }
                    }
                    None => return,
                },
                delta = delta_rx.recv() => match delta {
                    Some(delta) => {
                        if let Some(ref scope) = scope_turn_id {
                            if delta.turn_id.as_deref() != Some(scope.as_str()) {
                                continue;
                            }
                        }
                        if out_tx.send(token_delta_to_sse(delta)).await.is_err() {
                            return;
                        }
                    }
                    None => continue,
                },
            }
        }
    });

    let stream = ReceiverStream::new(out_rx).map(Ok::<_, std::convert::Infallible>);

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Wrap a transient [`TokenDelta`] in an envelope mirroring the persisted
/// `Event` shape so SSE consumers can branch on `payload.type` uniformly.
/// `sequence` is omitted because transient deltas are not persisted.
fn token_delta_to_sse(delta: TokenDelta) -> SseEvent {
    let envelope = serde_json::json!({
        "aggregate_type": "session",
        "aggregate_id": delta.session_id,
        "event_type": "llm.token.delta",
        "occurred_at": chrono::Utc::now(),
        "payload": {
            "type": "llm.token.delta",
            "call_id": delta.call_id,
            "attempt": delta.attempt,
            "seq": delta.seq,
            "agent_id": delta.agent_id,
            "turn_id": delta.turn_id,
            "text": delta.text,
            "finish_reason": delta.finish_reason,
        },
    });
    SseEvent::default()
        .event("llm.token.delta")
        .data(envelope.to_string())
}
