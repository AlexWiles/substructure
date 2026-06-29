use axum::extract::{Extension, Path, Query, State};
use axum::http::StatusCode;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::owner::SessionOwner;
use crate::session::command::SessionError;
use crate::session::decision::ClientPayload;
use crate::session::events::MessageTree;
use crate::session::subscriptions::{SessionSubscriptionSpec, SubscriptionScope};
use crate::transport::ag_ui::snapshot::snapshot_events;
use crate::transport::ag_ui::translator::run_ag_ui_translation;
use crate::transport::ag_ui::types::RunAgentInput;
use crate::transport::session_sse::merge_session_stream;
use crate::{
    Caller, InterruptSessionInput, ResumeInterruptInput, RuntimeError, SubmitClientPayload,
    SubmitToolCallResult, SubmitToolCallResultInput,
};

use super::types::{
    InterruptSessionRequest, InterruptSessionResponse, ResumeInterruptRequest,
    ResumeInterruptResponse, StreamSessionEventsParams, SubmitClientPayloadRequest,
    SubmitClientPayloadResponse, SubmitToolCallResultRequest, SubmitToolCallResultResponse,
};
use super::ClientHttpState;

pub async fn submit_client_payload(
    State(state): State<ClientHttpState>,
    Extension(caller): Extension<Caller>,
    Extension(owner): Extension<SessionOwner>,
    Json(req): Json<SubmitClientPayloadRequest>,
) -> Response {
    let session_id = req.session_id.unwrap_or_else(|| Uuid::now_v7().to_string());

    let result = state
        .runtime
        .submit_client_payload(SubmitClientPayload {
            session_id,
            caller,
            owner,
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
    Extension(caller): Extension<Caller>,
    Path(session_id): Path<String>,
    Json(req): Json<SubmitToolCallResultRequest>,
) -> Response {
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

pub async fn interrupt_session(
    State(state): State<ClientHttpState>,
    Extension(caller): Extension<Caller>,
    Path(session_id): Path<String>,
    Json(req): Json<InterruptSessionRequest>,
) -> Response {
    let interrupt_id = req
        .interrupt_id
        .unwrap_or_else(|| Uuid::now_v7().to_string());

    let result = state
        .runtime
        .interrupt_session(InterruptSessionInput {
            session_id,
            interrupt_id: interrupt_id.clone(),
            reason: req.reason.unwrap_or_default(),
            payload: req.payload.unwrap_or(serde_json::Value::Null),
            caller,
            span: crate::span::SpanContext::root().child("client_interrupt"),
        })
        .await;

    match result {
        Ok(()) => Json(InterruptSessionResponse {
            ok: true,
            interrupt_id,
        })
        .into_response(),
        Err(e) => runtime_error_response(e),
    }
}

pub async fn resume_interrupt(
    State(state): State<ClientHttpState>,
    Extension(caller): Extension<Caller>,
    Path(session_id): Path<String>,
    Json(req): Json<ResumeInterruptRequest>,
) -> Response {
    let result = state
        .runtime
        .resume_interrupt(ResumeInterruptInput {
            session_id,
            interrupt_id: req.interrupt_id,
            payload: req.payload.unwrap_or(serde_json::Value::Null),
            caller,
            span: crate::span::SpanContext::root().child("client_resume_interrupt"),
        })
        .await;

    match result {
        Ok(()) => Json(ResumeInterruptResponse { ok: true }).into_response(),
        Err(e) => runtime_error_response(e),
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

pub(crate) fn runtime_error_response(err: RuntimeError) -> Response {
    let (status, message) = runtime_error_status(&err);
    (status, Json(serde_json::json!({"error": message}))).into_response()
}

pub async fn stream_session_events(
    State(state): State<ClientHttpState>,
    Extension(caller): Extension<Caller>,
    Path(session_id): Path<String>,
    Query(params): Query<StreamSessionEventsParams>,
) -> Response {
    let root_session_id = session_id.clone();
    let scope_turn_id = params.turn_id.clone();

    let delta_rx = state
        .runtime
        .subscribe_token_deltas(&caller, &root_session_id)
        .await;

    let spec = SessionSubscriptionSpec {
        root_session_id: session_id,
        caller,
        scope: match params.turn_id {
            Some(turn_id) => SubscriptionScope::Turn { turn_id },
            None => SubscriptionScope::All,
        },
    };

    let event_rx = match state.runtime.stream(spec, params.sequence_after).await {
        Ok(rx) => rx,
        Err(e) => return runtime_error_response(e),
    };

    let out_rx = merge_session_stream(event_rx, delta_rx, scope_turn_id, state.shutdown.clone());
    let stream = ReceiverStream::new(out_rx).map(Ok::<_, std::convert::Infallible>);

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

pub async fn ag_ui_run(
    State(state): State<ClientHttpState>,
    Extension(caller): Extension<Caller>,
    Extension(owner): Extension<SessionOwner>,
    Path(agent_id): Path<String>,
    Json(input): Json<RunAgentInput>,
) -> Response {
    let session_id = input.thread_id.clone();

    // Pure passthrough: resume entries answer a pending interrupt; everything
    // else forwards the client's full transcript and the aggregate classifies
    // it against effect state (new turn vs. client tool results).
    if input.resume.is_empty() && input.to_messages().is_empty() {
        let body = serde_json::json!({"error": "no messages or resume in RunAgentInput"});
        return (StatusCode::BAD_REQUEST, Json(body)).into_response();
    }

    let spec = SessionSubscriptionSpec {
        scope: SubscriptionScope::All,
        caller: caller.clone(),
        root_session_id: session_id.clone(),
    };
    let event_rx = match state.runtime.stream(spec, None).await {
        Ok(rx) => rx,
        Err(e) => return runtime_error_response(e),
    };
    let delta_rx = state
        .runtime
        .subscribe_token_deltas(&caller, &session_id)
        .await;

    let submit = if !input.resume.is_empty() {
        // AG-UI resume entries answer the session's pending interrupt. The
        // worker receives `{status, payload}` as the `interrupt.resumed`
        // trigger payload. Stale interrupt ids no-op in the core, which
        // keeps resumes idempotent as the spec requires.
        let mut outcome: Result<(), RuntimeError> = Ok(());
        for entry in input.resume.clone() {
            let payload = serde_json::json!({
                "status": entry.status.as_str(),
                "payload": entry.payload.unwrap_or(serde_json::Value::Null),
            });
            let r = state
                .runtime
                .resume_interrupt(ResumeInterruptInput {
                    session_id: session_id.clone(),
                    interrupt_id: entry.interrupt_id,
                    payload,
                    caller: caller.clone(),
                    span: crate::span::SpanContext::root().child("ag_ui_resume"),
                })
                .await;
            if let Err(e) = r {
                outcome = Err(e);
                break;
            }
        }
        outcome
    } else {
        state
            .runtime
            .submit_client_payload(SubmitClientPayload {
                session_id: session_id.clone(),
                caller,
                owner,
                agent_id,
                // Merge the client's full view so edits/branches reconcile into
                // the tree; trailing client tool results resolve their effects.
                payload: ClientPayload::Messages {
                    messages: input.to_messages(),
                    stream: true,
                },
                turn_id: Some(input.run_id.clone()),
            })
            .await
            .map(|_| ())
    };
    if let Err(e) = submit {
        return runtime_error_response(e);
    }

    let out_rx = run_ag_ui_translation(
        event_rx,
        delta_rx,
        input.thread_id,
        input.run_id,
        state.shutdown.clone(),
    );
    let stream = ReceiverStream::new(out_rx).map(Ok::<_, std::convert::Infallible>);
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

pub async fn ag_ui_connect(
    State(state): State<ClientHttpState>,
    Extension(caller): Extension<Caller>,
    Path(_agent_id): Path<String>,
    Json(input): Json<RunAgentInput>,
) -> Response {
    // Authorizes the read and carries the active interrupt (its payload lives
    // only in the event log, not the materialized state).
    let events = match state
        .runtime
        .read_session_events(&caller, &input.thread_id, None, None)
        .await
    {
        Ok(events) => events,
        Err(e) => return runtime_error_response(e),
    };

    // No events ⇒ the session isn't created yet (a fresh thread); its snapshot
    // is an empty tree. Otherwise load it and read the materialized tree.
    let tree = if events.is_empty() {
        MessageTree::default()
    } else {
        match state
            .runtime
            .get_session(caller.tenant_id(), &input.thread_id)
            .await
        {
            Ok((_, session)) => session.message_tree(),
            Err(e) => return runtime_error_response(e),
        }
    };

    let frames = snapshot_events(input.thread_id, input.run_id, &tree, &events)
        .into_iter()
        .map(|ev| Ok::<_, std::convert::Infallible>(ev.to_sse()))
        .collect::<Vec<_>>();

    Sse::new(futures_util::stream::iter(frames))
        .keep_alive(KeepAlive::default())
        .into_response()
}
