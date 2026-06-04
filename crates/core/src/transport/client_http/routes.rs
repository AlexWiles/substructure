use std::collections::HashMap;

use axum::extract::rejection::JsonRejection;
use axum::extract::{Extension, Path, Query, State};
use axum::http::StatusCode;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::event_store::EventFilter;
use crate::identity::ClientIdentity;
use crate::session::command::SessionError;
use crate::session::decision::ClientPayload;
use crate::session::subscriptions::SessionSubscriptionSpec;
use crate::transport::ag_ui::snapshot::{message_history, snapshot_events};
use crate::transport::ag_ui::translator::run_ag_ui_translation;
use crate::transport::ag_ui::types::{AgUiInput, RunAgentInput};
use crate::transport::auth::AuthPrincipal;
use crate::transport::session_sse::merge_session_stream;
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
    Extension(principal): Extension<AuthPrincipal>,
    Path(session_id): Path<String>,
    Query(params): Query<StreamSessionEventsParams>,
) -> Response {
    let root_session_id = session_id.clone();
    let scope_turn_id = params.turn_id.clone();
    let spec = match params.turn_id {
        Some(turn_id) => SessionSubscriptionSpec::Turn {
            tenant_id: principal.tenant_id.clone(),
            root_session_id: session_id,
            turn_id,
        },
        None => SessionSubscriptionSpec::All {
            tenant_id: principal.tenant_id.clone(),
            root_session_id: session_id,
        },
    };

    let event_rx = state.runtime.stream(spec, params.sequence_after).await;
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

/// `POST /api/client/ag-ui/agents/{agent_id}/run` — native AG-UI endpoint.
///
/// Accepts a [`RunAgentInput`], starts (or resumes) a turn for the path-selected
/// agent, and streams the turn's events back as the AG-UI SSE sequence. Identity
/// is stamped from the authenticated principal, never from the request body.
pub async fn ag_ui_run(
    State(state): State<ClientHttpState>,
    Extension(principal): Extension<AuthPrincipal>,
    Path(agent_id): Path<String>,
    payload: Result<Json<RunAgentInput>, JsonRejection>,
) -> Response {
    let Some(user_id) = principal.subject.clone() else {
        let body = serde_json::json!({"error": "client subject is required"});
        return (StatusCode::FORBIDDEN, Json(body)).into_response();
    };

    let input = match payload {
        Ok(Json(input)) => input,
        Err(rejection) => {
            let body = serde_json::json!({"error": format!("invalid RunAgentInput: {rejection}")});
            return (StatusCode::BAD_REQUEST, Json(body)).into_response();
        }
    };

    // The trailing message decides the action: a new user turn, or a frontend
    // tool result resuming the turn suspended on that call.
    let Some(action) = input.classify() else {
        let body = serde_json::json!({"error": "no user message or tool result in RunAgentInput"});
        return (StatusCode::BAD_REQUEST, Json(body)).into_response();
    };

    // threadId maps onto the session id; runId need not equal the engine turn id.
    let session_id = input.thread_id.clone();
    let tenant_id = principal.tenant_id.clone();

    // Subscribe live to the whole session BEFORE acting, so the translator sees
    // every event the submit/resume produces.
    let spec = SessionSubscriptionSpec::All {
        tenant_id: tenant_id.clone(),
        root_session_id: session_id.clone(),
    };
    let event_rx = state.runtime.stream(spec, None).await;
    let delta_rx = state
        .runtime
        .subscribe_token_deltas(&tenant_id, &session_id)
        .await;

    let caller = Caller::Frontend {
        tenant_id: tenant_id.clone(),
        user_id: user_id.clone(),
        attrs: principal.attrs.clone(),
    };

    let submit = match action {
        AgUiInput::UserTurn(message) => {
            let identity = ClientIdentity {
                tenant_id: tenant_id.clone(),
                id: Some(user_id),
                metadata: HashMap::new(),
            };
            state
                .runtime
                .submit_client_payload(SubmitClientPayload {
                    session_id: session_id.clone(),
                    tenant_id: tenant_id.clone(),
                    caller,
                    identity,
                    agent_id,
                    payload: ClientPayload::Message {
                        message,
                        stream: true,
                    },
                    turn_id: Some(input.run_id.clone()),
                })
                .await
                .map(|_| ())
        }
        AgUiInput::ToolResults(results) => {
            // A client echoes its whole result history on resume, including
            // already-resolved worker tools and prior submissions. Submit every
            // result; tolerate the benign "not pending" errors those produce
            // (skipping them keeps the genuinely-pending client result moving).
            let mut outcome: Result<(), RuntimeError> = Ok(());
            for item in results {
                let tool_call_id = item.tool_call_id.clone();
                let r = state
                    .runtime
                    .submit_tool_call_result(SubmitToolCallResultInput {
                        session_id: session_id.clone(),
                        tenant_id: tenant_id.clone(),
                        tool_call_id: item.tool_call_id,
                        attempt: 0,
                        result: SubmitToolCallResult::Result {
                            result: item.content,
                        },
                        caller: caller.clone(),
                        span: crate::span::SpanContext::root().child("ag_ui_tool_result"),
                    })
                    .await;
                match r {
                    Ok(()) => {}
                    Err(RuntimeError::Session(
                        SessionError::ToolCallNotPending
                        | SessionError::ToolCallNotFound
                        | SessionError::ToolCallAttemptMismatch
                        | SessionError::ToolCallWrongHandler,
                    )) => {
                        tracing::debug!(
                            %tool_call_id,
                            "ag_ui_run: skipping non-pending tool result on resume"
                        );
                    }
                    Err(e) => {
                        outcome = Err(e);
                        break;
                    }
                }
            }
            outcome
        }
    };
    if let Err(e) = submit {
        // The SSE body has not been opened yet, so a plain HTTP error is fine.
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

/// `GET /api/client/ag-ui/sessions/{session_id}/messages` — the session's history
/// as an AG-UI message list, for hydrating a thread (e.g. as `HttpAgent`'s
/// `initialMessages`). Tenant-scoped to the authenticated client. Read-only.
pub async fn ag_ui_session_messages(
    State(state): State<ClientHttpState>,
    Extension(principal): Extension<AuthPrincipal>,
    Path(session_id): Path<String>,
) -> Response {
    let filter = EventFilter {
        aggregate_id: Some(session_id),
        tenant_id: Some(principal.tenant_id.clone()),
        ..Default::default()
    };
    let events = match state.runtime.get_session_events(&filter).await {
        Ok(events) => events,
        Err(e) => return runtime_error_response(e),
    };
    Json(message_history(&events)).into_response()
}

/// `POST /api/client/ag-ui/agents/{agent_id}/connect` — replays a thread's history
/// as an AG-UI event stream (`RUN_STARTED` → `MESSAGES_SNAPSHOT` → `RUN_FINISHED`),
/// the standard "load thread" response a client expects on (re)connect.
///
/// AG-UI clients restore history by `connect`-ing to a thread, not by seeding
/// `initialMessages`: CopilotKit's `<CopilotChat threadId>` calls the agent's
/// `connect()`, which `POST`s a `RunAgentInput` here and applies the snapshot to
/// its message list. The body's `threadId`/`runId` are the only fields read —
/// `threadId` selects the session, `runId` labels the synthetic snapshot run.
/// Tenant-scoped to the authenticated client. Read-only (no turn is submitted).
pub async fn ag_ui_connect(
    State(state): State<ClientHttpState>,
    Extension(principal): Extension<AuthPrincipal>,
    Path(_agent_id): Path<String>,
    Json(input): Json<RunAgentInput>,
) -> Response {
    let filter = EventFilter {
        aggregate_id: Some(input.thread_id.clone()),
        tenant_id: Some(principal.tenant_id.clone()),
        ..Default::default()
    };
    let events = match state.runtime.get_session_events(&filter).await {
        Ok(events) => events,
        Err(e) => return runtime_error_response(e),
    };
    // A finite, ready-made stream — the snapshot is computed up front, not driven
    // by a live turn, so there is no channel/translator to run.
    let frames = snapshot_events(input.thread_id, input.run_id, &events)
        .into_iter()
        .map(|ev| Ok::<_, std::convert::Infallible>(ev.to_sse()))
        .collect::<Vec<_>>();
    Sse::new(futures_util::stream::iter(frames))
        .keep_alive(KeepAlive::default())
        .into_response()
}
