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
use crate::session::events::MessageTree;
use crate::session::subscriptions::{SessionSubscriptionSpec, SubscriptionScope};
use crate::session::wire::WireClientInput;
use crate::transport::ag_ui::snapshot::snapshot_events;
use crate::transport::ag_ui::translator::run_ag_ui_translation;
use crate::transport::ag_ui::types::RunAgentInput;
use crate::transport::session_sse::merge_session_stream;
use crate::{Caller, HandleClientInput, InterruptSessionInput, RuntimeError};

use super::types::{
    ClientInputRequest, ClientInputResponse, InterruptSessionRequest, InterruptSessionResponse,
    StreamSessionEventsParams,
};
use super::ClientHttpState;

/// The one client input endpoint. `input` is the tagged union of everything a client can
/// send, carrying its own addressing; a submit missing `agent_id` fails to deserialize (a
/// 422 from the JSON extractor). `handle_client_input` routes it; always 202 on success
/// with the resolved ids.
pub async fn client_input(
    State(state): State<ClientHttpState>,
    Extension(caller): Extension<Caller>,
    Extension(owner): Extension<SessionOwner>,
    Json(req): Json<ClientInputRequest>,
) -> Response {
    let session_id = req.session_id.unwrap_or_else(|| Uuid::now_v7().to_string());

    let result = state
        .runtime
        .handle_client_input(HandleClientInput {
            session_id,
            caller,
            owner,
            input: req.input,
            span: crate::span::SpanContext::root().child("client_input"),
        })
        .await;

    match result {
        Ok(output) => (
            StatusCode::ACCEPTED,
            Json(ClientInputResponse {
                session_id: output.session_id,
                turn_id: output.turn_id,
            }),
        )
            .into_response(),
        Err(e) => runtime_error_response(e),
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

/// Map RuntimeError variants to HTTP status codes for the client API.
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
            | SessionError::TurnAlreadyCompleted { .. }
            | SessionError::NoActiveTurn,
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

    // Passthrough; the core classifies new turn vs. client tool results.
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

    // AG-UI's input is already classified (a `resume` list vs. messages), so the inputs are
    // built directly with their addressing rather than parsed from untrusted tagged JSON.
    let submit = if !input.resume.is_empty() {
        // Stale interrupt ids no-op, keeping resumes idempotent.
        let mut outcome: Result<(), RuntimeError> = Ok(());
        for entry in input.resume.clone() {
            let payload = serde_json::json!({
                "status": entry.status.as_str(),
                "payload": entry.payload.unwrap_or(serde_json::Value::Null),
            });
            let r = state
                .runtime
                .handle_client_input(HandleClientInput {
                    session_id: session_id.clone(),
                    caller: caller.clone(),
                    owner: owner.clone(),
                    input: WireClientInput::InterruptResume {
                        interrupt_id: entry.interrupt_id,
                        payload,
                    },
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
            .handle_client_input(HandleClientInput {
                session_id: session_id.clone(),
                caller,
                owner,
                input: WireClientInput::Messages {
                    agent_id,
                    turn_id: Some(input.run_id.clone()),
                    // Full client view so edits/branches reconcile into the tree.
                    messages: input.to_messages(),
                    stream: true,
                },
                span: crate::span::SpanContext::root().child("ag_ui_run"),
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
    // Read events to authorize and recover the active interrupt.
    let events = match state
        .runtime
        .read_session_events(&caller, &input.thread_id, None, None)
        .await
    {
        Ok(events) => events,
        Err(e) => return runtime_error_response(e),
    };

    // No events ⇒ session not yet created; empty tree.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn status_of(err: SessionError) -> StatusCode {
        runtime_error_status(&RuntimeError::Session(err)).0
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
        assert_eq!(status_of(SessionError::NoActiveTurn), StatusCode::CONFLICT);
    }
}
