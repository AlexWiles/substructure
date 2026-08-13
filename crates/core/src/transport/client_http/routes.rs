use axum::extract::{Extension, Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::protocol::SessionOwner;
use crate::runtime::blob::{BlobError, BlobRef};
use crate::session::command::SessionError;
use crate::session::subscriptions::{SessionSubscriptionSpec, SubscriptionScope};
use crate::transport::http::{runtime_error_response, Body};
use crate::transport::session_sse::{merge_session_stream, resume_cursor};
use crate::{Caller, HandleClientInput, InterruptSessionInput, RuntimeError};

use super::types::{
    session_response, ClientInputRequest, ClientInputResponse, InterruptSessionRequest,
    InterruptSessionResponse, StreamSessionEventsParams,
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
    Body(req): Body<ClientInputRequest>,
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
                queued: output.queued,
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
    Body(req): Body<InterruptSessionRequest>,
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

/// The session resource: head-resolved status, open interrupts, and the full
/// message tree.
/// Serves a stored blob to its own tenant. `uri` is the `blob://` ref as it
/// appears in message content; a ref from another tenant reads as absent, so
/// the endpoint never confirms what exists elsewhere.
pub async fn get_blob(
    State(state): State<ClientHttpState>,
    Extension(caller): Extension<Caller>,
    Query(params): Query<super::types::GetBlobParams>,
) -> Response {
    let Some(blobs) = &state.blobs else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let Some(r) = BlobRef::parse(&params.uri) else {
        return (StatusCode::BAD_REQUEST, "not a blob uri").into_response();
    };
    if r.tenant_id != caller.tenant_id() {
        return StatusCode::NOT_FOUND.into_response();
    }
    match blobs.get(&r).await {
        Ok(bytes) => {
            let mime = r
                .mime
                .parse()
                .unwrap_or(axum::http::HeaderValue::from_static(
                    "application/octet-stream",
                ));
            let mut headers = HeaderMap::new();
            headers.insert(axum::http::header::CONTENT_TYPE, mime);
            // An id names one immutable object; cache hard, never shared.
            headers.insert(
                axum::http::header::CACHE_CONTROL,
                axum::http::HeaderValue::from_static("private, max-age=31536000, immutable"),
            );
            (headers, bytes).into_response()
        }
        Err(BlobError::NotFound) => StatusCode::NOT_FOUND.into_response(),
        Err(e) => {
            tracing::warn!(error = %e, "blob read failed");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

pub async fn get_session(
    State(state): State<ClientHttpState>,
    Extension(caller): Extension<Caller>,
    Path(session_id): Path<String>,
) -> Response {
    // Authorizes the read; an uncreated session has no events ⇒ 404.
    let events = match state
        .runtime
        .read_session_events(&caller, &session_id, None, Some(1))
        .await
    {
        Ok(events) => events,
        Err(e) => return runtime_error_response(e),
    };
    if events.is_empty() {
        return runtime_error_response(RuntimeError::Session(SessionError::SessionNotCreated));
    }

    let session = match state
        .runtime
        .get_session(caller.tenant_id(), &session_id)
        .await
    {
        Ok(session) => session.state,
        Err(e) => return runtime_error_response(e),
    };

    Json(session_response(session_id, &session)).into_response()
}

pub async fn stream_session_events(
    State(state): State<ClientHttpState>,
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
        Err(e) => return runtime_error_response(e),
    };

    let out_rx = merge_session_stream(event_rx, delta_rx, scope_turn_id, state.shutdown.clone());
    let stream = ReceiverStream::new(out_rx).map(Ok::<_, std::convert::Infallible>);

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}
