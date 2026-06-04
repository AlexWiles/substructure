use std::collections::HashMap;

use axum::extract::rejection::JsonRejection;
use axum::extract::{Extension, Path, State};
use axum::http::StatusCode;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use crate::identity::ClientIdentity;
use crate::session::command::SessionError;
use crate::session::decision::ClientPayload;
use crate::session::subscriptions::SessionSubscriptionSpec;
use crate::transport::auth::AuthPrincipal;
use crate::{
    Caller, RuntimeError, SubmitClientPayload, SubmitToolCallResult, SubmitToolCallResultInput,
};

use super::super::routes::runtime_error_response;
use super::super::ClientHttpState;
use super::translator::run_ag_ui_translation;
use super::types::{AgUiInput, RunAgentInput};

/// `POST /api/client/ag-ui/agents/{agent_id}/run` — native AG-UI endpoint.
///
/// Accepts a [`RunAgentInput`], starts a turn for the path-selected agent, and
/// streams the turn's events back as the AG-UI protocol SSE sequence. Reuses
/// the client-token auth of this router; identity is stamped from the
/// authenticated principal, never from the request body.
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
