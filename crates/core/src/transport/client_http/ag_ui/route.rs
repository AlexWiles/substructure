use std::collections::HashMap;

use axum::extract::{Extension, Path, State};
use axum::http::StatusCode;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use crate::identity::ClientIdentity;
use crate::session::decision::ClientPayload;
use crate::session::subscriptions::SessionSubscriptionSpec;
use crate::transport::auth::AuthPrincipal;
use crate::{Caller, SubmitClientPayload, SubmitToolCallResult, SubmitToolCallResultInput};

use super::super::routes::runtime_error_response;
use super::super::ClientHttpState;
use super::translator::{client_tool_signal_for, run_ag_ui_translation};
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
    Json(input): Json<RunAgentInput>,
) -> Response {
    let Some(user_id) = principal.subject.clone() else {
        let body = serde_json::json!({"error": "client subject is required"});
        return (StatusCode::FORBIDDEN, Json(body)).into_response();
    };
    // The trailing message decides the action: a new user turn, or a frontend
    // tool result resuming the turn suspended on that call.
    let Some(action) = input.classify() else {
        let body = serde_json::json!({"error": "no user message or tool result in RunAgentInput"});
        return (StatusCode::BAD_REQUEST, Json(body)).into_response();
    };

    // AG-UI threadId maps onto the session id. (The runId need not equal the
    // engine turn id: a resumed turn keeps the turn it suspended on.)
    let session_id = input.thread_id.clone();
    let tenant_id = principal.tenant_id.clone();

    // Subscribe live to the whole session FIRST (live-only), THEN act — so the
    // reducer sees every event the submit/resume produces. It follows the
    // stream and closes the run on `turn.completed` or a client-tool yield.
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
                    // Bind the new turn to the AG-UI runId.
                    turn_id: Some(input.run_id.clone()),
                })
                .await
                .map(|_| ())
        }
        AgUiInput::ToolResult {
            tool_call_id,
            content,
        } => {
            state
                .runtime
                .submit_tool_call_result(SubmitToolCallResultInput {
                    session_id: session_id.clone(),
                    tenant_id: tenant_id.clone(),
                    tool_call_id,
                    attempt: 0,
                    result: SubmitToolCallResult::Result { result: content },
                    caller,
                    span: crate::span::SpanContext::root().child("ag_ui_tool_result"),
                })
                .await
        }
    };
    if let Err(e) = submit {
        // The SSE body has not been opened yet, so a plain HTTP error is fine.
        return runtime_error_response(e);
    }

    // Pick the client dialect from the client's hint (spec-pure by default).
    let client_tool_signal = client_tool_signal_for(
        input
            .forwarded_props
            .as_ref()
            .and_then(|p| p.get("aguiClient"))
            .and_then(|v| v.as_str()),
    );

    let out_rx = run_ag_ui_translation(
        event_rx,
        delta_rx,
        input.thread_id,
        input.run_id,
        client_tool_signal,
        state.shutdown.clone(),
    );
    let stream = ReceiverStream::new(out_rx).map(Ok::<_, std::convert::Infallible>);
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}
