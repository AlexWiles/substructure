use std::sync::Arc;

use axum::extract::{Extension, Path, State};
use axum::http::StatusCode;
use axum::middleware;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use futures_util::stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::protocol::{ClientInput, InterruptResumption, SessionOwner};
use crate::session::subscriptions::{SessionSubscriptionSpec, SubscriptionScope};
use crate::transport::ag_ui::snapshot::snapshot_events;
use crate::transport::ag_ui::translator::run_ag_ui_translation;
use crate::transport::ag_ui::types::{ConnectInput, RunAgentInput};
use crate::transport::auth::AuthResolver;
use crate::transport::channel::{Channel, ChannelContext};
use crate::transport::http::{client_auth_middleware, client_cors, runtime_error_response};
use crate::{Caller, HandleClientInput};

pub struct AgUiChannel {
    auth: Arc<dyn AuthResolver>,
}

impl AgUiChannel {
    pub fn new(auth: Arc<dyn AuthResolver>) -> Self {
        Self { auth }
    }
}

impl Channel for AgUiChannel {
    fn kind(&self) -> &'static str {
        "ag-ui"
    }

    fn router(&self, ctx: ChannelContext) -> Option<Router> {
        Some(
            Router::new()
                .route("/agents/{agent_id}/run", post(run))
                .route("/agents/{agent_id}/connect", post(connect))
                .route_layer(middleware::from_fn_with_state(
                    self.auth.clone(),
                    client_auth_middleware,
                ))
                // Outermost so CORS preflight (OPTIONS) is answered before auth runs.
                .layer(client_cors())
                .with_state(ctx),
        )
    }
}

async fn run(
    State(ctx): State<ChannelContext>,
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
        session_id: session_id.clone(),
    };
    let event_rx = match ctx.stream(spec, None).await {
        Ok(rx) => rx,
        Err(e) => return runtime_error_response(e),
    };
    let delta_rx = ctx.subscribe_token_deltas(&caller, &session_id).await;

    // AG-UI's input is already classified (a `resume` list vs. messages), so the inputs are
    // built directly with their addressing rather than parsed from untrusted tagged JSON.
    // Resumes apply first, then the messages view — steerAway sends both.
    for entry in input.resume.clone() {
        let payload = serde_json::json!({
            "status": entry.status.as_str(),
            "payload": entry.payload.unwrap_or(serde_json::Value::Null),
        });
        let r = ctx
            .handle_client_input(HandleClientInput {
                session_id: session_id.clone(),
                caller: caller.clone(),
                owner: owner.clone(),
                input: ClientInput::InterruptResume {
                    resumption: InterruptResumption {
                        interrupt_id: entry.interrupt_id,
                        payload,
                    },
                },
                span: crate::span::SpanContext::root().child("ag_ui_resume"),
            })
            .await;
        if let Err(e) = r {
            return runtime_error_response(e);
        }
    }
    if !input.to_messages().is_empty() {
        let submit = ctx
            .handle_client_input(HandleClientInput {
                session_id: session_id.clone(),
                caller,
                owner,
                input: ClientInput::Messages {
                    agent_id,
                    turn_id: Some(input.run_id.clone()),
                    // Full client view so edits/branches reconcile into the tree.
                    messages: input.to_messages(),
                    stream: true,
                    // Client-declared tools/context/state, forwarded to the worker.
                    client: input.client_context(),
                },
                span: crate::span::SpanContext::root().child("ag_ui_run"),
            })
            .await;
        if let Err(e) = submit {
            return runtime_error_response(e);
        }
    }

    let out_rx = run_ag_ui_translation(
        event_rx,
        delta_rx,
        input.thread_id,
        input.run_id,
        ctx.shutdown.clone(),
    );
    let stream = ReceiverStream::new(out_rx).map(Ok::<_, std::convert::Infallible>);
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

async fn connect(
    State(ctx): State<ChannelContext>,
    Extension(caller): Extension<Caller>,
    Path(_agent_id): Path<String>,
    Json(input): Json<ConnectInput>,
) -> Response {
    // Authorizes the read; no events ⇒ session not yet created, empty snapshot.
    let events = match ctx
        .read_session_events(&caller, &input.thread_id, None, Some(1))
        .await
    {
        Ok(events) => events,
        Err(e) => return runtime_error_response(e),
    };

    let session = if events.is_empty() {
        None
    } else {
        match ctx.get_session(caller.tenant_id(), &input.thread_id).await {
            Ok(session) => Some(session.state),
            Err(e) => return runtime_error_response(e),
        }
    };

    let run_id = input.run_id.unwrap_or_else(|| Uuid::now_v7().to_string());
    let frames = snapshot_events(input.thread_id, run_id, session.as_ref())
        .into_iter()
        .map(|ev| Ok::<_, std::convert::Infallible>(ev.to_sse()))
        .collect::<Vec<_>>();

    Sse::new(futures_util::stream::iter(frames))
        .keep_alive(KeepAlive::default())
        .into_response()
}
