use std::sync::Arc;

use axum::extract::{Extension, Path, State};
use axum::middleware;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use futures_util::stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::protocol::{
    ClientInput, ClientMessages, ClientPayload, InterruptResumption, SessionOwner,
};
use crate::session::subscriptions::{SessionSubscriptionSpec, SubscriptionScope};
use crate::transport::ag_ui::events::AgUiEvent;
use crate::transport::ag_ui::snapshot::snapshot_events;
use crate::transport::ag_ui::translator::{run_ag_ui_translation, AgUiTranslator};
use crate::transport::ag_ui::types::{ConnectInput, RunAgentInput};
use crate::transport::auth::AuthResolver;
use crate::transport::channel::{Channel, ChannelContext, ChannelKind};
use crate::transport::http::{client_auth_middleware, client_cors, runtime_error_response};
use crate::{Caller, HandleClientInput, SubmitClientPayload};

pub struct AgUiChannel {
    auth: Arc<dyn AuthResolver>,
}

impl AgUiChannel {
    pub fn new(auth: Arc<dyn AuthResolver>) -> Self {
        Self { auth }
    }
}

impl Channel for AgUiChannel {
    fn kind(&self) -> ChannelKind {
        ChannelKind::AG_UI
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

/// AG-UI reports a rejected run on the stream, not as an HTTP status: "agents
/// receiving a non-conforming input must emit `RunError`". Only input the
/// protocol defines as non-conforming comes here — auth and stream-setup
/// failures stay HTTP errors, since no run ever began for them.
fn run_error(thread_id: String, run_id: String, message: String) -> Response {
    let mut t = AgUiTranslator::new(thread_id, run_id);
    let mut events: Vec<AgUiEvent> = t.start();
    events.extend(t.finalize_error(message));
    let stream = futures_util::stream::iter(
        events
            .into_iter()
            .map(|e| Ok::<_, std::convert::Infallible>(e.to_sse())),
    );
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

async fn run(
    State(ctx): State<ChannelContext>,
    Extension(caller): Extension<Caller>,
    Extension(owner): Extension<SessionOwner>,
    Path(agent_id): Path<String>,
    Json(input): Json<RunAgentInput>,
) -> Response {
    let session_id = input.thread_id.clone();
    let (thread_id, run_id) = (input.thread_id.clone(), input.run_id.clone());

    // Passthrough; the core classifies new turn vs. client tool results.
    if input.resume.is_empty() && input.to_messages().is_empty() {
        return run_error(
            thread_id,
            run_id,
            "no messages or resume in RunAgentInput".to_string(),
        );
    }

    // "If a thread has unresolved interrupts, any RunAgentInput on that thread
    // must include a resume addressing them", and "a single resume array must
    // address every open interrupt from the interrupted run. Partial resumes
    // are not supported."
    let open: Vec<String> = match ctx.get_session(caller.tenant_id(), &session_id).await {
        Ok(session) => session
            .state
            .at_head()
            .interrupts_for()
            .into_iter()
            .map(|i| i.interrupt_id.clone())
            .collect(),
        // No session yet: nothing can be open on it.
        Err(_) => Vec::new(),
    };
    let addressed: std::collections::HashSet<&str> = input
        .resume
        .iter()
        .map(|r| r.interrupt_id.as_str())
        .collect();
    let unaddressed: Vec<&str> = open
        .iter()
        .map(String::as_str)
        .filter(|id| !addressed.contains(id))
        .collect();
    if !unaddressed.is_empty() {
        return run_error(
            thread_id,
            run_id,
            format!(
                "resume must address every open interrupt; unaddressed: {}",
                unaddressed.join(", ")
            ),
        );
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
    // Resumes apply first, then the messages view — while an interrupt is open the protocol
    // requires both to arrive as one input, so a user who types instead of answering the
    // prompt lands here.
    let resuming = !input.resume.is_empty();
    for entry in input.resume.clone() {
        let payload = serde_json::to_value(crate::protocol::InterruptResolution {
            status: entry.status,
            payload: entry.payload.unwrap_or(serde_json::Value::Null),
            responder: Some(crate::protocol::InterruptResponder {
                channel: ChannelKind::AG_UI.to_string(),
                user: None,
                // Nobody picked an option. The client sends its own payload.
                label: None,
                style: None,
            }),
        })
        .unwrap_or_default();
        let r = ctx
            .handle_client_input(HandleClientInput {
                session_id: session_id.clone(),
                caller: caller.clone(),
                owner: owner.clone(),
                agent: None,
                worker: None,
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
            .submit_client_payload(SubmitClientPayload {
                session_id: session_id.clone(),
                caller,
                owner,
                agent_id,
                agent: None,
                worker: None,
                payload: ClientPayload::Messages(ClientMessages {
                    // Full client view so edits/branches reconcile into the tree.
                    messages: input.to_messages(),
                    stream: true,
                    // Client-declared tools/context/state, forwarded to the worker.
                    client: input.client_context(),
                }),
                turn_id: Some(input.run_id.clone()),
                // A resume did not end the turn it unpaused, so a view arriving
                // with one continues that turn rather than opening a second.
                continue_turn: resuming,
                // A full view cannot be frozen behind a turn: it settles
                // pending client calls at submit time.
                queue: false,
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
    let stream = ReceiverStream::new(out_rx).map(|e| Ok::<_, std::convert::Infallible>(e.to_sse()));
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
