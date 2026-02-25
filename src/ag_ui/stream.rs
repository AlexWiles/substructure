use chrono::Utc;
use ractor::Actor;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::Stream;
use uuid::Uuid;

use crate::domain::event::{Role, SessionAuth, SpanContext};
use crate::domain::session::{CommandPayload, SessionCommand};
use crate::runtime::Runtime;
use crate::runtime::{RuntimeError, SessionHandle};

use super::observer::{AgUiObserverActor, ObserverArgs};
use super::types::{AgUiEvent, Message, RunAgentInput};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum AgUiError {
    #[error("runtime error: {0}")]
    Runtime(#[from] RuntimeError),
    #[error("no user message found in input")]
    NoUserMessage,
    #[error("no resume info provided")]
    NoResumeInfo,
}

// ---------------------------------------------------------------------------
// run_session — create a new session and return an AG-UI event stream
// ---------------------------------------------------------------------------

/// Create a new session from AG-UI input and return a stream of AG-UI events.
pub async fn run_session(
    runtime: &Runtime,
    agent_name: &str,
    auth: SessionAuth,
    input: RunAgentInput,
) -> Result<impl Stream<Item = AgUiEvent> + Send, AgUiError> {
    // Extract the last user message
    let user_content = input
        .messages
        .iter()
        .rev()
        .find_map(|m| match m {
            Message::User { content, .. } => {
                let text = match content {
                    super::types::MessageContent::Text(t) => t.clone(),
                    super::types::MessageContent::Parts(parts) => parts
                        .iter()
                        .filter_map(|p| match p {
                            super::types::InputContent::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join(""),
                };
                Some(text)
            }
            _ => None,
        })
        .ok_or(AgUiError::NoUserMessage)?;

    let session = runtime.create_session_for(agent_name, auth).await?;
    let thread_id = input
        .thread_id
        .unwrap_or_else(|| session.session_id.to_string());
    let run_id = input.run_id.unwrap_or_else(|| Uuid::new_v4().to_string());

    let (tx, rx) = mpsc::channel(256);

    // Spawn observer BEFORE sending the command so we don't miss events
    let _observer = Actor::spawn(
        None,
        AgUiObserverActor,
        ObserverArgs {
            session_id: session.session_id,
            thread_id: thread_id.clone(),
            run_id: run_id.clone(),
            event_tx: tx,
            init_events: vec![AgUiEvent::RunStarted { thread_id, run_id }],
            skip_until: 0,
        },
    )
    .await
    .expect("failed to spawn AG-UI observer");

    // Send the user message
    session
        .send_command(SessionCommand {
            span: SpanContext::root(),
            occurred_at: Utc::now(),
            payload: CommandPayload::SendUserMessage {
                content: user_content,
                stream: true,
            },
        })
        .await?;

    Ok(ReceiverStream::new(rx))
}

// ---------------------------------------------------------------------------
// resume_run — resume an interrupted session
// ---------------------------------------------------------------------------

/// Resume a previously interrupted session and return a stream of AG-UI events.
pub async fn resume_run(
    session: &SessionHandle,
    input: RunAgentInput,
) -> Result<impl Stream<Item = AgUiEvent> + Send, AgUiError> {
    let resume = input.resume.ok_or(AgUiError::NoResumeInfo)?;
    let thread_id = input
        .thread_id
        .unwrap_or_else(|| session.session_id.to_string());
    let run_id = input.run_id.unwrap_or_else(|| Uuid::new_v4().to_string());

    let (tx, rx) = mpsc::channel(256);

    // Spawn observer BEFORE sending the command so we don't miss events
    let _observer = Actor::spawn(
        None,
        AgUiObserverActor,
        ObserverArgs {
            session_id: session.session_id,
            thread_id: thread_id.clone(),
            run_id: run_id.clone(),
            event_tx: tx,
            init_events: vec![AgUiEvent::RunStarted { thread_id, run_id }],
            skip_until: 0,
        },
    )
    .await
    .expect("failed to spawn AG-UI observer");

    // Send the ResumeInterrupt command
    session
        .send_command(SessionCommand {
            span: SpanContext::root(),
            occurred_at: Utc::now(),
            payload: CommandPayload::ResumeInterrupt {
                interrupt_id: resume.interrupt_id,
                payload: resume.payload,
            },
        })
        .await?;

    Ok(ReceiverStream::new(rx))
}

// ---------------------------------------------------------------------------
// observe_session — observe an existing session
// ---------------------------------------------------------------------------

/// Observe an existing session, yielding a snapshot followed by live events.
pub async fn observe_session(
    _runtime: &Runtime,
    session: &SessionHandle,
    run_id: String,
) -> Result<impl Stream<Item = AgUiEvent> + Send, AgUiError> {
    let state = session.get_state().await;
    let thread_id = session.session_id.to_string();

    // Convert domain messages to AG-UI messages
    let messages = state
        .messages
        .iter()
        .map(domain_message_to_ag_ui)
        .collect::<Vec<_>>();

    let last_applied = state.last_applied.unwrap_or(0);

    let (tx, rx) = mpsc::channel(256);

    let _observer = Actor::spawn(
        None,
        AgUiObserverActor,
        ObserverArgs {
            session_id: session.session_id,
            thread_id: thread_id.clone(),
            run_id: run_id.clone(),
            event_tx: tx,
            init_events: vec![AgUiEvent::MessagesSnapshot { messages }],
            skip_until: last_applied,
        },
    )
    .await
    .expect("failed to spawn AG-UI observer");

    Ok(ReceiverStream::new(rx))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a domain `Message` to an AG-UI `Message`.
fn domain_message_to_ag_ui(msg: &crate::domain::event::Message) -> Message {
    match msg.role {
        Role::User => Message::User {
            id: None,
            content: super::types::MessageContent::Text(msg.content.clone().unwrap_or_default()),
        },
        Role::Assistant => Message::Assistant {
            id: None,
            content: msg.content.clone(),
            tool_calls: msg
                .tool_calls
                .iter()
                .map(|tc| super::types::ToolCallInfo {
                    id: tc.id.clone(),
                    function: super::types::FunctionCall {
                        name: tc.name.clone(),
                        arguments: tc.arguments.clone(),
                    },
                })
                .collect(),
        },
        Role::System => Message::System {
            id: None,
            content: msg.content.clone().unwrap_or_default(),
        },
        Role::Tool => Message::Tool {
            id: None,
            tool_call_id: msg.tool_call_id.clone().unwrap_or_default(),
            content: msg.content.clone().unwrap_or_default(),
        },
    }
}
