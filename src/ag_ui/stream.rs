use std::sync::Mutex;

use chrono::Utc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::Stream;
use uuid::Uuid;

use crate::domain::event::{Event, Role, SessionAuth, SpanContext};
use crate::domain::openai;
use crate::domain::session::{AgentState, CommandPayload, IncomingMessage, SessionCommand};
use crate::runtime::{OnEvent, Runtime, RuntimeError, SessionMessage};

use super::translate::{EventTranslator, TranslateOutput};
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
// Callback builder
// ---------------------------------------------------------------------------

struct CallbackState {
    translator: EventTranslator,
    tx: Option<mpsc::Sender<AgUiEvent>>,
    thread_id: String,
    run_id: String,
    skip_until: u64,
}

fn build_ag_ui_callback(
    tx: mpsc::Sender<AgUiEvent>,
    thread_id: String,
    run_id: String,
    skip_until: u64,
) -> OnEvent {
    let state = Mutex::new(CallbackState {
        translator: EventTranslator::new(),
        tx: Some(tx),
        thread_id,
        run_id,
        skip_until,
    });

    Box::new(move |event: &Event| {
        let mut s = state.lock().expect("ag-ui callback lock poisoned");

        if s.tx.is_none() {
            return;
        }

        if s.skip_until > 0 && event.sequence <= s.skip_until {
            return;
        }

        let output = s.translator.translate(event);

        // Now borrow tx for sending
        let tx = s.tx.as_ref().unwrap();

        match output {
            TranslateOutput::Events(events) => {
                for e in events {
                    let _ = tx.try_send(e);
                }
            }
            TranslateOutput::Terminal(events) => {
                for e in events {
                    let _ = tx.try_send(e);
                }
                let _ = tx.try_send(AgUiEvent::RunFinished {
                    thread_id: s.thread_id.clone(),
                    run_id: s.run_id.clone(),
                    outcome: None,
                    interrupt: None,
                });
                s.tx = None;
            }
            TranslateOutput::Interrupt(events, info) => {
                for e in events {
                    let _ = tx.try_send(e);
                }
                let _ = tx.try_send(AgUiEvent::RunFinished {
                    thread_id: s.thread_id.clone(),
                    run_id: s.run_id.clone(),
                    outcome: Some("interrupt".into()),
                    interrupt: Some(info),
                });
                s.tx = None;
            }
        }
    })
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
    let new_messages = extract_new_messages(&input);
    if new_messages.is_empty() {
        return Err(AgUiError::NoUserMessage);
    }

    // Create the session (starts the SessionActor)
    let session = runtime.create_session_for(agent_name, auth.clone()).await?;
    let session_id = session.session_id;
    let thread_id = input.thread_id.unwrap_or_else(|| session_id.to_string());
    let run_id = input.run_id.unwrap_or_else(|| Uuid::new_v4().to_string());

    let (tx, rx) = mpsc::channel(256);
    let _ = tx.try_send(AgUiEvent::RunStarted {
        thread_id: thread_id.clone(),
        run_id: run_id.clone(),
    });

    // Connect with translation callback BEFORE sending the command
    let client = runtime
        .connect(
            session_id,
            auth,
            Some(build_ag_ui_callback(tx, thread_id, run_id, 0)),
        )
        .await?;

    // Set client tools on the session actor
    if !input.tools.is_empty() {
        send_client_tools(session_id, input.tools);
    }

    for message in new_messages {
        client
            .send_command(SessionCommand {
                span: SpanContext::root(),
                occurred_at: Utc::now(),
                payload: CommandPayload::SendMessage {
                    message,
                    stream: true,
                },
            })
            .await?;
    }

    Ok(ReceiverStream::new(rx))
}

// ---------------------------------------------------------------------------
// run_existing_session — send user input to an existing session
// ---------------------------------------------------------------------------

/// Send input to an existing session and return a stream of AG-UI events.
pub async fn run_existing_session(
    runtime: &Runtime,
    session_id: Uuid,
    auth: SessionAuth,
    input: RunAgentInput,
) -> Result<impl Stream<Item = AgUiEvent> + Send, AgUiError> {
    let new_messages = extract_new_messages(&input);
    if new_messages.is_empty() {
        return Err(AgUiError::NoUserMessage);
    }

    let thread_id = input.thread_id.unwrap_or_else(|| session_id.to_string());
    let run_id = input.run_id.unwrap_or_else(|| Uuid::new_v4().to_string());

    let skip_until = load_last_applied(runtime, session_id, &auth);

    let (tx, rx) = mpsc::channel(256);
    let _ = tx.try_send(AgUiEvent::RunStarted {
        thread_id: thread_id.clone(),
        run_id: run_id.clone(),
    });

    let client = runtime
        .connect(
            session_id,
            auth,
            Some(build_ag_ui_callback(tx, thread_id, run_id, skip_until)),
        )
        .await?;

    // Update client tools on the session actor
    if !input.tools.is_empty() {
        send_client_tools(session_id, input.tools);
    }

    for message in new_messages {
        client
            .send_command(SessionCommand {
                span: SpanContext::root(),
                occurred_at: Utc::now(),
                payload: CommandPayload::SendMessage {
                    message,
                    stream: true,
                },
            })
            .await?;
    }

    Ok(ReceiverStream::new(rx))
}

// ---------------------------------------------------------------------------
// resume_run — resume an interrupted session
// ---------------------------------------------------------------------------

/// Resume a previously interrupted session and return a stream of AG-UI events.
pub async fn resume_run(
    runtime: &Runtime,
    session_id: Uuid,
    auth: SessionAuth,
    input: RunAgentInput,
) -> Result<impl Stream<Item = AgUiEvent> + Send, AgUiError> {
    let resume = input.resume.ok_or(AgUiError::NoResumeInfo)?;
    let thread_id = input.thread_id.unwrap_or_else(|| session_id.to_string());
    let run_id = input.run_id.unwrap_or_else(|| Uuid::new_v4().to_string());

    let (tx, rx) = mpsc::channel(256);
    let _ = tx.try_send(AgUiEvent::RunStarted {
        thread_id: thread_id.clone(),
        run_id: run_id.clone(),
    });

    let client = runtime
        .connect(
            session_id,
            auth,
            Some(build_ag_ui_callback(tx, thread_id, run_id, 0)),
        )
        .await?;

    // Update client tools on the session actor (may change between runs)
    if !input.tools.is_empty() {
        send_client_tools(session_id, input.tools);
    }

    client
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
    runtime: &Runtime,
    session_id: Uuid,
    auth: SessionAuth,
    run_id: String,
) -> Result<impl Stream<Item = AgUiEvent> + Send, AgUiError> {
    let thread_id = session_id.to_string();

    let (state, last_applied) = load_state(runtime, session_id, &auth);

    let messages = state
        .messages
        .iter()
        .map(domain_message_to_ag_ui)
        .collect::<Vec<_>>();

    let (tx, rx) = mpsc::channel(256);
    let _ = tx.try_send(AgUiEvent::RunStarted {
        thread_id: thread_id.clone(),
        run_id: run_id.clone(),
    });
    let _ = tx.try_send(AgUiEvent::MessagesSnapshot { messages });

    let _client = runtime
        .connect(
            session_id,
            auth,
            Some(build_ag_ui_callback(tx, thread_id, run_id, last_applied)),
        )
        .await?;

    Ok(ReceiverStream::new(rx))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn load_state(runtime: &Runtime, session_id: Uuid, auth: &SessionAuth) -> (AgentState, u64) {
    match runtime.store().load(session_id, auth) {
        Ok(load) => {
            let last_applied = load.snapshot.last_applied.unwrap_or(0);
            (load.snapshot, last_applied)
        }
        Err(_) => (AgentState::new(session_id), 0),
    }
}

fn load_last_applied(runtime: &Runtime, session_id: Uuid, auth: &SessionAuth) -> u64 {
    load_state(runtime, session_id, auth).1
}

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
            error: None,
        },
    }
}

fn send_client_tools(session_id: Uuid, tools: Vec<super::types::Tool>) {
    let oai_tools: Vec<openai::Tool> = tools
        .into_iter()
        .map(|t| openai::Tool {
            tool_type: "function".to_string(),
            function: openai::ToolFunction {
                name: t.name,
                description: t.description.unwrap_or_default(),
                parameters: t.parameters,
            },
        })
        .collect();
    crate::runtime::send_to_session(session_id, SessionMessage::SetClientTools(oai_tools));
}

/// Extract new messages from the AG-UI input as domain `IncomingMessage`s.
/// Tool results come first, then the last user message (if any).
fn extract_new_messages(input: &RunAgentInput) -> Vec<IncomingMessage> {
    let mut messages = Vec::new();

    // Collect tool result messages
    for m in &input.messages {
        if let Message::Tool {
            tool_call_id,
            content,
            error,
            ..
        } = m
        {
            messages.push(IncomingMessage::ToolResult {
                tool_call_id: tool_call_id.clone(),
                content: content.clone(),
                error: error.clone(),
            });
        }
    }

    // Extract last user message
    if let Some(content) = input.messages.iter().rev().find_map(|m| match m {
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
    }) {
        messages.push(IncomingMessage::User { content });
    }

    messages
}
