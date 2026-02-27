use std::sync::Mutex;

use chrono::Utc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::Stream;
use uuid::Uuid;

use crate::domain::aggregate::DomainEvent;
use crate::domain::event::{Message as DomainMessage, Role, SessionAuth, SpanContext, ToolCall as DomainToolCall};
use crate::domain::openai;
use crate::domain::session::{AgentState, CommandPayload, SessionCommand};
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

    Box::new(move |event: &DomainEvent<AgentState>| {
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
    let messages: Vec<DomainMessage> = input.messages.iter().map(ag_ui_to_domain_message).collect();
    if messages.is_empty() {
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

    client
        .send_command(SessionCommand {
            span: SpanContext::root(),
            occurred_at: Utc::now(),
            payload: CommandPayload::SyncConversation {
                messages,
                stream: true,
            },
        })
        .await?;

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
    let messages: Vec<DomainMessage> = input.messages.iter().map(ag_ui_to_domain_message).collect();
    if messages.is_empty() {
        return Err(AgUiError::NoUserMessage);
    }

    let thread_id = input.thread_id.unwrap_or_else(|| session_id.to_string());
    let run_id = input.run_id.unwrap_or_else(|| Uuid::new_v4().to_string());

    let skip_until = load_state(runtime, session_id, &auth).1;

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

    client
        .send_command(SessionCommand {
            span: SpanContext::root(),
            occurred_at: Utc::now(),
            payload: CommandPayload::SyncConversation {
                messages,
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
    match runtime.store().load(session_id, &auth.tenant_id) {
        Ok(load) => {
            let state: AgentState = serde_json::from_value(load.snapshot)
                .unwrap_or_else(|_| AgentState::new(session_id));
            let last_applied = state.last_applied.unwrap_or(0);
            (state, last_applied)
        }
        Err(_) => (AgentState::new(session_id), 0),
    }
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

/// Convert an AG-UI `Message` to a domain `Message`.
pub(super) fn ag_ui_to_domain_message(msg: &Message) -> DomainMessage {
    match msg {
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
            DomainMessage {
                role: Role::User,
                content: Some(text),
                tool_calls: vec![],
                tool_call_id: None,
                call_id: None,
                token_count: None,
            }
        }
        Message::Assistant {
            content,
            tool_calls,
            ..
        } => DomainMessage {
            role: Role::Assistant,
            content: content.clone(),
            tool_calls: tool_calls
                .iter()
                .map(|tc| DomainToolCall {
                    id: tc.id.clone(),
                    name: tc.function.name.clone(),
                    arguments: tc.function.arguments.clone(),
                })
                .collect(),
            tool_call_id: None,
            call_id: None,
            token_count: None,
        },
        Message::Tool {
            tool_call_id,
            content,
            ..
        } => DomainMessage {
            role: Role::Tool,
            content: Some(content.clone()),
            tool_calls: vec![],
            tool_call_id: Some(tool_call_id.clone()),
            call_id: None,
            token_count: None,
        },
        Message::System { content, .. } | Message::Developer { content, .. } => DomainMessage {
            role: Role::System,
            content: Some(content.clone()),
            tool_calls: vec![],
            tool_call_id: None,
            call_id: None,
            token_count: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::types::*;

    #[test]
    fn user_message_with_parts_joins_text() {
        let msg = Message::User {
            id: None,
            content: MessageContent::Parts(vec![
                InputContent::Text { text: "hello ".into() },
                InputContent::ImageUrl { image_url: ImageUrl { url: "http://img".into() } },
                InputContent::Text { text: "world".into() },
            ]),
        };
        let domain = ag_ui_to_domain_message(&msg);
        assert_eq!(domain.role, Role::User);
        assert_eq!(domain.content.as_deref(), Some("hello world"));
    }

    #[test]
    fn assistant_message_preserves_tool_calls() {
        let msg = Message::Assistant {
            id: None,
            content: Some("thinking".into()),
            tool_calls: vec![
                ToolCallInfo {
                    id: "tc-1".into(),
                    function: FunctionCall { name: "read".into(), arguments: "{}".into() },
                },
                ToolCallInfo {
                    id: "tc-2".into(),
                    function: FunctionCall { name: "write".into(), arguments: r#"{"x":1}"#.into() },
                },
            ],
        };
        let domain = ag_ui_to_domain_message(&msg);
        assert_eq!(domain.role, Role::Assistant);
        assert_eq!(domain.content.as_deref(), Some("thinking"));
        assert_eq!(domain.tool_calls.len(), 2);
        assert_eq!(domain.tool_calls[0].id, "tc-1");
        assert_eq!(domain.tool_calls[0].name, "read");
        assert_eq!(domain.tool_calls[1].id, "tc-2");
        assert_eq!(domain.tool_calls[1].arguments, r#"{"x":1}"#);
    }

    #[test]
    fn tool_message_maps_tool_call_id() {
        let msg = Message::Tool {
            id: None,
            tool_call_id: "tc-1".into(),
            content: "42".into(),
            error: Some("oops".into()),
        };
        let domain = ag_ui_to_domain_message(&msg);
        assert_eq!(domain.role, Role::Tool);
        assert_eq!(domain.content.as_deref(), Some("42"));
        assert_eq!(domain.tool_call_id.as_deref(), Some("tc-1"));
        assert!(domain.tool_calls.is_empty());
    }
}
