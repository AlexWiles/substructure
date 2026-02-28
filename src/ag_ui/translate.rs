use std::collections::HashSet;

use crate::domain::event::{EventPayload, ToolCallMeta};
use crate::runtime::Notification;

use super::types::{AgUiEvent, InterruptInfo};

// ---------------------------------------------------------------------------
// TranslateOutput
// ---------------------------------------------------------------------------

/// Result of translating a single domain event.
pub enum TranslateOutput {
    /// Normal events — the run continues.
    Events(Vec<AgUiEvent>),
    /// Terminal events — the run is finished after emitting these.
    Terminal(Vec<AgUiEvent>),
    /// Interrupt — emit events then finish with interrupt data.
    Interrupt(Vec<AgUiEvent>, InterruptInfo),
}

// ---------------------------------------------------------------------------
// EventTranslator
// ---------------------------------------------------------------------------

/// Stateful translator from domain `Event` to AG-UI events.
///
/// Tracks open text messages, emitted tool calls, and emitted tool results
/// to handle deduplication between streaming and finalized events.
pub struct EventTranslator {
    /// If we're currently streaming text, the message_id.
    open_text_message: Option<String>,
    /// Message IDs that already streamed text chunks.
    streamed_text_messages: HashSet<String>,
    /// Tool call IDs for which we already emitted Start/Args/End.
    emitted_tool_calls: HashSet<String>,
    /// Tool call IDs for which we already emitted ToolCallResult.
    emitted_tool_results: HashSet<String>,
}

impl EventTranslator {
    pub fn new() -> Self {
        Self {
            open_text_message: None,
            streamed_text_messages: HashSet::new(),
            emitted_tool_calls: HashSet::new(),
            emitted_tool_results: HashSet::new(),
        }
    }

    /// Translate a transient notification into AG-UI events.
    pub fn translate_notification(&mut self, notification: &Notification) -> TranslateOutput {
        match notification {
            Notification::LlmStreamChunk {
                call_id,
                text,
                ..
            } => {
                if text.is_empty() {
                    return TranslateOutput::Events(vec![]);
                }

                let mut events = Vec::new();

                // Emit TextMessageStart on the first chunk
                if self.open_text_message.is_none() {
                    let message_id = call_id.clone();
                    self.open_text_message = Some(message_id.clone());
                    self.streamed_text_messages.insert(message_id.clone());
                    events.push(AgUiEvent::TextMessageStart { message_id });
                }

                let message_id = self
                    .open_text_message
                    .as_ref()
                    .expect("open_text_message must be set")
                    .clone();

                events.push(AgUiEvent::TextMessageContent {
                    message_id,
                    delta: text.clone(),
                });

                TranslateOutput::Events(events)
            }
        }
    }

    /// Translate a persisted domain event into zero or more AG-UI events.
    pub fn translate_event(&mut self, payload: &EventPayload) -> TranslateOutput {
        match payload {
            // SessionCreated is skipped — RunStarted is emitted by the stream wrapper
            EventPayload::SessionCreated(_) => TranslateOutput::Events(vec![]),

            // User messages are skipped — the user sent this
            EventPayload::MessageUser(_) => TranslateOutput::Events(vec![]),

            EventPayload::LlmCallRequested(req) => {
                TranslateOutput::Events(vec![AgUiEvent::StepStarted {
                    step_id: req.call_id.clone(),
                    step_name: "llm_call".into(),
                }])
            }

            EventPayload::LlmCallCompleted(completed) => {
                let mut events = Vec::new();

                // Close the text message stream if open
                if let Some(message_id) = self.open_text_message.take() {
                    events.push(AgUiEvent::TextMessageEnd { message_id });
                }

                events.push(AgUiEvent::StepFinished {
                    step_id: completed.call_id.clone(),
                    step_name: "llm_call".into(),
                });

                TranslateOutput::Events(events)
            }

            EventPayload::LlmCallErrored(errored) => {
                let mut events = Vec::new();

                // Close the text message stream if open
                if let Some(message_id) = self.open_text_message.take() {
                    events.push(AgUiEvent::TextMessageEnd { message_id });
                }

                events.push(AgUiEvent::StepFinished {
                    step_id: errored.call_id.clone(),
                    step_name: "llm_call".into(),
                });

                events.push(AgUiEvent::RunError {
                    message: errored.error.clone(),
                    code: None,
                });

                TranslateOutput::Terminal(events)
            }

            EventPayload::MessageAssistant(msg) => {
                let mut events = Vec::new();
                let has_text = msg.message.content.as_ref().is_some_and(|t| !t.is_empty());
                let has_tool_calls = !msg.message.tool_calls.is_empty();

                // Emit text message if not already streamed
                if has_text
                    && self.open_text_message.is_none()
                    && !self.streamed_text_messages.contains(&msg.call_id)
                {
                    let message_id = msg.call_id.clone();
                    let content = msg.message.content.clone().unwrap_or_default();

                    events.push(AgUiEvent::TextMessageStart {
                        message_id: message_id.clone(),
                    });
                    events.push(AgUiEvent::TextMessageContent {
                        message_id: message_id.clone(),
                        delta: content,
                    });
                    events.push(AgUiEvent::TextMessageEnd { message_id });
                }

                // Tool calls are emitted from ToolCallRequested (which follows
                // immediately in the event stream) to include meta like child_session_id.

                // Terminal detection: text content with no tool calls = run finished
                if has_text && !has_tool_calls {
                    TranslateOutput::Terminal(events)
                } else {
                    TranslateOutput::Events(events)
                }
            }

            EventPayload::ToolCallRequested(req) => {
                if !self.emitted_tool_calls.insert(req.tool_call_id.clone()) {
                    return TranslateOutput::Events(vec![]);
                }

                let child_session_id = match &req.meta {
                    Some(ToolCallMeta::SubAgent {
                        child_session_id, ..
                    }) => Some(child_session_id.to_string()),
                    _ => None,
                };
                let mut events = vec![AgUiEvent::ToolCallStart {
                    tool_call_id: req.tool_call_id.clone(),
                    tool_call_name: req.name.clone(),
                    parent_message_id: None,
                    child_session_id,
                }];
                if !req.arguments.is_empty() {
                    events.push(AgUiEvent::ToolCallArgs {
                        tool_call_id: req.tool_call_id.clone(),
                        delta: req.arguments.clone(),
                    });
                }
                events.push(AgUiEvent::ToolCallEnd {
                    tool_call_id: req.tool_call_id.clone(),
                });
                TranslateOutput::Events(events)
            }

            EventPayload::ToolCallCompleted(completed) => {
                if !self
                    .emitted_tool_results
                    .insert(completed.tool_call_id.clone())
                {
                    return TranslateOutput::Events(vec![]);
                }

                TranslateOutput::Events(vec![AgUiEvent::ToolCallResult {
                    message_id: format!("{}-result", completed.tool_call_id),
                    tool_call_id: completed.tool_call_id.clone(),
                    content: completed.result.clone(),
                    role: Some("tool".into()),
                    error: None,
                }])
            }

            EventPayload::ToolCallErrored(errored) => {
                if !self
                    .emitted_tool_results
                    .insert(errored.tool_call_id.clone())
                {
                    return TranslateOutput::Events(vec![]);
                }

                TranslateOutput::Events(vec![AgUiEvent::ToolCallResult {
                    message_id: format!("{}-result", errored.tool_call_id),
                    tool_call_id: errored.tool_call_id.clone(),
                    content: String::new(),
                    role: Some("tool".into()),
                    error: Some(errored.error.clone()),
                }])
            }

            EventPayload::SessionInterrupted(payload) => TranslateOutput::Interrupt(
                vec![],
                InterruptInfo {
                    id: payload.interrupt_id.clone(),
                    reason: payload.reason.clone(),
                    payload: payload.payload.clone(),
                },
            ),

            EventPayload::InterruptResumed(_) => TranslateOutput::Events(vec![]),

            EventPayload::StrategyStateChanged(_) => TranslateOutput::Events(vec![]),
            EventPayload::BudgetExceeded => TranslateOutput::Events(vec![]),
            EventPayload::SessionCancelled => TranslateOutput::Events(vec![]),
            EventPayload::SessionDone(_) => TranslateOutput::Events(vec![]),

            EventPayload::MessageTool(msg) => {
                // Skip if already emitted via ToolCallCompleted/Errored
                let tool_call_id = match &msg.message.tool_call_id {
                    Some(id) => id.clone(),
                    None => return TranslateOutput::Events(vec![]),
                };

                if !self.emitted_tool_results.insert(tool_call_id.clone()) {
                    return TranslateOutput::Events(vec![]);
                }

                TranslateOutput::Events(vec![AgUiEvent::ToolCallResult {
                    message_id: format!("{}-result", tool_call_id),
                    tool_call_id,
                    content: msg.message.content.clone().unwrap_or_default(),
                    role: Some("tool".into()),
                    error: None,
                }])
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::event::*;
    use chrono::Utc;
    use uuid::Uuid;

    fn assert_events(output: TranslateOutput) -> Vec<AgUiEvent> {
        match output {
            TranslateOutput::Events(e) => e,
            TranslateOutput::Terminal(_) => panic!("expected Events, got Terminal"),
            TranslateOutput::Interrupt(_, _) => panic!("expected Events, got Interrupt"),
        }
    }

    fn assert_terminal(output: TranslateOutput) -> Vec<AgUiEvent> {
        match output {
            TranslateOutput::Terminal(e) => e,
            TranslateOutput::Events(_) => panic!("expected Terminal, got Events"),
            TranslateOutput::Interrupt(_, _) => panic!("expected Terminal, got Interrupt"),
        }
    }

    // Helper to match event types
    fn event_type(event: &AgUiEvent) -> &'static str {
        match event {
            AgUiEvent::RunStarted { .. } => "RunStarted",
            AgUiEvent::RunFinished { .. } => "RunFinished",
            AgUiEvent::RunError { .. } => "RunError",
            AgUiEvent::StepStarted { .. } => "StepStarted",
            AgUiEvent::StepFinished { .. } => "StepFinished",
            AgUiEvent::TextMessageStart { .. } => "TextMessageStart",
            AgUiEvent::TextMessageContent { .. } => "TextMessageContent",
            AgUiEvent::TextMessageEnd { .. } => "TextMessageEnd",
            AgUiEvent::ToolCallStart { .. } => "ToolCallStart",
            AgUiEvent::ToolCallArgs { .. } => "ToolCallArgs",
            AgUiEvent::ToolCallEnd { .. } => "ToolCallEnd",
            AgUiEvent::ToolCallResult { .. } => "ToolCallResult",
            AgUiEvent::StateSnapshot { .. } => "StateSnapshot",
            AgUiEvent::StateDelta { .. } => "StateDelta",
            AgUiEvent::MessagesSnapshot { .. } => "MessagesSnapshot",
            AgUiEvent::Raw { .. } => "Raw",
            AgUiEvent::Custom { .. } => "Custom",
        }
    }

    #[test]
    fn test_streaming_text_path() {
        let mut translator = EventTranslator::new();
        let call_id = "call-1".to_string();

        // LlmCallRequested -> StepStarted
        let events = assert_events(translator.translate_event(
            &EventPayload::LlmCallRequested(LlmCallRequested {
                call_id: call_id.clone(),
                request: LlmRequest::OpenAi(crate::domain::openai::ChatCompletionRequest {
                    model: "test".into(),
                    messages: vec![],
                    tools: None,
                    tool_choice: None,
                    temperature: None,
                    max_tokens: None,
                }),
                stream: true,
                deadline: Utc::now() + chrono::Duration::hours(1),
            }),
        ));
        assert_eq!(events.len(), 1);
        assert_eq!(event_type(&events[0]), "StepStarted");

        // First chunk -> TextMessageStart + TextMessageContent
        let events = assert_events(translator.translate_notification(
            &Notification::LlmStreamChunk {
                call_id: call_id.clone(),
                chunk_index: 0,
                text: "Hello".into(),
            },
        ));
        assert_eq!(events.len(), 2);
        assert_eq!(event_type(&events[0]), "TextMessageStart");
        assert_eq!(event_type(&events[1]), "TextMessageContent");

        // Second chunk -> TextMessageContent only
        let events = assert_events(translator.translate_notification(
            &Notification::LlmStreamChunk {
                call_id: call_id.clone(),
                chunk_index: 1,
                text: " world".into(),
            },
        ));
        assert_eq!(events.len(), 1);
        assert_eq!(event_type(&events[0]), "TextMessageContent");

        // LlmCallCompleted -> TextMessageEnd + StepFinished
        let events = assert_events(translator.translate_event(
            &EventPayload::LlmCallCompleted(LlmCallCompleted {
                call_id: call_id.clone(),
                response: LlmResponse::OpenAi(crate::domain::openai::ChatCompletionResponse {
                    id: "resp-1".into(),
                    model: "test".into(),
                    choices: vec![],
                    usage: None,
                }),
            }),
        ));
        assert_eq!(events.len(), 2);
        assert_eq!(event_type(&events[0]), "TextMessageEnd");
        assert_eq!(event_type(&events[1]), "StepFinished");

        // MessageAssistant with text (already streamed) -> terminal with no text events
        let events = assert_terminal(translator.translate_event(
            &EventPayload::MessageAssistant(MessageAssistant {
                call_id: call_id.clone(),
                message: Message {
                    role: Role::Assistant,
                    content: Some("Hello world".into()),
                    tool_calls: vec![],
                    tool_call_id: None,
                    call_id: None,
                    token_count: None,
                },
            }),
        ));
        // The text was already streamed (tracked in streamed_text_messages),
        // so MessageAssistant emits no duplicate text events.
        assert_eq!(events.len(), 0);
    }

    #[test]
    fn test_non_streaming_text_path() {
        let mut translator = EventTranslator::new();
        let call_id = "call-1".to_string();

        // MessageAssistant with text, no tool calls -> terminal
        let events = assert_terminal(translator.translate_event(
            &EventPayload::MessageAssistant(MessageAssistant {
                call_id: call_id.clone(),
                message: Message {
                    role: Role::Assistant,
                    content: Some("Direct response".into()),
                    tool_calls: vec![],
                    tool_call_id: None,
                    call_id: None,
                    token_count: None,
                },
            }),
        ));
        assert_eq!(events.len(), 3);
        assert_eq!(event_type(&events[0]), "TextMessageStart");
        assert_eq!(event_type(&events[1]), "TextMessageContent");
        assert_eq!(event_type(&events[2]), "TextMessageEnd");

        // Verify content
        if let AgUiEvent::TextMessageContent { delta, .. } = &events[1] {
            assert_eq!(delta, "Direct response");
        } else {
            panic!("expected TextMessageContent");
        }
    }

    #[test]
    fn test_tool_call_emission() {
        let mut translator = EventTranslator::new();
        let call_id = "call-1".to_string();
        let tc_id = "tc-1".to_string();

        // MessageAssistant with tool calls — no longer emits ToolCallStart
        // (deferred to ToolCallRequested which carries meta)
        let events = assert_events(translator.translate_event(
            &EventPayload::MessageAssistant(MessageAssistant {
                call_id: call_id.clone(),
                message: Message {
                    role: Role::Assistant,
                    content: None,
                    tool_calls: vec![ToolCall {
                        id: tc_id.clone(),
                        name: "get_weather".into(),
                        arguments: r#"{"city":"NYC"}"#.into(),
                    }],
                    tool_call_id: None,
                    call_id: None,
                    token_count: None,
                },
            }),
        ));
        assert_eq!(events.len(), 0);

        // ToolCallRequested emits ToolCallStart/Args/End
        let events = assert_events(translator.translate_event(
            &EventPayload::ToolCallRequested(ToolCallRequested {
                tool_call_id: tc_id.clone(),
                name: "get_weather".into(),
                arguments: r#"{"city":"NYC"}"#.into(),
                deadline: Utc::now() + chrono::Duration::hours(1),
                handler: Default::default(),
                meta: None,
            }),
        ));
        assert_eq!(events.len(), 3);
        assert_eq!(event_type(&events[0]), "ToolCallStart");
        assert_eq!(event_type(&events[1]), "ToolCallArgs");
        assert_eq!(event_type(&events[2]), "ToolCallEnd");
    }

    #[test]
    fn test_tool_result_deduplication() {
        let mut translator = EventTranslator::new();
        let tc_id = "tc-1".to_string();

        // ToolCallCompleted -> ToolCallResult
        let events = assert_events(translator.translate_event(
            &EventPayload::ToolCallCompleted(ToolCallCompleted {
                tool_call_id: tc_id.clone(),
                name: "get_weather".into(),
                result: "Sunny, 72F".into(),
            }),
        ));
        assert_eq!(events.len(), 1);
        assert_eq!(event_type(&events[0]), "ToolCallResult");

        // MessageTool for the same tool_call_id -> deduplicated (skipped)
        let events = assert_events(translator.translate_event(&EventPayload::MessageTool(
            MessageTool {
                message: Message {
                    role: Role::Tool,
                    content: Some("Sunny, 72F".into()),
                    tool_calls: vec![],
                    tool_call_id: Some(tc_id.clone()),
                    call_id: None,
                    token_count: None,
                },
            },
        )));
        assert_eq!(events.len(), 0);
    }

    #[test]
    fn test_tool_call_error_result() {
        let mut translator = EventTranslator::new();
        let tc_id = "tc-err".to_string();

        let events = assert_events(translator.translate_event(
            &EventPayload::ToolCallErrored(ToolCallErrored {
                tool_call_id: tc_id.clone(),
                name: "bad_tool".into(),
                error: "tool not found".into(),
            }),
        ));
        assert_eq!(events.len(), 1);
        assert_eq!(event_type(&events[0]), "ToolCallResult");
        if let AgUiEvent::ToolCallResult { error, content, .. } = &events[0] {
            assert_eq!(error.as_deref(), Some("tool not found"));
            assert_eq!(content, "");
        } else {
            panic!("expected ToolCallResult");
        }
    }

    #[test]
    fn test_terminal_detection() {
        let mut translator = EventTranslator::new();

        // MessageAssistant with text and NO tool calls -> terminal
        let output = translator.translate_event(&EventPayload::MessageAssistant(
            MessageAssistant {
                call_id: "call-1".into(),
                message: Message {
                    role: Role::Assistant,
                    content: Some("Done!".into()),
                    tool_calls: vec![],
                    tool_call_id: None,
                    call_id: None,
                    token_count: None,
                },
            },
        ));
        assert!(matches!(output, TranslateOutput::Terminal(_)));

        // MessageAssistant with tool calls -> NOT terminal
        let mut translator2 = EventTranslator::new();
        let output = translator2.translate_event(&EventPayload::MessageAssistant(
            MessageAssistant {
                call_id: "call-2".into(),
                message: Message {
                    role: Role::Assistant,
                    content: Some("Let me check".into()),
                    tool_calls: vec![ToolCall {
                        id: "tc-1".into(),
                        name: "search".into(),
                        arguments: "{}".into(),
                    }],
                    tool_call_id: None,
                    call_id: None,
                    token_count: None,
                },
            },
        ));
        assert!(matches!(output, TranslateOutput::Events(_)));
    }

    #[test]
    fn test_llm_error_is_terminal() {
        let mut translator = EventTranslator::new();

        let output =
            translator.translate_event(&EventPayload::LlmCallErrored(LlmCallErrored {
                call_id: "call-1".into(),
                error: "API rate limit".into(),
                retryable: true,
                source: None,
            }));
        let events = assert_terminal(output);
        assert_eq!(events.len(), 2); // StepFinished + RunError
        assert_eq!(event_type(&events[0]), "StepFinished");
        assert_eq!(event_type(&events[1]), "RunError");
    }

    #[test]
    fn test_session_created_and_user_message_skipped() {
        let mut translator = EventTranslator::new();

        let events = assert_events(translator.translate_event(
            &EventPayload::SessionCreated(SessionCreated {
                agent: crate::domain::agent::AgentConfig {
                    id: Uuid::new_v4(),
                    name: "test".into(),
                    description: None,
                    llm: crate::domain::agent::LlmConfig {
                        client: "mock".into(),
                        params: Default::default(),
                    },
                    system_prompt: "You are helpful.".into(),
                    mcp_servers: vec![],
                    strategy: Default::default(),
                    retry: Default::default(),
                    token_budget: None,
                    sub_agents: vec![],
                },
                auth: ClientIdentity {
                    tenant_id: "t".into(),
                    sub: None,
                    attrs: Default::default(),
                },
                on_done: None,
            }),
        ));
        assert_eq!(events.len(), 0);

        let events = assert_events(translator.translate_event(&EventPayload::MessageUser(
            MessageUser {
                message: Message {
                    role: Role::User,
                    content: Some("Hi".into()),
                    tool_calls: vec![],
                    tool_call_id: None,
                    call_id: None,
                    token_count: None,
                },
                stream: true,
            },
        )));
        assert_eq!(events.len(), 0);
    }
}
