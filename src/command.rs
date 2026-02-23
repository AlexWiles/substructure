use crate::event::*;
use crate::session::{LlmCallStatus, SessionState};

#[derive(Debug, Clone)]
pub enum SessionCommand {
    CreateSession {
        agent: AgentConfig,
        auth: SessionAuth,
    },
    SendUserMessage {
        content: String,
    },
    SendAssistantMessage {
        content: Option<String>,
        tool_calls: Vec<ToolCall>,
        token_count: Option<u32>,
    },
    SendToolMessage {
        tool_call_id: String,
        content: String,
        token_count: Option<u32>,
    },
    RequestLlmCall {
        call_id: String,
        request: LlmRequest,
    },
    CompleteLlmCall {
        call_id: String,
        response: LlmResponse,
    },
    FailLlmCall {
        call_id: String,
        error: String,
    },
    StreamLlmChunk {
        call_id: String,
        chunk_index: u32,
        text: String,
    },
    RequestToolCall {
        tool_call_id: String,
        name: String,
        arguments: String,
    },
    CompleteToolCall {
        tool_call_id: String,
        name: String,
        result: String,
    },
    FailToolCall {
        tool_call_id: String,
        name: String,
        error: String,
    },
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum SessionError {
    #[error("session has not been created")]
    SessionNotCreated,
    #[error("session already exists")]
    SessionAlreadyCreated,
}

impl SessionState {
    pub fn handle(&self, cmd: SessionCommand) -> Result<Vec<EventPayload>, SessionError> {
        match (&self.agent, cmd) {
            // Uncreated session: only CreateSession is valid
            (None, SessionCommand::CreateSession { agent, auth }) => {
                Ok(vec![EventPayload::SessionCreated(SessionCreated { agent, auth })])
            }
            (Some(_), SessionCommand::CreateSession { .. }) => {
                Err(SessionError::SessionAlreadyCreated)
            }
            (None, _) => {
                Err(SessionError::SessionNotCreated)
            }

            // Active session
            (Some(_), SessionCommand::SendUserMessage { content }) => {
                Ok(vec![EventPayload::MessageUser(MessageUser {
                    message: Message {
                        role: Role::User,
                        content: Some(content),
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                        token_count: None,
                    },
                })])
            }
            (Some(_), SessionCommand::SendAssistantMessage { content, tool_calls, token_count }) => {
                Ok(vec![EventPayload::MessageAssistant(MessageAssistant {
                    message: Message {
                        role: Role::Assistant,
                        content,
                        tool_calls,
                        tool_call_id: None,
                        token_count,
                    },
                })])
            }
            (Some(_), SessionCommand::SendToolMessage { tool_call_id, content, token_count }) => {
                Ok(vec![EventPayload::MessageTool(MessageTool {
                    message: Message {
                        role: Role::Tool,
                        content: Some(content),
                        tool_calls: Vec::new(),
                        tool_call_id: Some(tool_call_id),
                        token_count,
                    },
                })])
            }
            // Command guards — use LLM call lifecycle to decide.
            //
            // RequestLlmCall: skip if call_id already known (duplicate) or
            // another call is already in flight.
            (Some(_), SessionCommand::RequestLlmCall { ref call_id, .. })
                if self.llm_calls.contains_key(call_id)
                    || self.active_llm_call().is_some() =>
            {
                Ok(vec![])
            }
            // CompleteLlmCall: only valid if this call_id is currently pending.
            (Some(_), SessionCommand::CompleteLlmCall { ref call_id, .. })
                if !matches!(
                    self.llm_calls.get(call_id).map(|c| &c.status),
                    Some(&LlmCallStatus::Pending)
                ) =>
            {
                Ok(vec![])
            }
            // FailLlmCall: same — only valid if this call_id is currently pending.
            (Some(_), SessionCommand::FailLlmCall { ref call_id, .. })
                if !matches!(
                    self.llm_calls.get(call_id).map(|c| &c.status),
                    Some(&LlmCallStatus::Pending)
                ) =>
            {
                Ok(vec![])
            }
            (Some(_), SessionCommand::RequestLlmCall { call_id, request }) => {
                Ok(vec![EventPayload::LlmCallRequested(LlmCallRequested { call_id, request })])
            }
            (Some(_), SessionCommand::CompleteLlmCall { call_id, response }) => {
                Ok(vec![EventPayload::LlmCallCompleted(LlmCallCompleted { call_id, response })])
            }
            (Some(_), SessionCommand::FailLlmCall { call_id, error }) => {
                Ok(vec![EventPayload::LlmCallErrored(LlmCallErrored { call_id, error })])
            }
            (Some(_), SessionCommand::StreamLlmChunk { call_id, chunk_index, text }) => {
                Ok(vec![EventPayload::LlmStreamChunk(LlmStreamChunk { call_id, chunk_index, text })])
            }
            (Some(_), SessionCommand::RequestToolCall { tool_call_id, name, arguments }) => {
                Ok(vec![EventPayload::ToolCallRequested(ToolCallRequested {
                    tool_call_id, name, arguments,
                })])
            }
            (Some(_), SessionCommand::CompleteToolCall { tool_call_id, name, result }) => {
                Ok(vec![EventPayload::ToolCallCompleted(ToolCallCompleted {
                    tool_call_id, name, result,
                })])
            }
            (Some(_), SessionCommand::FailToolCall { tool_call_id, name, error }) => {
                Ok(vec![EventPayload::ToolCallErrored(ToolCallErrored {
                    tool_call_id, name, error,
                })])
            }
        }
    }
}
