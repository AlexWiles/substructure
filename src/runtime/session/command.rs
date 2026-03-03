use std::fmt::Write;

use chrono::{DateTime, Utc};

use super::state::{
    new_call_id, LlmCallStatus, SessionContext, SessionState, SessionStatus, ToolCallState,
    ToolCallStatus,
};
use crate::runtime::aggregate::Emit;
use crate::runtime::event::*;

// ---------------------------------------------------------------------------
// Tool result truncation
// ---------------------------------------------------------------------------

/// Truncate a tool result string if it exceeds the limit.
///
/// `max_bytes = None` means no limit (disabled). Truncation respects UTF-8
/// char boundaries and appends a note with the original and limit sizes.
pub(super) fn truncate_tool_result(s: String, max_bytes: Option<usize>) -> String {
    let Some(max_bytes) = max_bytes else {
        return s;
    };
    if s.len() <= max_bytes {
        return s;
    }
    let total = s.len();
    let cut = s.floor_char_boundary(max_bytes);
    let mut out = s;
    out.truncate(cut);
    let _ = write!(
        out,
        "\n\n--- Output truncated ({total} bytes, limit: {max_bytes} bytes) ---"
    );
    out
}

// ---------------------------------------------------------------------------
// Command types
// ---------------------------------------------------------------------------

/// A message from an external client (AG-UI, HTTP, etc.).
#[derive(Debug, Clone)]
pub enum IncomingMessage {
    User {
        content: String,
    },
    ToolResult {
        tool_call_id: String,
        content: String,
        error: Option<String>,
    },
}

#[derive(Debug, Clone)]
pub struct SessionCommand {
    pub span: SpanContext,
    pub occurred_at: DateTime<Utc>,
    pub payload: CommandPayload,
}

#[derive(Debug, Clone)]
pub enum CommandPayload {
    CreateSession {
        agent: Box<AgentConfig>,
        auth: ClientIdentity,
        on_done: Option<CompletionDelivery>,
    },
    SendMessage {
        message: IncomingMessage,
        stream: bool,
    },
    RequestLlmCall {
        call_id: String,
        request: LlmRequest,
        stream: bool,
        deadline: DateTime<Utc>,
    },
    CompleteLlmCall {
        call_id: String,
        response: LlmResponse,
    },
    FailLlmCall {
        call_id: String,
        error: String,
        retryable: bool,
        source: Option<serde_json::Value>,
    },
    RequestToolCall {
        tool_call_id: String,
        name: String,
        arguments: String,
        deadline: DateTime<Utc>,
        #[allow(dead_code)]
        handler: ToolHandler,
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
    Interrupt {
        interrupt_id: String,
        reason: String,
        payload: serde_json::Value,
    },
    ResumeInterrupt {
        interrupt_id: String,
        payload: serde_json::Value,
    },
    UpdateStrategyState {
        state: serde_json::Value,
    },
    SyncConversation {
        messages: Vec<Message>,
        stream: bool,
    },
    CancelSession,
    MarkDone {
        artifacts: Vec<Artifact>,
    },
    Wake,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum SessionError {
    #[error("session has not been created")]
    SessionNotCreated,
    #[error("session already exists")]
    SessionAlreadyCreated,
    #[error("session is interrupted")]
    SessionInterrupted,
    #[error("session is busy")]
    SessionBusy,
    #[error("conversation has diverged")]
    ConversationDiverged,
}

// ---------------------------------------------------------------------------
// Command handling
// ---------------------------------------------------------------------------

impl SessionState {
    pub fn handle(
        &self,
        cmd: CommandPayload,
        ctx: &SessionContext,
    ) -> Result<Vec<Emit<EventPayload>>, SessionError> {
        match (&self.agent, cmd) {
            (
                None,
                CommandPayload::CreateSession {
                    agent,
                    auth,
                    on_done,
                },
            ) => Ok(vec![Emit::new(EventPayload::SessionCreated(Box::new(
                SessionCreated {
                    agent: (*agent).clone(),
                    auth,
                    on_done,
                },
            )))
            .label(&agent.name)]),
            (Some(_), CommandPayload::CreateSession { .. }) => {
                Err(SessionError::SessionAlreadyCreated)
            }
            (None, _) => Err(SessionError::SessionNotCreated),
            // Active session
            (Some(_), cmd) => self.handle_active(cmd, ctx),
        }
    }

    /// Command validation using SessionState for idempotency guards.
    fn handle_active(
        &self,
        cmd: CommandPayload,
        ctx: &SessionContext,
    ) -> Result<Vec<Emit<EventPayload>>, SessionError> {
        let state = self;
        match cmd {
            CommandPayload::CreateSession { .. } => {
                unreachable!("CreateSession is handled by SessionState::handle")
            }
            CommandPayload::SendMessage { message, stream } => match message {
                IncomingMessage::User { content } => match state.status {
                    SessionStatus::Interrupted { .. } => Err(SessionError::SessionInterrupted),
                    SessionStatus::Active => Err(SessionError::SessionBusy),
                    _ => {
                        let agent_name = state
                            .agent
                            .as_ref()
                            .map(|a| a.name.as_str())
                            .unwrap_or("unknown");
                        Ok(vec![Emit::new(EventPayload::MessageUser(MessageUser {
                            message: Message {
                                role: Role::User,
                                content: Some(content),
                                tool_calls: Vec::new(),
                                tool_call_id: None,
                                call_id: None,
                                token_count: None,
                            },
                            stream,
                        }))
                        .label(agent_name)])
                    }
                },
                IncomingMessage::ToolResult {
                    tool_call_id,
                    content,
                    error,
                } => {
                    let tc = state.tool_calls.get(&tool_call_id);

                    match tc {
                        // Case where we havent seen the tool call before. We ignore it.
                        None => Ok(vec![]),
                        // Have seen a terminal/retry state for this toolcall already
                        Some(ToolCallState {
                            status: ToolCallStatus::Completed | ToolCallStatus::Failed | ToolCallStatus::RetryScheduled,
                            ..
                        }) => Ok(vec![]),
                        // We are expecting a result
                        Some(ToolCallState {
                            status: ToolCallStatus::Pending,
                            name,
                            ..
                        }) => {
                            let max = state.tool_result_max_bytes(name, ctx);
                            if let Some(err) = error {
                                let err = truncate_tool_result(err, max);
                                Ok(vec![Emit::new(EventPayload::ToolCallErrored(
                                    ToolCallErrored {
                                        tool_call_id,
                                        name: name.clone(),
                                        error: err.clone(),
                                        retryable: false,
                                    },
                                ))
                                .with("tool.name", name.as_str())
                                .label(name.as_str())
                                .error(err)])
                            } else {
                                let content = truncate_tool_result(content, max);
                                Ok(vec![Emit::new(EventPayload::ToolCallCompleted(
                                    ToolCallCompleted {
                                        tool_call_id,
                                        name: name.clone(),
                                        result: content,
                                    },
                                ))
                                .with("tool.name", name.as_str())
                                .label(name.as_str())])
                            }
                        }
                    }
                }
            },
            CommandPayload::RequestLlmCall {
                call_id,
                request,
                stream,
                deadline,
            } => {
                if state.is_over_budget() {
                    return Ok(vec![EventPayload::BudgetExceeded.into()]);
                }
                let has_pending = state
                    .llm_calls
                    .values()
                    .any(|c| c.status == LlmCallStatus::Pending);

                if has_pending {
                    return Ok(vec![]);
                }

                let issue = match state.llm_calls.get(&call_id).map(|c| &c.status) {
                    // New call
                    None => true,
                    // Previously failed — retry
                    Some(&LlmCallStatus::Failed) => true,
                    // Retry scheduled but not yet fired — short-circuit the backoff
                    Some(&LlmCallStatus::RetryScheduled) => true,
                    // Already in flight or completed — skip
                    _ => false,
                };
                if issue {
                    Ok(vec![EventPayload::LlmCallRequested(LlmCallRequested {
                        call_id,
                        request,
                        stream,
                        deadline,
                    })
                    .into()])
                } else {
                    Ok(vec![])
                }
            }
            CommandPayload::CompleteLlmCall { call_id, response } => {
                match state.llm_calls.get(&call_id).map(|c| &c.status) {
                    // Pending call — complete it
                    Some(&LlmCallStatus::Pending) => {
                        let (content, tool_calls, token_count) = response.as_parts();

                        // Extract metadata before moving response
                        let model = match &response {
                            LlmResponse::OpenAi(r) => r.model.clone(),
                        };
                        let usage = match &response {
                            LlmResponse::OpenAi(r) => r.usage.clone(),
                        };

                        let mut completed =
                            Emit::new(EventPayload::LlmCallCompleted(LlmCallCompleted {
                                call_id: call_id.clone(),
                                response,
                            }))
                            .with("llm.model", &model)
                            .label(&model);
                        if let Some(u) = usage {
                            completed = completed
                                .with("llm.tokens.prompt", u.prompt_tokens.to_string())
                                .with("llm.tokens.completion", u.completion_tokens.to_string());
                        }

                        let mut events = vec![
                            completed,
                            EventPayload::MessageAssistant(MessageAssistant {
                                call_id: call_id.clone(),
                                message: Message {
                                    role: Role::Assistant,
                                    content,
                                    tool_calls: tool_calls.clone(),
                                    tool_call_id: None,
                                    call_id: Some(call_id),
                                    token_count,
                                },
                            })
                            .into(),
                        ];
                        for tc in &tool_calls {
                            let meta = self.tool_call_meta(&tc.name, &tc.id, &ctx.mcp_tools);
                            events.push(
                                Emit::new(EventPayload::ToolCallRequested(ToolCallRequested {
                                    tool_call_id: tc.id.clone(),
                                    name: tc.name.clone(),
                                    arguments: tc.arguments.clone(),
                                    deadline: self.tool_deadline(meta.as_ref()),
                                    handler: Default::default(),
                                    meta,
                                }))
                                .with("tool.name", &tc.name)
                                .label(&tc.name),
                            );
                        }
                        Ok(events)
                    }
                    // Not pending or unknown — skip
                    _ => Ok(vec![]),
                }
            }
            CommandPayload::FailLlmCall {
                call_id,
                error,
                retryable,
                source,
            } => match state.llm_calls.get(&call_id).map(|c| &c.status) {
                // Pending call — fail it
                Some(&LlmCallStatus::Pending) => Ok(vec![Emit::new(EventPayload::LlmCallErrored(
                    LlmCallErrored {
                        call_id,
                        error: error.clone(),
                        retryable,
                        source,
                    },
                ))
                .error(error)]),
                // Not pending or unknown — skip
                _ => Ok(vec![]),
            },
            CommandPayload::RequestToolCall {
                tool_call_id,
                name,
                arguments,
                deadline,
                handler,
            } => match state.tool_calls.get(&tool_call_id) {
                // Already tracked — skip
                Some(_) => Ok(vec![]),
                // New tool call
                None => {
                    let meta = self.tool_call_meta(&name, &tool_call_id, &ctx.mcp_tools);
                    Ok(vec![Emit::new(EventPayload::ToolCallRequested(
                        ToolCallRequested {
                            tool_call_id,
                            name: name.clone(),
                            arguments,
                            deadline,
                            handler,
                            meta,
                        },
                    ))
                    .with("tool.name", &name)
                    .label(name)])
                }
            },
            CommandPayload::CompleteToolCall {
                tool_call_id,
                name,
                result,
            } => match state.tool_calls.get(&tool_call_id).map(|tc| &tc.status) {
                // Pending — complete and emit tool message
                Some(&ToolCallStatus::Pending) => {
                    let max = state.tool_result_max_bytes(&name, ctx);
                    let result = truncate_tool_result(result, max);
                    Ok(vec![
                        Emit::new(EventPayload::ToolCallCompleted(ToolCallCompleted {
                            tool_call_id: tool_call_id.clone(),
                            name: name.clone(),
                            result: result.clone(),
                        }))
                        .with("tool.name", &name)
                        .label(&name),
                        EventPayload::MessageTool(MessageTool {
                            message: Message {
                                role: Role::Tool,
                                content: Some(result),
                                tool_calls: Vec::new(),
                                tool_call_id: Some(tool_call_id),
                                call_id: None,
                                token_count: None,
                            },
                        })
                        .into(),
                    ])
                }
                // Not pending or unknown — skip
                _ => Ok(vec![]),
            },
            CommandPayload::FailToolCall {
                tool_call_id,
                name,
                error,
            } => match state.tool_calls.get(&tool_call_id).map(|tc| &tc.status) {
                // Pending — fail and emit error tool message
                Some(&ToolCallStatus::Pending) => {
                    let max = state.tool_result_max_bytes(&name, ctx);
                    let error = truncate_tool_result(error, max);
                    let error_content = format!("Error: {}", error);
                    Ok(vec![
                        Emit::new(EventPayload::ToolCallErrored(ToolCallErrored {
                            tool_call_id: tool_call_id.clone(),
                            name: name.clone(),
                            error: error.clone(),
                            retryable: false,
                        }))
                        .with("tool.name", &name)
                        .label(&name)
                        .error(error),
                        EventPayload::MessageTool(MessageTool {
                            message: Message {
                                role: Role::Tool,
                                content: Some(error_content),
                                tool_calls: Vec::new(),
                                tool_call_id: Some(tool_call_id),
                                call_id: None,
                                token_count: None,
                            },
                        })
                        .into(),
                    ])
                }
                // Not pending or unknown — skip
                _ => Ok(vec![]),
            },
            CommandPayload::Interrupt {
                interrupt_id,
                reason,
                payload,
            } => match state.status {
                // Already interrupted — skip
                SessionStatus::Interrupted { .. } => Ok(vec![]),
                _ => Ok(vec![EventPayload::SessionInterrupted(SessionInterrupted {
                    interrupt_id,
                    reason,
                    payload,
                })
                .into()]),
            },
            CommandPayload::ResumeInterrupt {
                interrupt_id,
                payload,
            } => match state.active_interrupt() {
                // Matching active interrupt — resume
                Some(id) if id == interrupt_id => {
                    Ok(vec![EventPayload::InterruptResumed(InterruptResumed {
                        interrupt_id,
                        payload,
                    })
                    .into()])
                }
                // No active interrupt or wrong ID — skip
                _ => Ok(vec![]),
            },
            CommandPayload::UpdateStrategyState { state } => {
                Ok(vec![EventPayload::StrategyStateChanged(
                    StrategyStateChanged { state },
                )
                .into()])
            }
            CommandPayload::SyncConversation { messages, stream } => {
                self.handle_sync_conversation(messages, stream, ctx)
            }
            CommandPayload::CancelSession => Ok(vec![EventPayload::SessionCancelled.into()]),
            CommandPayload::MarkDone { artifacts } => {
                Ok(vec![
                    EventPayload::SessionDone(SessionDone { artifacts }).into()
                ])
            }
            CommandPayload::Wake => self.handle_wake(ctx),
        }
    }

    // -----------------------------------------------------------------------
    // Wake — inspects state and emits events for timeouts, retries, recovery
    // -----------------------------------------------------------------------

    fn handle_wake(&self, _ctx: &SessionContext) -> Result<Vec<Emit<EventPayload>>, SessionError> {
        let now = Utc::now();
        let state = self;

        // 1. Timed-out pending LLM calls → fail
        for call in state.llm_calls.values() {
            if call.status == LlmCallStatus::Pending && call.deadline <= now {
                return Ok(vec![Emit::new(EventPayload::LlmCallErrored(
                    LlmCallErrored {
                        call_id: call.call_id.clone(),
                        error: "deadline exceeded".to_string(),
                        retryable: true,
                        source: None,
                    },
                ))
                .error("deadline exceeded")]);
            }
        }

        // 2. Timed-out pending tool calls → fail (or re-issue for sub-agents)
        for tc in state.tool_calls.values() {
            if tc.status == ToolCallStatus::Pending
                && tc.deadline <= now
                && tc.handler != ToolHandler::Client
            {
                if tc.child_session_id().is_some() {
                    // Sub-agent: re-arm with fresh deadline (parent-driven retry)
                    let arguments = state
                        .messages
                        .iter()
                        .flat_map(|m| m.tool_calls.iter())
                        .find(|t| t.id == tc.tool_call_id)
                        .map(|t| t.arguments.clone())
                        .unwrap_or_default();
                    return Ok(vec![Emit::new(EventPayload::ToolCallRequested(
                        ToolCallRequested {
                            tool_call_id: tc.tool_call_id.clone(),
                            name: tc.name.clone(),
                            arguments,
                            deadline: self.tool_deadline(tc.meta.as_ref()),
                            handler: tc.handler.clone(),
                            meta: tc.meta.clone(),
                        },
                    ))
                    .with("tool.name", &tc.name)
                    .label(&tc.name)]);
                }
                // Check if retries remain
                if tc.retry.attempts < tc.retry_policy.max_retries {
                    // Retryable — emit only ToolCallErrored, apply_core will schedule retry
                    return Ok(vec![Emit::new(EventPayload::ToolCallErrored(
                        ToolCallErrored {
                            tool_call_id: tc.tool_call_id.clone(),
                            name: tc.name.clone(),
                            error: "deadline exceeded".to_string(),
                            retryable: true,
                        },
                    ))
                    .with("tool.name", &tc.name)
                    .label(&tc.name)
                    .error("deadline exceeded")]);
                } else {
                    // Retries exhausted — fail with MessageTool so LLM sees the error
                    let error = "deadline exceeded".to_string();
                    let error_content = format!("Error: {}", error);
                    return Ok(vec![
                        Emit::new(EventPayload::ToolCallErrored(ToolCallErrored {
                            tool_call_id: tc.tool_call_id.clone(),
                            name: tc.name.clone(),
                            error: error.clone(),
                            retryable: false,
                        }))
                        .with("tool.name", &tc.name)
                        .label(&tc.name)
                        .error(&error),
                        EventPayload::MessageTool(MessageTool {
                            message: Message {
                                role: Role::Tool,
                                content: Some(error_content),
                                tool_calls: Vec::new(),
                                tool_call_id: Some(tc.tool_call_id.clone()),
                                call_id: None,
                                token_count: None,
                            },
                        })
                        .into(),
                    ]);
                }
            }
        }

        // 3. RetryScheduled LLM call with next_at passed → re-issue
        for call in state.llm_calls.values() {
            if call.status == LlmCallStatus::RetryScheduled {
                if let Some(next_at) = call.retry.next_at {
                    if next_at <= now {
                        if state.is_over_budget() {
                            return Ok(vec![EventPayload::BudgetExceeded.into()]);
                        }
                        if let Some(request) = state.build_llm_request(None) {
                            return Ok(vec![EventPayload::LlmCallRequested(LlmCallRequested {
                                call_id: call.call_id.clone(),
                                request,
                                stream: true,
                                deadline: self.llm_deadline(),
                            })
                            .into()]);
                        }
                    }
                }
            }
        }

        // 3b. RetryScheduled tool call with next_at passed → re-issue
        for tc in state.tool_calls.values() {
            if tc.status == ToolCallStatus::RetryScheduled {
                if let Some(next_at) = tc.retry.next_at {
                    if next_at <= now {
                        let arguments = state
                            .messages
                            .iter()
                            .flat_map(|m| m.tool_calls.iter())
                            .find(|t| t.id == tc.tool_call_id)
                            .map(|t| t.arguments.clone())
                            .unwrap_or_default();
                        return Ok(vec![Emit::new(EventPayload::ToolCallRequested(
                            ToolCallRequested {
                                tool_call_id: tc.tool_call_id.clone(),
                                name: tc.name.clone(),
                                arguments,
                                deadline: self.tool_deadline(tc.meta.as_ref()),
                                handler: tc.handler.clone(),
                                meta: tc.meta.clone(),
                            },
                        ))
                        .with("tool.name", &tc.name)
                        .label(&tc.name)]);
                    }
                }
            }
        }

        // 4. Pending tool calls still in flight → re-emit (crash recovery)
        for tc in state.tool_calls.values() {
            if tc.status == ToolCallStatus::Pending
                && tc.deadline > now
                && tc.handler != ToolHandler::Client
            {
                return Ok(vec![Emit::new(EventPayload::ToolCallRequested(
                    ToolCallRequested {
                        tool_call_id: tc.tool_call_id.clone(),
                        name: tc.name.clone(),
                        arguments: state
                            .messages
                            .iter()
                            .flat_map(|m| m.tool_calls.iter())
                            .find(|t| t.id == tc.tool_call_id)
                            .map(|t| t.arguments.clone())
                            .unwrap_or_default(),
                        deadline: tc.deadline,
                        handler: tc.handler.clone(),
                        meta: tc.meta.clone(),
                    },
                ))
                .with("tool.name", &tc.name)
                .label(&tc.name)]);
            }
        }

        // 5. Pending LLM calls still in flight → re-emit (crash recovery)
        for call in state.llm_calls.values() {
            if call.status == LlmCallStatus::Pending && call.deadline > now {
                if state.is_over_budget() {
                    return Ok(vec![EventPayload::BudgetExceeded.into()]);
                }
                if let Some(request) = state.build_llm_request(None) {
                    return Ok(vec![EventPayload::LlmCallRequested(LlmCallRequested {
                        call_id: call.call_id.clone(),
                        request,
                        stream: true,
                        deadline: call.deadline,
                    })
                    .into()]);
                }
            }
        }

        // 6. All tools done, no next step → request next LLM call
        let last_is_tool = state.messages.last().is_some_and(|m| m.role == Role::Tool);
        if state.pending_tool_results() == 0
            && last_is_tool
            && !state
                .llm_calls
                .values()
                .any(|c| c.status == LlmCallStatus::Pending)
        {
            if state.is_over_budget() {
                return Ok(vec![EventPayload::BudgetExceeded.into()]);
            }
            if let Some(request) = state.build_llm_request(None) {
                return Ok(vec![EventPayload::LlmCallRequested(LlmCallRequested {
                    call_id: new_call_id(),
                    request,
                    stream: true,
                    deadline: self.llm_deadline(),
                })
                .into()]);
            }
        }

        Ok(vec![])
    }

    // -----------------------------------------------------------------------
    // SyncConversation — diff incoming history and emit events for the delta
    // -----------------------------------------------------------------------

    fn handle_sync_conversation(
        &self,
        incoming: Vec<Message>,
        stream: bool,
        ctx: &SessionContext,
    ) -> Result<Vec<Emit<EventPayload>>, SessionError> {
        let state = self;

        // Verify prefix matches current state
        let prefix_len = state.messages.len();
        if incoming.len() < prefix_len {
            return Err(SessionError::ConversationDiverged);
        }
        for (existing, incoming_msg) in state.messages.iter().zip(incoming.iter()) {
            if !messages_match(existing, incoming_msg) {
                return Err(SessionError::ConversationDiverged);
            }
        }

        // Delta = incoming messages beyond what we already have
        let delta = &incoming[prefix_len..];
        if delta.is_empty() {
            return Ok(vec![]);
        }

        let mut events = Vec::new();
        for msg in delta {
            match msg.role {
                Role::User => {
                    events.push(
                        EventPayload::MessageUser(MessageUser {
                            message: msg.clone(),
                            stream,
                        })
                        .into(),
                    );
                }
                Role::Assistant => {
                    let call_id = new_call_id();
                    events.push(
                        EventPayload::MessageAssistant(MessageAssistant {
                            call_id: call_id.clone(),
                            message: msg.clone(),
                        })
                        .into(),
                    );
                    for tc in &msg.tool_calls {
                        let meta = self.tool_call_meta(&tc.name, &tc.id, &ctx.mcp_tools);
                        events.push(
                            Emit::new(EventPayload::ToolCallRequested(ToolCallRequested {
                                tool_call_id: tc.id.clone(),
                                name: tc.name.clone(),
                                arguments: tc.arguments.clone(),
                                deadline: self.tool_deadline(meta.as_ref()),
                                handler: Default::default(),
                                meta,
                            }))
                            .with("tool.name", &tc.name)
                            .label(&tc.name),
                        );
                    }
                }
                Role::Tool => {
                    if let Some(ref tc_id) = msg.tool_call_id {
                        let name = state
                            .tool_calls
                            .get(tc_id)
                            .map(|tc| tc.name.clone())
                            .or_else(|| {
                                delta
                                    .iter()
                                    .flat_map(|m| m.tool_calls.iter())
                                    .find(|t| t.id == *tc_id)
                                    .map(|t| t.name.clone())
                            })
                            .unwrap_or_default();

                        events.push(
                            Emit::new(EventPayload::ToolCallCompleted(ToolCallCompleted {
                                tool_call_id: tc_id.clone(),
                                name: name.clone(),
                                result: msg.content.clone().unwrap_or_default(),
                            }))
                            .with("tool.name", &name)
                            .label(name),
                        );
                    }
                    events.push(
                        EventPayload::MessageTool(MessageTool {
                            message: msg.clone(),
                        })
                        .into(),
                    );
                }
                Role::System => {}
            }
        }

        Ok(events)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compare two messages for structural equality (role, content, tool_call_id,
/// and tool call IDs). Used by SyncConversation to verify the prefix.
fn messages_match(a: &Message, b: &Message) -> bool {
    if a.role != b.role {
        return false;
    }
    if a.content != b.content {
        return false;
    }
    if a.tool_call_id != b.tool_call_id {
        return false;
    }
    if a.tool_calls.len() != b.tool_calls.len() {
        return false;
    }
    a.tool_calls
        .iter()
        .zip(b.tool_calls.iter())
        .all(|(ta, tb)| ta.id == tb.id && ta.name == tb.name)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::aggregate::Aggregate;
    use crate::runtime::config::{AgentConfig, LlmConfig};
    use crate::runtime::llm::types as openai;
    use chrono::Utc;
    use uuid::Uuid;

    fn far_future() -> DateTime<Utc> {
        Utc::now() + chrono::Duration::hours(1)
    }

    fn test_agent() -> AgentConfig {
        AgentConfig {
            id: Uuid::new_v4(),
            name: "test".into(),
            description: None,
            llm: LlmConfig {
                client: "mock".into(),
                retry: Default::default(),
                params: Default::default(),
            },
            system_prompt: "test".into(),
            mcp_servers: vec![],
            strategy: Default::default(),
            token_budget: None,
            sub_agents: vec![],
            tool_result_max_bytes: None,
        }
    }

    fn test_auth() -> ClientIdentity {
        ClientIdentity {
            tenant_id: "t".into(),
            sub: None,
            attrs: Default::default(),
        }
    }

    fn mock_llm_request() -> LlmRequest {
        LlmRequest::OpenAi(openai::ChatCompletionRequest {
            model: "mock".into(),
            messages: vec![],
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
        })
    }

    fn created_state() -> Aggregate<SessionState> {
        let mut state = Aggregate::new(SessionState::new(Uuid::new_v4()));
        state.apply(
            &EventPayload::SessionCreated(Box::new(SessionCreated {
                agent: test_agent(),
                auth: test_auth(),
                on_done: None,
            })),
            1,
            Utc::now(),
        );
        state
    }

    fn apply_events(state: &mut Aggregate<SessionState>, emits: Vec<Emit<EventPayload>>) {
        let seq = state.last_applied.unwrap_or(0);
        for (s, emit) in (seq + 1..).zip(emits.iter()) {
            state.apply(&emit.event, s, Utc::now());
        }
    }

    fn default_ctx() -> SessionContext {
        SessionContext::default()
    }

    #[test]
    fn complete_llm_call_emits_message_and_tool_calls() {
        let mut state = created_state();
        let call_id = "call-1".to_string();

        // Request an LLM call
        let emits = state
            .state
            .handle(
                CommandPayload::RequestLlmCall {
                    call_id: call_id.clone(),
                    request: mock_llm_request(),
                    stream: false,
                    deadline: far_future(),
                },
                &default_ctx(),
            )
            .unwrap();
        apply_events(&mut state, emits);

        // Complete with a response that has tool calls
        let response = LlmResponse::OpenAi(openai::ChatCompletionResponse {
            id: "resp-1".into(),
            model: "mock".into(),
            choices: vec![openai::Choice {
                index: 0,
                message: openai::ChatMessage {
                    role: openai::Role::Assistant,
                    content: Some("thinking...".into()),
                    tool_calls: Some(vec![openai::ToolCall {
                        id: "tc-1".into(),
                        call_type: "function".into(),
                        function: openai::FunctionCall {
                            name: "read_file".into(),
                            arguments: r#"{"path":"foo.rs"}"#.into(),
                        },
                    }]),
                    tool_call_id: None,
                },
                finish_reason: Some("tool_calls".into()),
            }],
            usage: None,
        });

        let emits = state
            .state
            .handle(
                CommandPayload::CompleteLlmCall {
                    call_id: call_id.clone(),
                    response,
                },
                &default_ctx(),
            )
            .unwrap();

        assert_eq!(
            emits.len(),
            3,
            "expected LlmCallCompleted + MessageAssistant + ToolCallRequested"
        );
        assert!(matches!(&emits[0].event, EventPayload::LlmCallCompleted(_)));
        assert!(
            matches!(&emits[1].event, EventPayload::MessageAssistant(m) if m.message.content == Some("thinking...".into()))
        );
        assert!(
            matches!(&emits[2].event, EventPayload::ToolCallRequested(t) if t.tool_call_id == "tc-1" && t.name == "read_file")
        );
    }

    #[test]
    fn complete_tool_call_emits_message_tool() {
        let mut state = created_state();
        let call_id = "call-1".to_string();
        let tool_call_id = "tc-1".to_string();

        // Set up: LLM call -> complete (emits assistant message + tool call requested)
        let emits = state
            .state
            .handle(
                CommandPayload::RequestLlmCall {
                    call_id: call_id.clone(),
                    request: mock_llm_request(),
                    stream: false,
                    deadline: far_future(),
                },
                &default_ctx(),
            )
            .unwrap();
        apply_events(&mut state, emits);

        let response = LlmResponse::OpenAi(openai::ChatCompletionResponse {
            id: "resp-1".into(),
            model: "mock".into(),
            choices: vec![openai::Choice {
                index: 0,
                message: openai::ChatMessage {
                    role: openai::Role::Assistant,
                    content: None,
                    tool_calls: Some(vec![openai::ToolCall {
                        id: tool_call_id.clone(),
                        call_type: "function".into(),
                        function: openai::FunctionCall {
                            name: "test".into(),
                            arguments: "{}".into(),
                        },
                    }]),
                    tool_call_id: None,
                },
                finish_reason: Some("tool_calls".into()),
            }],
            usage: None,
        });

        let emits = state
            .state
            .handle(
                CommandPayload::CompleteLlmCall {
                    call_id: call_id.clone(),
                    response,
                },
                &default_ctx(),
            )
            .unwrap();
        apply_events(&mut state, emits);

        // Complete the tool call
        let emits = state
            .state
            .handle(
                CommandPayload::CompleteToolCall {
                    tool_call_id: tool_call_id.clone(),
                    name: "test".into(),
                    result: "ok".into(),
                },
                &default_ctx(),
            )
            .unwrap();

        assert_eq!(emits.len(), 2, "expected ToolCallCompleted + MessageTool");
        assert!(
            matches!(&emits[0].event, EventPayload::ToolCallCompleted(t) if t.tool_call_id == "tc-1")
        );
        assert!(
            matches!(&emits[1].event, EventPayload::MessageTool(m) if m.message.content == Some("ok".into()))
        );
    }

    #[test]
    fn interrupt_command_produces_session_interrupted_event() {
        let state = created_state();

        let emits = state
            .state
            .handle(
                CommandPayload::Interrupt {
                    interrupt_id: "int-1".into(),
                    reason: "approval_needed".into(),
                    payload: serde_json::json!({"tool": "delete_file"}),
                },
                &default_ctx(),
            )
            .unwrap();
        assert_eq!(emits.len(), 1);
        assert!(
            matches!(&emits[0].event, EventPayload::SessionInterrupted(p) if p.interrupt_id == "int-1")
        );
    }

    #[test]
    fn resume_interrupt_with_matching_id_produces_event() {
        let mut state = created_state();

        // Interrupt first
        let emits = state
            .state
            .handle(
                CommandPayload::Interrupt {
                    interrupt_id: "int-1".into(),
                    reason: "approval_needed".into(),
                    payload: serde_json::json!({}),
                },
                &default_ctx(),
            )
            .unwrap();
        apply_events(&mut state, emits);

        // Resume with matching ID
        let emits = state
            .state
            .handle(
                CommandPayload::ResumeInterrupt {
                    interrupt_id: "int-1".into(),
                    payload: serde_json::json!({"approved": true}),
                },
                &default_ctx(),
            )
            .unwrap();
        assert_eq!(emits.len(), 1);
        assert!(
            matches!(&emits[0].event, EventPayload::InterruptResumed(p) if p.interrupt_id == "int-1")
        );
    }

    #[test]
    fn resume_interrupt_with_wrong_id_is_skipped() {
        let mut state = created_state();

        // Interrupt first
        let emits = state
            .state
            .handle(
                CommandPayload::Interrupt {
                    interrupt_id: "int-1".into(),
                    reason: "approval_needed".into(),
                    payload: serde_json::json!({}),
                },
                &default_ctx(),
            )
            .unwrap();
        apply_events(&mut state, emits);

        // Resume with wrong ID
        let emits = state
            .state
            .handle(
                CommandPayload::ResumeInterrupt {
                    interrupt_id: "int-WRONG".into(),
                    payload: serde_json::json!({}),
                },
                &default_ctx(),
            )
            .unwrap();
        assert!(
            emits.is_empty(),
            "ResumeInterrupt with wrong ID should produce no events"
        );
    }

    #[test]
    fn request_llm_call_over_budget_emits_budget_exceeded() {
        let mut agent = test_agent();
        agent.token_budget = Some(100);

        let mut state = Aggregate::new(SessionState::new(Uuid::new_v4()));
        state.apply(
            &EventPayload::SessionCreated(Box::new(SessionCreated {
                agent,
                auth: test_auth(),
                on_done: None,
            })),
            1,
            Utc::now(),
        );

        // Simulate token usage exceeding budget
        state.state.set_token_usage_total(200);

        let emits = state
            .state
            .handle(
                CommandPayload::RequestLlmCall {
                    call_id: "call-1".into(),
                    request: mock_llm_request(),
                    stream: false,
                    deadline: far_future(),
                },
                &default_ctx(),
            )
            .unwrap();

        assert_eq!(emits.len(), 1);
        assert!(
            matches!(&emits[0].event, EventPayload::BudgetExceeded),
            "expected BudgetExceeded, got {:?}",
            emits[0].event,
        );
    }

    #[test]
    fn send_user_message_during_interrupt_returns_error() {
        let mut state = created_state();

        // Interrupt first
        let emits = state
            .state
            .handle(
                CommandPayload::Interrupt {
                    interrupt_id: "int-1".into(),
                    reason: "approval_needed".into(),
                    payload: serde_json::json!({}),
                },
                &default_ctx(),
            )
            .unwrap();
        apply_events(&mut state, emits);

        // SendMessage with user content should fail
        let result = state.state.handle(
            CommandPayload::SendMessage {
                message: IncomingMessage::User {
                    content: "hello".into(),
                },
                stream: true,
            },
            &default_ctx(),
        );
        assert!(matches!(result, Err(SessionError::SessionInterrupted)));
    }

    // -----------------------------------------------------------------------
    // truncate_tool_result tests
    // -----------------------------------------------------------------------

    #[test]
    fn truncate_none_limit_returns_unchanged() {
        let input = "x".repeat(1_000_000);
        let result = truncate_tool_result(input.clone(), None);
        assert_eq!(result, input);
    }

    #[test]
    fn truncate_under_limit_returns_unchanged() {
        let input = "hello world".to_string();
        let result = truncate_tool_result(input.clone(), Some(100));
        assert_eq!(result, input);
    }

    #[test]
    fn truncate_at_limit_returns_unchanged() {
        let input = "x".repeat(100);
        let result = truncate_tool_result(input.clone(), Some(100));
        assert_eq!(result, input);
    }

    #[test]
    fn truncate_over_limit() {
        let input = "x".repeat(200);
        let result = truncate_tool_result(input, Some(100));
        assert!(result.starts_with(&"x".repeat(100)));
        assert!(result.contains("Output truncated"));
        assert!(result.contains("200 bytes"));
        assert!(result.contains("limit: 100 bytes"));
    }

    #[test]
    fn truncate_respects_utf8_boundary() {
        // Each emoji is 4 bytes; 3 emojis = 12 bytes
        let input = "\u{1F600}\u{1F600}\u{1F600}".to_string();
        assert_eq!(input.len(), 12);
        // Limit at 5 — should cut to 4 bytes (1 complete emoji)
        let result = truncate_tool_result(input, Some(5));
        assert!(result.starts_with("\u{1F600}"));
        assert!(result.contains("Output truncated"));
    }

    #[test]
    fn complete_tool_call_truncates_large_result() {
        let mut agent = test_agent();
        agent.tool_result_max_bytes = Some(50);

        let mut state = Aggregate::new(SessionState::new(Uuid::new_v4()));
        state.apply(
            &EventPayload::SessionCreated(Box::new(SessionCreated {
                agent,
                auth: test_auth(),
                on_done: None,
            })),
            1,
            Utc::now(),
        );

        let call_id = "call-1".to_string();
        let tool_call_id = "tc-1".to_string();

        // Request LLM call
        let emits = state
            .state
            .handle(
                CommandPayload::RequestLlmCall {
                    call_id: call_id.clone(),
                    request: mock_llm_request(),
                    stream: false,
                    deadline: far_future(),
                },
                &default_ctx(),
            )
            .unwrap();
        apply_events(&mut state, emits);

        // Complete LLM call with a tool call
        let response = LlmResponse::OpenAi(openai::ChatCompletionResponse {
            id: "resp-1".into(),
            model: "mock".into(),
            choices: vec![openai::Choice {
                index: 0,
                message: openai::ChatMessage {
                    role: openai::Role::Assistant,
                    content: None,
                    tool_calls: Some(vec![openai::ToolCall {
                        id: tool_call_id.clone(),
                        call_type: "function".into(),
                        function: openai::FunctionCall {
                            name: "test_tool".into(),
                            arguments: "{}".into(),
                        },
                    }]),
                    tool_call_id: None,
                },
                finish_reason: Some("tool_calls".into()),
            }],
            usage: None,
        });
        let emits = state
            .state
            .handle(
                CommandPayload::CompleteLlmCall { call_id, response },
                &default_ctx(),
            )
            .unwrap();
        apply_events(&mut state, emits);

        // Complete tool call with a large result
        let large_result = "x".repeat(200);
        let emits = state
            .state
            .handle(
                CommandPayload::CompleteToolCall {
                    tool_call_id: tool_call_id.clone(),
                    name: "test_tool".into(),
                    result: large_result,
                },
                &default_ctx(),
            )
            .unwrap();

        // Verify the MessageTool content is truncated
        let tool_msg = emits
            .iter()
            .find(|e| matches!(&e.event, EventPayload::MessageTool(_)));
        assert!(tool_msg.is_some(), "expected MessageTool event");
        if let EventPayload::MessageTool(m) = &tool_msg.unwrap().event {
            let content = m.message.content.as_ref().unwrap();
            assert!(content.len() < 200);
            assert!(content.contains("Output truncated"));
        }
    }
}
