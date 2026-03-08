use chrono::{DateTime, Utc};

use super::decision::{
    DecisionTrigger, WorkerDecisionCompleted, WorkerDecisionRequested, WorkerStateUpdated,
};
use super::state::{
    new_call_id, LlmCallStatus, SessionContext, SessionState, SessionStatus, ToolCallState,
    ToolCallStatus,
};
use super::types::*;
use crate::runtime::aggregate::Emit;
use crate::runtime::budget;
use crate::runtime::config::ClientIdentity;
use crate::runtime::event::EventPayload;
use crate::runtime::llm::{
    LlmCallCompleted, LlmCallErrored, LlmCallRequested, LlmRequest, LlmResponse,
};
use crate::runtime::message::{Message, Role};
use crate::runtime::span::SpanContext;
use crate::worker as proto;

// ---------------------------------------------------------------------------
// Tool result truncation
// ---------------------------------------------------------------------------

/// Truncate a tool result string if it exceeds the limit.
///
/// `max_bytes = None` means no limit (disabled). Truncation respects UTF-8
/// char boundaries and appends a note with the original and limit sizes.
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
        agent_name: String,
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
        llm_client: String,
        timeout_secs: Option<u32>,
        max_retries: Option<u32>,
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
        handler: ToolHandler,
        /// Opaque context from the worker, passed through to transport dispatch.
        context: serde_json::Value,
        timeout_secs: Option<u32>,
        max_retries: Option<u32>,
    },
    CompleteToolCall {
        tool_call_id: String,
        name: String,
        result: String,
        /// Optional updated worker state returned alongside the tool result.
        worker_state: Option<Vec<u8>>,
    },
    FailToolCall {
        tool_call_id: String,
        name: String,
        error: String,
        /// Optional updated worker state returned alongside the tool error.
        worker_state: Option<Vec<u8>>,
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
    TriggerWorkerDecision {
        trigger: DecisionTrigger,
    },
    SubmitWorkerDecision {
        decision_id: String,
        actions: Vec<proto::WorkerAction>,
        /// Opaque worker state bytes — passed through without interpretation.
        state: Vec<u8>,
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
        match (&self.agent_name, cmd) {
            (
                None,
                CommandPayload::CreateSession {
                    agent_name,
                    auth,
                    on_done,
                },
            ) => Ok(vec![Emit::new(EventPayload::SessionCreated(Box::new(
                SessionCreated {
                    agent_name: agent_name.clone(),
                    auth,
                    on_done,
                },
            )))
            .label(&agent_name)]),
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
                        let agent_name = state.agent_name.as_deref().unwrap_or("unknown");
                        Ok(vec![Emit::new(EventPayload::MessageUser(MessageUser {
                            message: Message {
                                role: Role::User,
                                content: Some(content),
                                tool_calls: Vec::new(),
                                tool_call_id: None,
                                call_id: None,
                                usage: None,
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
                            status:
                                ToolCallStatus::Completed
                                | ToolCallStatus::Failed
                                | ToolCallStatus::RetryScheduled,
                            ..
                        }) => Ok(vec![]),
                        // We are expecting a result
                        Some(ToolCallState {
                            status: ToolCallStatus::Pending,
                            name,
                            ..
                        }) => {
                            if let Some(err) = error {
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
                llm_client,
                timeout_secs,
                max_retries,
            } => {
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
                        llm_client,
                        timeout_secs,
                        max_retries,
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
                        let content = response.content();
                        let tool_calls = response.tool_calls();
                        let usage = response.usage().cloned();
                        let model = response.model().to_string();

                        let mut completed =
                            Emit::new(EventPayload::LlmCallCompleted(LlmCallCompleted {
                                call_id: call_id.clone(),
                                response,
                            }))
                            .with("llm.model", &model)
                            .label(&model);
                        if let Some(ref u) = usage {
                            for (k, v) in budget::flatten_usage(u) {
                                completed = completed.with(format!("llm.usage.{k}"), v.to_string());
                            }
                        }

                        // Tool calls are no longer emitted here — the worker
                        // decides which tools to execute via the decision pipeline.
                        let events = vec![
                            completed,
                            EventPayload::MessageAssistant(MessageAssistant {
                                call_id: call_id.clone(),
                                message: Message {
                                    role: Role::Assistant,
                                    content,
                                    tool_calls,
                                    tool_call_id: None,
                                    call_id: Some(call_id),
                                    usage,
                                },
                            })
                            .into(),
                        ];
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
                context,
                timeout_secs,
                max_retries,
            } => match state.tool_calls.get(&tool_call_id) {
                // Already tracked — skip
                Some(_) => Ok(vec![]),
                // New tool call
                None => Ok(vec![Emit::new(EventPayload::ToolCallRequested(
                    ToolCallRequested {
                        tool_call_id,
                        name: name.clone(),
                        arguments,
                        deadline,
                        handler,
                        context,
                        timeout_secs,
                        max_retries,
                    },
                ))
                .with("tool.name", &name)
                .label(name)]),
            },
            CommandPayload::CompleteToolCall {
                tool_call_id,
                name,
                result,
                worker_state,
            } => match state.tool_calls.get(&tool_call_id).map(|tc| &tc.status) {
                // Pending — complete and emit tool message
                Some(&ToolCallStatus::Pending) => {
                    let mut events = vec![
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
                                usage: None,
                            },
                        })
                        .into(),
                    ];
                    if let Some(ws) = worker_state {
                        events.push(
                            EventPayload::WorkerStateUpdated(WorkerStateUpdated { state: ws })
                                .into(),
                        );
                    }
                    Ok(events)
                }
                // Not pending or unknown — skip
                _ => Ok(vec![]),
            },
            CommandPayload::FailToolCall {
                tool_call_id,
                name,
                error,
                worker_state,
            } => match state.tool_calls.get(&tool_call_id).map(|tc| &tc.status) {
                // Pending — fail and emit error tool message
                Some(&ToolCallStatus::Pending) => {
                    let error_content = format!("Error: {}", error);
                    let mut events = vec![
                        Emit::new(EventPayload::ToolCallErrored(ToolCallErrored {
                            tool_call_id: tool_call_id.clone(),
                            name: name.clone(),
                            error: error.clone(),
                            retryable: false,
                        }))
                        .with("tool.name", &name)
                        .label(&name)
                        .error(&error),
                        EventPayload::MessageTool(MessageTool {
                            message: Message {
                                role: Role::Tool,
                                content: Some(error_content),
                                tool_calls: Vec::new(),
                                tool_call_id: Some(tool_call_id),
                                call_id: None,
                                usage: None,
                            },
                        })
                        .into(),
                    ];
                    if let Some(ws) = worker_state {
                        events.push(
                            EventPayload::WorkerStateUpdated(WorkerStateUpdated { state: ws })
                                .into(),
                        );
                    }
                    Ok(events)
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
            CommandPayload::TriggerWorkerDecision { trigger } => {
                Ok(vec![Self::worker_decision_event(trigger)])
            }
            CommandPayload::SubmitWorkerDecision {
                decision_id,
                actions,
                state,
            } => {
                let mut events: Vec<Emit<EventPayload>> = vec![
                    EventPayload::WorkerDecisionCompleted(WorkerDecisionCompleted {
                        decision_id,
                        state,
                    })
                    .into(),
                ];
                for action in actions {
                    let Some(inner) = action.action else {
                        continue;
                    };
                    match inner {
                        proto::worker_action::Action::RequestLlm(r) => {
                            let Some(ref request) = r.request else {
                                continue;
                            };
                            let sub_cmd = CommandPayload::RequestLlmCall {
                                call_id: new_call_id(),
                                request: request.into(),
                                stream: r.stream,
                                deadline: self.llm_deadline(),
                                llm_client: r.llm_client.clone(),
                                timeout_secs: r.timeout_secs,
                                max_retries: r.max_retries,
                            };
                            if let Ok(sub_events) = self.handle(sub_cmd, ctx) {
                                events.extend(sub_events);
                            }
                        }
                        proto::worker_action::Action::RequestToolCalls(r) => {
                            for tca in &r.tool_calls {
                                let Some(tc) = &tca.tool_call else {
                                    continue;
                                };
                                let handler = ToolHandler::default();
                                let context = tca
                                    .context
                                    .as_ref()
                                    .and_then(|v| serde_json::to_value(v).ok())
                                    .unwrap_or(serde_json::Value::Null);
                                let sub = CommandPayload::RequestToolCall {
                                    tool_call_id: tc.id.clone(),
                                    name: tc.name.clone(),
                                    arguments: tc.arguments.clone(),
                                    deadline: self.tool_deadline(),
                                    handler,
                                    context,
                                    timeout_secs: tca.timeout_secs,
                                    max_retries: tca.max_retries,
                                };
                                if let Ok(sub_events) = self.handle(sub, ctx) {
                                    events.extend(sub_events);
                                }
                            }
                        }
                        proto::worker_action::Action::Done(d) => {
                            let sub_cmd = CommandPayload::MarkDone {
                                artifacts: d.artifacts.iter().map(Into::into).collect(),
                            };
                            if let Ok(sub_events) = self.handle(sub_cmd, ctx) {
                                events.extend(sub_events);
                            }
                        }
                        proto::worker_action::Action::RequestSubAgent(r) => {
                            let sub_cmd = CommandPayload::RequestToolCall {
                                tool_call_id: new_call_id(),
                                name: r.agent_name.clone(),
                                arguments: serde_json::json!({"message": r.message}).to_string(),
                                deadline: self.tool_deadline(),
                                handler: ToolHandler::SubAgent,
                                context: serde_json::Value::Null,
                                timeout_secs: None,
                                max_retries: None,
                            };
                            if let Ok(sub_events) = self.handle(sub_cmd, ctx) {
                                events.extend(sub_events);
                            }
                        }
                    }
                }
                Ok(events)
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

    /// Build a WorkerDecisionRequested emit from a trigger.
    fn worker_decision_event(trigger: DecisionTrigger) -> Emit<EventPayload> {
        EventPayload::WorkerDecisionRequested(WorkerDecisionRequested {
            decision_id: new_call_id(),
            trigger,
        })
        .into()
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

        // 2. Timed-out pending tool calls → fail
        for tc in state.tool_calls.values() {
            if tc.status == ToolCallStatus::Pending
                && tc.deadline <= now
                && tc.handler != ToolHandler::Client
            {
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
                                usage: None,
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
                        return Ok(vec![EventPayload::LlmCallRequested(LlmCallRequested {
                            call_id: call.call_id.clone(),
                            request: call.request.clone(),
                            stream: call.stream,
                            deadline: self.llm_deadline(),
                            llm_client: call.llm_client.clone(),
                            timeout_secs: None,
                            max_retries: None,
                        })
                        .into()]);
                    }
                }
            }
        }

        // 3b. RetryScheduled tool call with next_at passed → re-issue
        for tc in state.tool_calls.values() {
            if tc.status == ToolCallStatus::RetryScheduled {
                if let Some(next_at) = tc.retry.next_at {
                    if next_at <= now {
                        return Ok(vec![Emit::new(EventPayload::ToolCallRequested(
                            ToolCallRequested {
                                tool_call_id: tc.tool_call_id.clone(),
                                name: tc.name.clone(),
                                arguments: tc.arguments.clone(),
                                deadline: self.tool_deadline(),
                                handler: tc.handler.clone(),
                                context: tc.context.clone(),
                                timeout_secs: None,
                                max_retries: None,
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
                        arguments: tc.arguments.clone(),
                        deadline: tc.deadline,
                        handler: tc.handler.clone(),
                        context: tc.context.clone(),
                        timeout_secs: None,
                        max_retries: None,
                    },
                ))
                .with("tool.name", &tc.name)
                .label(&tc.name)]);
            }
        }

        // 5. Pending LLM calls still in flight → re-emit (crash recovery)
        for call in state.llm_calls.values() {
            if call.status == LlmCallStatus::Pending && call.deadline > now {
                return Ok(vec![EventPayload::LlmCallRequested(LlmCallRequested {
                    call_id: call.call_id.clone(),
                    request: call.request.clone(),
                    stream: call.stream,
                    deadline: call.deadline,
                    llm_client: call.llm_client.clone(),
                    timeout_secs: None,
                    max_retries: None,
                })
                .into()]);
            }
        }

        // 6. All tools done, no next step → trigger worker decision (stall recovery)
        if state.all_tools_resolved() && !state.has_pending_llm() {
            return Ok(vec![Self::worker_decision_event(DecisionTrigger::Stall)]);
        }

        Ok(vec![])
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::aggregate::Aggregate;
    use crate::runtime::llm::openai;
    use chrono::Utc;
    use uuid::Uuid;

    fn far_future() -> DateTime<Utc> {
        Utc::now() + chrono::Duration::hours(1)
    }

    fn test_auth() -> ClientIdentity {
        ClientIdentity {
            tenant_id: "t".into(),
            sub: None,
            attrs: Default::default(),
        }
    }

    fn mock_llm_request() -> LlmRequest {
        LlmRequest {
            model: "mock".into(),
            messages: vec![],
            tools: None,
            temperature: None,
            max_completion_tokens: None,
        }
    }

    fn created_state() -> Aggregate<SessionState> {
        let mut state = Aggregate::new(SessionState::new(Uuid::new_v4()));
        state.apply(
            &EventPayload::SessionCreated(Box::new(SessionCreated {
                agent_name: "test".into(),
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
        use crate::runtime::session::system::SessionSystem;

        SessionContext {
            session_id: Uuid::nil(),
            auth: ClientIdentity {
                tenant_id: String::new(),
                sub: None,
                attrs: Default::default(),
            },
            stream: false,
            system: SessionSystem::for_test(),
        }
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
                    llm_client: "mock".into(),
                    timeout_secs: None,
                    max_retries: None,
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
            2,
            "expected LlmCallCompleted + MessageAssistant (tool calls come from worker pipeline)"
        );
        assert!(matches!(&emits[0].event, EventPayload::LlmCallCompleted(_)));
        assert!(
            matches!(&emits[1].event, EventPayload::MessageAssistant(m) if m.message.content == Some("thinking...".into()))
        );
        // Tool call requests now come from SubmitWorkerDecision, not from CompleteLlmCall
        let msg = match &emits[1].event {
            EventPayload::MessageAssistant(m) => &m.message,
            _ => panic!("expected MessageAssistant"),
        };
        assert_eq!(msg.tool_calls.len(), 1);
        assert_eq!(msg.tool_calls[0].name, "read_file");
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
                    llm_client: "mock".into(),
                    timeout_secs: None,
                    max_retries: None,
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

        // Register the tool call (previously done by CompleteLlmCall, now via worker pipeline)
        let emits = state
            .state
            .handle(
                CommandPayload::RequestToolCall {
                    tool_call_id: tool_call_id.clone(),
                    name: "test".into(),
                    arguments: "{}".into(),
                    deadline: far_future(),
                    handler: ToolHandler::Worker,
                    context: serde_json::Value::Null,
                    timeout_secs: None,
                    max_retries: None,
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
                    worker_state: None,
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
}
