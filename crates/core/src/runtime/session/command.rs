use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;

use super::decision::{ClientPayload, DecisionTrigger, ToolResult, WorkerAction};
use super::events::*;
use super::message::{Content, ContentPart, ImageUrl, Message, Role};
use super::state::{new_call_id, EffectStatus, SessionState, SessionStatus};
use crate::runtime::aggregate::Caller;
use crate::runtime::llm::{ErrorCode, LlmRequest, LlmResponse};
use crate::runtime::owner::SessionOwner;
use crate::runtime::retry::RetryPolicy;

#[derive(Debug, Clone)]
pub enum CommandPayload {
    CreateSession {
        agent_id: String,
        owner: SessionOwner,
        ancestry: Vec<String>,
        worker_retry: RetryPolicy,
    },
    SubmitClientPayload {
        payload: ClientPayload,
        turn_id: Option<String>,
    },
    SendMessage {
        message: Message,
        #[allow(dead_code)]
        stream: bool,
        turn_id: Option<String>,
    },
    RequestLlmCall {
        call_id: String,
        request: LlmRequest,
        stream: bool,
        retry: RetryPolicy,
        handler: LlmHandler,
    },
    CompleteLlmCall {
        call_id: String,
        attempt: u32,
        response: LlmResponse,
    },
    FailLlmCall {
        call_id: String,
        attempt: u32,
        error: String,
        retryable: bool,
        code: Option<ErrorCode>,
        detail: Option<serde_json::Value>,
    },
    RequestToolCall {
        tool_call_id: String,
        name: String,
        arguments: String,
        handler: ToolHandler,
        retry: RetryPolicy,
    },
    CompleteToolCall {
        tool_call_id: String,
        attempt: u32,
        result: String,
        worker_state: Option<Vec<u8>>,
    },
    FailToolCall {
        tool_call_id: String,
        attempt: u32,
        error: String,
        retryable: bool,
        worker_state: Option<Vec<u8>>,
    },
    RequestSubAgent {
        session_id: String,
        agent_id: String,
        retry: RetryPolicy,
    },
    StartSubAgent {
        session_id: String,
    },
    FailSubAgent {
        session_id: String,
        error: String,
        retryable: bool,
    },
    CompleteSubAgentTurn {
        session_id: String,
        agent_id: String,
        turn_id: String,
        data: serde_json::Value,
        cost: Decimal,
        token_usage: BTreeMap<String, u64>,
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
    SubmitWorkerDecision {
        decision_id: String,
        actions: Vec<WorkerAction>,
        state: Vec<u8>,
    },
    FailWorkerDecision {
        decision_id: String,
        error: String,
        retryable: bool,
    },
    CancelSession,
    MarkDone {
        data: serde_json::Value,
    },
    Wake {
        now: DateTime<Utc>,
    },
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum SessionError {
    #[error("session has not been created")]
    SessionNotCreated,
    #[error("session already exists")]
    SessionAlreadyCreated,
    #[error("session is interrupted")]
    SessionInterrupted,
    #[error("turn already active: {turn_id}")]
    TurnAlreadyActive { turn_id: String },
    #[error("turn already completed: {turn_id}")]
    TurnAlreadyCompleted { turn_id: String },
    #[error("client subject is required")]
    MissingSubject,
    #[error("session access denied")]
    SessionAccessDenied,
    #[error("tool call not found")]
    ToolCallNotFound,
    #[error("tool call is not pending")]
    ToolCallNotPending,
    #[error("tool call attempt mismatch")]
    ToolCallAttemptMismatch,
    #[error("client may only complete client-handled tool calls")]
    ToolCallWrongHandler,
}

impl SessionState {
    /// Reject anything but [`Caller::System`]. For commands only ever
    /// dispatched by internal processors or recursive expansions.
    fn ensure_internal(caller: &Caller) -> Result<(), SessionError> {
        match caller {
            Caller::System { .. } => Ok(()),
            _ => Err(SessionError::SessionAccessDenied),
        }
    }

    /// Allow [`Caller::Machine`] and [`Caller::System`] only. For
    /// worker-flavored commands (worker decisions, admin actions) that
    /// end users should never reach.
    fn ensure_machine_or_system(caller: &Caller) -> Result<(), SessionError> {
        match caller {
            Caller::System { .. } | Caller::Machine { .. } => Ok(()),
            Caller::Frontend { .. } => Err(SessionError::SessionAccessDenied),
        }
    }

    /// Verify that the caller's tenant matches the given session-tenant.
    /// Catches transport bugs where a route builds a [`Caller`] with a
    /// different tenant than the one the session was created under.
    fn ensure_tenant_matches(caller: &Caller, tenant_id: &str) -> Result<(), SessionError> {
        match caller {
            Caller::System { .. } => Ok(()),
            Caller::Machine {
                tenant_id: caller_tenant,
                ..
            }
            | Caller::Frontend {
                tenant_id: caller_tenant,
                ..
            } => {
                if caller_tenant != tenant_id {
                    return Err(SessionError::SessionAccessDenied);
                }
                Ok(())
            }
        }
    }

    /// Verify that the caller is allowed to act on this session.
    /// `Frontend` callers must own the session (matching `user_id`).
    /// `Machine` and `System` callers are unconstrained.
    fn ensure_owns_session(&self, caller: &Caller) -> Result<(), SessionError> {
        match caller {
            Caller::System { .. } | Caller::Machine { .. } => Ok(()),
            Caller::Frontend { user_id, .. } => {
                let owner = self
                    .owner
                    .as_ref()
                    .ok_or(SessionError::SessionAccessDenied)?;
                if owner.id.as_deref() != Some(user_id.as_str()) {
                    return Err(SessionError::SessionAccessDenied);
                }
                Ok(())
            }
        }
    }

    fn check_tool_call_caller(
        &self,
        tc: &super::state::ToolCallState,
        caller: &Caller,
    ) -> Result<(), SessionError> {
        self.ensure_owns_session(caller)?;
        if matches!(caller, Caller::Frontend { .. }) && tc.handler != ToolHandler::Client {
            return Err(SessionError::ToolCallWrongHandler);
        }
        Ok(())
    }

    pub fn handle(
        &self,
        cmd: CommandPayload,
        caller: &Caller,
    ) -> Result<Vec<EventPayload>, SessionError> {
        match (&self.agent_id, cmd) {
            (
                None,
                CommandPayload::CreateSession {
                    agent_id,
                    owner,
                    ancestry,
                    worker_retry,
                },
            ) => {
                Self::ensure_tenant_matches(caller, &owner.tenant_id)?;
                if let Caller::Frontend { user_id, .. } = caller {
                    if owner.id.as_deref() != Some(user_id.as_str()) {
                        return Err(SessionError::SessionAccessDenied);
                    }
                }
                Ok(vec![EventPayload::SessionCreated(Box::new(
                    SessionCreated {
                        agent_id,
                        owner,
                        ancestry,
                        worker_retry,
                    },
                ))])
            }
            (Some(_), CommandPayload::CreateSession { .. }) => {
                Err(SessionError::SessionAlreadyCreated)
            }
            (None, _) => Err(SessionError::SessionNotCreated),
            (Some(_), cmd) => self.handle_active(cmd, caller),
        }
    }

    fn emit_decision_request(&self, trigger: DecisionTrigger) -> EventPayload {
        let decision_id = new_call_id();
        if self.has_pending_worker_decision() {
            EventPayload::DecisionRequestQueued(DecisionRequestQueued {
                decision_id,
                trigger,
            })
        } else {
            EventPayload::WorkerDecisionRequested(WorkerDecisionRequested {
                decision_id,
                trigger,
            })
        }
    }

    fn handle_active(
        &self,
        cmd: CommandPayload,
        caller: &Caller,
    ) -> Result<Vec<EventPayload>, SessionError> {
        if let Some(owner) = self.owner.as_ref() {
            Self::ensure_tenant_matches(caller, &owner.tenant_id)?;
        }
        match cmd {
            CommandPayload::CreateSession { .. } => Err(SessionError::SessionAlreadyCreated),

            CommandPayload::SubmitClientPayload { payload, turn_id } => {
                self.ensure_owns_session(caller)?;

                if let Some(ref tid) = turn_id {
                    if self.completed_turn_ids.contains(tid) {
                        return Err(SessionError::TurnAlreadyCompleted {
                            turn_id: tid.clone(),
                        });
                    }
                    if self.turn_id.as_ref() == Some(tid) {
                        return Err(SessionError::TurnAlreadyActive {
                            turn_id: tid.clone(),
                        });
                    }
                }

                let mut events: Vec<EventPayload> = Vec::new();
                if let Some(turn_id) = turn_id {
                    events.push(EventPayload::TurnStarted(TurnStarted { turn_id }));
                }

                match payload {
                    ClientPayload::Message { message, stream } => {
                        if matches!(self.status, SessionStatus::Interrupted { .. })
                            && message.role == Role::User
                        {
                            return Err(SessionError::SessionInterrupted);
                        }
                        events.push(EventPayload::NewMessage(NewMessage {
                            message: message.clone(),
                        }));
                        if message.role == Role::User {
                            events.push(self.emit_decision_request(DecisionTrigger::UserMessage {
                                stream,
                                message,
                            }));
                        }
                    }
                    ClientPayload::Action { action } => {
                        events.push(self.emit_decision_request(DecisionTrigger::ClientAction {
                            name: action.name,
                            args: action.args,
                        }));
                    }
                }

                Ok(events)
            }

            CommandPayload::SendMessage {
                message,
                stream,
                turn_id,
            } => {
                Self::ensure_internal(caller)?;
                // Idempotency guard: reject if this turn_id was already seen
                if let Some(ref tid) = turn_id {
                    if self.completed_turn_ids.contains(tid) {
                        return Err(SessionError::TurnAlreadyCompleted {
                            turn_id: tid.clone(),
                        });
                    }
                    if self.turn_id.as_ref() == Some(tid) {
                        return Err(SessionError::TurnAlreadyActive {
                            turn_id: tid.clone(),
                        });
                    }
                }

                match self.status {
                    SessionStatus::Interrupted { .. } if message.role == Role::User => {
                        Err(SessionError::SessionInterrupted)
                    }
                    _ => {
                        let mut events = vec![EventPayload::NewMessage(NewMessage {
                            message: message.clone(),
                        })];
                        if let Some(turn_id) = turn_id {
                            events.push(EventPayload::TurnStarted(TurnStarted { turn_id }));
                        }
                        if message.role == Role::User {
                            events.push(self.emit_decision_request(DecisionTrigger::UserMessage {
                                stream,
                                message,
                            }));
                        }
                        Ok(events)
                    }
                }
            }

            CommandPayload::RequestLlmCall {
                call_id,
                request,
                stream,
                retry,
                handler,
            } => {
                Self::ensure_internal(caller)?;
                if self.has_pending_llm() {
                    return Ok(vec![]);
                }
                let issue = matches!(
                    self.llm_calls.get(&call_id).map(|c| &c.tracking.status),
                    None | Some(&EffectStatus::Failed) | Some(&EffectStatus::RetryScheduled)
                );
                if issue {
                    let mut events = vec![EventPayload::LlmCallRequested(LlmCallRequested {
                        call_id: call_id.clone(),
                        attempt: 0,
                        request: request.clone(),
                        stream,
                        retry,
                        handler: handler.clone(),
                    })];
                    if handler == LlmHandler::Worker {
                        events.push(self.emit_decision_request(DecisionTrigger::LlmRequest {
                            call_id,
                            request,
                            stream,
                            attempt: 0,
                        }));
                    }
                    Ok(events)
                } else {
                    Ok(vec![])
                }
            }

            CommandPayload::CompleteLlmCall {
                call_id,
                attempt,
                response,
            } => {
                Self::ensure_internal(caller)?;
                match self.llm_calls.get(&call_id).map(|c| &c.tracking.status) {
                    Some(&EffectStatus::Pending) => {
                        let truncated = response.finish_reason.as_deref() == Some("length");
                        let usage = response.usage.clone();
                        let cost = response.cost;
                        let tool_calls = if response.tool_calls.is_empty() {
                            None
                        } else {
                            Some(response.tool_calls.clone())
                        };
                        let content = if response.images.is_empty() {
                            response.content.clone().map(Content::Text)
                        } else {
                            let mut parts: Vec<ContentPart> = Vec::new();
                            if let Some(text) = &response.content {
                                parts.push(ContentPart::Text { text: text.clone() });
                            }
                            for img in &response.images {
                                parts.push(ContentPart::ImageUrl {
                                    image_url: ImageUrl {
                                        url: img.url.clone(),
                                    },
                                });
                            }
                            Some(Content::Parts(parts))
                        };
                        let message = Message {
                            role: Role::Assistant,
                            content,
                            tool_calls,
                            tool_call_id: None,
                            name: None,
                        };
                        Ok(vec![
                            EventPayload::LlmCallCompleted(LlmCallCompleted {
                                call_id: call_id.clone(),
                                attempt,
                                response,
                            }),
                            EventPayload::NewMessage(NewMessage {
                                message: message.clone(),
                            }),
                            self.emit_decision_request(DecisionTrigger::LlmResponse {
                                call_id,
                                message,
                                truncated,
                                usage,
                                cost,
                            }),
                        ])
                    }
                    _ => Ok(vec![]),
                }
            }

            CommandPayload::FailLlmCall {
                call_id,
                attempt,
                error,
                retryable,
                code,
                detail,
            } => {
                Self::ensure_internal(caller)?;
                let Some(call) = self.llm_calls.get(&call_id) else {
                    return Ok(vec![]);
                };
                if call.tracking.status != EffectStatus::Pending {
                    return Ok(vec![]);
                }
                let mut events = vec![EventPayload::LlmCallErrored(LlmCallErrored {
                    call_id: call_id.clone(),
                    attempt,
                    error: error.clone(),
                    retryable,
                    code: code.clone(),
                    detail: detail.clone(),
                })];
                if call
                    .tracking
                    .retry_policy
                    .exhausted(&call.tracking.retry, retryable)
                {
                    events.push(self.emit_decision_request(DecisionTrigger::LlmError {
                        call_id,
                        error,
                        code,
                        detail,
                    }));
                }
                Ok(events)
            }

            CommandPayload::RequestToolCall {
                tool_call_id,
                name,
                arguments,
                handler,
                retry,
            } => {
                Self::ensure_internal(caller)?;
                match self.tool_calls.get(&tool_call_id) {
                    Some(_) => Ok(vec![]),
                    None => {
                        let mut events = vec![EventPayload::ToolCallRequested(ToolCallRequested {
                            tool_call_id: tool_call_id.clone(),
                            attempt: 0,
                            name: name.clone(),
                            arguments: arguments.clone(),
                            handler: handler.clone(),
                            retry: retry.clone(),
                        })];
                        if handler == ToolHandler::Worker {
                            events.push(self.emit_decision_request(DecisionTrigger::ToolExecute {
                                tool_call_id,
                                name,
                                arguments,
                                attempt: 0,
                                deadline: retry.deadline(chrono::Utc::now()),
                            }));
                        }
                        Ok(events)
                    }
                }
            }

            CommandPayload::CompleteToolCall {
                tool_call_id,
                attempt,
                result,
                worker_state,
            } => {
                let tc = self
                    .tool_calls
                    .get(&tool_call_id)
                    .ok_or(SessionError::ToolCallNotFound)?;
                if tc.tracking.status != EffectStatus::Pending {
                    return Err(SessionError::ToolCallNotPending);
                }
                if tc.tracking.retry.attempts != attempt {
                    return Err(SessionError::ToolCallAttemptMismatch);
                }
                self.check_tool_call_caller(tc, caller)?;
                let name = tc.name.clone();
                let mut events = vec![
                    EventPayload::ToolCallCompleted(ToolCallCompleted {
                        tool_call_id: tool_call_id.clone(),
                        name: name.clone(),
                        result: result.clone(),
                    }),
                    EventPayload::NewMessage(NewMessage {
                        message: Message {
                            role: Role::Tool,
                            content: Some(Content::Text(result.clone())),
                            tool_calls: None,
                            tool_call_id: Some(tool_call_id.clone()),
                            name: Some(name.clone()),
                        },
                    }),
                ];
                if let Some(ws) = worker_state {
                    events.push(EventPayload::WorkerStateUpdated(WorkerStateUpdated {
                        state: ws,
                    }));
                }
                events.push(self.emit_decision_request(DecisionTrigger::ToolResult {
                    result: ToolResult {
                        tool_call_id,
                        name,
                        content: result,
                        is_error: false,
                    },
                }));
                Ok(events)
            }

            CommandPayload::FailToolCall {
                tool_call_id,
                attempt,
                error,
                retryable,
                worker_state,
            } => {
                let tc = self
                    .tool_calls
                    .get(&tool_call_id)
                    .ok_or(SessionError::ToolCallNotFound)?;
                if tc.tracking.status != EffectStatus::Pending {
                    return Err(SessionError::ToolCallNotPending);
                }
                if tc.tracking.retry.attempts != attempt {
                    return Err(SessionError::ToolCallAttemptMismatch);
                }
                self.check_tool_call_caller(tc, caller)?;
                let name = tc.name.clone();
                let mut events = vec![EventPayload::ToolCallErrored(ToolCallErrored {
                    tool_call_id: tool_call_id.clone(),
                    name: name.clone(),
                    error: error.clone(),
                    retryable,
                })];
                if let Some(ws) = worker_state {
                    events.push(EventPayload::WorkerStateUpdated(WorkerStateUpdated {
                        state: ws,
                    }));
                }
                if tc
                    .tracking
                    .retry_policy
                    .exhausted(&tc.tracking.retry, retryable)
                {
                    events.push(self.emit_decision_request(DecisionTrigger::ToolResult {
                        result: ToolResult {
                            tool_call_id,
                            name,
                            content: error,
                            is_error: true,
                        },
                    }));
                }
                Ok(events)
            }

            CommandPayload::RequestSubAgent {
                session_id,
                agent_id,
                retry,
            } => {
                Self::ensure_internal(caller)?;
                match self.sub_agent_calls.get(&session_id) {
                    Some(_) => Ok(vec![]),
                    None => Ok(vec![EventPayload::SubAgentRequested(SubAgentRequested {
                        session_id,
                        agent_id,
                        retry,
                    })]),
                }
            }

            CommandPayload::StartSubAgent { session_id } => {
                Self::ensure_internal(caller)?;
                match self
                    .sub_agent_calls
                    .get(&session_id)
                    .map(|c| &c.tracking.status)
                {
                    Some(&EffectStatus::Pending) => {
                        Ok(vec![EventPayload::SubAgentStarted(SubAgentStarted {
                            session_id,
                        })])
                    }
                    _ => Ok(vec![]),
                }
            }

            CommandPayload::FailSubAgent {
                session_id,
                error,
                retryable,
            } => {
                Self::ensure_internal(caller)?;
                let Some(sa) = self.sub_agent_calls.get(&session_id) else {
                    return Ok(vec![]);
                };
                if sa.tracking.status != EffectStatus::Pending {
                    return Ok(vec![]);
                }
                let mut events = vec![EventPayload::SubAgentErrored(SubAgentErrored {
                    session_id: session_id.clone(),
                    error: error.clone(),
                    retryable,
                })];
                if sa
                    .tracking
                    .retry_policy
                    .exhausted(&sa.tracking.retry, retryable)
                {
                    events.push(self.emit_decision_request(DecisionTrigger::SubAgentError {
                        session_id,
                        agent_id: sa.agent_id.clone(),
                        error,
                    }));
                }
                Ok(events)
            }

            CommandPayload::CompleteSubAgentTurn {
                session_id,
                agent_id,
                turn_id,
                data,
                cost,
                token_usage,
            } => {
                Self::ensure_internal(caller)?;
                // Only fire if we know about this sub-agent
                if self.sub_agent_calls.contains_key(&session_id) {
                    Ok(vec![
                        EventPayload::SubAgentTurnCompleted(SubAgentTurnCompleted {
                            session_id: session_id.clone(),
                            cost,
                            token_usage,
                        }),
                        self.emit_decision_request(DecisionTrigger::SubAgentTurnComplete {
                            session_id,
                            agent_id,
                            turn_id,
                            data,
                        }),
                    ])
                } else {
                    Ok(vec![])
                }
            }

            CommandPayload::Interrupt {
                interrupt_id,
                reason,
                payload,
            } => {
                Self::ensure_machine_or_system(caller)?;
                match self.status {
                    SessionStatus::Interrupted { .. } => Ok(vec![]),
                    _ => Ok(vec![EventPayload::SessionInterrupted(SessionInterrupted {
                        interrupt_id,
                        reason,
                        payload,
                    })]),
                }
            }

            CommandPayload::ResumeInterrupt {
                interrupt_id,
                payload,
            } => {
                Self::ensure_machine_or_system(caller)?;
                match self.active_interrupt() {
                    Some(id) if id == interrupt_id => Ok(vec![
                        EventPayload::InterruptResumed(InterruptResumed {
                            interrupt_id: interrupt_id.clone(),
                            payload,
                        }),
                        self.emit_decision_request(DecisionTrigger::InterruptResumed {
                            interrupt_id,
                        }),
                    ]),
                    _ => Ok(vec![]),
                }
            }

            CommandPayload::SubmitWorkerDecision {
                decision_id,
                actions,
                state,
            } => {
                Self::ensure_machine_or_system(caller)?;
                match self
                    .worker_decisions
                    .get(&decision_id)
                    .map(|d| &d.tracking.status)
                {
                    Some(&EffectStatus::Pending) => {}
                    _ => return Ok(vec![]),
                }
                let mut events: Vec<EventPayload> = vec![EventPayload::WorkerDecisionCompleted(
                    WorkerDecisionCompleted { decision_id, state },
                )];
                for action in actions {
                    let sub_events = match action {
                        WorkerAction::CallLlm {
                            request,
                            stream,
                            retry,
                            handler,
                        } => self.handle(
                            CommandPayload::RequestLlmCall {
                                call_id: new_call_id(),
                                request,
                                stream,
                                retry,
                                handler,
                            },
                            &Caller::System {
                                tenant_id: "tenant-a".to_string(),
                            },
                        ),
                        WorkerAction::CallTool {
                            tool_call_id,
                            name,
                            arguments,
                            handler,
                            retry,
                        } => self.handle(
                            CommandPayload::RequestToolCall {
                                tool_call_id,
                                name,
                                arguments,
                                handler,
                                retry,
                            },
                            &Caller::System {
                                tenant_id: "tenant-a".to_string(),
                            },
                        ),
                        WorkerAction::SpawnSubAgent {
                            session_id,
                            agent_id,
                            retry,
                        } => self.handle(
                            CommandPayload::RequestSubAgent {
                                session_id,
                                agent_id,
                                retry,
                            },
                            &Caller::System {
                                tenant_id: "tenant-a".to_string(),
                            },
                        ),
                        WorkerAction::SendMessage {
                            session_id,
                            message,
                        } => Ok(vec![EventPayload::SessionMessageRequested(
                            SessionMessageRequested {
                                target_session_id: session_id,
                                message,
                            },
                        )]),
                        WorkerAction::ReturnToolResult {
                            tool_call_id,
                            result,
                            attempt,
                        } => self.handle(
                            CommandPayload::CompleteToolCall {
                                tool_call_id,
                                attempt,
                                result,
                                worker_state: None,
                            },
                            &Caller::System {
                                tenant_id: "tenant-a".to_string(),
                            },
                        ),
                        WorkerAction::ReturnToolError {
                            tool_call_id,
                            error,
                            retryable,
                            attempt,
                        } => self.handle(
                            CommandPayload::FailToolCall {
                                tool_call_id,
                                attempt,
                                error,
                                retryable,
                                worker_state: None,
                            },
                            &Caller::System {
                                tenant_id: "tenant-a".to_string(),
                            },
                        ),
                        WorkerAction::ReturnLlmResult {
                            call_id,
                            response,
                            attempt,
                        } => self.handle(
                            CommandPayload::CompleteLlmCall {
                                call_id,
                                attempt,
                                response,
                            },
                            &Caller::System {
                                tenant_id: "tenant-a".to_string(),
                            },
                        ),
                        WorkerAction::ReturnLlmError {
                            call_id,
                            error,
                            retryable,
                            code,
                            detail,
                            attempt,
                        } => self.handle(
                            CommandPayload::FailLlmCall {
                                call_id,
                                attempt,
                                error,
                                retryable,
                                code,
                                detail,
                            },
                            &Caller::System {
                                tenant_id: "tenant-a".to_string(),
                            },
                        ),
                        WorkerAction::Done { data } => self.handle(
                            CommandPayload::MarkDone { data },
                            &Caller::System {
                                tenant_id: "tenant-a".to_string(),
                            },
                        ),
                    };
                    if let Ok(sub) = sub_events {
                        events.extend(sub);
                    }
                }

                // Promote the next queued decision inline so it dispatches
                // in the same commit rather than waiting for a wake cycle.
                let next_decision = self
                    .next_queued_decision()
                    .map(|d| (d.decision_id.clone(), d.trigger.clone()))
                    .or_else(|| {
                        events.iter().find_map(|e| match e {
                            EventPayload::DecisionRequestQueued(q) => {
                                Some((q.decision_id.clone(), q.trigger.clone()))
                            }
                            _ => None,
                        })
                    });

                if let Some((decision_id, trigger)) = next_decision {
                    events.push(EventPayload::WorkerDecisionRequested(
                        WorkerDecisionRequested {
                            decision_id,
                            trigger,
                        },
                    ));
                }

                Ok(events)
            }

            CommandPayload::FailWorkerDecision {
                decision_id,
                error,
                retryable,
            } => {
                Self::ensure_machine_or_system(caller)?;
                match self
                    .worker_decisions
                    .get(&decision_id)
                    .map(|d| &d.tracking.status)
                {
                    Some(&EffectStatus::Pending) => {}
                    _ => return Ok(vec![]),
                }
                Ok(vec![EventPayload::WorkerDecisionErrored(
                    WorkerDecisionErrored {
                        decision_id,
                        error,
                        retryable,
                    },
                )])
            }

            CommandPayload::CancelSession => {
                Self::ensure_machine_or_system(caller)?;
                Ok(vec![EventPayload::SessionCancelled])
            }

            CommandPayload::MarkDone { data } => {
                Self::ensure_internal(caller)?;
                let mut events = Vec::new();
                if let Some(turn_id) = &self.turn_id {
                    events.push(EventPayload::TurnCompleted(TurnCompleted {
                        turn_id: turn_id.clone(),
                        data,
                        turn_cost: self.turn_cost,
                        turn_token_usage: self.turn_token_usage.clone(),
                    }));
                }
                events.push(EventPayload::SessionDone(SessionDone {}));
                Ok(events)
            }

            CommandPayload::Wake { now } => {
                Self::ensure_internal(caller)?;
                self.handle_wake(now)
            }
        }
    }

    fn handle_wake(&self, now: DateTime<Utc>) -> Result<Vec<EventPayload>, SessionError> {
        // 1. Timed-out pending LLM calls → fail
        for call in self.llm_calls.values() {
            if call.tracking.status == EffectStatus::Pending
                && call.tracking.deadline.is_some_and(|d| d <= now)
            {
                return Ok(vec![EventPayload::LlmCallErrored(LlmCallErrored {
                    call_id: call.call_id.clone(),
                    attempt: call.tracking.retry.attempts,
                    error: "deadline exceeded".to_string(),
                    retryable: true,
                    code: Some(ErrorCode::DeadlineExceeded),
                    detail: None,
                })]);
            }
        }

        // 2. Timed-out pending tool calls → fail
        for tc in self.tool_calls.values() {
            if tc.tracking.status == EffectStatus::Pending
                && tc.tracking.deadline.is_some_and(|d| d <= now)
            {
                return Ok(vec![EventPayload::ToolCallErrored(ToolCallErrored {
                    tool_call_id: tc.tool_call_id.clone(),
                    name: tc.name.clone(),
                    error: "deadline exceeded".to_string(),
                    retryable: true,
                })]);
            }
        }

        // 3. RetryScheduled LLM calls ready to re-issue
        for call in self.llm_calls.values() {
            if call.tracking.status == EffectStatus::RetryScheduled {
                if let Some(next_at) = call.tracking.retry.next_at {
                    if next_at <= now {
                        let mut events = vec![EventPayload::LlmCallRequested(LlmCallRequested {
                            call_id: call.call_id.clone(),
                            attempt: call.tracking.retry.attempts,
                            request: call.request.clone(),
                            stream: call.stream,
                            retry: call.tracking.retry_policy.clone(),
                            handler: call.handler.clone(),
                        })];
                        if call.handler == LlmHandler::Worker {
                            events.push(self.emit_decision_request(DecisionTrigger::LlmRequest {
                                call_id: call.call_id.clone(),
                                request: call.request.clone(),
                                stream: call.stream,
                                attempt: call.tracking.retry.attempts,
                            }));
                        }
                        return Ok(events);
                    }
                }
            }
        }

        // 4. RetryScheduled tool calls ready to re-issue
        for tc in self.tool_calls.values() {
            if tc.tracking.status == EffectStatus::RetryScheduled {
                if let Some(next_at) = tc.tracking.retry.next_at {
                    if next_at <= now {
                        let mut events = vec![EventPayload::ToolCallRequested(ToolCallRequested {
                            tool_call_id: tc.tool_call_id.clone(),
                            attempt: tc.tracking.retry.attempts,
                            name: tc.name.clone(),
                            arguments: tc.arguments.clone(),
                            handler: tc.handler.clone(),
                            retry: tc.tracking.retry_policy.clone(),
                        })];
                        if tc.handler == ToolHandler::Worker {
                            events.push(self.emit_decision_request(DecisionTrigger::ToolExecute {
                                tool_call_id: tc.tool_call_id.clone(),
                                name: tc.name.clone(),
                                arguments: tc.arguments.clone(),
                                attempt: tc.tracking.retry.attempts,
                                deadline: tc.tracking.retry_policy.deadline(now),
                            }));
                        }
                        return Ok(events);
                    }
                }
            }
        }

        // 5. Timed-out pending sub-agent calls → fail
        for sa in self.sub_agent_calls.values() {
            if sa.tracking.status == EffectStatus::Pending
                && sa.tracking.deadline.is_some_and(|d| d <= now)
            {
                return Ok(vec![EventPayload::SubAgentErrored(SubAgentErrored {
                    session_id: sa.session_id.clone(),
                    error: "deadline exceeded".to_string(),
                    retryable: true,
                })]);
            }
        }

        // 6. RetryScheduled sub-agent calls ready to re-issue
        for sa in self.sub_agent_calls.values() {
            if sa.tracking.status == EffectStatus::RetryScheduled {
                if let Some(next_at) = sa.tracking.retry.next_at {
                    if next_at <= now {
                        return Ok(vec![EventPayload::SubAgentRequested(SubAgentRequested {
                            session_id: sa.session_id.clone(),
                            agent_id: sa.agent_id.clone(),
                            retry: sa.tracking.retry_policy.clone(),
                        })]);
                    }
                }
            }
        }

        // 7. Timed-out pending worker decisions → fail
        for wd in self.worker_decisions.values() {
            if wd.tracking.status == EffectStatus::Pending
                && wd.tracking.deadline.is_some_and(|d| d <= now)
            {
                return Ok(vec![EventPayload::WorkerDecisionErrored(
                    WorkerDecisionErrored {
                        decision_id: wd.decision_id.clone(),
                        error: "deadline exceeded".to_string(),
                        retryable: true,
                    },
                )]);
            }
        }

        // 8. RetryScheduled worker decisions ready to re-issue
        for wd in self.worker_decisions.values() {
            if wd.tracking.status == EffectStatus::RetryScheduled {
                if let Some(next_at) = wd.tracking.retry.next_at {
                    if next_at <= now {
                        return Ok(vec![EventPayload::WorkerDecisionRequested(
                            WorkerDecisionRequested {
                                decision_id: wd.decision_id.clone(),
                                trigger: wd.trigger.clone(),
                            },
                        )]);
                    }
                }
            }
        }

        // 9. Promote the next queued worker decision
        if let Some(wd) = self.next_queued_decision() {
            return Ok(vec![EventPayload::WorkerDecisionRequested(
                WorkerDecisionRequested {
                    decision_id: wd.decision_id.clone(),
                    trigger: wd.trigger.clone(),
                },
            )]);
        }

        // 10. All tools done, no next step → stall recovery
        if self.all_tools_resolved()
            && !self.has_pending_llm()
            && !self.has_queued_worker_decision()
        {
            return Ok(vec![EventPayload::WorkerDecisionRequested(
                WorkerDecisionRequested {
                    decision_id: new_call_id(),
                    trigger: DecisionTrigger::Stall,
                },
            )]);
        }

        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use chrono::Utc;

    use super::*;
    use crate::runtime::aggregate::{Aggregate, Caller, CommitContext};
    use crate::runtime::llm::{LlmRequest, LlmResponse};
    use crate::runtime::owner::SessionOwner;
    use crate::runtime::retry::RetryPolicy;
    use crate::runtime::session::decision::{ClientPayload, DecisionTrigger, WorkerAction};
    use crate::runtime::session::events::{EventPayload, LlmHandler, ToolHandler};
    use crate::runtime::session::message::{Content, Message, Role};
    use crate::runtime::session::state::{EffectStatus, SessionState, SessionStatus};
    use crate::runtime::span::SpanContext;

    /// Run a command through the handler and commit the resulting events.
    /// Mirrors what `execute()` does in production (minus the store
    /// round-trip), so test setup exercises the real command + commit
    /// paths including stream version, sequence numbers, and derived
    /// state tracking. Returns the emitted event payloads so tests can
    /// pluck out generated ids (decision_id, etc.) without peeking into
    /// state.
    fn dispatch(
        agg: &mut Aggregate<SessionState>,
        cmd: CommandPayload,
        caller: &Caller,
    ) -> Vec<EventPayload> {
        let events = agg.state.handle(cmd, caller).expect("setup command failed");
        let ctx = CommitContext {
            span: SpanContext::root(),
            occurred_at: Utc::now(),
        };
        agg.commit(events.clone(), &ctx);
        events
    }

    /// Build an empty `Aggregate<SessionState>` for the given session and
    /// run `CreateSession` against it. Returns the aggregate ready for
    /// further `dispatch(...)` calls.
    fn create_session(session_id: &str, tenant_id: &str, user_id: &str) -> Aggregate<SessionState> {
        let mut agg = Aggregate::new(
            session_id.to_string(),
            tenant_id.to_string(),
            SessionState::new(session_id.to_string()),
        );
        dispatch(
            &mut agg,
            CommandPayload::CreateSession {
                agent_id: "agent-1".to_string(),
                owner: SessionOwner {
                    tenant_id: tenant_id.to_string(),
                    id: Some(user_id.to_string()),
                    metadata: HashMap::new(),
                },
                ancestry: vec![],
                worker_retry: RetryPolicy::no_retry(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        agg
    }

    #[test]
    fn frontend_can_complete_own_client_handled_tool_call() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::RequestToolCall {
                tool_call_id: "tc-1".to_string(),
                name: "my_tool".to_string(),
                arguments: "{}".to_string(),
                handler: ToolHandler::Client,
                retry: RetryPolicy::no_retry(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        let events = dispatch(
            &mut agg,
            CommandPayload::CompleteToolCall {
                tool_call_id: "tc-1".to_string(),
                attempt: 0,
                result: "ok".to_string(),
                worker_state: None,
            },
            &Caller::Frontend {
                tenant_id: "tenant-a".to_string(),
                user_id: "user-1".to_string(),
                attrs: HashMap::new(),
            },
        );

        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::ToolCallCompleted(_),
                    EventPayload::NewMessage(_),
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "expected [ToolCallCompleted, NewMessage, WorkerDecisionRequested]; got {events:?}"
        );

        let tc = agg.state.tool_calls.get("tc-1").expect("tool call present");
        assert_eq!(tc.tracking.status, EffectStatus::Completed);
        assert_eq!(tc.result.as_deref(), Some("ok"));
        assert!(!tc.is_error);
    }

    #[test]
    fn frontend_with_mismatched_user_id_is_denied() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::RequestToolCall {
                tool_call_id: "tc-1".to_string(),
                name: "my_tool".to_string(),
                arguments: "{}".to_string(),
                handler: ToolHandler::Client,
                retry: RetryPolicy::no_retry(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        let caller = Caller::Frontend {
            tenant_id: "tenant-a".to_string(),
            user_id: "other-user".to_string(),
            attrs: HashMap::new(),
        };

        let err = agg
            .state
            .handle(
                CommandPayload::CompleteToolCall {
                    tool_call_id: "tc-1".to_string(),
                    attempt: 0,
                    result: "ok".to_string(),
                    worker_state: None,
                },
                &caller,
            )
            .expect_err("mismatched user_id should be rejected");

        assert!(
            matches!(err, SessionError::SessionAccessDenied),
            "expected SessionAccessDenied; got {err:?}"
        );
    }

    #[test]
    fn frontend_cannot_complete_worker_handled_tool_call() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::RequestToolCall {
                tool_call_id: "tc-1".to_string(),
                name: "my_tool".to_string(),
                arguments: "{}".to_string(),
                handler: ToolHandler::Worker,
                retry: RetryPolicy::no_retry(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        let caller = Caller::Frontend {
            tenant_id: "tenant-a".to_string(),
            user_id: "user-1".to_string(),
            attrs: HashMap::new(),
        };

        let err = agg
            .state
            .handle(
                CommandPayload::CompleteToolCall {
                    tool_call_id: "tc-1".to_string(),
                    attempt: 0,
                    result: "ok".to_string(),
                    worker_state: None,
                },
                &caller,
            )
            .expect_err("frontend should not complete worker-handled tool calls");

        assert!(
            matches!(err, SessionError::ToolCallWrongHandler),
            "expected ToolCallWrongHandler; got {err:?}"
        );
    }

    #[test]
    fn request_tool_call_with_client_handler_does_not_queue_worker_decision() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");

        let events = dispatch(
            &mut agg,
            CommandPayload::RequestToolCall {
                tool_call_id: "tc-1".to_string(),
                name: "my_tool".to_string(),
                arguments: "{}".to_string(),
                handler: ToolHandler::Client,
                retry: RetryPolicy::no_retry(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        assert!(
            matches!(events.as_slice(), [EventPayload::ToolCallRequested(_)]),
            "client-handled tool call should emit ToolCallRequested only; got {events:?}"
        );

        let tc = agg.state.tool_calls.get("tc-1").expect("tool call present");
        assert_eq!(tc.tracking.status, EffectStatus::Pending);
        assert_eq!(tc.handler, ToolHandler::Client);
        assert_eq!(agg.state.status, SessionStatus::Idle);
    }

    #[test]
    fn request_tool_call_with_worker_handler_emits_decision_to_execute() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");

        let events = dispatch(
            &mut agg,
            CommandPayload::RequestToolCall {
                tool_call_id: "tc-1".to_string(),
                name: "my_tool".to_string(),
                arguments: "{}".to_string(),
                handler: ToolHandler::Worker,
                retry: RetryPolicy::no_retry(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::ToolCallRequested(_),
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "worker-handled tool call should also queue a worker decision; got {events:?}"
        );

        let tc = agg.state.tool_calls.get("tc-1").expect("tool call present");
        assert_eq!(tc.handler, ToolHandler::Worker);
    }

    #[test]
    fn machine_completes_worker_handled_tool_call_after_worker_releases_decision() {
        // Realistic async-tool flow: worker sees the ToolExecute decision,
        // acknowledges it with no actions ("I've handed off"), then the
        // machine that's actually running the work completes the tool
        // call out-of-band. Result emerges cleanly with a fresh follow-up
        // worker decision.
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let request_events = dispatch(
            &mut agg,
            CommandPayload::RequestToolCall {
                tool_call_id: "tc-1".to_string(),
                name: "my_tool".to_string(),
                arguments: "{}".to_string(),
                handler: ToolHandler::Worker,
                retry: RetryPolicy::no_retry(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        let d1 = request_events
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.decision_id.clone()),
                _ => None,
            })
            .expect("worker-handled tool call emits a ToolExecute decision");

        let machine = Caller::Machine {
            tenant_id: "tenant-a".to_string(),
            key_id: "prod-key-1".to_string(),
        };

        // Worker releases its decision with no actions.
        dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: d1,
                actions: vec![],
                state: vec![],
            },
            &machine,
        );

        // Now the machine completes the tool call directly.
        let events = dispatch(
            &mut agg,
            CommandPayload::CompleteToolCall {
                tool_call_id: "tc-1".to_string(),
                attempt: 0,
                result: "ok".to_string(),
                worker_state: None,
            },
            &machine,
        );

        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::ToolCallCompleted(_),
                    EventPayload::NewMessage(_),
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "expected [ToolCallCompleted, NewMessage, WorkerDecisionRequested]; got {events:?}"
        );

        let tc = agg.state.tool_calls.get("tc-1").expect("tool call present");
        assert_eq!(tc.tracking.status, EffectStatus::Completed);
    }

    #[test]
    fn machine_completes_worker_handled_tool_call_before_worker_releases_decision() {
        // Race case: tool result arrives before the worker has acknowledged
        // its ToolExecute decision. The pending decision is still in flight,
        // so the follow-up ToolResult decision is queued behind it. It will
        // be promoted later — either by the worker's eventual response or
        // by the wake cycle timing out the original decision.
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::RequestToolCall {
                tool_call_id: "tc-1".to_string(),
                name: "my_tool".to_string(),
                arguments: "{}".to_string(),
                handler: ToolHandler::Worker,
                retry: RetryPolicy::no_retry(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        let events = dispatch(
            &mut agg,
            CommandPayload::CompleteToolCall {
                tool_call_id: "tc-1".to_string(),
                attempt: 0,
                result: "ok".to_string(),
                worker_state: None,
            },
            &Caller::Machine {
                tenant_id: "tenant-a".to_string(),
                key_id: "prod-key-1".to_string(),
            },
        );

        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::ToolCallCompleted(_),
                    EventPayload::NewMessage(_),
                    EventPayload::DecisionRequestQueued(_),
                ]
            ),
            "expected [ToolCallCompleted, NewMessage, DecisionRequestQueued]; got {events:?}"
        );

        let tc = agg.state.tool_calls.get("tc-1").expect("tool call present");
        assert_eq!(tc.tracking.status, EffectStatus::Completed);
    }

    #[test]
    fn complete_tool_call_with_wrong_attempt_fails() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::RequestToolCall {
                tool_call_id: "tc-1".to_string(),
                name: "my_tool".to_string(),
                arguments: "{}".to_string(),
                handler: ToolHandler::Worker,
                retry: RetryPolicy::no_retry(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        let caller = Caller::Machine {
            tenant_id: "tenant-a".to_string(),
            key_id: "prod-key-1".to_string(),
        };

        let err = agg
            .state
            .handle(
                CommandPayload::CompleteToolCall {
                    tool_call_id: "tc-1".to_string(),
                    attempt: 7,
                    result: "ok".to_string(),
                    worker_state: None,
                },
                &caller,
            )
            .expect_err("wrong attempt should be rejected");

        assert!(
            matches!(err, SessionError::ToolCallAttemptMismatch),
            "expected ToolCallAttemptMismatch; got {err:?}"
        );
    }

    #[test]
    fn submit_client_payload_with_active_turn_id_is_rejected() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let payload = ClientPayload::Message {
            message: Message {
                role: Role::User,
                content: Some(Content::Text("hello".to_string())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            stream: false,
        };

        dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: payload.clone(),
                turn_id: Some("turn-1".to_string()),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        let err = agg
            .state
            .handle(
                CommandPayload::SubmitClientPayload {
                    payload,
                    turn_id: Some("turn-1".to_string()),
                },
                &Caller::System {
                    tenant_id: "tenant-a".to_string(),
                },
            )
            .expect_err("re-submitting an active turn_id should be rejected");

        match err {
            SessionError::TurnAlreadyActive { turn_id } => assert_eq!(turn_id, "turn-1"),
            other => panic!("expected TurnAlreadyActive; got {other:?}"),
        }
    }

    #[test]
    fn submit_worker_decision_dispatches_action_and_completes_decision() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let setup_events = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message {
                    message: Message {
                        role: Role::User,
                        content: Some(Content::Text("hi".to_string())),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    },
                    stream: false,
                },
                turn_id: Some("turn-1".to_string()),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        let decision_id = setup_events
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.decision_id.clone()),
                _ => None,
            })
            .expect("user message should request a worker decision");

        let machine = Caller::Machine {
            tenant_id: "tenant-a".to_string(),
            key_id: "prod-key-1".to_string(),
        };

        let events = agg
            .state
            .handle(
                CommandPayload::SubmitWorkerDecision {
                    decision_id,
                    actions: vec![WorkerAction::CallTool {
                        tool_call_id: "tc-1".to_string(),
                        name: "my_tool".to_string(),
                        arguments: "{}".to_string(),
                        handler: ToolHandler::Worker,
                        retry: RetryPolicy::no_retry(),
                    }],
                    state: vec![],
                },
                &machine,
            )
            .expect("submit worker decision should succeed");

        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::WorkerDecisionCompleted(_))),
            "expected WorkerDecisionCompleted; got {events:?}"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::ToolCallRequested(_))),
            "CallTool action should expand into a ToolCallRequested event; got {events:?}"
        );
    }

    #[test]
    fn duplicate_submit_worker_decision_is_no_op() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let setup_events = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message {
                    message: Message {
                        role: Role::User,
                        content: Some(Content::Text("hi".to_string())),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    },
                    stream: false,
                },
                turn_id: Some("turn-1".to_string()),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        let decision_id = setup_events
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.decision_id.clone()),
                _ => None,
            })
            .expect("user message should request a worker decision");

        let machine = Caller::Machine {
            tenant_id: "tenant-a".to_string(),
            key_id: "prod-key-1".to_string(),
        };

        // First submission completes the decision.
        dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: decision_id.clone(),
                actions: vec![],
                state: vec![],
            },
            &machine,
        );

        // Second submission with the same decision_id should be a no-op.
        let events = agg
            .state
            .handle(
                CommandPayload::SubmitWorkerDecision {
                    decision_id,
                    actions: vec![WorkerAction::CallTool {
                        tool_call_id: "tc-1".to_string(),
                        name: "my_tool".to_string(),
                        arguments: "{}".to_string(),
                        handler: ToolHandler::Worker,
                        retry: RetryPolicy::no_retry(),
                    }],
                    state: vec![],
                },
                &machine,
            )
            .expect("duplicate submission should not error");

        assert!(
            events.is_empty(),
            "duplicate worker decision submission should emit no events; got {events:?}"
        );
    }

    #[test]
    fn user_message_rejected_while_session_interrupted() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::Interrupt {
                interrupt_id: "int-1".to_string(),
                reason: "paused".to_string(),
                payload: serde_json::Value::Null,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        let user_message = ClientPayload::Message {
            message: Message {
                role: Role::User,
                content: Some(Content::Text("hello".to_string())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            stream: false,
        };

        let err = agg
            .state
            .handle(
                CommandPayload::SubmitClientPayload {
                    payload: user_message,
                    turn_id: Some("turn-1".to_string()),
                },
                &Caller::System {
                    tenant_id: "tenant-a".to_string(),
                },
            )
            .expect_err("user messages should be rejected while interrupted");

        assert!(
            matches!(err, SessionError::SessionInterrupted),
            "expected SessionInterrupted; got {err:?}"
        );
    }

    #[test]
    fn complete_unknown_tool_call_fails() {
        let agg = create_session("sess-1", "tenant-a", "user-1");

        let caller = Caller::Machine {
            tenant_id: "tenant-a".to_string(),
            key_id: "prod-key-1".to_string(),
        };

        let err = agg
            .state
            .handle(
                CommandPayload::CompleteToolCall {
                    tool_call_id: "tc-unknown".to_string(),
                    attempt: 0,
                    result: "ok".to_string(),
                    worker_state: None,
                },
                &caller,
            )
            .expect_err("unknown tool call should be rejected");

        assert!(
            matches!(err, SessionError::ToolCallNotFound),
            "expected ToolCallNotFound; got {err:?}"
        );
    }

    #[test]
    fn send_message_emits_new_message() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");

        let events = dispatch(
            &mut agg,
            CommandPayload::SendMessage {
                message: Message {
                    role: Role::User,
                    content: Some(Content::Text("hi".to_string())),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
                stream: false,
                turn_id: None,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        // A user message wakes a worker decision so the LLM can respond.
        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::NewMessage(_),
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "expected [NewMessage, WorkerDecisionRequested]; got {events:?}"
        );
        assert_eq!(agg.state.status, SessionStatus::Idle);
    }

    #[test]
    fn cancel_session_emits_cancelled() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");

        let events = dispatch(
            &mut agg,
            CommandPayload::CancelSession,
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        assert!(
            matches!(events.as_slice(), [EventPayload::SessionCancelled]),
            "expected [SessionCancelled]; got {events:?}"
        );
        assert_eq!(agg.state.status, SessionStatus::Done);
    }

    #[test]
    fn mark_done_emits_session_done() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");

        let events = dispatch(
            &mut agg,
            CommandPayload::MarkDone {
                data: serde_json::Value::Null,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        // No active turn, no ancestry → just SessionDone, status returns to Idle.
        assert!(
            matches!(events.as_slice(), [EventPayload::SessionDone(_)]),
            "expected [SessionDone]; got {events:?}"
        );
        assert_eq!(agg.state.status, SessionStatus::Idle);
    }

    #[test]
    fn wake_with_no_pending_effects_is_noop() {
        let agg = create_session("sess-1", "tenant-a", "user-1");

        let events = agg
            .state
            .handle(
                CommandPayload::Wake { now: Utc::now() },
                &Caller::System {
                    tenant_id: "tenant-a".to_string(),
                },
            )
            .expect("wake should succeed");

        assert!(
            events.is_empty(),
            "wake on idle session should be a no-op; got {events:?}"
        );
    }

    #[test]
    fn request_llm_call_emits_requested() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");

        let events = dispatch(
            &mut agg,
            CommandPayload::RequestLlmCall {
                call_id: "llm-1".to_string(),
                request: LlmRequest {
                    model: "test-model".to_string(),
                    messages: vec![],
                    tools: None,
                    temperature: None,
                    max_completion_tokens: None,
                    reasoning: None,
                },
                stream: false,
                retry: RetryPolicy::no_retry(),
                handler: LlmHandler::Server,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        assert!(
            matches!(events.as_slice(), [EventPayload::LlmCallRequested(_)]),
            "expected [LlmCallRequested]; got {events:?}"
        );
        let call = agg.state.llm_calls.get("llm-1").expect("llm call present");
        assert_eq!(call.tracking.status, EffectStatus::Pending);
    }

    #[test]
    fn complete_llm_call_emits_completed() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::RequestLlmCall {
                call_id: "llm-1".to_string(),
                request: LlmRequest {
                    model: "test-model".to_string(),
                    messages: vec![],
                    tools: None,
                    temperature: None,
                    max_completion_tokens: None,
                    reasoning: None,
                },
                stream: false,
                retry: RetryPolicy::no_retry(),
                handler: LlmHandler::Server,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        let events = dispatch(
            &mut agg,
            CommandPayload::CompleteLlmCall {
                call_id: "llm-1".to_string(),
                attempt: 0,
                response: LlmResponse {
                    model: "test-model".to_string(),
                    content: Some("hello".to_string()),
                    tool_calls: vec![],
                    finish_reason: Some("stop".to_string()),
                    usage: None,
                    cost: None,
                    images: vec![],
                },
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::LlmCallCompleted(_),
                    EventPayload::NewMessage(_),
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "expected [LlmCallCompleted, NewMessage, WorkerDecisionRequested]; got {events:?}"
        );
        let call = agg.state.llm_calls.get("llm-1").expect("llm call present");
        assert_eq!(call.tracking.status, EffectStatus::Completed);
    }

    #[test]
    fn fail_llm_call_emits_errored() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::RequestLlmCall {
                call_id: "llm-1".to_string(),
                request: LlmRequest {
                    model: "test-model".to_string(),
                    messages: vec![],
                    tools: None,
                    temperature: None,
                    max_completion_tokens: None,
                    reasoning: None,
                },
                stream: false,
                retry: RetryPolicy::no_retry(),
                handler: LlmHandler::Server,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        let events = dispatch(
            &mut agg,
            CommandPayload::FailLlmCall {
                call_id: "llm-1".to_string(),
                attempt: 0,
                error: "provider down".to_string(),
                retryable: false,
                code: None,
                detail: None,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        // Retry policy exhausted (no_retry + retryable=false) so the handler
        // fires a follow-up worker decision with the error.
        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::LlmCallErrored(_),
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "expected [LlmCallErrored, WorkerDecisionRequested]; got {events:?}"
        );
        let call = agg.state.llm_calls.get("llm-1").expect("llm call present");
        assert_eq!(call.tracking.status, EffectStatus::Failed);
    }

    fn test_llm_request() -> LlmRequest {
        LlmRequest {
            model: "test-model".to_string(),
            messages: vec![],
            tools: None,
            temperature: None,
            max_completion_tokens: None,
            reasoning: None,
        }
    }

    #[test]
    fn worker_handled_llm_call_emits_request_trigger() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let events = dispatch(
            &mut agg,
            CommandPayload::RequestLlmCall {
                call_id: "llm-1".to_string(),
                request: test_llm_request(),
                stream: false,
                retry: RetryPolicy::no_retry(),
                handler: LlmHandler::Worker,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::LlmCallRequested(_),
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "expected [LlmCallRequested, WorkerDecisionRequested]; got {events:?}"
        );
        let trigger = events
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(&p.trigger),
                _ => None,
            })
            .expect("worker decision present");
        assert!(
            matches!(trigger, DecisionTrigger::LlmRequest { call_id, .. } if call_id == "llm-1"),
            "expected an llm.request trigger; got {trigger:?}"
        );
        let call = agg.state.llm_calls.get("llm-1").expect("llm call present");
        assert_eq!(call.handler, LlmHandler::Worker);
        assert_eq!(call.tracking.status, EffectStatus::Pending);
    }

    #[test]
    fn server_handled_llm_call_does_not_emit_request_trigger() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let events = dispatch(
            &mut agg,
            CommandPayload::RequestLlmCall {
                call_id: "llm-1".to_string(),
                request: test_llm_request(),
                stream: false,
                retry: RetryPolicy::no_retry(),
                handler: LlmHandler::Server,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        assert!(
            matches!(events.as_slice(), [EventPayload::LlmCallRequested(_)]),
            "expected just [LlmCallRequested]; got {events:?}"
        );
        let call = agg.state.llm_calls.get("llm-1").expect("llm call present");
        assert_eq!(call.handler, LlmHandler::Server);
    }

    #[test]
    fn return_llm_result_completes_worker_handled_call() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let request_events = dispatch(
            &mut agg,
            CommandPayload::RequestLlmCall {
                call_id: "llm-1".to_string(),
                request: test_llm_request(),
                stream: false,
                retry: RetryPolicy::no_retry(),
                handler: LlmHandler::Worker,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        let decision_id = request_events
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.decision_id.clone()),
                _ => None,
            })
            .expect("worker-handled llm call emits an llm.request decision");

        let machine = Caller::Machine {
            tenant_id: "tenant-a".to_string(),
            key_id: "prod-key-1".to_string(),
        };

        let events = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id,
                actions: vec![WorkerAction::ReturnLlmResult {
                    call_id: "llm-1".to_string(),
                    response: LlmResponse {
                        model: "test-model".to_string(),
                        content: Some("hello from the worker".to_string()),
                        tool_calls: vec![],
                        finish_reason: Some("stop".to_string()),
                        usage: None,
                        cost: None,
                        images: vec![],
                    },
                    attempt: 0,
                }],
                state: vec![],
            },
            &machine,
        );

        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::LlmCallCompleted(_))),
            "expected an LlmCallCompleted event; got {events:?}"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::NewMessage(_))),
            "expected a NewMessage event; got {events:?}"
        );
        let call = agg.state.llm_calls.get("llm-1").expect("llm call present");
        assert_eq!(call.tracking.status, EffectStatus::Completed);
    }

    #[test]
    fn fail_tool_call_emits_errored() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let request_events = dispatch(
            &mut agg,
            CommandPayload::RequestToolCall {
                tool_call_id: "tc-1".to_string(),
                name: "my_tool".to_string(),
                arguments: "{}".to_string(),
                handler: ToolHandler::Worker,
                retry: RetryPolicy::no_retry(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        let d1 = request_events
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.decision_id.clone()),
                _ => None,
            })
            .expect("worker-handled tool call emits a ToolExecute decision");

        let machine = Caller::Machine {
            tenant_id: "tenant-a".to_string(),
            key_id: "prod-key-1".to_string(),
        };

        // Worker releases its decision so the failure cleanly emits a fresh
        // follow-up rather than queueing behind a phantom pending decision.
        dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: d1,
                actions: vec![],
                state: vec![],
            },
            &machine,
        );

        let events = dispatch(
            &mut agg,
            CommandPayload::FailToolCall {
                tool_call_id: "tc-1".to_string(),
                attempt: 0,
                error: "boom".to_string(),
                retryable: false,
                worker_state: None,
            },
            &machine,
        );

        // Retry exhausted (no_retry + retryable=false) so the failure
        // surfaces a follow-up worker decision with is_error=true.
        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::ToolCallErrored(_),
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "expected [ToolCallErrored, WorkerDecisionRequested]; got {events:?}"
        );
        let tc = agg.state.tool_calls.get("tc-1").expect("tool call present");
        assert_eq!(tc.tracking.status, EffectStatus::Failed);
        assert!(tc.is_error);
    }

    #[test]
    fn request_sub_agent_emits_requested() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");

        let events = dispatch(
            &mut agg,
            CommandPayload::RequestSubAgent {
                session_id: "child-1".to_string(),
                agent_id: "agent-2".to_string(),
                retry: RetryPolicy::no_retry(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        assert!(
            matches!(events.as_slice(), [EventPayload::SubAgentRequested(_)]),
            "expected [SubAgentRequested]; got {events:?}"
        );
        let sa = agg
            .state
            .sub_agent_calls
            .get("child-1")
            .expect("sub-agent recorded");
        assert_eq!(sa.tracking.status, EffectStatus::Pending);
    }

    #[test]
    fn start_sub_agent_emits_started() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::RequestSubAgent {
                session_id: "child-1".to_string(),
                agent_id: "agent-2".to_string(),
                retry: RetryPolicy::no_retry(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        let events = dispatch(
            &mut agg,
            CommandPayload::StartSubAgent {
                session_id: "child-1".to_string(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        assert!(
            matches!(events.as_slice(), [EventPayload::SubAgentStarted(_)]),
            "expected [SubAgentStarted]; got {events:?}"
        );
    }

    #[test]
    fn fail_sub_agent_emits_errored() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::RequestSubAgent {
                session_id: "child-1".to_string(),
                agent_id: "agent-2".to_string(),
                retry: RetryPolicy::no_retry(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        let events = dispatch(
            &mut agg,
            CommandPayload::FailSubAgent {
                session_id: "child-1".to_string(),
                error: "child crashed".to_string(),
                retryable: false,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        // Retry exhausted → follow-up worker decision with the error.
        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::SubAgentErrored(_),
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "expected [SubAgentErrored, WorkerDecisionRequested]; got {events:?}"
        );
        let sa = agg
            .state
            .sub_agent_calls
            .get("child-1")
            .expect("sub-agent present");
        assert_eq!(sa.tracking.status, EffectStatus::Failed);
    }

    #[test]
    fn complete_sub_agent_turn_emits_completed() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::RequestSubAgent {
                session_id: "child-1".to_string(),
                agent_id: "agent-2".to_string(),
                retry: RetryPolicy::no_retry(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        let events = dispatch(
            &mut agg,
            CommandPayload::CompleteSubAgentTurn {
                session_id: "child-1".to_string(),
                agent_id: "agent-2".to_string(),
                turn_id: "turn-x".to_string(),
                data: serde_json::Value::Null,
                cost: rust_decimal::Decimal::ZERO,
                token_usage: std::collections::BTreeMap::new(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::SubAgentTurnCompleted(_),
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "expected [SubAgentTurnCompleted, WorkerDecisionRequested]; got {events:?}"
        );
    }

    #[test]
    fn resume_interrupt_emits_resumed() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::Interrupt {
                interrupt_id: "int-1".to_string(),
                reason: "paused".to_string(),
                payload: serde_json::Value::Null,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        let events = dispatch(
            &mut agg,
            CommandPayload::ResumeInterrupt {
                interrupt_id: "int-1".to_string(),
                payload: serde_json::Value::Null,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::InterruptResumed(_),
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "expected [InterruptResumed, WorkerDecisionRequested]; got {events:?}"
        );
        assert_eq!(agg.state.status, SessionStatus::Idle);
    }

    #[test]
    fn fail_worker_decision_emits_errored() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let setup_events = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message {
                    message: Message {
                        role: Role::User,
                        content: Some(Content::Text("hi".to_string())),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    },
                    stream: false,
                },
                turn_id: Some("turn-1".to_string()),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        let decision_id = setup_events
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.decision_id.clone()),
                _ => None,
            })
            .expect("user message should request a worker decision");

        let events = dispatch(
            &mut agg,
            CommandPayload::FailWorkerDecision {
                decision_id: decision_id.clone(),
                error: "worker offline".to_string(),
                retryable: false,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        assert!(
            matches!(events.as_slice(), [EventPayload::WorkerDecisionErrored(_)]),
            "expected [WorkerDecisionErrored]; got {events:?}"
        );
        let wd = agg
            .state
            .worker_decisions
            .get(&decision_id)
            .expect("decision present");
        assert_eq!(wd.tracking.status, EffectStatus::Failed);
    }

    #[test]
    fn machine_caller_from_wrong_tenant_is_denied() {
        let agg = create_session("sess-1", "tenant-a", "user-1");

        let cross_tenant_machine = Caller::Machine {
            tenant_id: "tenant-b".to_string(),
            key_id: "key-from-tenant-b".to_string(),
        };

        let err = agg
            .state
            .handle(
                CommandPayload::CompleteToolCall {
                    tool_call_id: "tc-1".to_string(),
                    attempt: 0,
                    result: "ok".to_string(),
                    worker_state: None,
                },
                &cross_tenant_machine,
            )
            .expect_err("machine from a different tenant should be rejected");

        assert!(
            matches!(err, SessionError::SessionAccessDenied),
            "expected SessionAccessDenied; got {err:?}"
        );
    }

    #[test]
    fn frontend_caller_with_mismatched_tenant_on_create_session_is_denied() {
        let session_id = "sess-1".to_string();
        let agg = Aggregate::new(
            session_id.clone(),
            "tenant-a".to_string(),
            SessionState::new(session_id),
        );

        let caller = Caller::Frontend {
            tenant_id: "tenant-a".to_string(),
            user_id: "user-1".to_string(),
            attrs: HashMap::new(),
        };

        let err = agg
            .state
            .handle(
                CommandPayload::CreateSession {
                    agent_id: "agent-1".to_string(),
                    owner: SessionOwner {
                        tenant_id: "tenant-b".to_string(),
                        id: Some("user-1".to_string()),
                        metadata: HashMap::new(),
                    },
                    ancestry: vec![],
                    worker_retry: RetryPolicy::no_retry(),
                },
                &caller,
            )
            .expect_err("creating a session in a different tenant should be rejected");

        assert!(
            matches!(err, SessionError::SessionAccessDenied),
            "expected SessionAccessDenied; got {err:?}"
        );
    }
}
