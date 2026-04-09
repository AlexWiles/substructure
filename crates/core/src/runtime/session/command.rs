use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;

use super::decision::{ClientPayload, DecisionTrigger, ToolResult, WorkerAction};
use super::events::*;
use super::message::{Content, ContentPart, ImageUrl, Message, Role};
use super::state::{new_call_id, EffectStatus, SessionState, SessionStatus};
use crate::runtime::identity::ClientIdentity;
use crate::runtime::llm::{LlmRequest, LlmResponse};
use crate::runtime::retry::RetryPolicy;

#[derive(Debug, Clone)]
pub enum CommandPayload {
    CreateSession {
        agent_id: String,
        auth: ClientIdentity,
        ancestry: Vec<String>,
        worker_retry: RetryPolicy,
    },
    SubmitClientPayload {
        payload: ClientPayload,
        auth: ClientIdentity,
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
        llm_client: String,
        retry: RetryPolicy,
    },
    CompleteLlmCall {
        call_id: String,
        response: LlmResponse,
    },
    FailLlmCall {
        call_id: String,
        error: String,
        retryable: bool,
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
        result: String,
        worker_state: Option<Vec<u8>>,
    },
    FailToolCall {
        tool_call_id: String,
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
}

impl SessionState {
    pub fn handle(&self, cmd: CommandPayload) -> Result<Vec<EventPayload>, SessionError> {
        match (&self.agent_id, cmd) {
            (
                None,
                CommandPayload::CreateSession {
                    agent_id,
                    auth,
                    ancestry,
                    worker_retry,
                },
            ) => Ok(vec![EventPayload::SessionCreated(Box::new(
                SessionCreated {
                    agent_id,
                    auth,
                    ancestry,
                    worker_retry,
                },
            ))]),
            (Some(_), CommandPayload::CreateSession { .. }) => {
                Err(SessionError::SessionAlreadyCreated)
            }
            (None, _) => Err(SessionError::SessionNotCreated),
            (Some(_), cmd) => self.handle_active(cmd),
        }
    }

    fn handle_active(&self, cmd: CommandPayload) -> Result<Vec<EventPayload>, SessionError> {
        match cmd {
            CommandPayload::CreateSession { .. } => Err(SessionError::SessionAlreadyCreated),

            CommandPayload::SubmitClientPayload {
                payload,
                auth,
                turn_id,
            } => {
                let Some(subject) = auth.sub.as_deref() else {
                    return Err(SessionError::MissingSubject);
                };
                let Some(existing_auth) = self.auth.as_ref() else {
                    return Err(SessionError::SessionAccessDenied);
                };
                if existing_auth.tenant_id != auth.tenant_id
                    || existing_auth.sub.as_deref() != Some(subject)
                {
                    return Err(SessionError::SessionAccessDenied);
                }

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
                            events.push(EventPayload::WorkerDecisionRequested(
                                WorkerDecisionRequested {
                                    decision_id: new_call_id(),
                                    trigger: DecisionTrigger::UserMessage { stream, message },
                                },
                            ));
                        }
                    }
                    ClientPayload::Action { action } => {
                        events.push(EventPayload::WorkerDecisionRequested(
                            WorkerDecisionRequested {
                                decision_id: new_call_id(),
                                trigger: DecisionTrigger::ClientAction {
                                    name: action.name,
                                    args: action.args,
                                },
                            },
                        ));
                    }
                }

                Ok(events)
            }

            CommandPayload::SendMessage {
                message,
                stream,
                turn_id,
            } => {
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
                            events.push(EventPayload::WorkerDecisionRequested(
                                WorkerDecisionRequested {
                                    decision_id: new_call_id(),
                                    trigger: DecisionTrigger::UserMessage { stream, message },
                                },
                            ));
                        }
                        Ok(events)
                    }
                }
            }

            CommandPayload::RequestLlmCall {
                call_id,
                request,
                stream,
                llm_client,
                retry,
            } => {
                if self.has_pending_llm() {
                    return Ok(vec![]);
                }
                let issue = matches!(
                    self.llm_calls.get(&call_id).map(|c| &c.tracking.status),
                    None | Some(&EffectStatus::Failed) | Some(&EffectStatus::RetryScheduled)
                );
                if issue {
                    Ok(vec![EventPayload::LlmCallRequested(LlmCallRequested {
                        call_id,
                        request,
                        stream,
                        llm_client,
                        retry,
                    })])
                } else {
                    Ok(vec![])
                }
            }

            CommandPayload::CompleteLlmCall { call_id, response } => {
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
                                response,
                            }),
                            EventPayload::NewMessage(NewMessage {
                                message: message.clone(),
                            }),
                            EventPayload::WorkerDecisionRequested(WorkerDecisionRequested {
                                decision_id: new_call_id(),
                                trigger: DecisionTrigger::LlmResponse {
                                    call_id,
                                    message,
                                    truncated,
                                    usage,
                                    cost,
                                },
                            }),
                        ])
                    }
                    _ => Ok(vec![]),
                }
            }

            CommandPayload::FailLlmCall {
                call_id,
                error,
                retryable,
            } => {
                let Some(call) = self.llm_calls.get(&call_id) else {
                    return Ok(vec![]);
                };
                if call.tracking.status != EffectStatus::Pending {
                    return Ok(vec![]);
                }
                let mut events = vec![EventPayload::LlmCallErrored(LlmCallErrored {
                    call_id: call_id.clone(),
                    error: error.clone(),
                    retryable,
                    source: None,
                })];
                if call
                    .tracking
                    .retry_policy
                    .exhausted(&call.tracking.retry, retryable)
                {
                    events.push(EventPayload::WorkerDecisionRequested(
                        WorkerDecisionRequested {
                            decision_id: new_call_id(),
                            trigger: DecisionTrigger::LlmError { call_id, error },
                        },
                    ));
                }
                Ok(events)
            }

            CommandPayload::RequestToolCall {
                tool_call_id,
                name,
                arguments,
                handler,
                retry,
            } => match self.tool_calls.get(&tool_call_id) {
                Some(_) => Ok(vec![]),
                None => {
                    let mut events = vec![EventPayload::ToolCallRequested(ToolCallRequested {
                        tool_call_id: tool_call_id.clone(),
                        name: name.clone(),
                        arguments: arguments.clone(),
                        handler: handler.clone(),
                        retry: retry.clone(),
                    })];
                    if handler == ToolHandler::Worker {
                        events.push(EventPayload::WorkerDecisionRequested(
                            WorkerDecisionRequested {
                                decision_id: new_call_id(),
                                trigger: DecisionTrigger::ToolExecute {
                                    tool_call_id,
                                    name,
                                    arguments,
                                    attempt: 0,
                                    deadline: retry.deadline(chrono::Utc::now()),
                                },
                            },
                        ));
                    }
                    Ok(events)
                }
            },

            CommandPayload::CompleteToolCall {
                tool_call_id,
                result,
                worker_state,
            } => {
                let Some(tc) = self.tool_calls.get(&tool_call_id) else {
                    return Ok(vec![]);
                };
                if tc.tracking.status != EffectStatus::Pending {
                    return Ok(vec![]);
                }
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
                            name: None,
                        },
                    }),
                ];
                if let Some(ws) = worker_state {
                    events.push(EventPayload::WorkerStateUpdated(WorkerStateUpdated {
                        state: ws,
                    }));
                }
                events.push(EventPayload::WorkerDecisionRequested(
                    WorkerDecisionRequested {
                        decision_id: new_call_id(),
                        trigger: DecisionTrigger::ToolResult {
                            result: ToolResult {
                                tool_call_id,
                                name,
                                content: result,
                                is_error: false,
                            },
                        },
                    },
                ));
                Ok(events)
            }

            CommandPayload::FailToolCall {
                tool_call_id,
                error,
                retryable,
                worker_state,
            } => {
                let Some(tc) = self.tool_calls.get(&tool_call_id) else {
                    return Ok(vec![]);
                };
                if tc.tracking.status != EffectStatus::Pending {
                    return Ok(vec![]);
                }
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
                    events.push(EventPayload::WorkerDecisionRequested(
                        WorkerDecisionRequested {
                            decision_id: new_call_id(),
                            trigger: DecisionTrigger::ToolResult {
                                result: ToolResult {
                                    tool_call_id,
                                    name,
                                    content: error,
                                    is_error: true,
                                },
                            },
                        },
                    ));
                }
                Ok(events)
            }

            CommandPayload::RequestSubAgent {
                session_id,
                agent_id,
                retry,
            } => match self.sub_agent_calls.get(&session_id) {
                Some(_) => Ok(vec![]),
                None => Ok(vec![EventPayload::SubAgentRequested(SubAgentRequested {
                    session_id,
                    agent_id,
                    retry,
                })]),
            },

            CommandPayload::StartSubAgent { session_id } => {
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
                    events.push(EventPayload::WorkerDecisionRequested(
                        WorkerDecisionRequested {
                            decision_id: new_call_id(),
                            trigger: DecisionTrigger::SubAgentError {
                                session_id,
                                agent_id: sa.agent_id.clone(),
                                error,
                            },
                        },
                    ));
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
                // Only fire if we know about this sub-agent
                if self.sub_agent_calls.contains_key(&session_id) {
                    Ok(vec![
                        EventPayload::SubAgentTurnCompleted(SubAgentTurnCompleted {
                            session_id: session_id.clone(),
                            cost,
                            token_usage,
                        }),
                        EventPayload::WorkerDecisionRequested(WorkerDecisionRequested {
                            decision_id: new_call_id(),
                            trigger: DecisionTrigger::SubAgentTurnComplete {
                                session_id,
                                agent_id,
                                turn_id,
                                data,
                            },
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
            } => match self.status {
                SessionStatus::Interrupted { .. } => Ok(vec![]),
                _ => Ok(vec![EventPayload::SessionInterrupted(SessionInterrupted {
                    interrupt_id,
                    reason,
                    payload,
                })]),
            },

            CommandPayload::ResumeInterrupt {
                interrupt_id,
                payload,
            } => match self.active_interrupt() {
                Some(id) if id == interrupt_id => Ok(vec![
                    EventPayload::InterruptResumed(InterruptResumed {
                        interrupt_id: interrupt_id.clone(),
                        payload,
                    }),
                    EventPayload::WorkerDecisionRequested(WorkerDecisionRequested {
                        decision_id: new_call_id(),
                        trigger: DecisionTrigger::InterruptResumed { interrupt_id },
                    }),
                ]),
                _ => Ok(vec![]),
            },

            CommandPayload::SubmitWorkerDecision {
                decision_id,
                actions,
                state,
            } => {
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
                            llm_client,
                            retry,
                        } => self.handle(CommandPayload::RequestLlmCall {
                            call_id: new_call_id(),
                            request,
                            stream,
                            llm_client,
                            retry,
                        }),
                        WorkerAction::CallTool {
                            tool_call_id,
                            name,
                            arguments,
                            handler,
                            retry,
                        } => self.handle(CommandPayload::RequestToolCall {
                            tool_call_id,
                            name,
                            arguments,
                            handler,
                            retry,
                        }),
                        WorkerAction::SpawnSubAgent {
                            session_id,
                            agent_id,
                            retry,
                        } => self.handle(CommandPayload::RequestSubAgent {
                            session_id,
                            agent_id,
                            retry,
                        }),
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
                        } => match self.tool_calls.get(&tool_call_id) {
                            Some(tc)
                                if tc.tracking.status == EffectStatus::Pending
                                    && tc.tracking.retry.attempts == attempt =>
                            {
                                self.handle(CommandPayload::CompleteToolCall {
                                    tool_call_id,
                                    result,
                                    worker_state: None,
                                })
                            }
                            _ => Ok(vec![]),
                        },
                        WorkerAction::ReturnToolError {
                            tool_call_id,
                            error,
                            retryable,
                            attempt,
                        } => match self.tool_calls.get(&tool_call_id) {
                            Some(tc)
                                if tc.tracking.status == EffectStatus::Pending
                                    && tc.tracking.retry.attempts == attempt =>
                            {
                                self.handle(CommandPayload::FailToolCall {
                                    tool_call_id,
                                    error,
                                    retryable,
                                    worker_state: None,
                                })
                            }
                            _ => Ok(vec![]),
                        },
                        WorkerAction::Done { data } => {
                            self.handle(CommandPayload::MarkDone { data })
                        }
                    };
                    if let Ok(sub) = sub_events {
                        events.extend(sub);
                    }
                }
                Ok(events)
            }

            CommandPayload::CancelSession => Ok(vec![EventPayload::SessionCancelled]),

            CommandPayload::MarkDone { data } => {
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

            CommandPayload::Wake { now } => self.handle_wake(now),
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
                    error: "deadline exceeded".to_string(),
                    retryable: true,
                    source: None,
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
                        return Ok(vec![EventPayload::LlmCallRequested(LlmCallRequested {
                            call_id: call.call_id.clone(),
                            request: call.request.clone(),
                            stream: call.stream,
                            llm_client: call.llm_client.clone(),
                            retry: call.tracking.retry_policy.clone(),
                        })]);
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
                            name: tc.name.clone(),
                            arguments: tc.arguments.clone(),
                            handler: tc.handler.clone(),
                            retry: tc.tracking.retry_policy.clone(),
                        })];
                        if tc.handler == ToolHandler::Worker {
                            events.push(EventPayload::WorkerDecisionRequested(
                                WorkerDecisionRequested {
                                    decision_id: new_call_id(),
                                    trigger: DecisionTrigger::ToolExecute {
                                        tool_call_id: tc.tool_call_id.clone(),
                                        name: tc.name.clone(),
                                        arguments: tc.arguments.clone(),
                                        attempt: tc.tracking.retry.attempts + 1,
                                        deadline: tc.tracking.retry_policy.deadline(now),
                                    },
                                },
                            ));
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

        // 9. All tools done, no next step → stall recovery
        if self.all_tools_resolved() && !self.has_pending_llm() {
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
