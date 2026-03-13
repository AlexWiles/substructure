use chrono::{DateTime, Utc};
use uuid::Uuid;

use super::decision::{DecisionTrigger, ToolResult, WorkerAction};
use super::events::*;
use super::message::{Message, Role};
use super::state::{new_call_id, EffectStatus, SessionState, SessionStatus};
use crate::runtime::identity::ClientIdentity;
use crate::runtime::llm::{LlmRequest, LlmResponse};
use crate::runtime::retry::RetryPolicy;

#[derive(Debug, Clone)]
pub enum CommandPayload {
    CreateSession {
        agent_id: String,
        auth: ClientIdentity,
        ancestry: Vec<Uuid>,
        worker_retry: RetryPolicy,
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
        session_id: Uuid,
        agent_id: String,
        retry: RetryPolicy,
    },
    StartSubAgent {
        session_id: Uuid,
    },
    FailSubAgent {
        session_id: Uuid,
        error: String,
        retryable: bool,
    },
    CompleteSubAgentTurn {
        session_id: Uuid,
        agent_id: String,
        turn_id: String,
        artifacts: Vec<Artifact>,
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
        artifacts: Vec<Artifact>,
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
            CommandPayload::CreateSession { .. } => {
                Err(SessionError::SessionAlreadyCreated)
            }

            CommandPayload::SendMessage { message, stream, turn_id } => match self.status {
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
                        let truncated =
                            response.finish_reason.as_deref() == Some("length");
                        let tool_calls = if response.tool_calls.is_empty() {
                            None
                        } else {
                            Some(response.tool_calls.clone())
                        };
                        let message = Message {
                            role: Role::Assistant,
                            content: response.content.clone(),
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
                if call.tracking.retry_policy.exhausted(&call.tracking.retry, retryable) {
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
                            content: Some(result.clone()),
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
                if tc.tracking.retry_policy.exhausted(&tc.tracking.retry, retryable) {
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
            } => {
                let key = session_id.to_string();
                match self.sub_agent_calls.get(&key) {
                    Some(_) => Ok(vec![]),
                    None => Ok(vec![EventPayload::SubAgentRequested(SubAgentRequested {
                        session_id,
                        agent_id,
                        retry,
                    })]),
                }
            }

            CommandPayload::StartSubAgent { session_id } => {
                let key = session_id.to_string();
                match self.sub_agent_calls.get(&key).map(|c| &c.tracking.status) {
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
                let key = session_id.to_string();
                let Some(sa) = self.sub_agent_calls.get(&key) else {
                    return Ok(vec![]);
                };
                if sa.tracking.status != EffectStatus::Pending {
                    return Ok(vec![]);
                }
                let mut events = vec![EventPayload::SubAgentErrored(SubAgentErrored {
                    session_id,
                    error: error.clone(),
                    retryable,
                })];
                if sa.tracking.retry_policy.exhausted(&sa.tracking.retry, retryable) {
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
                artifacts,
            } => {
                let key = session_id.to_string();
                // Only fire if we know about this sub-agent
                if self.sub_agent_calls.contains_key(&key) {
                    Ok(vec![EventPayload::WorkerDecisionRequested(
                        WorkerDecisionRequested {
                            decision_id: new_call_id(),
                            trigger: DecisionTrigger::SubAgentTurnComplete {
                                session_id,
                                agent_id,
                                turn_id,
                                artifacts,
                            },
                        },
                    )])
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
                Some(id) if id == interrupt_id => {
                    Ok(vec![
                        EventPayload::InterruptResumed(InterruptResumed {
                            interrupt_id: interrupt_id.clone(),
                            payload,
                        }),
                        EventPayload::WorkerDecisionRequested(WorkerDecisionRequested {
                            decision_id: new_call_id(),
                            trigger: DecisionTrigger::InterruptResumed { interrupt_id },
                        }),
                    ])
                }
                _ => Ok(vec![]),
            },

            CommandPayload::SubmitWorkerDecision {
                decision_id,
                actions,
                state,
            } => {
                match self.worker_decisions.get(&decision_id).map(|d| &d.tracking.status) {
                    Some(&EffectStatus::Pending) => {}
                    _ => return Ok(vec![]),
                }
                let mut events: Vec<EventPayload> = vec![
                    EventPayload::WorkerDecisionCompleted(WorkerDecisionCompleted {
                        decision_id,
                        state,
                    }),
                ];
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
                        } => {
                            match self.tool_calls.get(&tool_call_id) {
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
                            }
                        }
                        WorkerAction::ReturnToolError {
                            tool_call_id,
                            error,
                            retryable,
                            attempt,
                        } => {
                            match self.tool_calls.get(&tool_call_id) {
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
                            }
                        }
                        WorkerAction::ResolveRemoteTool {
                            session_id,
                            tool_call_id,
                            result,
                        } => Ok(vec![EventPayload::ToolCallResolutionRequested(
                            ToolCallResolutionRequested {
                                target_session_id: session_id,
                                tool_call_id,
                                result,
                            },
                        )]),
                        WorkerAction::Done { artifacts } => {
                            self.handle(CommandPayload::MarkDone { artifacts })
                        }
                    };
                    if let Ok(sub) = sub_events {
                        events.extend(sub);
                    }
                }
                Ok(events)
            }

            CommandPayload::CancelSession => Ok(vec![EventPayload::SessionCancelled]),

            CommandPayload::MarkDone { artifacts } => {
                let mut events = vec![EventPayload::SessionDone(SessionDone { artifacts })];
                if let Some(turn_id) = &self.turn_id {
                    events.push(EventPayload::TurnCompleted(TurnCompleted {
                        turn_id: turn_id.clone(),
                    }));
                }
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
                        let mut events =
                            vec![EventPayload::ToolCallRequested(ToolCallRequested {
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
                    session_id: sa.session_id,
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
                            session_id: sa.session_id,
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
