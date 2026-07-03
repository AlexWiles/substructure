use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;

use super::decision::{
    ClientPayload, DecisionTrigger, EffectOutcome, EffectResultPayload, EffectWork, WorkKind,
    WorkerAction,
};
use super::events::*;
use super::message::{Content, ContentPart, ImageUrl, Message, Role};
use super::state::{
    json_to_string, new_call_id, new_message_id, EffectStatus, SessionState, SessionStatus,
};
use crate::runtime::aggregate::Caller;
use crate::runtime::llm::{ErrorCode, LlmRequest, LlmResponse};
use crate::runtime::owner::SessionOwner;
use crate::runtime::retry::RetryPolicy;
use crate::runtime::worker::WorkerState;

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
        parent_id: Option<String>,
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
        worker_state: Option<WorkerState>,
    },
    FailToolCall {
        tool_call_id: String,
        attempt: u32,
        error: String,
        retryable: bool,
        worker_state: Option<WorkerState>,
    },
    RequestSubAgent {
        session_id: String,
        agent_id: String,
        tool_call_id: String,
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
        /// The worker's flat transcript; the engine reconciles it into the tree.
        transcript: Vec<Message>,
        actions: Vec<WorkerAction>,
        state: WorkerState,
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
    #[error("effect not found")]
    EffectNotFound,
    #[error("effect is not pending")]
    EffectNotPending,
    #[error("effect attempt mismatch")]
    EffectAttemptMismatch,
    #[error("caller may not settle this effect")]
    EffectWrongHandler,
}

fn assign_id(mut message: Message) -> Message {
    if message.id.is_empty() {
        message.id = new_message_id();
    }
    message
}

impl SessionState {
    fn ensure_internal(caller: &Caller) -> Result<(), SessionError> {
        match caller {
            Caller::System { .. } => Ok(()),
            _ => Err(SessionError::SessionAccessDenied),
        }
    }

    fn ensure_machine_or_system(caller: &Caller) -> Result<(), SessionError> {
        match caller {
            Caller::System { .. } | Caller::Machine { .. } => Ok(()),
            Caller::Frontend { .. } => Err(SessionError::SessionAccessDenied),
        }
    }

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

    fn caller_interrupt_origin(caller: &Caller) -> InterruptOrigin {
        match caller {
            Caller::System { .. } => InterruptOrigin::System,
            Caller::Machine { .. } => InterruptOrigin::Machine,
            Caller::Frontend { .. } => InterruptOrigin::Frontend,
        }
    }

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
            return Err(SessionError::EffectWrongHandler);
        }
        Ok(())
    }

    /// Who may settle an llm call out-of-band. The tenant is already checked in
    /// `handle_active`. System (executor, internal dispatch) always passes. A
    /// Machine caller may settle only worker-handled calls — an engine-handled
    /// call is the executor's to settle, so an external completion would race
    /// it. Frontends never settle model calls.
    fn check_llm_call_caller(
        &self,
        call: Option<&super::state::LlmCallState>,
        caller: &Caller,
    ) -> Result<(), SessionError> {
        match caller {
            Caller::System { .. } => Ok(()),
            Caller::Frontend { .. } => Err(SessionError::EffectWrongHandler),
            Caller::Machine { .. } => match call {
                Some(c) if c.handler == LlmHandler::Worker => Ok(()),
                Some(_) => Err(SessionError::EffectWrongHandler),
                None => Err(SessionError::EffectNotFound),
            },
        }
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

    /// Reconcile a worker's returned transcript into the tree: known ids are the
    /// path it continues, unknown/id-less messages append (chaining under the
    /// previous), forking automatically when the parent already has a child.
    fn reconcile_transcript(&self, transcript: Vec<Message>) -> Vec<EventPayload> {
        if transcript.is_empty() {
            return vec![];
        }
        let existing: std::collections::HashSet<String> =
            self.nodes.iter().map(|n| n.id().to_string()).collect();
        let mut events = Vec::new();
        let mut parent: Option<String> = None;
        for msg in transcript {
            if !msg.id.is_empty() && existing.contains(&msg.id) {
                parent = Some(msg.id);
            } else {
                let msg = assign_id(msg);
                let id = msg.id.clone();
                events.push(EventPayload::NewMessage(NewMessage {
                    message: msg,
                    parent_id: parent.take(),
                }));
                parent = Some(id);
            }
        }
        events
    }

    /// A completed effect fires an `effect.settled` decision as it lands. The
    /// worker decides when to prompt from `effects` on the request.
    fn emit_tool_result(
        &self,
        tool_call_id: String,
        name: String,
        result: String,
        is_error: bool,
    ) -> EventPayload {
        self.emit_decision_request(DecisionTrigger::EffectSettled {
            id: tool_call_id,
            ok: !is_error,
            outcome: EffectOutcome::ToolCall { name, result },
        })
    }

    fn emit_sub_agent_result(
        &self,
        session_id: String,
        tool_call_id: String,
        agent_id: String,
        result: String,
        is_error: bool,
    ) -> EventPayload {
        self.emit_decision_request(DecisionTrigger::EffectSettled {
            id: session_id,
            ok: !is_error,
            outcome: EffectOutcome::SubAgent {
                tool_call_id,
                agent_id,
                result,
            },
        })
    }

    /// `(tool_call_id, name, content)` for a transcript tool message answering a pending client tool call.
    fn pending_client_result(&self, m: &Message) -> Option<(String, String, String)> {
        if m.role != Role::Tool {
            return None;
        }
        let id = m.tool_call_id.as_deref()?;
        let tc = self.tool_calls.get(id)?;
        if tc.handler != ToolHandler::Client || tc.tracking.status != EffectStatus::Pending {
            return None;
        }
        let content = m
            .content
            .as_ref()
            .map(Content::text_owned)
            .unwrap_or_default();
        Some((id.to_string(), tc.name.clone(), content))
    }

    fn emit_decision_request(&self, trigger: DecisionTrigger) -> EventPayload {
        let decision_id = new_call_id();
        // Queue while another decision is in flight, or while interrupted —
        // an interrupted session records what happened (e.g. a late tool
        // result) but defers acting on it until resume promotes the queue.
        if self.has_pending_worker_decision()
            || matches!(self.status, SessionStatus::Interrupted { .. })
        {
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
                    ClientPayload::Message { message, stream: _ } => {
                        if matches!(self.status, SessionStatus::Interrupted { .. })
                            && message.role == Role::User
                        {
                            return Err(SessionError::SessionInterrupted);
                        }
                        if message.role == Role::User {
                            events.push(self.emit_decision_request(DecisionTrigger::UserMessage {
                                message: assign_id(message),
                            }));
                        }
                    }
                    ClientPayload::Messages {
                        messages,
                        stream: _,
                    } => {
                        let messages: Vec<Message> = messages.into_iter().map(assign_id).collect();

                        // A transcript ending in a tool message answers pending client tools, not a new turn.
                        if matches!(messages.last(), Some(m) if m.role == Role::Tool) {
                            let completions: Vec<(String, String, String)> = messages
                                .iter()
                                .filter_map(|m| self.pending_client_result(m))
                                .collect();
                            // Pair each completion with its decision. The live
                            // decision's snapshot is then taken before later
                            // siblings complete, so they still count as in-flight
                            // `pending` and it won't prompt before they land.
                            let mut live = self.has_pending_worker_decision()
                                || matches!(self.status, SessionStatus::Interrupted { .. });
                            for (tool_call_id, name, content) in completions {
                                events.push(EventPayload::ToolCallCompleted(ToolCallCompleted {
                                    tool_call_id: tool_call_id.clone(),
                                    name: name.clone(),
                                    result: content.clone(),
                                }));
                                let trigger = DecisionTrigger::EffectSettled {
                                    id: tool_call_id,
                                    ok: true,
                                    outcome: EffectOutcome::ToolCall {
                                        name,
                                        result: content,
                                    },
                                };
                                let decision_id = new_call_id();
                                if live {
                                    events.push(EventPayload::DecisionRequestQueued(
                                        DecisionRequestQueued {
                                            decision_id,
                                            trigger,
                                        },
                                    ));
                                } else {
                                    live = true;
                                    events.push(EventPayload::WorkerDecisionRequested(
                                        WorkerDecisionRequested {
                                            decision_id,
                                            trigger,
                                        },
                                    ));
                                }
                            }
                        } else {
                            if matches!(self.status, SessionStatus::Interrupted { .. }) {
                                return Err(SessionError::SessionInterrupted);
                            }
                            events.push(self.emit_decision_request(
                                DecisionTrigger::UserTranscript { messages },
                            ));
                        }
                    }
                    ClientPayload::Action { action } => {
                        if matches!(self.status, SessionStatus::Interrupted { .. }) {
                            return Err(SessionError::SessionInterrupted);
                        }
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
                stream: _,
                turn_id,
                parent_id: _,
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
                        let mut events = Vec::new();
                        if let Some(turn_id) = turn_id {
                            events.push(EventPayload::TurnStarted(TurnStarted { turn_id }));
                        }
                        if message.role == Role::User {
                            events.push(self.emit_decision_request(DecisionTrigger::UserMessage {
                                message: assign_id(message),
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

                // The id is the idempotency key: a Pending/Completed id no-ops
                // (redelivery-safe), while Failed/RetryScheduled re-issues.
                // Multiple calls may be in flight — there is no single-pending gate.
                let issue = matches!(
                    self.llm_calls.get(&call_id).map(|c| &c.tracking.status),
                    None | Some(&EffectStatus::Failed) | Some(&EffectStatus::RetryScheduled)
                );

                if issue {
                    let request = LlmRequest {
                        messages: request.messages.into_iter().map(assign_id).collect(),
                        ..request
                    };
                    // The prompt is the worker's disposable message list, stored verbatim
                    // for retries; it is never reconstructed from the tree.
                    let mut events = vec![EventPayload::LlmCallRequested(LlmCallRequested {
                        call_id: call_id.clone(),
                        attempt: 0,
                        request: request.clone(),
                        stream,
                        retry: retry.clone(),
                        handler: handler.clone(),
                    })];

                    if handler == LlmHandler::Worker {
                        events.push(self.emit_decision_request(DecisionTrigger::EffectExecute {
                            id: call_id,
                            attempt: 0,
                            deadline: retry.deadline(chrono::Utc::now()),
                            work: EffectWork::LlmCall { request, stream },
                        }));
                    }

                    Ok(events)
                } else {
                    tracing::debug!(
                        %call_id,
                        "llm call id already issued; request no-ops (idempotent)"
                    );
                    Ok(vec![])
                }
            }

            CommandPayload::CompleteLlmCall {
                call_id,
                attempt,
                response,
            } => {
                let is_system = matches!(caller, Caller::System { .. });
                let call = self.llm_calls.get(&call_id);
                self.check_llm_call_caller(call, caller)?;

                match call {
                    Some(c)
                        if c.tracking.status == EffectStatus::Pending
                            && c.tracking.retry.attempts == attempt =>
                    {
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
                        // The worker folds this assistant into its transcript; the
                        // engine reconciles it (id-less → a fresh node), so no parent
                        // pointer is needed here.
                        let message = Message {
                            id: String::new(),
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
                            self.emit_decision_request(DecisionTrigger::EffectSettled {
                                id: call_id,
                                ok: true,
                                outcome: EffectOutcome::llm_ok(message, truncated, usage, cost),
                            }),
                        ])
                    }
                    // Late/duplicate/stale settlements: silent for the executor
                    // (System) to keep completions idempotent; explicit errors for
                    // an out-of-band Machine caller with no other way to learn its
                    // settlement was dropped.
                    _ if is_system => Ok(vec![]),
                    None => Err(SessionError::EffectNotFound),
                    Some(c) if c.tracking.status != EffectStatus::Pending => {
                        Err(SessionError::EffectNotPending)
                    }
                    Some(_) => Err(SessionError::EffectAttemptMismatch),
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
                let is_system = matches!(caller, Caller::System { .. });
                let call = self.llm_calls.get(&call_id);
                self.check_llm_call_caller(call, caller)?;
                let call = match call {
                    Some(c)
                        if c.tracking.status == EffectStatus::Pending
                            && c.tracking.retry.attempts == attempt =>
                    {
                        c
                    }
                    _ if is_system => return Ok(vec![]),
                    None => return Err(SessionError::EffectNotFound),
                    Some(c) if c.tracking.status != EffectStatus::Pending => {
                        return Err(SessionError::EffectNotPending)
                    }
                    Some(_) => return Err(SessionError::EffectAttemptMismatch),
                };
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
                    events.push(self.emit_decision_request(DecisionTrigger::EffectSettled {
                        id: call_id,
                        ok: false,
                        outcome: EffectOutcome::llm_err(error, code, detail),
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
                            events.push(self.emit_decision_request(
                                DecisionTrigger::EffectExecute {
                                    id: tool_call_id,
                                    attempt: 0,
                                    deadline: retry.deadline(chrono::Utc::now()),
                                    work: EffectWork::ToolCall { name, arguments },
                                },
                            ));
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
                    .ok_or(SessionError::EffectNotFound)?;

                if tc.tracking.status != EffectStatus::Pending {
                    return Err(SessionError::EffectNotPending);
                }

                if tc.tracking.retry.attempts != attempt {
                    return Err(SessionError::EffectAttemptMismatch);
                }

                self.check_tool_call_caller(tc, caller)?;

                let name = tc.name.clone();

                let mut events = vec![EventPayload::ToolCallCompleted(ToolCallCompleted {
                    tool_call_id: tool_call_id.clone(),
                    name: name.clone(),
                    result: result.clone(),
                })];
                if let Some(ws) = worker_state {
                    events.push(EventPayload::WorkerStateUpdated(WorkerStateUpdated {
                        state: ws,
                    }));
                }
                events.push(self.emit_tool_result(tool_call_id, name, result, false));
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
                    .ok_or(SessionError::EffectNotFound)?;
                if tc.tracking.status != EffectStatus::Pending {
                    return Err(SessionError::EffectNotPending);
                }
                if tc.tracking.retry.attempts != attempt {
                    return Err(SessionError::EffectAttemptMismatch);
                }
                self.check_tool_call_caller(tc, caller)?;
                let name = tc.name.clone();
                let exhausted = tc
                    .tracking
                    .retry_policy
                    .exhausted(&tc.tracking.retry, retryable);
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
                if exhausted {
                    events.push(self.emit_tool_result(tool_call_id, name, error, true));
                }
                Ok(events)
            }

            CommandPayload::RequestSubAgent {
                session_id,
                agent_id,
                tool_call_id,
                retry,
            } => {
                Self::ensure_internal(caller)?;
                match self.sub_agent_calls.get(&session_id) {
                    Some(_) => Ok(vec![]),
                    None => Ok(vec![EventPayload::SubAgentRequested(SubAgentRequested {
                        session_id,
                        agent_id,
                        tool_call_id,
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
                let tool_call_id = sa.tool_call_id.clone();
                let agent_id = sa.agent_id.clone();
                let exhausted = sa
                    .tracking
                    .retry_policy
                    .exhausted(&sa.tracking.retry, retryable);
                let mut events = vec![EventPayload::SubAgentErrored(SubAgentErrored {
                    session_id: session_id.clone(),
                    error: error.clone(),
                    retryable,
                })];
                if exhausted {
                    events.push(self.emit_sub_agent_result(
                        session_id,
                        tool_call_id,
                        agent_id,
                        error,
                        true,
                    ));
                }
                Ok(events)
            }

            CommandPayload::CompleteSubAgentTurn {
                session_id,
                data,
                cost,
                token_usage,
                ..
            } => {
                Self::ensure_internal(caller)?;
                // Only fire if we know about this sub-agent.
                let Some(sa) = self.sub_agent_calls.get(&session_id) else {
                    return Ok(vec![]);
                };
                let tool_call_id = sa.tool_call_id.clone();
                let agent_id = sa.agent_id.clone();
                let result = json_to_string(&data);
                let mut events = vec![EventPayload::SubAgentTurnCompleted(SubAgentTurnCompleted {
                    session_id: session_id.clone(),
                    cost,
                    token_usage,
                    data,
                })];
                events.push(self.emit_sub_agent_result(
                    session_id,
                    tool_call_id,
                    agent_id,
                    result,
                    false,
                ));
                Ok(events)
            }

            CommandPayload::Interrupt {
                interrupt_id,
                reason,
                payload,
            } => {
                self.ensure_owns_session(caller)?;
                match self.status {
                    SessionStatus::Interrupted { .. } => Ok(vec![]),
                    _ => Ok(vec![EventPayload::SessionInterrupted(SessionInterrupted {
                        interrupt_id,
                        origin: Self::caller_interrupt_origin(caller),
                        reason,
                        payload,
                    })]),
                }
            }

            CommandPayload::ResumeInterrupt {
                interrupt_id,
                payload,
            } => {
                self.ensure_owns_session(caller)?;
                match self.active_interrupt() {
                    Some((id, origin)) if id == interrupt_id => {
                        if Self::caller_interrupt_origin(caller).privilege() < origin.privilege() {
                            return Err(SessionError::SessionAccessDenied);
                        }
                        // Emitted directly rather than via emit_decision_request:
                        // pre-event state is still Interrupted (which would queue
                        // it), and nothing can be pending — interrupt voided any
                        // in-flight decision. Decisions queued during the pause
                        // promote after this one completes.
                        Ok(vec![
                            EventPayload::InterruptResumed(InterruptResumed {
                                interrupt_id: interrupt_id.clone(),
                                payload: payload.clone(),
                            }),
                            EventPayload::WorkerDecisionRequested(WorkerDecisionRequested {
                                decision_id: new_call_id(),
                                trigger: DecisionTrigger::InterruptResumed {
                                    interrupt_id,
                                    payload,
                                },
                            }),
                        ])
                    }
                    _ => Ok(vec![]),
                }
            }

            CommandPayload::SubmitWorkerDecision {
                decision_id,
                transcript,
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
                // The engine's only tree write: reconcile the worker's returned
                // transcript before acting on its requests.
                events.extend(self.reconcile_transcript(transcript));
                let system = Caller::System {
                    tenant_id: self
                        .owner
                        .as_ref()
                        .map(|o| o.tenant_id.clone())
                        .unwrap_or_default(),
                };
                for action in actions {
                    let sub_events = match action {
                        WorkerAction::CallLlm {
                            id,
                            request,
                            stream,
                            retry,
                            handler,
                        } => self.handle(
                            CommandPayload::RequestLlmCall {
                                call_id: id,
                                request,
                                stream,
                                retry,
                                handler,
                            },
                            &system,
                        ),
                        WorkerAction::CallTool {
                            id,
                            name,
                            arguments,
                            handler,
                            retry,
                        } => self.handle(
                            CommandPayload::RequestToolCall {
                                tool_call_id: id,
                                name,
                                arguments,
                                handler,
                                retry,
                            },
                            &system,
                        ),
                        WorkerAction::SpawnSubAgent {
                            session_id,
                            agent_id,
                            tool_call_id,
                            retry,
                        } => self.handle(
                            CommandPayload::RequestSubAgent {
                                session_id,
                                agent_id,
                                tool_call_id,
                                retry,
                            },
                            &system,
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
                        WorkerAction::Interrupt {
                            interrupt_id,
                            reason,
                            payload,
                        } => match self.status {
                            SessionStatus::Interrupted { .. } => Ok(vec![]),
                            _ => Ok(vec![EventPayload::SessionInterrupted(SessionInterrupted {
                                interrupt_id: if interrupt_id.is_empty() {
                                    new_call_id()
                                } else {
                                    interrupt_id
                                },
                                origin: InterruptOrigin::Frontend,
                                reason,
                                payload,
                            })]),
                        },
                        WorkerAction::EffectResult {
                            id,
                            attempt,
                            result,
                        } => match result {
                            EffectResultPayload::ToolCall { result } => self.handle(
                                CommandPayload::CompleteToolCall {
                                    tool_call_id: id,
                                    attempt,
                                    result,
                                    worker_state: None,
                                },
                                &system,
                            ),
                            EffectResultPayload::LlmCall { response } => self.handle(
                                CommandPayload::CompleteLlmCall {
                                    call_id: id,
                                    attempt,
                                    response,
                                },
                                &system,
                            ),
                        },
                        WorkerAction::EffectError {
                            kind,
                            id,
                            attempt,
                            error,
                            retryable,
                            code,
                            detail,
                        } => match kind {
                            WorkKind::ToolCall => self.handle(
                                CommandPayload::FailToolCall {
                                    tool_call_id: id,
                                    attempt,
                                    error,
                                    retryable,
                                    worker_state: None,
                                },
                                &system,
                            ),
                            WorkKind::LlmCall => self.handle(
                                CommandPayload::FailLlmCall {
                                    call_id: id,
                                    attempt,
                                    error,
                                    retryable,
                                    code,
                                    detail,
                                },
                                &system,
                            ),
                        },
                        WorkerAction::Done { data } => {
                            self.handle(CommandPayload::MarkDone { data }, &system)
                        }
                    };
                    if let Ok(sub) = sub_events {
                        events.extend(sub);
                    }
                }

                // Promote the next queued decision inline so it dispatches
                // in the same commit rather than waiting for a wake cycle —
                // unless an action in this batch interrupted the session, in
                // which case the queue waits for resume.
                let interrupted_in_batch = events
                    .iter()
                    .any(|e| matches!(e, EventPayload::SessionInterrupted(_)));
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
                    if !interrupted_in_batch {
                        events.push(EventPayload::WorkerDecisionRequested(
                            WorkerDecisionRequested {
                                decision_id,
                                trigger,
                            },
                        ));
                    }
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
        // Timed-out pending LLM calls → fail
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

        // Timed-out pending tool calls → fail
        for tc in self.tool_calls.values() {
            if tc.tracking.status == EffectStatus::Pending
                && tc.tracking.deadline.is_some_and(|d| d <= now)
            {
                let mut events = vec![EventPayload::ToolCallErrored(ToolCallErrored {
                    tool_call_id: tc.tool_call_id.clone(),
                    name: tc.name.clone(),
                    error: "deadline exceeded".to_string(),
                    retryable: true,
                })];
                if tc.tracking.retry_policy.exhausted(&tc.tracking.retry, true) {
                    events.push(self.emit_tool_result(
                        tc.tool_call_id.clone(),
                        tc.name.clone(),
                        "deadline exceeded".to_string(),
                        true,
                    ));
                }
                return Ok(events);
            }
        }

        // RetryScheduled LLM calls ready to re-issue
        for call in self.llm_calls.values() {
            if call.tracking.status == EffectStatus::RetryScheduled {
                if let Some(next_at) = call.tracking.retry.next_at {
                    if next_at <= now {
                        let request = call.spec.to_request(call.prompt.clone());
                        let mut events = vec![EventPayload::LlmCallRequested(LlmCallRequested {
                            call_id: call.call_id.clone(),
                            attempt: call.tracking.retry.attempts,
                            request: request.clone(),
                            stream: call.stream,
                            retry: call.tracking.retry_policy.clone(),
                            handler: call.handler.clone(),
                        })];
                        if call.handler == LlmHandler::Worker {
                            events.push(self.emit_decision_request(
                                DecisionTrigger::EffectExecute {
                                    id: call.call_id.clone(),
                                    attempt: call.tracking.retry.attempts,
                                    deadline: call.tracking.retry_policy.deadline(now),
                                    work: EffectWork::LlmCall {
                                        request,
                                        stream: call.stream,
                                    },
                                },
                            ));
                        }
                        return Ok(events);
                    }
                }
            }
        }

        // RetryScheduled tool calls ready to re-issue
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
                            events.push(self.emit_decision_request(
                                DecisionTrigger::EffectExecute {
                                    id: tc.tool_call_id.clone(),
                                    attempt: tc.tracking.retry.attempts,
                                    deadline: tc.tracking.retry_policy.deadline(now),
                                    work: EffectWork::ToolCall {
                                        name: tc.name.clone(),
                                        arguments: tc.arguments.clone(),
                                    },
                                },
                            ));
                        }
                        return Ok(events);
                    }
                }
            }
        }

        // Timed-out pending sub-agent calls → fail
        for sa in self.sub_agent_calls.values() {
            if sa.tracking.status == EffectStatus::Pending
                && sa.tracking.deadline.is_some_and(|d| d <= now)
            {
                let mut events = vec![EventPayload::SubAgentErrored(SubAgentErrored {
                    session_id: sa.session_id.clone(),
                    error: "deadline exceeded".to_string(),
                    retryable: true,
                })];
                if sa.tracking.retry_policy.exhausted(&sa.tracking.retry, true) {
                    events.push(self.emit_sub_agent_result(
                        sa.session_id.clone(),
                        sa.tool_call_id.clone(),
                        sa.agent_id.clone(),
                        "deadline exceeded".to_string(),
                        true,
                    ));
                }
                return Ok(events);
            }
        }

        // RetryScheduled sub-agent calls ready to re-issue
        for sa in self.sub_agent_calls.values() {
            if sa.tracking.status == EffectStatus::RetryScheduled {
                if let Some(next_at) = sa.tracking.retry.next_at {
                    if next_at <= now {
                        return Ok(vec![EventPayload::SubAgentRequested(SubAgentRequested {
                            session_id: sa.session_id.clone(),
                            agent_id: sa.agent_id.clone(),
                            tool_call_id: sa.tool_call_id.clone(),
                            retry: sa.tracking.retry_policy.clone(),
                        })]);
                    }
                }
            }
        }

        // Timed-out pending worker decisions → fail
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

        // RetryScheduled worker decisions ready to re-issue
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

        // Lost `effect.settled`s are recovered by work-queue redelivery, not by re-firing here.
        if let Some(wd) = self.next_queued_decision() {
            return Ok(vec![EventPayload::WorkerDecisionRequested(
                WorkerDecisionRequested {
                    decision_id: wd.decision_id.clone(),
                    trigger: wd.trigger.clone(),
                },
            )]);
        }

        // Nothing left in flight → stall recovery.
        if self.all_tools_resolved()
            && !self.has_pending_llm()
            && !self.has_queued_worker_decision()
            && !self.has_pending_worker_decision()
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
    use crate::runtime::session::decision::{
        ClientPayload, DecisionTrigger, EffectOutcome, EffectResultPayload, EffectWork,
        WorkerAction,
    };
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
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "expected [ToolCallCompleted, WorkerDecisionRequested]; got {events:?}"
        );
        assert_eq!(fired_tool_result(&events), vec!["tc-1".to_string()]);

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
            matches!(err, SessionError::EffectWrongHandler),
            "expected EffectWrongHandler; got {err:?}"
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
        // Realistic async-tool flow: worker sees the effect.execute decision,
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
            .expect("worker-handled tool call emits an effect.execute decision");

        let machine = Caller::Machine {
            tenant_id: "tenant-a".to_string(),
            key_id: "prod-key-1".to_string(),
        };

        // Worker releases its decision with no actions.
        dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: d1,
                transcript: vec![],
                actions: vec![],
                state: vec![].into(),
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
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "expected [ToolCallCompleted, WorkerDecisionRequested]; got {events:?}"
        );
        assert_eq!(fired_tool_result(&events), vec!["tc-1".to_string()]);

        let tc = agg.state.tool_calls.get("tc-1").expect("tool call present");
        assert_eq!(tc.tracking.status, EffectStatus::Completed);
    }

    #[test]
    fn machine_completes_worker_handled_tool_call_before_worker_releases_decision() {
        // Tool result arrives before the worker has acknowledged its
        // effect.execute decision.
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
                    EventPayload::DecisionRequestQueued(_),
                ]
            ),
            "expected [ToolCallCompleted, DecisionRequestQueued]; got {events:?}"
        );
        assert_eq!(fired_tool_result(&events), vec!["tc-1".to_string()]);

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
            matches!(err, SessionError::EffectAttemptMismatch),
            "expected EffectAttemptMismatch; got {err:?}"
        );
    }

    #[test]
    fn submit_client_payload_with_active_turn_id_is_rejected() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let payload = ClientPayload::Message {
            message: Message {
                id: String::new(),
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
                        id: String::new(),
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
                    transcript: vec![],
                    actions: vec![WorkerAction::CallTool {
                        id: "tc-1".to_string(),
                        name: "my_tool".to_string(),
                        arguments: "{}".to_string(),
                        handler: ToolHandler::Worker,
                        retry: RetryPolicy::no_retry(),
                    }],
                    state: vec![].into(),
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
                        id: String::new(),
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
                transcript: vec![],
                actions: vec![],
                state: vec![].into(),
            },
            &machine,
        );

        // Second submission with the same decision_id should be a no-op.
        let events = agg
            .state
            .handle(
                CommandPayload::SubmitWorkerDecision {
                    decision_id,
                    transcript: vec![],
                    actions: vec![WorkerAction::CallTool {
                        id: "tc-1".to_string(),
                        name: "my_tool".to_string(),
                        arguments: "{}".to_string(),
                        handler: ToolHandler::Worker,
                        retry: RetryPolicy::no_retry(),
                    }],
                    state: vec![].into(),
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
                id: String::new(),
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
            matches!(err, SessionError::EffectNotFound),
            "expected EffectNotFound; got {err:?}"
        );
    }

    #[test]
    fn send_message_wakes_a_decision() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");

        let events = dispatch(
            &mut agg,
            CommandPayload::SendMessage {
                message: Message {
                    id: String::new(),
                    role: Role::User,
                    content: Some(Content::Text("hi".to_string())),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
                stream: false,
                turn_id: None,
                parent_id: None,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        assert!(
            matches!(
                events.as_slice(),
                [EventPayload::WorkerDecisionRequested(_)]
            ),
            "expected [WorkerDecisionRequested]; got {events:?}"
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

    fn node_msg(id: &str, role: Role, content: &str) -> Message {
        Message {
            id: id.into(),
            role,
            content: Some(Content::Text(content.into())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    fn request_with(messages: Vec<Message>) -> LlmRequest {
        LlmRequest {
            model: "test-model".to_string(),
            messages,
            tools: None,
            temperature: None,
            max_completion_tokens: None,
            reasoning: None,
        }
    }

    /// Drive a worker `Append` action onto the tree (the worker is the sole
    /// tree author). A user message opens a decision the worker answers with
    /// the nodes.
    fn append_via_worker(agg: &mut Aggregate<SessionState>, nodes: Vec<NewMessage>) {
        let setup = dispatch(
            agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message {
                    message: node_msg("seed", Role::User, "seed"),
                    stream: false,
                },
                turn_id: None,
            },
            &system(),
        );
        let decision_id = setup
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.decision_id.clone()),
                _ => None,
            })
            .expect("user message requests a worker decision");
        dispatch(
            agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id,
                transcript: nodes.into_iter().map(|n| n.message).collect(),
                actions: vec![],
                state: vec![].into(),
            },
            &machine(),
        );
    }

    #[test]
    fn request_llm_call_stores_prompt_without_minting_nodes() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let events = dispatch(
            &mut agg,
            CommandPayload::RequestLlmCall {
                call_id: "llm-1".to_string(),
                request: request_with(vec![
                    node_msg("sys", Role::System, "be helpful"),
                    node_msg("u1", Role::User, "hi"),
                ]),
                stream: false,
                retry: RetryPolicy::no_retry(),
                handler: LlmHandler::Server,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        // The worker is the sole tree author; the call records the verbatim
        // prompt the worker sent but mints no tree nodes itself.
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, EventPayload::NewMessage(_))),
            "request mints no tree nodes; got {events:?}"
        );
        let call = agg.state.llm_calls.get("llm-1").expect("llm call present");
        assert_eq!(
            call.prompt
                .iter()
                .map(|m| m.id.as_str())
                .collect::<Vec<_>>(),
            vec!["sys", "u1"]
        );
    }

    #[test]
    fn submit_messages_forwards_a_replace_submission() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let events = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Messages {
                    messages: vec![
                        node_msg("c1", Role::User, "hi"),
                        node_msg("a1", Role::Assistant, "hello"),
                        node_msg("c2", Role::User, "more"),
                    ],
                    stream: false,
                },
                turn_id: None,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        assert!(!events
            .iter()
            .any(|e| matches!(e, EventPayload::NewMessage(_))));
        let trigger = events
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(w) => Some(&w.trigger),
                EventPayload::DecisionRequestQueued(q) => Some(&q.trigger),
                _ => None,
            })
            .expect("a decision request");
        match trigger {
            DecisionTrigger::UserTranscript { messages } => {
                assert_eq!(
                    messages.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
                    vec!["c1", "a1", "c2"]
                );
            }
            t => panic!("expected a UserTranscript trigger; got {t:?}"),
        }
    }

    fn tool_msg(tool_call_id: &str, content: &str) -> Message {
        Message {
            id: String::new(),
            role: Role::Tool,
            content: Some(Content::Text(content.into())),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            name: None,
        }
    }

    fn submit_messages(
        agg: &mut Aggregate<SessionState>,
        messages: Vec<Message>,
    ) -> Vec<EventPayload> {
        dispatch(
            agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Messages {
                    messages,
                    stream: false,
                },
                turn_id: None,
            },
            &system(),
        )
    }

    #[test]
    fn submit_messages_resolves_a_client_tool_result() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_client_tool(&mut agg, "tc-1");

        let events = submit_messages(&mut agg, vec![tool_msg("tc-1", "the answer")]);

        assert!(events
            .iter()
            .any(|e| matches!(e, EventPayload::ToolCallCompleted(_))));
        assert_eq!(fired_tool_result(&events), vec!["tc-1".to_string()]);
        assert!(
            decision_with(&events, |t| matches!(
                t,
                DecisionTrigger::UserMessage { .. }
            ))
            .is_none(),
            "a tool-result submission is not a user turn"
        );

        let tc = agg.state.tool_calls.get("tc-1").expect("tool call present");
        assert_eq!(tc.tracking.status, EffectStatus::Completed);
        assert_eq!(tc.result.as_deref(), Some("the answer"));
    }

    #[test]
    fn submit_messages_fires_a_tool_result_per_client_completion() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_client_tool(&mut agg, "a");
        request_client_tool(&mut agg, "b");

        let first = submit_messages(&mut agg, vec![tool_msg("a", "RA")]);
        assert_eq!(fired_tool_result(&first), vec!["a".to_string()]);
        assert_eq!(
            agg.state.tool_calls.get("a").unwrap().result.as_deref(),
            Some("RA")
        );

        let second = submit_messages(&mut agg, vec![tool_msg("b", "RB")]);
        assert_eq!(fired_tool_result(&second), vec!["b".to_string()]);
        assert_eq!(
            agg.state.tool_calls.get("b").unwrap().result.as_deref(),
            Some("RB")
        );
    }

    #[test]
    fn submit_messages_resolving_several_client_tools_fires_a_tool_result_each() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_client_tool(&mut agg, "a");
        request_client_tool(&mut agg, "b");

        let events = submit_messages(&mut agg, vec![tool_msg("a", "RA"), tool_msg("b", "RB")]);
        // Each completion is immediately followed by its decision, so the live
        // decision's snapshot is taken while the next sibling is still in flight
        // (counted in `pending`) — the worker won't prompt before it lands.
        let sequence: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                EventPayload::ToolCallCompleted(_) => Some("complete"),
                EventPayload::WorkerDecisionRequested(p)
                    if matches!(
                        p.trigger,
                        DecisionTrigger::EffectSettled {
                            outcome: EffectOutcome::ToolCall { .. },
                            ..
                        }
                    ) =>
                {
                    Some("live")
                }
                EventPayload::DecisionRequestQueued(p)
                    if matches!(
                        p.trigger,
                        DecisionTrigger::EffectSettled {
                            outcome: EffectOutcome::ToolCall { .. },
                            ..
                        }
                    ) =>
                {
                    Some("queued")
                }
                _ => None,
            })
            .collect();
        assert_eq!(sequence, vec!["complete", "live", "complete", "queued"]);
        assert_eq!(
            agg.state.tool_calls.get("a").unwrap().result.as_deref(),
            Some("RA")
        );
        assert_eq!(
            agg.state.tool_calls.get("b").unwrap().result.as_deref(),
            Some("RB")
        );
    }

    #[test]
    fn submit_messages_echoing_a_resolved_tool_result_is_inert() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_client_tool(&mut agg, "tc-1");
        submit_messages(&mut agg, vec![tool_msg("tc-1", "done")]);

        // A re-sent, already-resolved tool result matches nothing pending and must not fabricate a response.
        let echo = submit_messages(&mut agg, vec![tool_msg("tc-1", "done")]);
        assert!(
            !echo
                .iter()
                .any(|e| matches!(e, EventPayload::ToolCallCompleted(_))),
            "nothing to complete; got {echo:?}"
        );
        assert!(
            decision_with(&echo, |_| true).is_none(),
            "no worker decision; got {echo:?}"
        );
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
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "expected [LlmCallCompleted, WorkerDecisionRequested]; got {events:?}"
        );
        assert!(
            decision_with(&events, |t| matches!(
                t,
                DecisionTrigger::EffectSettled {
                    id,
                    ok: true,
                    outcome: EffectOutcome::LlmCall { .. },
                } if id == "llm-1"
            ))
            .is_some(),
            "completion fires an effect.settled trigger; got {events:?}"
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

    #[test]
    fn llm_retry_reuses_the_stored_prompt() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let retry = RetryPolicy {
            timeout_secs: None,
            max_retries: 3,
            backoff_base_secs: 1,
            backoff_max_secs: 1,
        };
        // The call stores its prompt verbatim; the retry re-issues that stored
        // prompt rather than rebuilding it from the tree.
        append_via_worker(
            &mut agg,
            vec![
                NewMessage {
                    parent_id: None,
                    message: node_msg("sys", Role::System, "sys prompt"),
                },
                NewMessage {
                    parent_id: Some("sys".to_string()),
                    message: node_msg("u1", Role::User, "hi"),
                },
            ],
        );
        dispatch(
            &mut agg,
            CommandPayload::RequestLlmCall {
                call_id: "llm-1".to_string(),
                request: request_with(vec![
                    node_msg("sys", Role::System, "sys prompt"),
                    node_msg("u1", Role::User, "hi"),
                ]),
                stream: false,
                retry,
                handler: LlmHandler::Server,
            },
            &system(),
        );

        let call = agg.state.llm_calls.get("llm-1").expect("call present");
        assert_eq!(
            call.prompt
                .iter()
                .map(|m| m.id.as_str())
                .collect::<Vec<_>>(),
            vec!["sys", "u1"]
        );

        dispatch(
            &mut agg,
            CommandPayload::FailLlmCall {
                call_id: "llm-1".to_string(),
                attempt: 0,
                error: "provider hiccup".to_string(),
                retryable: true,
                code: None,
                detail: None,
            },
            &system(),
        );
        assert_eq!(
            agg.state.llm_calls.get("llm-1").map(|c| &c.tracking.status),
            Some(&EffectStatus::RetryScheduled)
        );

        let later = Utc::now() + chrono::Duration::seconds(120);
        let events = dispatch(&mut agg, CommandPayload::Wake { now: later }, &system());
        let reissued = events
            .iter()
            .find_map(|e| match e {
                EventPayload::LlmCallRequested(p) => Some(p),
                _ => None,
            })
            .expect("retry re-issues the llm call");

        assert_eq!(reissued.request.model, "test-model");
        assert_eq!(reissued.request.messages.len(), 2);
        assert_eq!(reissued.request.messages[0].id, "sys");
        assert_eq!(reissued.request.messages[1].id, "u1");
        assert!(matches!(
            &reissued.request.messages[1].content,
            Some(Content::Text(t)) if t == "hi"
        ));
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
            matches!(
                trigger,
                DecisionTrigger::EffectExecute {
                    id,
                    work: EffectWork::LlmCall { .. },
                    ..
                } if id == "llm-1"
            ),
            "expected an effect.execute trigger for the llm call; got {trigger:?}"
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
            .expect("worker-handled llm call emits an effect.execute decision");

        let machine = Caller::Machine {
            tenant_id: "tenant-a".to_string(),
            key_id: "prod-key-1".to_string(),
        };

        let events = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id,
                transcript: vec![],
                actions: vec![WorkerAction::EffectResult {
                    id: "llm-1".to_string(),
                    attempt: 0,
                    result: EffectResultPayload::LlmCall {
                        response: LlmResponse {
                            model: "test-model".to_string(),
                            content: Some("hello from the worker".to_string()),
                            tool_calls: vec![],
                            finish_reason: Some("stop".to_string()),
                            usage: None,
                            cost: None,
                            images: vec![],
                        },
                    },
                }],
                state: vec![].into(),
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
            decision_with(&events, |t| matches!(
                t,
                DecisionTrigger::EffectSettled {
                    id,
                    ok: true,
                    outcome: EffectOutcome::LlmCall { .. },
                } if id == "llm-1"
            ))
            .is_some(),
            "completion fires an effect.settled trigger; got {events:?}"
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
            .expect("worker-handled tool call emits an effect.execute decision");

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
                transcript: vec![],
                actions: vec![],
                state: vec![].into(),
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
        let trigger = events
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(&p.trigger),
                EventPayload::DecisionRequestQueued(p) => Some(&p.trigger),
                _ => None,
            })
            .expect("an effect.settled decision");
        assert!(
            matches!(
                trigger,
                DecisionTrigger::EffectSettled {
                    id,
                    ok: false,
                    outcome: EffectOutcome::ToolCall { .. },
                } if id == "tc-1"
            ),
            "expected an errored effect.settled for tc-1; got {trigger:?}"
        );
        let tc = agg.state.tool_calls.get("tc-1").expect("tool call present");
        assert_eq!(tc.tracking.status, EffectStatus::Failed);
        assert!(tc.is_error);
        assert_eq!(tc.result.as_deref(), Some("boom"));
    }

    #[test]
    fn request_sub_agent_emits_requested() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");

        let events = dispatch(
            &mut agg,
            CommandPayload::RequestSubAgent {
                session_id: "child-1".to_string(),
                agent_id: "agent-2".to_string(),
                tool_call_id: "call-sa".to_string(),
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
                tool_call_id: "call-sa".to_string(),
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
                tool_call_id: "call-sa".to_string(),
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
        assert_eq!(fired_tool_result(&events), vec!["call-sa".to_string()]);
        let sa = agg
            .state
            .sub_agent_calls
            .get("child-1")
            .expect("sub-agent present");
        assert_eq!(sa.tracking.status, EffectStatus::Failed);
        assert_eq!(sa.result.as_deref(), Some("child crashed"));
        assert!(sa.is_error);
    }

    #[test]
    fn complete_sub_agent_turn_emits_completed() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::RequestSubAgent {
                session_id: "child-1".to_string(),
                agent_id: "agent-2".to_string(),
                tool_call_id: "call-sa".to_string(),
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
                data: serde_json::json!("done"),
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
        assert_eq!(fired_tool_result(&events), vec!["call-sa".to_string()]);
        let sa = agg
            .state
            .sub_agent_calls
            .get("child-1")
            .expect("sub-agent present");
        assert_eq!(sa.result.as_deref(), Some("done"));
        assert!(!sa.is_error);
    }

    // ── Batched effect completion ────────────────────────────────────────

    fn machine() -> Caller {
        Caller::Machine {
            tenant_id: "tenant-a".to_string(),
            key_id: "prod-key-1".to_string(),
        }
    }

    fn system() -> Caller {
        Caller::System {
            tenant_id: "tenant-a".to_string(),
        }
    }

    fn request_client_tool(agg: &mut Aggregate<SessionState>, id: &str) {
        dispatch(
            agg,
            CommandPayload::RequestToolCall {
                tool_call_id: id.to_string(),
                name: format!("tool_{id}"),
                arguments: "{}".to_string(),
                handler: ToolHandler::Client,
                retry: RetryPolicy::no_retry(),
            },
            &system(),
        );
    }

    fn complete_tool(
        agg: &mut Aggregate<SessionState>,
        id: &str,
        result: &str,
    ) -> Vec<EventPayload> {
        dispatch(
            agg,
            CommandPayload::CompleteToolCall {
                tool_call_id: id.to_string(),
                attempt: 0,
                result: result.to_string(),
                worker_state: None,
            },
            &machine(),
        )
    }

    fn wake(agg: &mut Aggregate<SessionState>) -> Vec<EventPayload> {
        dispatch(agg, CommandPayload::Wake { now: Utc::now() }, &system())
    }

    /// The `tool_call_id`s of every tool/sub-agent `effect.settled` trigger in
    /// the events, in order. A queued decision that is promoted in the same
    /// commit surfaces as both `DecisionRequestQueued` and
    /// `WorkerDecisionRequested`; dedup by `decision_id` so each settle counts
    /// once.
    fn fired_tool_result(events: &[EventPayload]) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        events
            .iter()
            .filter_map(|e| {
                let (decision_id, trigger) = match e {
                    EventPayload::WorkerDecisionRequested(p) => (&p.decision_id, &p.trigger),
                    EventPayload::DecisionRequestQueued(p) => (&p.decision_id, &p.trigger),
                    _ => return None,
                };
                let tool_call_id = match trigger {
                    DecisionTrigger::EffectSettled {
                        id,
                        outcome: EffectOutcome::ToolCall { .. },
                        ..
                    } => id,
                    DecisionTrigger::EffectSettled {
                        outcome: EffectOutcome::SubAgent { tool_call_id, .. },
                        ..
                    } => tool_call_id,
                    _ => return None,
                };
                seen.insert(decision_id.clone())
                    .then(|| tool_call_id.clone())
            })
            .collect()
    }

    fn decision_with(
        events: &[EventPayload],
        pred: impl Fn(&DecisionTrigger) -> bool,
    ) -> Option<String> {
        events.iter().find_map(|e| {
            let (id, trigger) = match e {
                EventPayload::WorkerDecisionRequested(p) => (&p.decision_id, &p.trigger),
                EventPayload::DecisionRequestQueued(p) => (&p.decision_id, &p.trigger),
                _ => return None,
            };
            pred(trigger).then(|| id.clone())
        })
    }

    #[test]
    fn each_completion_fires_a_tool_result() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_client_tool(&mut agg, "a");
        request_client_tool(&mut agg, "b");

        let first = complete_tool(&mut agg, "a", "RA");
        assert_eq!(fired_tool_result(&first), vec!["a".to_string()]);
        assert_eq!(
            agg.state.tool_calls.get("a").unwrap().result.as_deref(),
            Some("RA")
        );

        let second = complete_tool(&mut agg, "b", "RB");
        assert_eq!(fired_tool_result(&second), vec!["b".to_string()]);
        assert_eq!(
            agg.state.tool_calls.get("b").unwrap().result.as_deref(),
            Some("RB")
        );
    }

    #[test]
    fn worker_tool_fires_tool_result_in_the_completion_commit() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let setup = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message {
                    message: Message {
                        id: String::new(),
                        role: Role::User,
                        content: Some(Content::Text("go".to_string())),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    },
                    stream: false,
                },
                turn_id: Some("turn-1".to_string()),
            },
            &system(),
        );
        let decision_id =
            decision_with(&setup, |t| matches!(t, DecisionTrigger::UserMessage { .. }))
                .expect("user message decision");

        let dispatched = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id,
                transcript: vec![],
                actions: vec![WorkerAction::CallTool {
                    id: "t1".to_string(),
                    name: "getWeather".to_string(),
                    arguments: "{}".to_string(),
                    handler: ToolHandler::Worker,
                    retry: RetryPolicy::no_retry(),
                }],
                state: vec![].into(),
            },
            &machine(),
        );
        let exec = decision_with(&dispatched, |t| {
            matches!(
                t,
                DecisionTrigger::EffectExecute {
                    work: EffectWork::ToolCall { .. },
                    ..
                }
            )
        })
        .expect("effect.execute decision");

        let completed = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: exec,
                transcript: vec![],
                actions: vec![WorkerAction::EffectResult {
                    id: "t1".to_string(),
                    attempt: 0,
                    result: EffectResultPayload::ToolCall {
                        result: "RA".to_string(),
                    },
                }],
                state: vec![].into(),
            },
            &machine(),
        );
        assert_eq!(
            fired_tool_result(&completed),
            vec!["t1".to_string()],
            "effect.settled fires in the completion commit; got {completed:?}"
        );
        assert_eq!(
            agg.state.tool_calls.get("t1").unwrap().result.as_deref(),
            Some("RA")
        );
        assert!(
            fired_tool_result(&wake(&mut agg)).is_empty(),
            "no wake needed"
        );
    }

    #[test]
    fn batch_mixes_tool_and_sub_agent() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_client_tool(&mut agg, "t1");
        dispatch(
            &mut agg,
            CommandPayload::RequestSubAgent {
                session_id: "child-1".to_string(),
                agent_id: "researcher".to_string(),
                tool_call_id: "s1".to_string(),
                retry: RetryPolicy::no_retry(),
            },
            &system(),
        );
        dispatch(
            &mut agg,
            CommandPayload::StartSubAgent {
                session_id: "child-1".to_string(),
            },
            &system(),
        );

        let tool_done = complete_tool(&mut agg, "t1", "TOOL");
        assert_eq!(fired_tool_result(&tool_done), vec!["t1".to_string()]);
        assert_eq!(
            agg.state.tool_calls.get("t1").unwrap().result.as_deref(),
            Some("TOOL")
        );

        let sub_done = dispatch(
            &mut agg,
            CommandPayload::CompleteSubAgentTurn {
                session_id: "child-1".to_string(),
                agent_id: "researcher".to_string(),
                turn_id: "turn-1".to_string(),
                data: serde_json::json!("FINDINGS"),
                cost: rust_decimal::Decimal::ZERO,
                token_usage: std::collections::BTreeMap::new(),
            },
            &system(),
        );

        assert_eq!(fired_tool_result(&sub_done), vec!["s1".to_string()]);
        let sa = agg
            .state
            .sub_agent_calls
            .get("child-1")
            .expect("sub-agent present");
        assert_eq!(sa.tool_call_id, "s1");
        assert_eq!(sa.agent_id, "researcher");
        assert_eq!(sa.result.as_deref(), Some("FINDINGS"));
    }

    #[test]
    fn tool_and_sub_agent_from_one_turn_dispatch_concurrently() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let setup = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message {
                    message: Message {
                        id: String::new(),
                        role: Role::User,
                        content: Some(Content::Text("go".to_string())),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    },
                    stream: false,
                },
                turn_id: Some("turn-1".to_string()),
            },
            &system(),
        );
        let decision_id = setup
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.decision_id.clone()),
                _ => None,
            })
            .expect("user message requests a worker decision");

        let events = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id,
                transcript: vec![],
                actions: vec![
                    WorkerAction::CallTool {
                        id: "t1".to_string(),
                        name: "getWeather".to_string(),
                        arguments: "{}".to_string(),
                        handler: ToolHandler::Worker,
                        retry: RetryPolicy::no_retry(),
                    },
                    WorkerAction::SpawnSubAgent {
                        session_id: "child-1".to_string(),
                        agent_id: "researcher".to_string(),
                        tool_call_id: "s1".to_string(),
                        retry: RetryPolicy::no_retry(),
                    },
                ],
                state: vec![].into(),
            },
            &machine(),
        );

        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::ToolCallRequested(_))),
            "tool dispatched; got {events:?}"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::SubAgentRequested(_))),
            "sub-agent dispatched; got {events:?}"
        );
        assert!(
            events.iter().any(|e| matches!(
                e,
                EventPayload::WorkerDecisionRequested(p)
                    if matches!(
                        p.trigger,
                        DecisionTrigger::EffectExecute {
                            work: EffectWork::ToolCall { .. },
                            ..
                        }
                    )
            )),
            "tool's effect.execute decision dispatched; got {events:?}"
        );

        assert_eq!(
            agg.state.tool_calls.get("t1").map(|t| &t.tracking.status),
            Some(&EffectStatus::Pending)
        );
        assert_eq!(
            agg.state
                .sub_agent_calls
                .get("child-1")
                .map(|s| &s.tracking.status),
            Some(&EffectStatus::Pending)
        );
        assert!(
            fired_tool_result(&events).is_empty(),
            "nothing has completed yet"
        );
    }

    #[test]
    fn timed_out_effect_fires_tool_result_via_wake() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::RequestToolCall {
                tool_call_id: "t1".to_string(),
                name: "tool_t1".to_string(),
                arguments: "{}".to_string(),
                handler: ToolHandler::Client,
                retry: RetryPolicy {
                    timeout_secs: Some(60),
                    max_retries: 0,
                    backoff_base_secs: 0,
                    backoff_max_secs: 0,
                },
            },
            &system(),
        );

        let past = Utc::now() + chrono::Duration::seconds(120);
        let errored = dispatch(&mut agg, CommandPayload::Wake { now: past }, &system());
        assert!(
            errored
                .iter()
                .any(|e| matches!(e, EventPayload::ToolCallErrored(_))),
            "deadline exceeded errors the tool; got {errored:?}"
        );
        assert_eq!(
            fired_tool_result(&errored),
            vec!["t1".to_string()],
            "the timed-out effect fires an effect.settled"
        );
        assert!(
            fired_tool_result(&wake(&mut agg)).is_empty(),
            "the next wake does not re-fire"
        );
    }

    #[test]
    fn completion_fires_tool_result_once_and_wake_does_not_re_fire() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_client_tool(&mut agg, "a");

        assert!(
            !fired_tool_result(&complete_tool(&mut agg, "a", "RA")).is_empty(),
            "fires on completion"
        );
        assert!(
            fired_tool_result(&wake(&mut agg)).is_empty(),
            "wake does not re-fire"
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
    fn machine_cannot_resume_system_interrupt() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::Interrupt {
                interrupt_id: "int-1".to_string(),
                reason: "budget_exhausted".to_string(),
                payload: serde_json::Value::Null,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        let err = agg
            .state
            .handle(
                CommandPayload::ResumeInterrupt {
                    interrupt_id: "int-1".to_string(),
                    payload: serde_json::Value::Null,
                },
                &Caller::Machine {
                    tenant_id: "tenant-a".to_string(),
                    key_id: "prod-key-1".to_string(),
                },
            )
            .expect_err("machine caller should not resume a system interrupt");
        assert!(
            matches!(err, SessionError::SessionAccessDenied),
            "expected SessionAccessDenied; got {err:?}"
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
            "system caller should resume a system interrupt; got {events:?}"
        );
    }

    #[test]
    fn machine_resumes_machine_interrupt() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let machine = Caller::Machine {
            tenant_id: "tenant-a".to_string(),
            key_id: "prod-key-1".to_string(),
        };
        dispatch(
            &mut agg,
            CommandPayload::Interrupt {
                interrupt_id: "int-1".to_string(),
                reason: "awaiting approval".to_string(),
                payload: serde_json::Value::Null,
            },
            &machine,
        );
        assert!(matches!(
            agg.state.status,
            SessionStatus::Interrupted {
                origin: InterruptOrigin::Machine,
                ..
            }
        ));

        let events = dispatch(
            &mut agg,
            CommandPayload::ResumeInterrupt {
                interrupt_id: "int-1".to_string(),
                payload: serde_json::Value::Null,
            },
            &machine,
        );
        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::InterruptResumed(_),
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "machine caller should resume its own interrupt; got {events:?}"
        );
        assert_eq!(agg.state.status, SessionStatus::Idle);
    }

    fn frontend_caller(tenant_id: &str, user_id: &str) -> Caller {
        Caller::Frontend {
            tenant_id: tenant_id.to_string(),
            user_id: user_id.to_string(),
            attrs: HashMap::new(),
        }
    }

    #[test]
    fn frontend_interrupts_and_resumes_own_session() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let frontend = frontend_caller("tenant-a", "user-1");

        dispatch(
            &mut agg,
            CommandPayload::Interrupt {
                interrupt_id: "int-1".to_string(),
                reason: "user paused".to_string(),
                payload: serde_json::Value::Null,
            },
            &frontend,
        );
        assert!(matches!(
            agg.state.status,
            SessionStatus::Interrupted {
                origin: InterruptOrigin::Frontend,
                ..
            }
        ));

        let events = dispatch(
            &mut agg,
            CommandPayload::ResumeInterrupt {
                interrupt_id: "int-1".to_string(),
                payload: serde_json::Value::Null,
            },
            &frontend,
        );
        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::InterruptResumed(_),
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "frontend owner should resume its own interrupt; got {events:?}"
        );
        assert_eq!(agg.state.status, SessionStatus::Idle);
    }

    #[test]
    fn non_owner_frontend_cannot_interrupt() {
        let agg = create_session("sess-1", "tenant-a", "user-1");

        let err = agg
            .state
            .handle(
                CommandPayload::Interrupt {
                    interrupt_id: "int-1".to_string(),
                    reason: "user paused".to_string(),
                    payload: serde_json::Value::Null,
                },
                &frontend_caller("tenant-a", "user-2"),
            )
            .expect_err("frontend caller should not interrupt another user's session");
        assert!(
            matches!(err, SessionError::SessionAccessDenied),
            "expected SessionAccessDenied; got {err:?}"
        );
    }

    #[test]
    fn non_owner_frontend_cannot_resume() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::Interrupt {
                interrupt_id: "int-1".to_string(),
                reason: "user paused".to_string(),
                payload: serde_json::Value::Null,
            },
            &frontend_caller("tenant-a", "user-1"),
        );

        let err = agg
            .state
            .handle(
                CommandPayload::ResumeInterrupt {
                    interrupt_id: "int-1".to_string(),
                    payload: serde_json::Value::Null,
                },
                &frontend_caller("tenant-a", "user-2"),
            )
            .expect_err("frontend caller should not resume another user's session");
        assert!(
            matches!(err, SessionError::SessionAccessDenied),
            "expected SessionAccessDenied; got {err:?}"
        );

        let err = agg
            .state
            .handle(
                CommandPayload::ResumeInterrupt {
                    interrupt_id: "int-1".to_string(),
                    payload: serde_json::Value::Null,
                },
                &frontend_caller("tenant-b", "user-1"),
            )
            .expect_err("frontend caller from another tenant should be denied");
        assert!(
            matches!(err, SessionError::SessionAccessDenied),
            "expected SessionAccessDenied; got {err:?}"
        );
    }

    #[test]
    fn frontend_cannot_resume_machine_interrupt() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::Interrupt {
                interrupt_id: "int-1".to_string(),
                reason: "awaiting approval".to_string(),
                payload: serde_json::Value::Null,
            },
            &Caller::Machine {
                tenant_id: "tenant-a".to_string(),
                key_id: "prod-key-1".to_string(),
            },
        );

        let err = agg
            .state
            .handle(
                CommandPayload::ResumeInterrupt {
                    interrupt_id: "int-1".to_string(),
                    payload: serde_json::Value::Null,
                },
                &frontend_caller("tenant-a", "user-1"),
            )
            .expect_err("frontend caller should not resume a machine interrupt");
        assert!(
            matches!(err, SessionError::SessionAccessDenied),
            "expected SessionAccessDenied; got {err:?}"
        );
    }

    #[test]
    fn machine_resumes_frontend_interrupt() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::Interrupt {
                interrupt_id: "int-1".to_string(),
                reason: "user paused".to_string(),
                payload: serde_json::Value::Null,
            },
            &frontend_caller("tenant-a", "user-1"),
        );

        let events = dispatch(
            &mut agg,
            CommandPayload::ResumeInterrupt {
                interrupt_id: "int-1".to_string(),
                payload: serde_json::Value::Null,
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
                    EventPayload::InterruptResumed(_),
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "machine caller should resume a frontend interrupt; got {events:?}"
        );
    }

    #[test]
    fn client_action_rejected_while_interrupted() {
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

        let err = agg
            .state
            .handle(
                CommandPayload::SubmitClientPayload {
                    payload: ClientPayload::Action {
                        action: crate::session::decision::ClientAction {
                            name: "refresh".to_string(),
                            args: None,
                        },
                    },
                    turn_id: None,
                },
                &Caller::System {
                    tenant_id: "tenant-a".to_string(),
                },
            )
            .expect_err("client actions should be rejected while interrupted");
        assert!(
            matches!(err, SessionError::SessionInterrupted),
            "expected SessionInterrupted; got {err:?}"
        );
    }

    #[test]
    fn cancel_voids_pending_effects() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let setup_events = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message {
                    message: Message {
                        id: String::new(),
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

        dispatch(
            &mut agg,
            CommandPayload::CancelSession,
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        assert!(!agg.state.has_pending_worker_decision());

        let stale = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id,
                transcript: vec![],
                actions: vec![WorkerAction::Done {
                    data: serde_json::Value::Null,
                }],
                state: vec![].into(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        assert!(
            stale.is_empty(),
            "stale submission after cancel should no-op; got {stale:?}"
        );
    }

    #[test]
    fn interrupt_voids_pending_worker_decision() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let setup_events = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message {
                    message: Message {
                        id: String::new(),
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

        dispatch(
            &mut agg,
            CommandPayload::Interrupt {
                interrupt_id: "int-1".to_string(),
                reason: "quota_exhausted".to_string(),
                payload: serde_json::Value::Null,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        assert!(!agg.state.has_pending_worker_decision());

        // A late submission from the worker is a no-op, not an error.
        let stale = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id,
                transcript: vec![],
                actions: vec![WorkerAction::Done {
                    data: serde_json::Value::Null,
                }],
                state: vec![].into(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        assert!(
            stale.is_empty(),
            "stale submission should no-op; got {stale:?}"
        );

        // Resume requests a fresh decision immediately rather than queueing
        // it behind the voided one.
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
            "expected immediate WorkerDecisionRequested; got {events:?}"
        );
    }

    #[test]
    fn tool_result_during_interrupt_queues_until_resume() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let system = Caller::System {
            tenant_id: "tenant-a".to_string(),
        };
        let setup_events = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message {
                    message: Message {
                        id: String::new(),
                        role: Role::User,
                        content: Some(Content::Text("crawl the site".to_string())),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    },
                    stream: false,
                },
                turn_id: Some("turn-1".to_string()),
            },
            &system,
        );
        let decision_id = setup_events
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.decision_id.clone()),
                _ => None,
            })
            .expect("user message should request a worker decision");

        dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id,
                transcript: vec![],
                actions: vec![WorkerAction::CallTool {
                    id: "tc-1".to_string(),
                    name: "crawl".to_string(),
                    arguments: "{}".to_string(),
                    handler: ToolHandler::Worker,
                    retry: RetryPolicy::no_retry(),
                }],
                state: vec![].into(),
            },
            &system,
        );

        dispatch(
            &mut agg,
            CommandPayload::Interrupt {
                interrupt_id: "int-1".to_string(),
                reason: "quota_exhausted".to_string(),
                payload: serde_json::Value::Null,
            },
            &system,
        );

        // The late tool result is recorded but the decision it drains into
        // is queued, not delivered.
        let events = dispatch(
            &mut agg,
            CommandPayload::CompleteToolCall {
                tool_call_id: "tc-1".to_string(),
                attempt: 0,
                result: "done".to_string(),
                worker_state: None,
            },
            &system,
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::ToolCallCompleted(_))),
            "tool result should be recorded; got {events:?}"
        );
        assert_eq!(
            agg.state.tool_calls.get("tc-1").unwrap().result.as_deref(),
            Some("done")
        );
        // The effect.settled trigger is queued while interrupted; only its delivery is deferred.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::DecisionRequestQueued(_))),
            "decision should be queued while interrupted; got {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, EventPayload::WorkerDecisionRequested(_))),
            "no decision should be delivered while interrupted; got {events:?}"
        );

        // Resume delivers interrupt.resumed first...
        let events = dispatch(
            &mut agg,
            CommandPayload::ResumeInterrupt {
                interrupt_id: "int-1".to_string(),
                payload: serde_json::Value::Null,
            },
            &system,
        );
        let resumed_decision_id = events
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p)
                    if matches!(p.trigger, DecisionTrigger::InterruptResumed { .. }) =>
                {
                    Some(p.decision_id.clone())
                }
                _ => None,
            })
            .expect("resume should request an interrupt.resumed decision");

        let events = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: resumed_decision_id,
                transcript: vec![],
                actions: vec![],
                state: vec![].into(),
            },
            &system,
        );
        let trigger = events
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.trigger.clone()),
                _ => None,
            })
            .expect("queued decision should promote after the resumed decision completes");
        assert!(
            matches!(
                trigger,
                DecisionTrigger::EffectSettled {
                    outcome: EffectOutcome::ToolCall { .. },
                    ..
                }
            ),
            "expected an effect.settled trigger; got {trigger:?}"
        );
    }

    #[test]
    fn worker_interrupt_action_pauses_session_and_resume_carries_payload() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let setup_events = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message {
                    message: Message {
                        id: String::new(),
                        role: Role::User,
                        content: Some(Content::Text("send the email".to_string())),
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
            CommandPayload::SubmitWorkerDecision {
                decision_id,
                transcript: vec![],
                actions: vec![WorkerAction::Interrupt {
                    interrupt_id: "int-1".to_string(),
                    reason: "confirmation".to_string(),
                    payload: serde_json::json!({"message": "Send the email?"}),
                }],
                state: vec![].into(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::SessionInterrupted(_))),
            "interrupt action should emit SessionInterrupted; got {events:?}"
        );
        assert!(matches!(
            agg.state.status,
            SessionStatus::Interrupted {
                origin: InterruptOrigin::Frontend,
                ..
            }
        ));

        let events = dispatch(
            &mut agg,
            CommandPayload::ResumeInterrupt {
                interrupt_id: "int-1".to_string(),
                payload: serde_json::json!({"approved": true}),
            },
            &frontend_caller("tenant-a", "user-1"),
        );
        let trigger = events
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.trigger.clone()),
                _ => None,
            })
            .expect("resume should request a worker decision");
        match trigger {
            DecisionTrigger::InterruptResumed {
                interrupt_id,
                payload,
            } => {
                assert_eq!(interrupt_id, "int-1");
                assert_eq!(payload, serde_json::json!({"approved": true}));
            }
            other => panic!("expected InterruptResumed trigger; got {other:?}"),
        }
    }

    #[test]
    fn fail_worker_decision_emits_errored() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let setup_events = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message {
                    message: Message {
                        id: String::new(),
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

    #[test]
    fn parallel_tool_results_record_in_completion_order() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let sys = Caller::System {
            tenant_id: "tenant-a".to_string(),
        };

        let request = |agg: &mut Aggregate<SessionState>, id: &str| {
            dispatch(
                agg,
                CommandPayload::RequestToolCall {
                    tool_call_id: id.to_string(),
                    name: "t".to_string(),
                    arguments: "{}".to_string(),
                    handler: ToolHandler::Client,
                    retry: RetryPolicy::no_retry(),
                },
                &sys,
            );
        };
        request(&mut agg, "tc-a");
        request(&mut agg, "tc-b");

        let complete = |agg: &mut Aggregate<SessionState>, id: &str| {
            dispatch(
                agg,
                CommandPayload::CompleteToolCall {
                    tool_call_id: id.to_string(),
                    attempt: 0,
                    result: format!("result-{id}"),
                    worker_state: None,
                },
                &sys,
            )
        };

        let first = complete(&mut agg, "tc-b");
        assert_eq!(fired_tool_result(&first), vec!["tc-b".to_string()]);
        assert_eq!(
            agg.state.tool_calls.get("tc-b").unwrap().result.as_deref(),
            Some("result-tc-b")
        );

        let second = complete(&mut agg, "tc-a");
        assert_eq!(fired_tool_result(&second), vec!["tc-a".to_string()]);
        assert_eq!(
            agg.state.tool_calls.get("tc-a").unwrap().result.as_deref(),
            Some("result-tc-a")
        );
    }

    #[test]
    fn worker_append_action_writes_a_tree_node() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let setup = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message {
                    message: node_msg("", Role::User, "hi"),
                    stream: false,
                },
                turn_id: Some("turn-1".to_string()),
            },
            &system(),
        );
        let decision_id =
            decision_with(&setup, |t| matches!(t, DecisionTrigger::UserMessage { .. }))
                .expect("user message decision");

        let events = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id,
                transcript: vec![node_msg("u1", Role::User, "hi")],
                actions: vec![],
                state: vec![].into(),
            },
            &machine(),
        );

        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::NewMessage(m) if m.message.id == "u1")),
            "append writes a NewMessage; got {events:?}"
        );
        assert_eq!(agg.state.head_id.as_deref(), Some("u1"));
    }

    #[test]
    fn complete_tool_call_fires_tool_result_without_appending() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_client_tool(&mut agg, "tc-1");

        let events = complete_tool(&mut agg, "tc-1", "ok");

        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::ToolCallCompleted(_))),
            "expected a ToolCallCompleted event; got {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, EventPayload::NewMessage(_))),
            "the engine appends no node on completion; got {events:?}"
        );
        assert_eq!(fired_tool_result(&events), vec!["tc-1".to_string()]);

        let tc = agg.state.tool_calls.get("tc-1").expect("tool call present");
        assert_eq!(tc.tracking.status, EffectStatus::Completed);
        assert_eq!(tc.result.as_deref(), Some("ok"));
        assert!(!tc.is_error);
    }

    // ── Concurrent LLM calls (fan-out) ───────────────────────────────────

    fn call_llm_action(id: &str, handler: LlmHandler) -> WorkerAction {
        WorkerAction::CallLlm {
            id: id.to_string(),
            request: request_with(vec![]),
            stream: false,
            retry: RetryPolicy::no_retry(),
            handler,
        }
    }

    fn llm_response(content: &str) -> LlmResponse {
        LlmResponse {
            model: "test-model".to_string(),
            content: Some(content.to_string()),
            tool_calls: vec![],
            finish_reason: Some("stop".to_string()),
            usage: None,
            cost: None,
            images: vec![],
        }
    }

    fn request_llm(
        agg: &mut Aggregate<SessionState>,
        id: &str,
        handler: LlmHandler,
    ) -> Vec<EventPayload> {
        dispatch(
            agg,
            CommandPayload::RequestLlmCall {
                call_id: id.to_string(),
                request: request_with(vec![]),
                stream: false,
                retry: RetryPolicy::no_retry(),
                handler,
            },
            &system(),
        )
    }

    fn complete_llm(
        agg: &mut Aggregate<SessionState>,
        id: &str,
        attempt: u32,
        caller: &Caller,
    ) -> Vec<EventPayload> {
        dispatch(
            agg,
            CommandPayload::CompleteLlmCall {
                call_id: id.to_string(),
                attempt,
                response: llm_response("ok"),
            },
            caller,
        )
    }

    /// Open a worker decision (via a user message) and answer it with `actions`.
    fn submit_decision_with(
        agg: &mut Aggregate<SessionState>,
        actions: Vec<WorkerAction>,
    ) -> Vec<EventPayload> {
        let setup = dispatch(
            agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message {
                    message: node_msg("seed", Role::User, "seed"),
                    stream: false,
                },
                turn_id: None,
            },
            &system(),
        );
        let decision_id = setup
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.decision_id.clone()),
                _ => None,
            })
            .expect("user message opens a decision");
        dispatch(
            agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id,
                transcript: vec![],
                actions,
                state: vec![].into(),
            },
            &machine(),
        )
    }

    /// The effect ids of every llm `effect.execute` trigger, deduped by
    /// decision id (a promoted queue entry surfaces as both queued + requested).
    fn fired_llm_execute(events: &[EventPayload]) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        events
            .iter()
            .filter_map(|e| {
                let (decision_id, trigger) = match e {
                    EventPayload::WorkerDecisionRequested(p) => (&p.decision_id, &p.trigger),
                    EventPayload::DecisionRequestQueued(p) => (&p.decision_id, &p.trigger),
                    _ => return None,
                };
                match trigger {
                    DecisionTrigger::EffectExecute {
                        id,
                        work: EffectWork::LlmCall { .. },
                        ..
                    } if seen.insert(decision_id.clone()) => Some(id.clone()),
                    _ => None,
                }
            })
            .collect()
    }

    /// The effect ids of every llm `effect.settled` trigger, deduped by decision id.
    fn settled_llm_ids(events: &[EventPayload]) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        events
            .iter()
            .filter_map(|e| {
                let (decision_id, trigger) = match e {
                    EventPayload::WorkerDecisionRequested(p) => (&p.decision_id, &p.trigger),
                    EventPayload::DecisionRequestQueued(p) => (&p.decision_id, &p.trigger),
                    _ => return None,
                };
                match trigger {
                    DecisionTrigger::EffectSettled {
                        id,
                        outcome: EffectOutcome::LlmCall { .. },
                        ..
                    } if seen.insert(decision_id.clone()) => Some(id.clone()),
                    _ => None,
                }
            })
            .collect()
    }

    #[test]
    fn two_server_llm_calls_in_one_decision_both_issue() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let events = submit_decision_with(
            &mut agg,
            vec![
                call_llm_action("llm-1", LlmHandler::Server),
                call_llm_action("llm-2", LlmHandler::Server),
            ],
        );
        let requested: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                EventPayload::LlmCallRequested(r) => Some(r.call_id.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            requested,
            vec!["llm-1".to_string(), "llm-2".to_string()],
            "both calls issue — the single-pending gate is gone; got {events:?}"
        );
        assert_eq!(agg.state.llm_calls.len(), 2);
        assert!(agg
            .state
            .llm_calls
            .values()
            .all(|c| c.tracking.status == EffectStatus::Pending));
        assert_eq!(agg.state.effects().len(), 2, "both in flight");
    }

    #[test]
    fn worker_handled_llm_fanout_delegates_both_executes() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let events = submit_decision_with(
            &mut agg,
            vec![
                call_llm_action("llm-1", LlmHandler::Worker),
                call_llm_action("llm-2", LlmHandler::Worker),
            ],
        );
        let mut execs = fired_llm_execute(&events);
        execs.sort();
        assert_eq!(
            execs,
            vec!["llm-1".to_string(), "llm-2".to_string()],
            "each worker-handled call delegates an effect.execute; got {events:?}"
        );
        assert_eq!(agg.state.effects().len(), 2);
    }

    #[test]
    fn reverse_order_llm_completion_settles_independently() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_llm(&mut agg, "llm-1", LlmHandler::Server);
        request_llm(&mut agg, "llm-2", LlmHandler::Server);
        assert_eq!(agg.state.effects().len(), 2);

        let e2 = complete_llm(&mut agg, "llm-2", 0, &system());
        assert!(settled_llm_ids(&e2).contains(&"llm-2".to_string()));
        let remaining: Vec<String> = agg.state.effects().iter().map(|e| e.id.clone()).collect();
        assert_eq!(
            remaining,
            vec!["llm-1".to_string()],
            "only the unsettled call remains in flight"
        );

        let e1 = complete_llm(&mut agg, "llm-1", 0, &system());
        assert!(settled_llm_ids(&e1).contains(&"llm-1".to_string()));
        assert!(agg.state.effects().is_empty());
    }

    #[test]
    fn wake_suppresses_stall_while_an_llm_call_is_pending() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        // A resolved tool would otherwise trigger stall recovery; a still-pending
        // llm call must suppress it.
        dispatch(
            &mut agg,
            CommandPayload::RequestToolCall {
                tool_call_id: "tc-1".to_string(),
                name: "t".to_string(),
                arguments: "{}".to_string(),
                handler: ToolHandler::Worker,
                retry: RetryPolicy::no_retry(),
            },
            &system(),
        );
        complete_tool(&mut agg, "tc-1", "done");
        request_llm(&mut agg, "llm-1", LlmHandler::Server);

        let woken = wake(&mut agg);
        assert!(
            !woken
                .iter()
                .any(|e| matches!(e, EventPayload::WorkerDecisionRequested(p)
                    if matches!(p.trigger, DecisionTrigger::Stall))),
            "a pending llm call suppresses the stall trigger; got {woken:?}"
        );
    }

    #[test]
    fn re_requesting_a_completed_llm_id_noops() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_llm(&mut agg, "llm-1", LlmHandler::Server);
        complete_llm(&mut agg, "llm-1", 0, &system());
        let again = request_llm(&mut agg, "llm-1", LlmHandler::Server);
        assert!(
            again.is_empty(),
            "re-request of a Completed id is an idempotent no-op; got {again:?}"
        );
    }

    #[test]
    fn llm_and_tool_with_the_same_id_coexist() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_client_tool(&mut agg, "shared");
        request_llm(&mut agg, "shared", LlmHandler::Server);
        assert!(agg.state.tool_calls.contains_key("shared"));
        assert!(agg.state.llm_calls.contains_key("shared"));
        assert_eq!(
            agg.state.effects().len(),
            2,
            "distinct maps keep a tool and an llm call with the same id apart"
        );
    }

    #[test]
    fn reissue_llm_after_interrupt_with_the_same_id() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_llm(&mut agg, "llm-1", LlmHandler::Server);
        dispatch(
            &mut agg,
            CommandPayload::Interrupt {
                interrupt_id: "int-1".to_string(),
                reason: String::new(),
                payload: serde_json::Value::Null,
            },
            &system(),
        );
        assert_eq!(
            agg.state.llm_calls.get("llm-1").unwrap().tracking.status,
            EffectStatus::Failed,
            "interrupt voids the pending call"
        );
        dispatch(
            &mut agg,
            CommandPayload::ResumeInterrupt {
                interrupt_id: "int-1".to_string(),
                payload: serde_json::Value::Null,
            },
            &system(),
        );
        let events = request_llm(&mut agg, "llm-1", LlmHandler::Server);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::LlmCallRequested(_))),
            "a Failed id re-issues on the same key; got {events:?}"
        );
        assert_eq!(
            agg.state.llm_calls.get("llm-1").unwrap().tracking.status,
            EffectStatus::Pending
        );
    }

    #[test]
    fn machine_settles_worker_handled_llm_call() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_llm(&mut agg, "llm-1", LlmHandler::Worker);
        let events = complete_llm(&mut agg, "llm-1", 0, &machine());
        assert!(events
            .iter()
            .any(|e| matches!(e, EventPayload::LlmCallCompleted(_))));
        assert!(settled_llm_ids(&events).contains(&"llm-1".to_string()));
        assert_eq!(
            agg.state.llm_calls.get("llm-1").unwrap().tracking.status,
            EffectStatus::Completed
        );
    }

    #[test]
    fn machine_fails_worker_handled_llm_call() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_llm(&mut agg, "llm-1", LlmHandler::Worker);
        let events = dispatch(
            &mut agg,
            CommandPayload::FailLlmCall {
                call_id: "llm-1".to_string(),
                attempt: 0,
                error: "boom".to_string(),
                retryable: false,
                code: None,
                detail: None,
            },
            &machine(),
        );
        assert!(events
            .iter()
            .any(|e| matches!(e, EventPayload::LlmCallErrored(_))));
        assert!(
            settled_llm_ids(&events).contains(&"llm-1".to_string()),
            "no-retry failure settles the call; got {events:?}"
        );
    }

    #[test]
    fn machine_cannot_settle_engine_handled_llm_call() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_llm(&mut agg, "llm-1", LlmHandler::Server);
        let err = agg
            .state
            .handle(
                CommandPayload::CompleteLlmCall {
                    call_id: "llm-1".to_string(),
                    attempt: 0,
                    response: llm_response("hi"),
                },
                &machine(),
            )
            .expect_err("an engine-handled call is not the machine's to settle");
        assert!(
            matches!(err, SessionError::EffectWrongHandler),
            "expected EffectWrongHandler; got {err:?}"
        );
    }

    #[test]
    fn machine_wrong_attempt_on_worker_llm_is_mismatch() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_llm(&mut agg, "llm-1", LlmHandler::Worker);
        let err = agg
            .state
            .handle(
                CommandPayload::CompleteLlmCall {
                    call_id: "llm-1".to_string(),
                    attempt: 7,
                    response: llm_response("hi"),
                },
                &machine(),
            )
            .expect_err("a stale attempt is a mismatch");
        assert!(
            matches!(err, SessionError::EffectAttemptMismatch),
            "expected EffectAttemptMismatch; got {err:?}"
        );
    }

    #[test]
    fn machine_settle_of_unknown_llm_is_not_found() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let err = agg
            .state
            .handle(
                CommandPayload::CompleteLlmCall {
                    call_id: "nope".to_string(),
                    attempt: 0,
                    response: llm_response("hi"),
                },
                &machine(),
            )
            .expect_err("unknown effect");
        assert!(
            matches!(err, SessionError::EffectNotFound),
            "expected EffectNotFound; got {err:?}"
        );
    }

    #[test]
    fn system_duplicate_llm_completion_is_silent_noop() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_llm(&mut agg, "llm-1", LlmHandler::Server);
        complete_llm(&mut agg, "llm-1", 0, &system());
        let again = complete_llm(&mut agg, "llm-1", 0, &system());
        assert!(
            again.is_empty(),
            "a duplicate executor completion stays a silent no-op; got {again:?}"
        );
    }

    #[test]
    fn done_then_late_llm_completion_still_settles() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_llm(&mut agg, "llm-1", LlmHandler::Server);
        dispatch(
            &mut agg,
            CommandPayload::MarkDone {
                data: serde_json::Value::Null,
            },
            &system(),
        );
        let events = complete_llm(&mut agg, "llm-1", 0, &system());
        assert!(
            settled_llm_ids(&events).contains(&"llm-1".to_string()),
            "a late settle after done still fires a decision; got {events:?}"
        );
    }
}
