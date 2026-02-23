use uuid::Uuid;

use serde_json::Value;

use super::call_tracker::{CallTracker, LlmCallStatus, ToolCallStatus};
use super::command::{CommandPayload, SessionError};
use super::agent_state::AgentState;
use super::effect::Effect;
use super::react::{extract_assistant_message, extract_response_summary};
use super::snapshot::SessionSnapshot;
use super::strategy::{Action, RecoveryEffects, Strategy, Turn};
use crate::domain::event::*;
use crate::domain::openai;

fn new_call_id() -> String {
    Uuid::new_v4().to_string()
}

pub struct AgentSession {
    pub core: AgentState,
    pub strategy_state: Value,
    tracker: CallTracker,
    strategy: Box<dyn Strategy>,
}

impl AgentSession {
    pub fn new(session_id: Uuid, strategy: Box<dyn Strategy>, strategy_state: Value) -> Self {
        AgentSession {
            core: AgentState::new(session_id),
            strategy_state,
            tracker: CallTracker::new(),
            strategy,
        }
    }

    /// Build from a pre-populated core and replay events through the tracker
    /// and strategy state.
    pub fn from_core(
        core: AgentState,
        strategy: Box<dyn Strategy>,
        strategy_state: Value,
        events: &[Event],
    ) -> Self {
        let mut tracker = CallTracker::new();
        let mut current_state = strategy_state;
        for event in events {
            tracker.apply(event);
            if let EventPayload::StrategyStateChanged(payload) = &event.payload {
                current_state = payload.state.clone();
            }
        }
        AgentSession {
            core,
            tracker,
            strategy,
            strategy_state: current_state,
        }
    }

    /// Build from a snapshot, applying any remaining events that occurred
    /// after the snapshot was taken.
    pub fn from_snapshot(
        snapshot: SessionSnapshot,
        strategy: Box<dyn Strategy>,
        remaining_events: &[Event],
    ) -> Self {
        let mut session = AgentSession {
            core: snapshot.core,
            tracker: snapshot.tracker,
            strategy,
            strategy_state: snapshot.strategy_state,
        };
        for event in remaining_events {
            session.apply(event);
        }
        session
    }

    /// Take a snapshot of the current session state.
    pub fn snapshot(&self) -> SessionSnapshot {
        SessionSnapshot {
            core: self.core.clone(),
            tracker: self.tracker.clone(),
            strategy_state: self.strategy_state.clone(),
        }
    }

    pub fn apply(&mut self, event: &Event) {
        if !self.core.apply_core(event) {
            return;
        }
        self.tracker.apply(event);
        if let EventPayload::StrategyStateChanged(payload) = &event.payload {
            self.strategy_state = payload.state.clone();
        }
    }

    pub fn handle(&self, cmd: CommandPayload) -> Result<Vec<EventPayload>, SessionError> {
        match (&self.core.agent, cmd) {
            // Uncreated session: only CreateSession is valid
            (None, CommandPayload::CreateSession { agent, auth }) => {
                Ok(vec![EventPayload::SessionCreated(SessionCreated {
                    agent,
                    auth,
                })])
            }
            (Some(_), CommandPayload::CreateSession { .. }) => {
                Err(SessionError::SessionAlreadyCreated)
            }
            (None, _) => Err(SessionError::SessionNotCreated),
            // Active session — use tracker for all guards
            (Some(_), cmd) => self.handle_active(cmd),
        }
    }

    /// Command validation using the CallTracker for idempotency guards.
    fn handle_active(&self, cmd: CommandPayload) -> Result<Vec<EventPayload>, SessionError> {
        match cmd {
            CommandPayload::CreateSession { .. } => {
                unreachable!("CreateSession is handled by SessionState::handle")
            }
            // SendUserMessage guard: reject if session is interrupted
            CommandPayload::SendUserMessage { .. } if self.tracker.active_interrupt().is_some() => {
                Err(SessionError::SessionInterrupted)
            }
            CommandPayload::SendUserMessage { content, stream } => {
                Ok(vec![EventPayload::MessageUser(MessageUser {
                    message: Message {
                        role: Role::User,
                        content: Some(content),
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                        call_id: None,
                        token_count: None,
                    },
                    stream,
                })])
            }
            // SendAssistantMessage guard: skip if message already extracted for this call_id
            CommandPayload::SendAssistantMessage { ref call_id, .. }
                if self.tracker.is_response_processed(call_id) =>
            {
                Ok(vec![])
            }
            CommandPayload::SendAssistantMessage {
                call_id,
                content,
                tool_calls,
                token_count,
            } => Ok(vec![EventPayload::MessageAssistant(MessageAssistant {
                call_id: call_id.clone(),
                message: Message {
                    role: Role::Assistant,
                    content,
                    tool_calls,
                    tool_call_id: None,
                    call_id: Some(call_id),
                    token_count,
                },
            })]),
            // SendToolMessage guard: skip if a tool message with this tool_call_id already exists
            CommandPayload::SendToolMessage {
                ref tool_call_id, ..
            } if self
                .core
                .messages
                .iter()
                .any(|m| m.tool_call_id.as_ref() == Some(tool_call_id)) =>
            {
                Ok(vec![])
            }
            CommandPayload::SendToolMessage {
                tool_call_id,
                content,
                token_count,
            } => Ok(vec![EventPayload::MessageTool(MessageTool {
                message: Message {
                    role: Role::Tool,
                    content: Some(content),
                    tool_calls: Vec::new(),
                    tool_call_id: Some(tool_call_id),
                    call_id: None,
                    token_count,
                },
            })]),
            // RequestLlmCall: skip if call_id already known or another call is in flight
            CommandPayload::RequestLlmCall { ref call_id, .. }
                if self.tracker.has_llm_call(call_id)
                    || self.tracker.active_llm_call().is_some() =>
            {
                Ok(vec![])
            }
            // CompleteLlmCall: only valid if this call_id is currently pending
            CommandPayload::CompleteLlmCall { ref call_id, .. }
                if !matches!(
                    self.tracker.llm_call_status(call_id),
                    Some(&LlmCallStatus::Pending)
                ) =>
            {
                Ok(vec![])
            }
            // FailLlmCall: only valid if this call_id is currently pending
            CommandPayload::FailLlmCall { ref call_id, .. }
                if !matches!(
                    self.tracker.llm_call_status(call_id),
                    Some(&LlmCallStatus::Pending)
                ) =>
            {
                Ok(vec![])
            }
            CommandPayload::RequestLlmCall {
                call_id,
                request,
                stream,
            } => Ok(vec![EventPayload::LlmCallRequested(LlmCallRequested {
                call_id,
                request,
                stream,
            })]),
            CommandPayload::CompleteLlmCall { call_id, response } => {
                Ok(vec![EventPayload::LlmCallCompleted(LlmCallCompleted {
                    call_id,
                    response,
                })])
            }
            CommandPayload::FailLlmCall { call_id, error } => {
                Ok(vec![EventPayload::LlmCallErrored(LlmCallErrored {
                    call_id,
                    error,
                })])
            }
            // StreamLlmChunk guard: skip if this call_id is not Pending
            CommandPayload::StreamLlmChunk { ref call_id, .. }
                if !matches!(
                    self.tracker.llm_call_status(call_id),
                    Some(&LlmCallStatus::Pending)
                ) =>
            {
                Ok(vec![])
            }
            CommandPayload::StreamLlmChunk {
                call_id,
                chunk_index,
                text,
            } => Ok(vec![EventPayload::LlmStreamChunk(LlmStreamChunk {
                call_id,
                chunk_index,
                text,
            })]),
            // Tool call guards — skip if tool_call_id already known (duplicate)
            CommandPayload::RequestToolCall {
                ref tool_call_id, ..
            } if self.tracker.has_tool_call(tool_call_id) => Ok(vec![]),
            // CompleteToolCall: only valid if this tool_call_id is currently Pending
            CommandPayload::CompleteToolCall {
                ref tool_call_id, ..
            } if !matches!(
                self.tracker.tool_call_status(tool_call_id),
                Some(&ToolCallStatus::Pending)
            ) =>
            {
                Ok(vec![])
            }
            // FailToolCall: only valid if this tool_call_id is currently Pending
            CommandPayload::FailToolCall {
                ref tool_call_id, ..
            } if !matches!(
                self.tracker.tool_call_status(tool_call_id),
                Some(&ToolCallStatus::Pending)
            ) =>
            {
                Ok(vec![])
            }
            CommandPayload::RequestToolCall {
                tool_call_id,
                name,
                arguments,
            } => Ok(vec![EventPayload::ToolCallRequested(ToolCallRequested {
                tool_call_id,
                name,
                arguments,
            })]),
            CommandPayload::CompleteToolCall {
                tool_call_id,
                name,
                result,
            } => Ok(vec![EventPayload::ToolCallCompleted(ToolCallCompleted {
                tool_call_id,
                name,
                result,
            })]),
            CommandPayload::FailToolCall {
                tool_call_id,
                name,
                error,
            } => Ok(vec![EventPayload::ToolCallErrored(ToolCallErrored {
                tool_call_id,
                name,
                error,
            })]),
            // Interrupt guard: skip if already interrupted
            CommandPayload::Interrupt { .. } if self.tracker.active_interrupt().is_some() => {
                Ok(vec![])
            }
            CommandPayload::Interrupt {
                interrupt_id,
                reason,
                payload,
            } => Ok(vec![EventPayload::SessionInterrupted(SessionInterrupted {
                interrupt_id,
                reason,
                payload,
            })]),
            // ResumeInterrupt guard: reject if not interrupted or ID mismatch
            CommandPayload::ResumeInterrupt {
                ref interrupt_id, ..
            } if self.tracker.active_interrupt() != Some(interrupt_id.as_str()) => Ok(vec![]),
            CommandPayload::ResumeInterrupt {
                interrupt_id,
                payload,
            } => Ok(vec![EventPayload::InterruptResumed(InterruptResumed {
                interrupt_id,
                payload,
            })]),
            CommandPayload::UpdateStrategyState { state } => {
                Ok(vec![EventPayload::StrategyStateChanged(
                    StrategyStateChanged { state },
                )])
            }
        }
    }

    // -----------------------------------------------------------------------
    // React — runtime reactor with strategy hooks at decision points
    // -----------------------------------------------------------------------

    pub fn react(&mut self, tools: Option<Vec<openai::Tool>>, event: &Event) -> Vec<Effect> {
        let session_id = self.core.session_id;

        match &event.payload {
            // --- Infrastructure: always handled by runtime ---
            EventPayload::SessionCreated(payload) => {
                if payload.agent.mcp_servers.is_empty() {
                    vec![]
                } else {
                    vec![Effect::StartMcpServers(payload.agent.mcp_servers.clone())]
                }
            }

            EventPayload::LlmCallRequested(payload) => {
                println!(
                    "[session:{}] LlmCallRequested [{}] -> calling LLM client",
                    session_id, payload.call_id,
                );
                vec![Effect::CallLlm {
                    call_id: payload.call_id.clone(),
                    request: payload.request.clone(),
                    stream: payload.stream,
                }]
            }

            EventPayload::ToolCallRequested(payload) => {
                let args: serde_json::Value =
                    serde_json::from_str(&payload.arguments).unwrap_or_default();
                println!(
                    "[session:{}] ToolCallRequested [{}] -> dispatching MCP tool '{}'",
                    session_id, payload.tool_call_id, payload.name,
                );
                vec![Effect::CallMcpTool {
                    tool_call_id: payload.tool_call_id.clone(),
                    name: payload.name.clone(),
                    arguments: args,
                }]
            }

            EventPayload::ToolCallCompleted(payload) => {
                println!(
                    "[session:{}] ToolCallCompleted [{}] -> sending tool message",
                    session_id, payload.tool_call_id,
                );
                vec![Effect::Command(CommandPayload::SendToolMessage {
                    tool_call_id: payload.tool_call_id.clone(),
                    content: payload.result.clone(),
                    token_count: None,
                })]
            }

            EventPayload::ToolCallErrored(payload) => {
                println!(
                    "[session:{}] ToolCallErrored [{}] -> sending tool error message",
                    session_id, payload.tool_call_id,
                );
                vec![Effect::Command(CommandPayload::SendToolMessage {
                    tool_call_id: payload.tool_call_id.clone(),
                    content: format!("Error: {}", payload.error),
                    token_count: None,
                })]
            }

            // --- Decision points: consult strategy ---
            EventPayload::MessageUser(_) => {
                if self.tracker.active_llm_call().is_some() {
                    println!(
                        "[session:{}] MessageUser -> LLM call already pending, dirty",
                        session_id
                    );
                    vec![]
                } else {
                    let turn = self.strategy.on_user_message(
                        &self.strategy_state,
                        &self.core,
                        tools.as_deref(),
                    );
                    self.apply_turn(turn, tools)
                }
            }

            EventPayload::LlmCallCompleted(payload) => {
                if self.tracker.dirty {
                    println!(
                        "[session:{}] LlmCallCompleted [{}] -> stale (dirty), re-triggering",
                        session_id, payload.call_id,
                    );
                    self.tracker.dirty = false;
                    let turn = self.strategy.on_user_message(
                        &self.strategy_state,
                        &self.core,
                        tools.as_deref(),
                    );
                    self.apply_turn(turn, tools)
                } else {
                    let summary = extract_response_summary(payload);

                    println!(
                        "[session:{}] LlmCallCompleted [{}] -> sending assistant message",
                        session_id, payload.call_id,
                    );

                    // Always emit SendAssistantMessage (mechanical)
                    let mut effects = vec![Effect::Command(CommandPayload::SendAssistantMessage {
                        call_id: summary.call_id.clone(),
                        content: summary.content.clone(),
                        tool_calls: summary.tool_calls.clone(),
                        token_count: summary.token_count,
                    })];

                    // Consult strategy for next step
                    let turn = self.strategy.on_llm_response(
                        &self.strategy_state,
                        &self.core,
                        tools.as_deref(),
                        &summary,
                    );
                    effects.extend(self.apply_turn(turn, tools));
                    effects
                }
            }

            EventPayload::LlmCallErrored(payload) => {
                if self.tracker.dirty {
                    println!(
                        "[session:{}] LlmCallErrored [{}] -> dirty, re-triggering",
                        session_id, payload.call_id,
                    );
                    self.tracker.dirty = false;
                    let turn = self.strategy.on_user_message(
                        &self.strategy_state,
                        &self.core,
                        tools.as_deref(),
                    );
                    self.apply_turn(turn, tools)
                } else {
                    println!(
                        "[session:{}] LlmCallErrored [{}] -> no action",
                        session_id, payload.call_id,
                    );
                    vec![]
                }
            }

            EventPayload::MessageTool(_) => {
                if self.tracker.pending_tool_results == 0 {
                    println!(
                        "[session:{}] MessageTool -> all tool results in, consulting strategy",
                        session_id,
                    );
                    let results = self.tracker.take_tool_result_batch();
                    let turn = self.strategy.on_tool_results(
                        &self.strategy_state,
                        &self.core,
                        tools.as_deref(),
                        &results,
                    );
                    self.apply_turn(turn, tools)
                } else {
                    println!(
                        "[session:{}] MessageTool -> {} tool results still pending",
                        session_id, self.tracker.pending_tool_results,
                    );
                    vec![]
                }
            }

            EventPayload::SessionInterrupted(_) => vec![],

            EventPayload::InterruptResumed(payload) => {
                let turn = self.strategy.on_interrupt_resume(
                    &self.strategy_state,
                    &self.core,
                    tools.as_deref(),
                    &payload.interrupt_id,
                    &payload.payload,
                );
                self.apply_turn(turn, tools)
            }

            EventPayload::StrategyStateChanged(_) => vec![],

            _ => vec![],
        }
    }

    /// Absorb strategy state update + translate action into Effects.
    fn apply_turn(&mut self, turn: Turn, tools: Option<Vec<openai::Tool>>) -> Vec<Effect> {
        let mut effects = Vec::new();
        if turn.state != self.strategy_state {
            self.strategy_state = turn.state.clone();
            effects.push(Effect::Command(CommandPayload::UpdateStrategyState {
                state: turn.state,
            }));
        }
        effects.extend(self.execute_action(turn.action, tools));
        effects
    }

    /// Translate a strategy Action into Effects.
    fn execute_action(&self, action: Action, tools: Option<Vec<openai::Tool>>) -> Vec<Effect> {
        match action {
            Action::CallLlm(params) => {
                let stream = params.stream.unwrap_or(self.tracker.stream);
                match self.core.build_llm_request(tools) {
                    Some(request) => vec![Effect::Command(CommandPayload::RequestLlmCall {
                        call_id: new_call_id(),
                        request,
                        stream,
                    })],
                    None => vec![],
                }
            }
            Action::ExecuteTools(plan) => plan
                .calls
                .iter()
                .map(|tc| {
                    Effect::Command(CommandPayload::RequestToolCall {
                        tool_call_id: tc.id.clone(),
                        name: tc.name.clone(),
                        arguments: tc.arguments.clone(),
                    })
                })
                .collect(),
            Action::Done => vec![],
            Action::Interrupt(req) => vec![Effect::Command(CommandPayload::Interrupt {
                interrupt_id: req.id,
                reason: req.reason,
                payload: req.payload,
            })],
        }
    }

    // -----------------------------------------------------------------------
    // Recover — uses tracker only, no strategy involvement
    // -----------------------------------------------------------------------

    pub fn recover(&self, tools: Option<Vec<openai::Tool>>) -> RecoveryEffects {
        let mut setup = Vec::new();
        let mut resume = Vec::new();

        // MCP servers not running
        if let Some(agent) = &self.core.agent {
            if !agent.mcp_servers.is_empty() {
                setup.push(Effect::StartMcpServers(agent.mcp_servers.clone()));
            }
        }

        // If interrupted, skip agent-loop recovery — session is paused
        if self.tracker.active_interrupt().is_some() {
            return RecoveryEffects { setup, resume };
        }

        // Pending LLM call — re-issue the call
        for call in self.tracker.llm_calls() {
            if call.status == LlmCallStatus::Pending {
                resume.push(Effect::CallLlm {
                    call_id: call.call_id.clone(),
                    request: call.request.clone(),
                    stream: self.tracker.stream,
                });
            }
        }

        // Completed LLM call, message not extracted — re-extract from stored response
        for call in self.tracker.llm_calls() {
            if call.status == LlmCallStatus::Completed && !call.response_processed {
                if let Some(ref response) = call.response {
                    let (content, tool_calls, token_count) = extract_assistant_message(response);
                    let mut effects = vec![Effect::Command(CommandPayload::SendAssistantMessage {
                        call_id: call.call_id.clone(),
                        content,
                        tool_calls: tool_calls.clone(),
                        token_count,
                    })];
                    for tc in &tool_calls {
                        effects.push(Effect::Command(CommandPayload::RequestToolCall {
                            tool_call_id: tc.id.clone(),
                            name: tc.name.clone(),
                            arguments: tc.arguments.clone(),
                        }));
                    }
                    resume.extend(effects);
                }
            }
        }

        // Pending tool calls — re-issue
        for tc in self.tracker.tool_calls() {
            if tc.status == ToolCallStatus::Pending {
                let args: serde_json::Value = self
                    .core
                    .messages
                    .iter()
                    .flat_map(|m| m.tool_calls.iter())
                    .find(|t| t.id == tc.tool_call_id)
                    .and_then(|t| serde_json::from_str(&t.arguments).ok())
                    .unwrap_or_default();
                resume.push(Effect::CallMcpTool {
                    tool_call_id: tc.tool_call_id.clone(),
                    name: tc.name.clone(),
                    arguments: args,
                });
            }
        }

        // All tools done, last message is Tool, no pending LLM call → trigger next LLM call
        let last_is_tool = self
            .core
            .messages
            .last()
            .is_some_and(|m| m.role == Role::Tool);
        if self.tracker.pending_tool_results == 0 && last_is_tool && !self.tracker.has_pending_llm()
        {
            if let Some(request) = self.core.build_llm_request(tools) {
                resume.push(Effect::Command(CommandPayload::RequestLlmCall {
                    call_id: Uuid::new_v4().to_string(),
                    request,
                    stream: self.tracker.stream,
                }));
            }
        }

        RecoveryEffects { setup, resume }
    }
}
