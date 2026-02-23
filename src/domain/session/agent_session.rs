use std::cmp::min;

use chrono::Utc;
use uuid::Uuid;

use serde_json::Value;

use super::agent_state::{AgentState, LlmCallStatus, SessionStatus, ToolCallStatus};
use super::command::{CommandPayload, SessionError};
use super::effect::Effect;
use super::react::{extract_assistant_message, extract_response_summary};
use super::strategy::{Action, RecoveryEffects, Strategy, Turn};
use crate::domain::event::*;
use crate::domain::openai;

fn new_call_id() -> String {
    Uuid::new_v4().to_string()
}

pub struct AgentSession {
    pub agent_state: AgentState,
    strategy: Box<dyn Strategy>,
}

impl AgentSession {
    pub fn new(session_id: Uuid, strategy: Box<dyn Strategy>, strategy_state: Value) -> Self {
        let mut agent_state = AgentState::new(session_id);
        agent_state.strategy_state = strategy_state;
        AgentSession {
            agent_state,
            strategy,
        }
    }

    /// Build from a pre-populated core (cold replay already applied strategy state via apply_core).
    pub fn from_core(
        mut core: AgentState,
        strategy: Box<dyn Strategy>,
        default_strategy_state: Value,
    ) -> Self {
        if core.strategy_state.is_null() {
            core.strategy_state = default_strategy_state;
        }
        AgentSession {
            agent_state: core,
            strategy,
        }
    }

    /// Build from a snapshot, applying any remaining events.
    pub fn from_snapshot(
        snapshot: AgentState,
        strategy: Box<dyn Strategy>,
        remaining_events: &[Event],
    ) -> Self {
        let mut session = AgentSession {
            agent_state: snapshot,
            strategy,
        };
        for event in remaining_events {
            session.apply(event);
        }
        session
    }

    /// Take a snapshot of the current session state.
    pub fn snapshot(&self) -> AgentState {
        self.agent_state.clone()
    }

    pub fn apply(&mut self, event: &Event) {
        self.agent_state.apply_core(event);
    }

    pub fn handle(&self, cmd: CommandPayload) -> Result<Vec<EventPayload>, SessionError> {
        match (&self.agent_state.agent, cmd) {
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
            // Active session
            (Some(_), cmd) => self.handle_active(cmd),
        }
    }

    /// Command validation using AgentState for idempotency guards.
    fn handle_active(&self, cmd: CommandPayload) -> Result<Vec<EventPayload>, SessionError> {
        match cmd {
            CommandPayload::CreateSession { .. } => {
                unreachable!("CreateSession is handled by SessionState::handle")
            }
            // SendUserMessage guard: reject if session is interrupted
            CommandPayload::SendUserMessage { .. }
                if matches!(self.agent_state.status, SessionStatus::Interrupted { .. }) =>
            {
                Err(SessionError::SessionInterrupted)
            }
            // SendUserMessage guard: reject if session is active (work in flight)
            CommandPayload::SendUserMessage { .. }
                if matches!(self.agent_state.status, SessionStatus::Active) =>
            {
                Err(SessionError::SessionBusy)
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
                if self.agent_state.is_response_processed(call_id) =>
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
                .agent_state
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
                if self.agent_state.has_llm_call(call_id)
                    || self.agent_state.active_llm_call().is_some() =>
            {
                Ok(vec![])
            }
            // CompleteLlmCall: only valid if this call_id is currently pending
            CommandPayload::CompleteLlmCall { ref call_id, .. }
                if !matches!(
                    self.agent_state.llm_call_status(call_id),
                    Some(&LlmCallStatus::Pending)
                ) =>
            {
                Ok(vec![])
            }
            // FailLlmCall: only valid if this call_id is currently pending
            CommandPayload::FailLlmCall { ref call_id, .. }
                if !matches!(
                    self.agent_state.llm_call_status(call_id),
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
                    self.agent_state.llm_call_status(call_id),
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
            } if self.agent_state.has_tool_call(tool_call_id) => Ok(vec![]),
            // CompleteToolCall: only valid if this tool_call_id is currently Pending
            CommandPayload::CompleteToolCall {
                ref tool_call_id, ..
            } if !matches!(
                self.agent_state.tool_call_status(tool_call_id),
                Some(&ToolCallStatus::Pending)
            ) =>
            {
                Ok(vec![])
            }
            // FailToolCall: only valid if this tool_call_id is currently Pending
            CommandPayload::FailToolCall {
                ref tool_call_id, ..
            } if !matches!(
                self.agent_state.tool_call_status(tool_call_id),
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
            CommandPayload::Interrupt { .. }
                if matches!(self.agent_state.status, SessionStatus::Interrupted { .. }) =>
            {
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
            } if self.agent_state.active_interrupt() != Some(interrupt_id.as_str()) => Ok(vec![]),
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
        let session_id = self.agent_state.session_id;

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
                self.agent_state.status = SessionStatus::Active;
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
                self.agent_state.status = SessionStatus::Active;
                let turn = self.strategy.on_user_message(
                    &self.agent_state.strategy_state,
                    &self.agent_state,
                    tools.as_deref(),
                );
                self.apply_turn(turn, tools)
            }

            EventPayload::LlmCallCompleted(payload) => {
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
                    &self.agent_state.strategy_state,
                    &self.agent_state,
                    tools.as_deref(),
                    &summary,
                );
                effects.extend(self.apply_turn(turn, tools));
                effects
            }

            EventPayload::LlmCallErrored(payload) => {
                let attempts = self
                    .agent_state
                    .llm_call(&payload.call_id)
                    .map(|c| c.retry.attempts)
                    .unwrap_or(1);
                let backoff_secs = min(2u64.pow(attempts), 60);
                let wake_at = Utc::now() + chrono::Duration::seconds(backoff_secs as i64);

                println!(
                    "[session:{}] LlmCallErrored [{}] -> idle, retry in {}s",
                    session_id, payload.call_id, backoff_secs,
                );

                self.agent_state.status = SessionStatus::Idle {
                    wake_at: Some(wake_at),
                };
                vec![Effect::ScheduleWake { wake_at }]
            }

            EventPayload::MessageTool(_) => {
                if self.agent_state.pending_tool_results() == 0 {
                    println!(
                        "[session:{}] MessageTool -> all tool results in, consulting strategy",
                        session_id,
                    );
                    let results = self.agent_state.collect_tool_results();
                    let turn = self.strategy.on_tool_results(
                        &self.agent_state.strategy_state,
                        &self.agent_state,
                        tools.as_deref(),
                        &results,
                    );
                    self.apply_turn(turn, tools)
                } else {
                    println!(
                        "[session:{}] MessageTool -> tool results still pending",
                        session_id,
                    );
                    vec![]
                }
            }

            EventPayload::SessionInterrupted(_) => vec![],

            EventPayload::InterruptResumed(payload) => {
                let turn = self.strategy.on_interrupt_resume(
                    &self.agent_state.strategy_state,
                    &self.agent_state,
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
        if turn.state != self.agent_state.strategy_state {
            self.agent_state.strategy_state = turn.state.clone();
            effects.push(Effect::Command(CommandPayload::UpdateStrategyState {
                state: turn.state,
            }));
        }
        let action_effects = self.execute_action(turn.action, tools);
        effects.extend(action_effects);
        effects
    }

    /// Translate a strategy Action into Effects.
    fn execute_action(&mut self, action: Action, tools: Option<Vec<openai::Tool>>) -> Vec<Effect> {
        match action {
            Action::CallLlm(params) => {
                let stream = params.stream.unwrap_or(true);
                match self.agent_state.build_llm_request(tools) {
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
            Action::Done => {
                self.agent_state.status = SessionStatus::Done;
                vec![]
            }
            Action::Interrupt(req) => vec![Effect::Command(CommandPayload::Interrupt {
                interrupt_id: req.id,
                reason: req.reason,
                payload: req.payload,
            })],
        }
    }

    // -----------------------------------------------------------------------
    // Wake — resume from Idle state
    // -----------------------------------------------------------------------

    pub fn wake(&mut self, tools: Option<Vec<openai::Tool>>) -> Vec<Effect> {
        self.agent_state.status = SessionStatus::Active;
        if let Some(request) = self.agent_state.build_llm_request(tools) {
            vec![Effect::Command(CommandPayload::RequestLlmCall {
                call_id: new_call_id(),
                request,
                stream: true,
            })]
        } else {
            vec![]
        }
    }

    // -----------------------------------------------------------------------
    // Recover — derive in-flight state from AgentState
    // -----------------------------------------------------------------------

    pub fn recover(&self, tools: Option<Vec<openai::Tool>>) -> RecoveryEffects {
        let mut setup = Vec::new();
        let mut resume = Vec::new();

        // MCP servers not running
        if let Some(agent) = &self.agent_state.agent {
            if !agent.mcp_servers.is_empty() {
                setup.push(Effect::StartMcpServers(agent.mcp_servers.clone()));
            }
        }

        // If interrupted, skip agent-loop recovery — session is paused
        if matches!(self.agent_state.status, SessionStatus::Interrupted { .. }) {
            return RecoveryEffects { setup, resume };
        }

        // If idle with a wake-up time, schedule it
        if let SessionStatus::Idle {
            wake_at: Some(wake_at),
        } = self.agent_state.status
        {
            if wake_at <= Utc::now() {
                // Wake-up time already passed — resume immediately
                if let Some(request) = self.agent_state.build_llm_request(tools.clone()) {
                    resume.push(Effect::Command(CommandPayload::RequestLlmCall {
                        call_id: new_call_id(),
                        request,
                        stream: true,
                    }));
                }
            } else {
                setup.push(Effect::ScheduleWake { wake_at });
            }
            return RecoveryEffects { setup, resume };
        }

        // If idle without wake-up or done, nothing to recover
        if matches!(
            self.agent_state.status,
            SessionStatus::Idle { .. } | SessionStatus::Done
        ) {
            return RecoveryEffects { setup, resume };
        }

        // Active — recover in-flight work
        // Pending LLM call — re-issue the call
        for call in self.agent_state.llm_calls() {
            if call.status == LlmCallStatus::Pending {
                resume.push(Effect::CallLlm {
                    call_id: call.call_id.clone(),
                    request: call.request.clone(),
                    stream: true,
                });
            }
        }

        // Completed LLM call, message not extracted — re-extract from stored response
        for call in self.agent_state.llm_calls() {
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
        for tc in self.agent_state.tool_call_states() {
            if tc.status == ToolCallStatus::Pending {
                let args: serde_json::Value = self
                    .agent_state
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
            .agent_state
            .messages
            .last()
            .is_some_and(|m| m.role == Role::Tool);
        if self.agent_state.pending_tool_results() == 0
            && last_is_tool
            && !self.agent_state.has_pending_llm()
        {
            if let Some(request) = self.agent_state.build_llm_request(tools) {
                resume.push(Effect::Command(CommandPayload::RequestLlmCall {
                    call_id: Uuid::new_v4().to_string(),
                    request,
                    stream: true,
                }));
            }
        }

        RecoveryEffects { setup, resume }
    }
}
