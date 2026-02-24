use chrono::Utc;
use uuid::Uuid;

use super::agent_state::{AgentState, LlmCallStatus, SessionStatus, ToolCallStatus};
use super::command::{CommandPayload, SessionCommand, SessionError};
use super::effect::Effect;
use super::react::{extract_assistant_message, extract_response_summary};
use super::strategy::{Action, Strategy, Turn};
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
    pub fn new(session_id: Uuid, strategy: Box<dyn Strategy>) -> Self {
        let mut agent_state = AgentState::new(session_id);
        agent_state.strategy_state = strategy.default_state();
        AgentSession {
            agent_state,
            strategy,
        }
    }

    /// Build from a stored snapshot.
    pub fn from_snapshot(snapshot: AgentState, strategy: Box<dyn Strategy>) -> Self {
        AgentSession {
            agent_state: snapshot,
            strategy,
        }
    }

    /// Take a snapshot of the current session state.
    pub fn snapshot(&self) -> AgentState {
        self.agent_state.clone()
    }

    pub fn apply(&mut self, event: &Event) {
        self.agent_state.apply_core(event);
    }

    /// Compute the derived state envelope for the current session state.
    pub fn derived_state(&self) -> DerivedState {
        DerivedState {
            status: self.agent_state.status.clone(),
            wake_at: self.agent_state.wake_at(),
        }
    }

    /// Process a command: validate, build events, apply, and stamp derived state.
    ///
    /// Returns the fully-stamped events and a post-apply snapshot, ready for
    /// persistence. The caller only needs to persist — no domain logic leaks out.
    pub fn process_command(
        &mut self,
        cmd: SessionCommand,
        tenant_id: &str,
    ) -> Result<(Vec<Event>, AgentState), SessionError> {
        let payloads = self.handle(cmd.payload)?;
        if payloads.is_empty() {
            return Ok((vec![], self.snapshot()));
        }

        let base_seq = self.agent_state.stream_version + 1;
        let mut events: Vec<Event> = payloads
            .into_iter()
            .enumerate()
            .map(|(i, payload)| Event {
                id: Uuid::new_v4(),
                tenant_id: tenant_id.to_string(),
                session_id: self.agent_state.session_id,
                sequence: base_seq + i as u64,
                span: cmd.span.clone(),
                occurred_at: cmd.occurred_at,
                payload,
                derived: None,
            })
            .collect();

        for event in &events {
            self.apply(event);
        }
        let derived = Some(self.derived_state());
        for event in &mut events {
            event.derived = derived.clone();
        }

        Ok((events, self.snapshot()))
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
                if self
                    .agent_state
                    .llm_calls
                    .get(call_id)
                    .is_some_and(|c| c.response_processed) =>
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
                if self.agent_state.llm_calls.contains_key(call_id)
                    || self
                        .agent_state
                        .llm_calls
                        .values()
                        .any(|c| c.status == LlmCallStatus::Pending) =>
            {
                Ok(vec![])
            }
            // CompleteLlmCall: only valid if this call_id is currently pending
            CommandPayload::CompleteLlmCall { ref call_id, .. }
                if !matches!(
                    self.agent_state.llm_calls.get(call_id).map(|c| &c.status),
                    Some(&LlmCallStatus::Pending)
                ) =>
            {
                Ok(vec![])
            }
            // FailLlmCall: only valid if this call_id is currently pending
            CommandPayload::FailLlmCall { ref call_id, .. }
                if !matches!(
                    self.agent_state.llm_calls.get(call_id).map(|c| &c.status),
                    Some(&LlmCallStatus::Pending)
                ) =>
            {
                Ok(vec![])
            }
            CommandPayload::RequestLlmCall {
                call_id,
                request,
                stream,
                deadline,
            } => Ok(vec![EventPayload::LlmCallRequested(LlmCallRequested {
                call_id,
                request,
                stream,
                deadline,
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
                    self.agent_state.llm_calls.get(call_id).map(|c| &c.status),
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
            } if self.agent_state.tool_calls.contains_key(tool_call_id) => Ok(vec![]),
            // CompleteToolCall: only valid if this tool_call_id is currently Pending
            CommandPayload::CompleteToolCall {
                ref tool_call_id, ..
            } if !matches!(
                self.agent_state
                    .tool_calls
                    .get(tool_call_id)
                    .map(|tc| &tc.status),
                Some(&ToolCallStatus::Pending)
            ) =>
            {
                Ok(vec![])
            }
            // FailToolCall: only valid if this tool_call_id is currently Pending
            CommandPayload::FailToolCall {
                ref tool_call_id, ..
            } if !matches!(
                self.agent_state
                    .tool_calls
                    .get(tool_call_id)
                    .map(|tc| &tc.status),
                Some(&ToolCallStatus::Pending)
            ) =>
            {
                Ok(vec![])
            }
            CommandPayload::RequestToolCall {
                tool_call_id,
                name,
                arguments,
                deadline,
            } => Ok(vec![EventPayload::ToolCallRequested(ToolCallRequested {
                tool_call_id,
                name,
                arguments,
                deadline,
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
                println!(
                    "[session:{}] LlmCallErrored [{}] -> idle, wake scheduler handles retry",
                    session_id, payload.call_id,
                );
                self.agent_state.status = SessionStatus::Idle;
                vec![] // wake scheduler handles retry timing
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

    /// Compute LLM call deadline from agent config.
    fn llm_deadline(&self) -> chrono::DateTime<Utc> {
        let timeout = self
            .agent_state
            .agent
            .as_ref()
            .map(|a| a.retry.llm_timeout_secs)
            .unwrap_or(60);
        Utc::now() + chrono::Duration::seconds(timeout as i64)
    }

    /// Compute tool call deadline from agent config.
    fn tool_deadline(&self) -> chrono::DateTime<Utc> {
        let timeout = self
            .agent_state
            .agent
            .as_ref()
            .map(|a| a.retry.tool_timeout_secs)
            .unwrap_or(120);
        Utc::now() + chrono::Duration::seconds(timeout as i64)
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
                        deadline: self.llm_deadline(),
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
                        deadline: self.tool_deadline(),
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
    // Wake — single "figure out what to do next" method
    // Handles both normal retries (from Idle) and crash recovery (from Active)
    // -----------------------------------------------------------------------

    pub fn wake(&mut self, tools: Option<Vec<openai::Tool>>) -> Vec<Effect> {
        let now = Utc::now();
        let max_retries = self
            .agent_state
            .agent
            .as_ref()
            .map(|a| a.retry.max_retries)
            .unwrap_or(3);

        // 1. Timed-out pending LLM calls → fail, then retry if within limits
        for call in self
            .agent_state
            .llm_calls
            .values()
            .cloned()
            .collect::<Vec<_>>()
        {
            if call.status == LlmCallStatus::Pending && call.deadline <= now {
                let mut effects = vec![Effect::Command(CommandPayload::FailLlmCall {
                    call_id: call.call_id.clone(),
                    error: "deadline exceeded".to_string(),
                })];
                // After failing, retry if within limits (the FailLlmCall will increment attempts via apply_core)
                if call.retry.attempts < max_retries {
                    if let Some(request) = self.agent_state.build_llm_request(tools.clone()) {
                        effects.push(Effect::Command(CommandPayload::RequestLlmCall {
                            call_id: new_call_id(),
                            request,
                            stream: true,
                            deadline: self.llm_deadline(),
                        }));
                    }
                }
                return effects;
            }
        }

        // 2. Timed-out pending tool calls → fail
        for tc in self
            .agent_state
            .tool_calls
            .values()
            .cloned()
            .collect::<Vec<_>>()
        {
            if tc.status == ToolCallStatus::Pending && tc.deadline <= now {
                return vec![Effect::Command(CommandPayload::FailToolCall {
                    tool_call_id: tc.tool_call_id.clone(),
                    name: tc.name.clone(),
                    error: "deadline exceeded".to_string(),
                })];
            }
        }

        // 3. Failed LLM call with retry.next_at passed → re-issue with new deadline
        for call in self
            .agent_state
            .llm_calls
            .values()
            .cloned()
            .collect::<Vec<_>>()
        {
            if call.status == LlmCallStatus::Failed {
                if call.retry.attempts >= max_retries {
                    // Past retry limit → done
                    self.agent_state.status = SessionStatus::Done;
                    return vec![];
                }
                if let Some(next_at) = call.retry.next_at {
                    if next_at <= now {
                        if let Some(request) = self.agent_state.build_llm_request(tools.clone()) {
                            self.agent_state.status = SessionStatus::Active;
                            return vec![Effect::Command(CommandPayload::RequestLlmCall {
                                call_id: new_call_id(),
                                request,
                                stream: true,
                                deadline: self.llm_deadline(),
                            })];
                        }
                    }
                }
            }
        }

        // 4. Completed LLM call, response not processed → emit SendAssistantMessage + tool calls
        for call in self
            .agent_state
            .llm_calls
            .values()
            .cloned()
            .collect::<Vec<_>>()
        {
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
                            deadline: self.tool_deadline(),
                        }));
                    }
                    return effects;
                }
            }
        }

        // 5. Pending tool calls still in flight → re-issue them (crash recovery)
        for tc in self
            .agent_state
            .tool_calls
            .values()
            .cloned()
            .collect::<Vec<_>>()
        {
            if tc.status == ToolCallStatus::Pending && tc.deadline > now {
                let args: serde_json::Value = self
                    .agent_state
                    .messages
                    .iter()
                    .flat_map(|m| m.tool_calls.iter())
                    .find(|t| t.id == tc.tool_call_id)
                    .and_then(|t| serde_json::from_str(&t.arguments).ok())
                    .unwrap_or_default();
                return vec![Effect::CallMcpTool {
                    tool_call_id: tc.tool_call_id.clone(),
                    name: tc.name.clone(),
                    arguments: args,
                }];
            }
        }

        // 6. Pending LLM calls still in flight → re-issue (crash recovery)
        for call in self
            .agent_state
            .llm_calls
            .values()
            .cloned()
            .collect::<Vec<_>>()
        {
            if call.status == LlmCallStatus::Pending && call.deadline > now {
                return vec![Effect::CallLlm {
                    call_id: call.call_id.clone(),
                    request: call.request.clone(),
                    stream: true,
                }];
            }
        }

        // 7. All tools done, no next step → consult strategy (same as react does for MessageTool)
        let last_is_tool = self
            .agent_state
            .messages
            .last()
            .is_some_and(|m| m.role == Role::Tool);
        if self.agent_state.pending_tool_results() == 0
            && last_is_tool
            && !self
                .agent_state
                .llm_calls
                .values()
                .any(|c| c.status == LlmCallStatus::Pending)
        {
            self.agent_state.status = SessionStatus::Active;
            if let Some(request) = self.agent_state.build_llm_request(tools) {
                return vec![Effect::Command(CommandPayload::RequestLlmCall {
                    call_id: new_call_id(),
                    request,
                    stream: true,
                    deadline: self.llm_deadline(),
                })];
            }
        }

        vec![]
    }
}
