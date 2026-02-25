use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use super::agent_state::{AgentState, LlmCallStatus, ToolCallStatus};
use crate::domain::event::{EventPayload, LlmCallCompleted, Message, Role, ToolCall};

// ---------------------------------------------------------------------------
// DefaultStrategy config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultStrategyConfig {
    #[serde(default)]
    pub compaction: CompactionConfig,
}

impl Default for DefaultStrategyConfig {
    fn default() -> Self {
        Self {
            compaction: CompactionConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionConfig {
    /// Enable compaction. Default: false.
    #[serde(default)]
    pub enabled: bool,
    /// Message count threshold to trigger compaction. Default: 50.
    #[serde(default = "CompactionConfig::default_threshold")]
    pub threshold: usize,
    /// System prompt for the summarization LLM call.
    #[serde(default)]
    pub prompt: Option<String>,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold: 50,
            prompt: None,
        }
    }
}

impl CompactionConfig {
    fn default_threshold() -> usize {
        50
    }
}

// ---------------------------------------------------------------------------
// DefaultStrategy internal state (stored in strategy_state)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct DefaultStrategyState {
    #[serde(default)]
    phase: CompactionPhase,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    compacted_summary: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    compacted_at_message_index: Option<usize>,
    /// When compaction is triggered by a new user message, the reply is
    /// deferred until compaction finishes. This stores the stream preference
    /// and the message index so the new message is excluded from compaction
    /// but included in the follow-up LLM context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pending_reply: Option<PendingReply>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PendingReply {
    /// Index of the buffered user message in state.messages.
    message_index: usize,
    /// Stream preference from the original SendUserMessage command.
    stream: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum CompactionPhase {
    #[default]
    Normal,
    Compacting,
}

const DEFAULT_COMPACTION_PROMPT: &str = "\
Summarize the following conversation concisely. \
Preserve key facts, decisions, tool results, and any context needed to continue the conversation. \
Respond with only the summary.";

// ---------------------------------------------------------------------------
// Strategy trait
// ---------------------------------------------------------------------------

#[async_trait]
pub trait Strategy: Send + Sync {
    /// Create a strategy instance from the config params map.
    fn init(params: &serde_json::Map<String, Value>) -> Result<Box<dyn Strategy>, String>
    where
        Self: Sized;

    /// Initial strategy state for a new or cold-replayed session.
    fn default_state(&self) -> Value {
        Value::Null
    }

    /// Called for every event after the event handler has done mechanical work
    /// (dispatching LLM/tool calls, emitting assistant/tool messages).
    ///
    /// Return `Some(Turn)` to issue an action, or `None` to do nothing.
    async fn on_event(
        &self,
        state: &AgentState,
        event: &EventPayload,
    ) -> Option<Turn>;
}

// ---------------------------------------------------------------------------
// Turn — strategy decision plus updated state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    pub action: Option<Action>,
    pub state: Value,
}

// ---------------------------------------------------------------------------
// Action — what the strategy tells the runtime to do next
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Action {
    /// Call the LLM.
    CallLlm(LlmTurnParams),
    /// Execute tool calls, then consult strategy again.
    ExecuteTools(ToolExecutionPlan),
    /// Agent loop is done — idle until next external input.
    Done,
    /// Pause the session and ask the client to resume with a response.
    Interrupt(InterruptRequest),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterruptRequest {
    pub id: String,
    pub reason: String,
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LlmTurnParams {
    /// None = default context (system prompt + full message history).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<Message>>,
    /// None = use the runtime's default (from the user message stream preference).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

// ---------------------------------------------------------------------------
// Response extraction helpers
// ---------------------------------------------------------------------------

pub fn extract_response_summary(payload: &LlmCallCompleted) -> LlmResponseSummary {
    let (content, tool_calls, token_count) = match &payload.response {
        crate::domain::event::LlmResponse::OpenAi(resp) => {
            let choice = &resp.choices[0];
            let content = choice.message.content.clone();
            let tool_calls = choice
                .message
                .tool_calls
                .as_ref()
                .map(|tcs| {
                    tcs.iter()
                        .map(|tc| ToolCall {
                            id: tc.id.clone(),
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.clone(),
                        })
                        .collect()
                })
                .unwrap_or_default();
            let token_count = resp.usage.as_ref().map(|u| u.total_tokens);
            (content, tool_calls, token_count)
        }
    };
    LlmResponseSummary {
        call_id: payload.call_id.clone(),
        content,
        tool_calls,
        token_count,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponseSummary {
    pub call_id: String,
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub token_count: Option<u32>,
}

// ---------------------------------------------------------------------------
// Tool execution types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub name: String,
    pub content: String,
    pub is_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecutionPlan {
    pub calls: Vec<ToolCall>,
    #[serde(default)]
    pub mode: ToolExecutionMode,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolExecutionMode {
    #[default]
    Concurrent,
    Sequential,
}

// ---------------------------------------------------------------------------
// DefaultStrategy — concurrent ReAct with full history
// ---------------------------------------------------------------------------

pub struct DefaultStrategy {
    config: DefaultStrategyConfig,
}

impl Default for DefaultStrategy {
    fn default() -> Self {
        Self {
            config: DefaultStrategyConfig::default(),
        }
    }
}

impl DefaultStrategy {
    pub fn new(config: DefaultStrategyConfig) -> Self {
        Self { config }
    }

    /// Check if all pending tool calls are done.
    fn all_tools_done(state: &AgentState) -> bool {
        !state.tool_calls.values().any(|tc| tc.status == ToolCallStatus::Pending)
    }

    fn parse_strategy_state(raw: &Value) -> DefaultStrategyState {
        serde_json::from_value(raw.clone()).unwrap_or_default()
    }

    fn serialize_state(ss: &DefaultStrategyState) -> Value {
        serde_json::to_value(ss).unwrap_or(Value::Null)
    }

    /// Build LlmTurnParams, using compacted context when available.
    fn llm_params(&self, state: &AgentState, stream: Option<bool>) -> LlmTurnParams {
        if !self.config.compaction.enabled {
            return LlmTurnParams {
                context: None,
                stream,
            };
        }
        let ss = Self::parse_strategy_state(&state.strategy_state);
        self.llm_params_with_state(state, &ss, stream)
    }

    /// Like `llm_params` but with an explicit strategy state (used when the
    /// caller has a freshly computed state that isn't persisted yet).
    fn llm_params_with_state(
        &self,
        state: &AgentState,
        ss: &DefaultStrategyState,
        stream: Option<bool>,
    ) -> LlmTurnParams {
        LlmTurnParams {
            context: self.build_compacted_context(state, ss),
            stream,
        }
    }

    /// Build a custom context from the compacted summary + recent messages.
    /// Returns None if no compaction has occurred (caller uses full history).
    fn build_compacted_context(
        &self,
        state: &AgentState,
        ss: &DefaultStrategyState,
    ) -> Option<Vec<Message>> {
        let summary = ss.compacted_summary.as_ref()?;
        let at = ss.compacted_at_message_index?;
        let agent = state.agent.as_ref()?;

        let mut context = vec![
            Message {
                role: Role::System,
                content: Some(agent.system_prompt.clone()),
                tool_calls: vec![],
                tool_call_id: None,
                call_id: None,
                token_count: None,
            },
            Message {
                role: Role::User,
                content: Some(format!("[Conversation summary]\n{}", summary)),
                tool_calls: vec![],
                tool_call_id: None,
                call_id: None,
                token_count: None,
            },
        ];

        if at < state.messages.len() {
            context.extend(state.messages[at..].iter().cloned());
        }

        Some(context)
    }

    /// Check if compaction should trigger, and if so return the compaction Turn.
    ///
    /// `compact_end` limits which messages are included in the compaction
    /// context. When `None`, all messages from the compaction start are
    /// included. When `Some(idx)`, only `messages[start..idx]` are included
    /// (used to exclude a newly buffered user message).
    fn maybe_compact(
        &self,
        state: &AgentState,
        ss: &DefaultStrategyState,
        compact_end: Option<usize>,
    ) -> Option<Turn> {
        if matches!(ss.phase, CompactionPhase::Compacting) {
            return None;
        }

        let end = compact_end.unwrap_or(state.messages.len());
        let start = ss.compacted_at_message_index.unwrap_or(0);

        let messages_since = end.saturating_sub(start);

        if messages_since < self.config.compaction.threshold {
            return None;
        }

        let prompt = self
            .config
            .compaction
            .prompt
            .clone()
            .unwrap_or_else(|| DEFAULT_COMPACTION_PROMPT.to_string());

        let mut context = vec![Message {
            role: Role::System,
            content: Some(prompt),
            tool_calls: vec![],
            tool_call_id: None,
            call_id: None,
            token_count: None,
        }];

        context.extend(state.messages[start..end].iter().cloned());

        let new_ss = DefaultStrategyState {
            phase: CompactionPhase::Compacting,
            ..ss.clone()
        };

        Some(Turn {
            action: Some(Action::CallLlm(LlmTurnParams {
                context: Some(context),
                stream: Some(false),
            })),
            state: Self::serialize_state(&new_ss),
        })
    }

    /// Handle LlmCallCompleted when in compacting phase.
    fn handle_compaction_response(
        &self,
        state: &AgentState,
        ss: &DefaultStrategyState,
        payload: &LlmCallCompleted,
    ) -> Option<Turn> {
        let summary = extract_response_summary(payload);
        let summary_text = summary.content.unwrap_or_default();
        let pending = ss.pending_reply.clone();

        // +1 because the assistant message (compaction summary) will be added
        // by the mechanical handler before the CallLlm effect is processed.
        //
        // When a user message was buffered, set the index to the buffered
        // message so that build_compacted_context includes it as a recent
        // message after the summary.
        let compacted_at = match &pending {
            Some(pr) => pr.message_index,
            None => state.messages.len() + 1,
        };

        let new_ss = DefaultStrategyState {
            phase: CompactionPhase::Normal,
            compacted_summary: Some(summary_text.clone()),
            compacted_at_message_index: Some(compacted_at),
            pending_reply: None,
        };

        if let Some(pr) = pending {
            // A user message was buffered — use llm_params which builds
            // compacted context (summary + recent messages including the
            // buffered user message).
            return Some(Turn {
                action: Some(Action::CallLlm(self.llm_params_with_state(
                    state,
                    &new_ss,
                    Some(pr.stream),
                ))),
                state: Self::serialize_state(&new_ss),
            });
        }

        // Mid-turn compaction (after tools) — continue with summary context.
        let mut context = Vec::new();
        if let Some(agent) = state.agent.as_ref() {
            context.push(Message {
                role: Role::System,
                content: Some(agent.system_prompt.clone()),
                tool_calls: vec![],
                tool_call_id: None,
                call_id: None,
                token_count: None,
            });
        }
        context.push(Message {
            role: Role::User,
            content: Some(format!("[Conversation summary]\n{}", summary_text)),
            tool_calls: vec![],
            tool_call_id: None,
            call_id: None,
            token_count: None,
        });

        Some(Turn {
            action: Some(Action::CallLlm(LlmTurnParams {
                context: Some(context),
                stream: Some(true),
            })),
            state: Self::serialize_state(&new_ss),
        })
    }
}

#[async_trait]
impl Strategy for DefaultStrategy {
    fn init(params: &serde_json::Map<String, Value>) -> Result<Box<dyn Strategy>, String> {
        let config = if params.is_empty() {
            DefaultStrategyConfig::default()
        } else {
            serde_json::from_value::<DefaultStrategyConfig>(Value::Object(params.clone()))
                .map_err(|e| e.to_string())?
        };
        Ok(Box::new(Self::new(config)))
    }

    async fn on_event(
        &self,
        state: &AgentState,
        event: &EventPayload,
    ) -> Option<Turn> {
        let ss = Self::parse_strategy_state(&state.strategy_state);

        match event {
            EventPayload::MessageUser(payload) => {
                // The new user message is the last entry in state.messages.
                // Check if we should compact the prior history first.
                if self.config.compaction.enabled {
                    let msg_idx = state.messages.len().saturating_sub(1);
                    if let Some(mut turn) = self.maybe_compact(state, &ss, Some(msg_idx)) {
                        // Stash the pending reply so handle_compaction_response
                        // can issue the real LLM call afterwards.
                        let mut compact_ss: DefaultStrategyState =
                            serde_json::from_value(turn.state.clone()).unwrap_or_default();
                        compact_ss.pending_reply = Some(PendingReply {
                            message_index: msg_idx,
                            stream: payload.stream,
                        });
                        turn.state = Self::serialize_state(&compact_ss);
                        return Some(turn);
                    }
                }
                Some(Turn {
                    action: Some(Action::CallLlm(self.llm_params(state, Some(payload.stream)))),
                    state: Self::serialize_state(&ss),
                })
            }

            EventPayload::LlmCallCompleted(payload) => {
                // Handle compaction phase: the LLM response is the summary.
                if matches!(ss.phase, CompactionPhase::Compacting) {
                    return self.handle_compaction_response(state, &ss, payload);
                }

                // Normal flow.
                let summary = extract_response_summary(payload);
                let action = if summary.tool_calls.is_empty() {
                    Action::Done
                } else {
                    Action::ExecuteTools(ToolExecutionPlan {
                        calls: summary.tool_calls,
                        mode: ToolExecutionMode::Concurrent,
                    })
                };
                Some(Turn {
                    action: Some(action),
                    state: Self::serialize_state(&ss),
                })
            }

            EventPayload::LlmCallErrored(payload) => {
                // Only act when retries are exhausted (status == Failed).
                let call = state.llm_calls.get(&payload.call_id)?;
                if call.status != LlmCallStatus::Failed {
                    return None;
                }
                // If we were compacting and the call failed, reset to normal phase.
                // If a user message was buffered, still issue the LLM call.
                if matches!(ss.phase, CompactionPhase::Compacting) {
                    let pending = ss.pending_reply.clone();
                    let new_ss = DefaultStrategyState {
                        phase: CompactionPhase::Normal,
                        pending_reply: None,
                        ..ss
                    };
                    let action = match pending {
                        Some(pr) => Action::CallLlm(self.llm_params(state, Some(pr.stream))),
                        None => Action::Done,
                    };
                    return Some(Turn {
                        action: Some(action),
                        state: Self::serialize_state(&new_ss),
                    });
                }
                Some(Turn {
                    action: Some(Action::Done),
                    state: Self::serialize_state(&ss),
                })
            }

            EventPayload::MessageTool(_) => {
                // Wait until all tool calls are done before deciding next step.
                if !Self::all_tools_done(state) {
                    return None;
                }

                // Check if compaction should trigger.
                if self.config.compaction.enabled {
                    if let Some(turn) = self.maybe_compact(state, &ss, None) {
                        return Some(turn);
                    }
                }

                Some(Turn {
                    action: Some(Action::CallLlm(self.llm_params(state, Some(true)))),
                    state: Self::serialize_state(&ss),
                })
            }

            EventPayload::InterruptResumed(_) => Some(Turn {
                action: Some(Action::CallLlm(self.llm_params(state, None))),
                state: Self::serialize_state(&ss),
            }),

            EventPayload::BudgetExceeded => Some(Turn {
                action: Some(Action::Interrupt(InterruptRequest {
                    id: Uuid::new_v4().to_string(),
                    reason: "token_budget_exceeded".to_string(),
                    payload: serde_json::json!({
                        "total_tokens": state.token_usage.total_tokens,
                        "limit": state.token_budget.limit,
                    }),
                })),
                state: Self::serialize_state(&ss),
            }),

            // All other events: no strategy decision needed.
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::agent::{AgentConfig, LlmConfig};
    use crate::domain::event::*;
    use crate::domain::session::AgentState;
    use uuid::Uuid;

    fn test_agent() -> AgentConfig {
        AgentConfig {
            id: Uuid::new_v4(),
            name: "test".into(),
            llm: LlmConfig {
                client: "mock".into(),
                params: Default::default(),
            },
            system_prompt: "You are helpful.".into(),
            mcp_servers: vec![],
            strategy: Default::default(),
            retry: Default::default(),
            token_budget: None,

        }
    }

    fn test_auth() -> SessionAuth {
        SessionAuth {
            tenant_id: "t".into(),
            client_id: "c".into(),
            sub: None,
        }
    }

    fn make_session(agent: &AgentConfig) -> AgentState {
        let mut state = AgentState::new(Uuid::new_v4());
        let event = Event {
            id: Uuid::new_v4(),
            tenant_id: "t".into(),
            session_id: state.session_id,
            sequence: 1,
            span: SpanContext::root(),
            occurred_at: chrono::Utc::now(),
            payload: EventPayload::SessionCreated(SessionCreated {
                agent: agent.clone(),
                auth: test_auth(),
            }),
            derived: None,
        };
        state.apply_core(&event);
        state
    }

    fn apply(state: &mut AgentState, payloads: Vec<EventPayload>) {
        let seq = state.last_applied.unwrap_or(0);
        for (i, payload) in payloads.into_iter().enumerate() {
            let event = Event {
                id: Uuid::new_v4(),
                tenant_id: "t".into(),
                session_id: state.session_id,
                sequence: seq + 1 + i as u64,
                span: SpanContext::root(),
                occurred_at: chrono::Utc::now(),
                payload,
                derived: None,
            };
            state.apply_core(&event);
        }
    }

    fn user_msg(content: &str) -> EventPayload {
        EventPayload::MessageUser(MessageUser {
            message: Message {
                role: Role::User,
                content: Some(content.into()),
                tool_calls: vec![],
                tool_call_id: None,
                call_id: None,
                token_count: None,
            },
            stream: false,
        })
    }

    fn assistant_msg(call_id: &str, content: &str) -> EventPayload {
        EventPayload::MessageAssistant(MessageAssistant {
            call_id: call_id.into(),
            message: Message {
                role: Role::Assistant,
                content: Some(content.into()),
                tool_calls: vec![ToolCall {
                    id: format!("tc-{}", call_id),
                    name: "test_tool".into(),
                    arguments: "{}".into(),
                }],
                tool_call_id: None,
                call_id: Some(call_id.into()),
                token_count: None,
            },
        })
    }

    fn tool_msg(tool_call_id: &str) -> EventPayload {
        EventPayload::MessageTool(MessageTool {
            message: Message {
                role: Role::Tool,
                content: Some("ok".into()),
                tool_calls: vec![],
                tool_call_id: Some(tool_call_id.into()),
                call_id: None,
                token_count: None,
            },
        })
    }

    fn tool_call_completed(tool_call_id: &str) -> EventPayload {
        EventPayload::ToolCallCompleted(ToolCallCompleted {
            tool_call_id: tool_call_id.into(),
            name: "test_tool".into(),
            result: "ok".into(),
        })
    }

    fn mock_llm_completed(call_id: &str, content: &str) -> EventPayload {
        use crate::domain::openai;
        EventPayload::LlmCallCompleted(LlmCallCompleted {
            call_id: call_id.into(),
            response: LlmResponse::OpenAi(openai::ChatCompletionResponse {
                id: format!("resp-{}", call_id),
                model: "mock".into(),
                choices: vec![openai::Choice {
                    index: 0,
                    message: openai::ChatMessage {
                        role: openai::Role::Assistant,
                        content: Some(content.into()),
                        tool_calls: None,
                        tool_call_id: None,
                    },
                    finish_reason: Some("stop".into()),
                }],
                usage: None,
            }),
        })
    }

    // -----------------------------------------------------------------------
    // Default behavior (compaction disabled)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn default_strategy_unchanged_when_compaction_disabled() {
        let strategy = DefaultStrategy::default();
        let agent = test_agent();
        let state = make_session(&agent);

        let turn = strategy.on_event(&state, &user_msg("hi")).await.unwrap();
        assert!(matches!(turn.action, Some(Action::CallLlm(ref p)) if p.context.is_none()));
    }

    // -----------------------------------------------------------------------
    // Compaction triggers
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn compaction_triggers_when_threshold_exceeded() {
        let config = DefaultStrategyConfig {
            compaction: CompactionConfig {
                enabled: true,
                threshold: 5,
                prompt: Some("Summarize.".into()),
            },
        };
        let strategy = DefaultStrategy::new(config);
        let agent = test_agent();
        let mut state = make_session(&agent);

        // Build up messages to exceed threshold (5).
        // Each iteration adds: user + assistant (with tool call) + tool_call_completed + tool_msg = 3 messages
        let mut payloads = Vec::new();
        for i in 0..3 {
            let call_id = format!("call-{}", i);
            let tc_id = format!("tc-call-{}", i);
            payloads.push(user_msg(&format!("msg {}", i)));
            payloads.push(assistant_msg(&call_id, &format!("reply {}", i)));
            payloads.push(EventPayload::ToolCallRequested(ToolCallRequested {
                tool_call_id: tc_id.clone(),
                name: "test_tool".into(),
                arguments: "{}".into(),
                deadline: chrono::Utc::now() + chrono::Duration::hours(1),
            }));
            payloads.push(tool_call_completed(&tc_id));
            payloads.push(tool_msg(&tc_id));
        }
        apply(&mut state, payloads);

        // state.messages should now have 9 messages (3 user + 3 assistant + 3 tool)
        assert!(
            state.messages.len() >= 5,
            "should have enough messages: {}",
            state.messages.len()
        );

        // The last tool_msg triggers compaction check.
        let last_tool_event = tool_msg("tc-call-2");

        let turn = strategy.on_event(&state, &last_tool_event).await.unwrap();

        // Should enter compaction phase.
        let new_state: DefaultStrategyState =
            serde_json::from_value(turn.state.clone()).unwrap();
        assert!(
            matches!(new_state.phase, CompactionPhase::Compacting),
            "should enter compacting phase"
        );

        // Action should be CallLlm with custom context containing the compaction prompt.
        match &turn.action {
            Some(Action::CallLlm(params)) => {
                let ctx = params.context.as_ref().expect("should have custom context");
                // First message should be the compaction system prompt.
                assert_eq!(ctx[0].role, Role::System);
                assert_eq!(ctx[0].content.as_deref(), Some("Summarize."));
                // Should include the conversation messages.
                assert!(ctx.len() > 1);
                // Should not stream compaction.
                assert_eq!(params.stream, Some(false));
            }
            other => panic!("expected CallLlm, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn compaction_response_stores_summary_and_continues() {
        let config = DefaultStrategyConfig {
            compaction: CompactionConfig {
                enabled: true,
                threshold: 5,
                prompt: None,
            },
        };
        let strategy = DefaultStrategy::new(config);
        let agent = test_agent();
        let mut state = make_session(&agent);

        // Add enough messages.
        let mut payloads = Vec::new();
        for i in 0..3 {
            let call_id = format!("call-{}", i);
            let tc_id = format!("tc-call-{}", i);
            payloads.push(user_msg(&format!("msg {}", i)));
            payloads.push(assistant_msg(&call_id, &format!("reply {}", i)));
            payloads.push(EventPayload::ToolCallRequested(ToolCallRequested {
                tool_call_id: tc_id.clone(),
                name: "test_tool".into(),
                arguments: "{}".into(),
                deadline: chrono::Utc::now() + chrono::Duration::hours(1),
            }));
            payloads.push(tool_call_completed(&tc_id));
            payloads.push(tool_msg(&tc_id));
        }
        apply(&mut state, payloads);

        let message_count_before = state.messages.len();

        // Simulate compaction phase state.
        state.strategy_state = serde_json::to_value(DefaultStrategyState {
            phase: CompactionPhase::Compacting,
            compacted_summary: None,
            compacted_at_message_index: None,
            pending_reply: None,
        })
        .unwrap();

        // Simulate the compaction LLM response.
        let compaction_response = mock_llm_completed("compact-1", "Summary of conversation.");

        let turn = strategy.on_event(&state, &compaction_response).await.unwrap();

        // State should be back to normal with summary stored.
        let new_state: DefaultStrategyState =
            serde_json::from_value(turn.state.clone()).unwrap();
        assert!(
            matches!(new_state.phase, CompactionPhase::Normal),
            "should return to normal phase"
        );
        assert_eq!(
            new_state.compacted_summary.as_deref(),
            Some("Summary of conversation.")
        );
        assert_eq!(
            new_state.compacted_at_message_index,
            Some(message_count_before + 1)
        );

        // Action should be CallLlm with context containing summary.
        match &turn.action {
            Some(Action::CallLlm(params)) => {
                let ctx = params.context.as_ref().expect("should have custom context");
                // System prompt from agent config.
                assert_eq!(ctx[0].role, Role::System);
                assert_eq!(ctx[0].content.as_deref(), Some("You are helpful."));
                // Summary as user message.
                assert_eq!(ctx[1].role, Role::User);
                assert!(ctx[1]
                    .content
                    .as_ref()
                    .unwrap()
                    .contains("Summary of conversation."));
                assert_eq!(params.stream, Some(true));
            }
            other => panic!("expected CallLlm, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn compacted_context_used_in_subsequent_turns() {
        let config = DefaultStrategyConfig {
            compaction: CompactionConfig {
                enabled: true,
                threshold: 100, // High threshold so we don't re-compact.
                prompt: None,
            },
        };
        let strategy = DefaultStrategy::new(config);
        let agent = test_agent();
        let mut state = make_session(&agent);

        // Add some messages.
        apply(
            &mut state,
            vec![user_msg("hello"), user_msg("world")],
        );

        // Simulate state after compaction completed.
        state.strategy_state = serde_json::to_value(DefaultStrategyState {
            phase: CompactionPhase::Normal,
            compacted_summary: Some("User said hello and world.".into()),
            compacted_at_message_index: Some(2), // Compact point after 2 messages.
            pending_reply: None,
        })
        .unwrap();

        // New user message arrives.
        let new_msg = user_msg("how are you?");
        apply(&mut state, vec![new_msg.clone()]);

        let turn = strategy.on_event(&state, &new_msg).await.unwrap();

        match &turn.action {
            Some(Action::CallLlm(params)) => {
                let ctx = params.context.as_ref().expect("should use compacted context");
                // System prompt.
                assert_eq!(ctx[0].role, Role::System);
                // Summary.
                assert_eq!(ctx[1].role, Role::User);
                assert!(ctx[1].content.as_ref().unwrap().contains("hello and world"));
                // Recent messages (after compaction point).
                assert!(ctx.len() >= 3, "should include recent messages, got {}", ctx.len());
                assert_eq!(ctx[2].role, Role::User);
                assert_eq!(ctx[2].content.as_deref(), Some("how are you?"));
            }
            other => panic!("expected CallLlm, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn no_compaction_below_threshold() {
        let config = DefaultStrategyConfig {
            compaction: CompactionConfig {
                enabled: true,
                threshold: 100,
                prompt: None,
            },
        };
        let strategy = DefaultStrategy::new(config);
        let agent = test_agent();
        let mut state = make_session(&agent);

        // Add a few messages (below threshold).
        let tc_id = "tc-1";
        apply(
            &mut state,
            vec![
                user_msg("hi"),
                EventPayload::ToolCallRequested(ToolCallRequested {
                    tool_call_id: tc_id.into(),
                    name: "test_tool".into(),
                    arguments: "{}".into(),
                    deadline: chrono::Utc::now() + chrono::Duration::hours(1),
                }),
                tool_call_completed(tc_id),
                tool_msg(tc_id),
            ],
        );

        let turn = strategy.on_event(&state, &tool_msg(tc_id)).await.unwrap();

        // Should NOT enter compaction, just normal CallLlm.
        assert!(
            matches!(turn.action, Some(Action::CallLlm(ref p)) if p.context.is_none()),
            "should not compact below threshold"
        );
        assert_eq!(turn.state, DefaultStrategy::serialize_state(&DefaultStrategyState::default()));
    }
}
