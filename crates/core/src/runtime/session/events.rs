use std::collections::{BTreeMap, HashMap};

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use super::decision::{LlmHandler, ToolHandler, Trigger};
use crate::connectors::registry::ConnectionPath;
use crate::connectors::{AuthNeed, RemoteTool};
pub use crate::protocol::EffectKind;
use crate::protocol::{
    AgentConfig, ConnectorToolKind, DeferToolsStrategy, DraftMessage, ErrorInfo, InterruptOrigin,
    LlmFormat, LlmRequest, LlmResponse, Message, MessageTree, NewMessage, RetryPolicy,
    SessionOwner, SpawnMode, StoredResult, Usage, WorkerRef, WorkerState,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum EventPayload {
    #[serde(rename = "session.created")]
    SessionCreated(Box<SessionCreated>),
    #[serde(rename = "message.new")]
    NewMessage(NewMessage),
    #[serde(rename = "head.moved")]
    HeadMoved(HeadMoved),
    #[serde(rename = "llm.call.requested")]
    LlmCallRequested(LlmCallRequested),
    #[serde(rename = "llm.call.dispatched")]
    LlmCallDispatched(LlmCallDispatched),
    #[serde(rename = "llm.call.completed")]
    LlmCallCompleted(LlmCallCompleted),
    #[serde(rename = "llm.call.errored")]
    LlmCallErrored(LlmCallErrored),
    #[serde(rename = "tool.call.requested")]
    ToolCallRequested(ToolCallRequested),
    #[serde(rename = "tool.call.dispatched")]
    ToolCallDispatched(ToolCallDispatched),
    #[serde(rename = "tool.call.completed")]
    ToolCallCompleted(ToolCallCompleted),
    #[serde(rename = "tool.call.errored")]
    ToolCallErrored(ToolCallErrored),
    #[serde(rename = "connector.sync.requested")]
    ConnectorSyncRequested(ConnectorSyncRequested),
    #[serde(rename = "connector.sync.completed")]
    ConnectorSyncCompleted(Box<ConnectorSyncCompleted>),
    #[serde(rename = "connector.sync.errored")]
    ConnectorSyncErrored(ConnectorSyncErrored),
    #[serde(rename = "connector.auth.failed")]
    ConnectorAuthFailed(ConnectorAuthFailed),
    #[serde(rename = "subagent.requested")]
    SubagentRequested(SubagentRequested),
    #[serde(rename = "subagent.dispatched")]
    SubagentDispatched(SubagentDispatched),
    #[serde(rename = "subagent.started")]
    SubagentStarted(SubagentStarted),
    #[serde(rename = "subagent.errored")]
    SubagentErrored(SubagentErrored),
    #[serde(rename = "session.interrupted")]
    SessionInterrupted(SessionInterrupted),
    #[serde(rename = "session.interrupt_resumed")]
    InterruptResumed(InterruptResumed),
    #[serde(rename = "decision.dispatched")]
    DecisionDispatched(DecisionDispatched),
    #[serde(rename = "decision.completed")]
    DecisionCompleted(DecisionCompleted),
    #[serde(rename = "decision.errored")]
    DecisionErrored(DecisionErrored),
    #[serde(rename = "session.message_requested")]
    SessionMessageRequested(SessionMessageRequested),
    #[serde(rename = "worker.state.updated")]
    WorkerStateUpdated(WorkerStateUpdated),
    #[serde(rename = "agent.updated")]
    AgentConfigUpdated(AgentConfigUpdated),
    #[serde(rename = "subagent.turn_completed")]
    SubagentTurnCompleted(SubagentTurnCompleted),
    #[serde(rename = "decision.queued")]
    DecisionQueued(DecisionQueued),
    #[serde(rename = "decision.dropped")]
    DecisionDropped(DecisionDropped),
    #[serde(rename = "call.voided")]
    CallVoided(CallVoided),
    #[serde(rename = "channels.updated")]
    ChannelsUpdated(ChannelsUpdated),
    #[serde(rename = "session.cancelled")]
    SessionCancelled,
    #[serde(rename = "session.done")]
    SessionDone(SessionDone),
    #[serde(rename = "turn.started")]
    TurnStarted(TurnStarted),
    #[serde(rename = "turn.completed")]
    TurnCompleted(TurnCompleted),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCreated {
    pub agent_id: String,
    pub identity: SessionOwner,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ancestry: Vec<String>,
    pub worker_retry: RetryPolicy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker: Option<WorkerRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionDone {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadMoved {
    pub head_id: String,
}

impl MessageTree {
    pub fn path_to(&self, leaf: &str) -> Vec<Message> {
        let mut by_id: HashMap<&str, &NewMessage> = self
            .nodes
            .iter()
            .map(|n| (n.message.id.as_str(), n))
            .collect();
        let mut path = Vec::new();
        let mut cursor = Some(leaf.to_string());
        while let Some(id) = cursor.take() {
            let Some(node) = by_id.remove(id.as_str()) else {
                break;
            };
            path.push(node.message.clone());
            cursor = node.parent_id.clone();
        }
        path.reverse();
        path
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCallRequested {
    pub id: String,
    pub attempt: u32,
    pub llm: String,
    pub request: LlmRequest,
    pub stream: bool,
    pub retry: RetryPolicy,
    #[serde(default)]
    pub handler: LlmHandler,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<LlmFormat>,
    #[serde(default)]
    pub defer_tools_strategy: DeferToolsStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCallDispatched {
    pub id: String,
    pub attempt: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDispatched {
    pub id: String,
    pub attempt: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentDispatched {
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCallCompleted {
    pub id: String,
    pub attempt: u32,
    pub response: LlmResponse,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCallErrored {
    pub id: String,
    pub attempt: u32,
    pub error: ErrorInfo,
    #[serde(default = "default_true")]
    pub retryable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorSyncRequested {
    pub path: ConnectionPath,
    pub attempt: u32,
    pub retry: RetryPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorSyncCompleted {
    pub path: ConnectionPath,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server: Option<String>,
    pub tools: Vec<RemoteTool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorSyncErrored {
    pub path: ConnectionPath,
    pub error: ErrorInfo,
    #[serde(default)]
    pub retryable: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth: Option<AuthNeed>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorAuthFailed {
    pub path: ConnectionPath,
    pub auth: AuthNeed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentRequested {
    pub id: String,
    pub agent_id: String,
    #[serde(default)]
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<DraftMessage>,
    pub retry: RetryPolicy,
    #[serde(default, skip_serializing_if = "is_blocking")]
    pub mode: SpawnMode,
}

fn is_blocking(mode: &SpawnMode) -> bool {
    *mode == SpawnMode::Blocking
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentStarted {
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentErrored {
    pub id: String,
    pub error: ErrorInfo,
    #[serde(default)]
    pub retryable: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConnectorTarget {
    Remote {
        path: ConnectionPath,
        remote_name: String,
    },
    Find,
    Call,
    Skill {
        plugin: String,
        skill: String,
    },
}

impl ConnectorTarget {
    pub fn kind(&self) -> ConnectorToolKind {
        match self {
            Self::Remote { .. } => ConnectorToolKind::Remote,
            Self::Find => ConnectorToolKind::Find,
            Self::Call => ConnectorToolKind::Call,
            Self::Skill { .. } => ConnectorToolKind::Skill,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRequested {
    pub id: String,
    pub attempt: u32,
    pub name: String,
    pub arguments: String,
    #[serde(default)]
    pub handler: ToolHandler,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<ConnectorTarget>,
    pub retry: RetryPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallCompleted {
    pub id: String,
    pub name: String,
    pub result: StoredResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallErrored {
    pub id: String,
    pub name: String,
    pub error: ErrorInfo,
    #[serde(default)]
    pub retryable: bool,
}

impl InterruptOrigin {
    pub fn privilege(self) -> u8 {
        match self {
            InterruptOrigin::System => 3,
            InterruptOrigin::Operator => 2,
            InterruptOrigin::Machine => 1,
            InterruptOrigin::Frontend => 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInterrupted {
    pub interrupt_id: String,
    pub origin: InterruptOrigin,
    pub reason: String,
    pub payload: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterruptResumed {
    pub interrupt_id: String,
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionDispatched {
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionCompleted {
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionErrored {
    pub id: String,
    pub error: ErrorInfo,
    #[serde(default = "default_true")]
    pub retryable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMessageRequested {
    pub target_session_id: String,
    pub message: DraftMessage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerStateUpdated {
    pub state: WorkerState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfigUpdated {
    pub config: AgentConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnStarted {
    pub turn_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnCompleted {
    pub turn_id: String,
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub data: serde_json::Value,
    #[serde(default)]
    pub turn_cost: Decimal,
    #[serde(default)]
    pub turn_token_usage: Usage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionQueued {
    pub id: String,
    pub trigger: Trigger,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionDropped {
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelsUpdated {
    pub decision_id: String,
    #[serde(default)]
    pub finishes_turn: bool,
    pub channels: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallVoided {
    pub kind: EffectKind,
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentTurnCompleted {
    pub id: String,
    #[serde(default)]
    pub cost: Decimal,
    #[serde(default)]
    pub token_usage: Usage,
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub data: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorInfo>,
}
