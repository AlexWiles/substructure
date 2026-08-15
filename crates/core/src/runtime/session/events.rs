use std::collections::{BTreeMap, HashMap};

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use super::decision::{LlmHandler, ToolHandler, Trigger};
use crate::connectors::{AuthNeed, RemoteTool};
pub use crate::protocol::EffectKind;
use crate::protocol::{
    AgentConfig, ConnectorToolKind, DeferToolsStrategy, DraftMessage, ErrorInfo, InterruptOrigin,
    LlmFormat, LlmRequest, LlmResponse, Message, MessageTree, NewMessage, RetryPolicy,
    SessionOwner, StoredResult, Usage, WorkerState,
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
    #[serde(rename = "sub_agent.requested")]
    SubAgentRequested(SubAgentRequested),
    #[serde(rename = "sub_agent.dispatched")]
    SubAgentDispatched(SubAgentDispatched),
    #[serde(rename = "sub_agent.started")]
    SubAgentStarted(SubAgentStarted),
    #[serde(rename = "sub_agent.errored")]
    SubAgentErrored(SubAgentErrored),
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
    #[serde(rename = "plugin.enabled")]
    PluginEnabled(PluginEnabled),
    #[serde(rename = "sub_agent.turn_completed")]
    SubAgentTurnCompleted(SubAgentTurnCompleted),
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
}

/// Pure lifecycle marker — no data. Turn output lives in TurnCompleted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionDone {}

/// The head moved to an existing node (truncation or branch switch); the only
/// non-append head writer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadMoved {
    pub head_id: String,
}

impl MessageTree {
    /// Root-to-`leaf` messages; empty if `leaf` is unknown.
    pub fn path_to(&self, leaf: &str) -> Vec<Message> {
        let mut by_id: HashMap<&str, &NewMessage> = self
            .nodes
            .iter()
            .map(|n| (n.message.id.as_str(), n))
            .collect();
        let mut path = Vec::new();
        let mut cursor = Some(leaf.to_string());
        // `remove` so a malformed parent cycle can't loop forever.
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
    /// The `[llm.*]` block the call runs on.
    pub llm: String,
    pub request: LlmRequest,
    pub stream: bool,
    pub retry: RetryPolicy,
    #[serde(default)]
    pub handler: LlmHandler,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<LlmFormat>,
    /// How a deferred tool reaches the model, from the block this call names.
    #[serde(default)]
    pub defer_tools_strategy: DeferToolsStrategy,
}

/// Dispatch marker: the queued call cleared its gates and starts executing.
/// The payload lives on the call's `*Requested` event and in state; executors
/// key off this marker and read the call from state, mirroring how worker
/// delivery reads a promoted decision. For an LLM call, applying this merges
/// the connector tools in force into the stored spec — dispatch time is when
/// derived context is current.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCallDispatched {
    pub id: String,
    pub attempt: u32,
}

/// Dispatch marker; see [`LlmCallDispatched`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDispatched {
    pub id: String,
    pub attempt: u32,
}

/// Dispatch marker; see [`LlmCallDispatched`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubAgentDispatched {
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

/// Fetch one connection's tool list. Keyed on the connection id, not on the
/// agent version that asked for it: the offer is a fact about the remote, and
/// a config rewritten for unrelated reasons must not cost another round trip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorSyncRequested {
    pub id: String,
    pub attempt: u32,
    pub retry: RetryPolicy,
}

/// What a connection offered, verbatim and unfiltered, plus the prefix its
/// tools are expanded under. The prefix is recorded rather than read back from
/// config so that editing `substructure.toml` cannot rename tools underneath a
/// live session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorSyncCompleted {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix: Option<String>,
    pub tools: Vec<RemoteTool>,
    /// What the server said it is for at the handshake.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorSyncErrored {
    pub id: String,
    pub error: ErrorInfo,
    #[serde(default)]
    pub retryable: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth: Option<AuthNeed>,
}

/// A tool call found that the credential is not valid. The call settles
/// separately with its own error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorAuthFailed {
    pub id: String,
    pub auth: AuthNeed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubAgentRequested {
    pub id: String,
    pub agent_id: String,
    #[serde(default)]
    pub tool_call_id: String,
    /// The child's opening message, sent once its session is created.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<DraftMessage>,
    pub retry: RetryPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubAgentStarted {
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubAgentErrored {
    pub id: String,
    pub error: ErrorInfo,
    #[serde(default)]
    pub retryable: bool,
}

/// Where an engine-executed call lands: the connection, and the name that
/// connection knows the tool by. Frozen onto the call beside its handler, so a
/// config change cannot reroute a call already in flight — and so the audit
/// record names the connection rather than only the model's prefixed alias.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConnectorTarget {
    pub connector: String,
    /// Empty for a `find_tools`, and for a `call_tool` whose name the filter
    /// refused. An empty name is what keeps the connection out of the call.
    pub remote_name: String,
    #[serde(default)]
    pub kind: ConnectorToolKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRequested {
    pub id: String,
    pub attempt: u32,
    pub name: String,
    pub arguments: String,
    #[serde(default)]
    pub handler: ToolHandler,
    /// Set exactly when `handler` is `Server`.
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
    /// Head when raised; `None` parks every path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterruptResumed {
    pub interrupt_id: String,
    pub payload: serde_json::Value,
}

/// Promotion marker: the queued decision is now live. The trigger lives on
/// the decision's `DecisionQueued` event, which always precedes this.
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
    /// Active head when written; `None` if the tree was empty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfigUpdated {
    pub config: AgentConfig,
    /// Active head when written; `None` if the tree was empty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
}

/// A plugin the agent started using. Anchored: a fork from before this point
/// does not inherit it, and a rewind takes it back.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginEnabled {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnStarted {
    pub turn_id: String,
}

/// The run terminal (pass 2): the turn's `turn.finished` finalizer has settled, so
/// the turn — output plus post-turn side effects — is done. `error` is set when the
/// finalizer failed terminally; the output is still durable but the run failed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnCompleted {
    pub turn_id: String,
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub data: serde_json::Value,
    #[serde(default)]
    pub turn_cost: Decimal,
    #[serde(default)]
    pub turn_token_usage: Usage,
    /// Set when the run failed. The turn terminal is the only event every
    /// consumer watches, so this is where a renderer decides how a failure
    /// reads — by its `code`, not by matching on the sentence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionQueued {
    pub id: String,
    pub trigger: Trigger,
}

/// A settle decision dropped when its branch was forked away.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionDropped {
    pub id: String,
}

/// How each channel shows a settled decision, keyed by channel kind. Opaque
/// to the engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelsUpdated {
    pub decision_id: String,
    /// True when the decision answered `turn.finished`.
    #[serde(default)]
    pub finishes_turn: bool,
    pub channels: BTreeMap<String, serde_json::Value>,
}

/// An abandoned in-flight call, named the way its kind names itself: a
/// sub-agent's `id` is the child session, a connector sync's the connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallVoided {
    pub kind: EffectKind,
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubAgentTurnCompleted {
    pub id: String,
    #[serde(default)]
    pub cost: Decimal,
    #[serde(default)]
    pub token_usage: Usage,
    /// The child's turn result.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub data: serde_json::Value,
}
