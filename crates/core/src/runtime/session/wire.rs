//! The wire forms of the decision protocol and the two boundaries that convert them
//! to and from the internal representation.
//!
//! - Inbound (worker → engine): [`WireAction`] is deserialized from untrusted worker
//!   output; `resolve_actions` is the single seam where it becomes the strict internal
//!   [`Action`] the core consumes.
//! - Outbound (engine → worker): [`WireTrigger`] is the materialized projection of the
//!   internal [`DecisionTrigger`] a worker sees; `to_wire_trigger` is the single seam
//!   that produces it. It has no `ClientMessage` — that variant can never reach a worker.

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use super::decision::{Action, DecisionTrigger};
use super::events::{LlmHandler, MessageTree, ToolHandler};
use super::message::{Content, Message, Role, ToolCall};
use super::reconcile::news_start;
use super::state::{new_call_id, new_message_id, Effect};
use crate::runtime::llm::{ErrorCode, LlmRequest, LlmResponse};
use crate::runtime::owner::SessionOwner;
use crate::runtime::retry::RetryPolicy;
use crate::runtime::worker::{WorkerDecisionRequest, WorkerState};

/// The wire form of a [`Message`]: `id` is optional because a client-submitted or
/// worker-authored message is not yet recorded. `record`/`rerecord` are the seams
/// that lower it to the internal [`Message`] (id always present) at recording time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireMessage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub role: Role,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Content>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl From<Message> for WireMessage {
    fn from(m: Message) -> Self {
        WireMessage {
            id: Some(m.id),
            role: m.role,
            content: m.content,
            tool_calls: m.tool_calls,
            tool_call_id: m.tool_call_id,
            name: m.name,
        }
    }
}

impl WireMessage {
    /// Record with the wire id if present, else a minted one.
    pub fn record(self) -> Message {
        Message {
            id: self.id.unwrap_or_else(new_message_id),
            role: self.role,
            content: self.content,
            tool_calls: self.tool_calls,
            tool_call_id: self.tool_call_id,
            name: self.name,
        }
    }

    /// Record as a fresh node, ignoring any wire id (for reconcile re-record).
    pub fn rerecord(self) -> Message {
        Message {
            id: new_message_id(),
            role: self.role,
            content: self.content,
            tool_calls: self.tool_calls,
            tool_call_id: self.tool_call_id,
            name: self.name,
        }
    }
}

/// The trigger a worker sees on the wire — the materialized projection of the internal
/// [`DecisionTrigger`]. It has no `ClientMessage`: a bare client message is always
/// materialized to `ClientTranscript` by `to_wire_trigger` before delivery, so an
/// unmaterialized message can never reach a worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WireTrigger {
    #[serde(rename = "client.messages")]
    ClientTranscript {
        messages: Vec<WireMessage>,
        #[serde(default)]
        new_from: usize,
    },
    #[serde(rename = "client.action")]
    ClientAction {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        args: Option<serde_json::Value>,
    },
    /// Answer with `tool.result`/`tool.error`.
    #[serde(rename = "tool.execute")]
    ToolExecute {
        id: String,
        name: String,
        arguments: String,
        attempt: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        deadline: Option<DateTime<Utc>>,
    },
    #[serde(rename = "tool.finished")]
    ToolFinished {
        id: String,
        ok: bool,
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        result: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Answer with `llm.result`/`llm.error`.
    #[serde(rename = "llm.execute")]
    LlmExecute {
        id: String,
        request: LlmRequest,
        #[serde(default)]
        stream: bool,
        attempt: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        deadline: Option<DateTime<Utc>>,
    },
    #[serde(rename = "llm.finished")]
    LlmFinished {
        id: String,
        ok: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<WireMessage>,
        #[serde(default)]
        truncated: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<serde_json::Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cost: Option<Decimal>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        code: Option<ErrorCode>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<serde_json::Value>,
    },
    #[serde(rename = "sub_agent.finished")]
    SubAgentFinished {
        id: String,
        ok: bool,
        session_id: String,
        agent_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        result: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    #[serde(rename = "interrupt.resumed")]
    InterruptResumed {
        interrupt_id: String,
        #[serde(default)]
        payload: serde_json::Value,
    },
}

/// The action a worker authors on the wire. Mirrors [`Action`], but a settle's effect
/// id may be omitted: on the sync/pull paths the answered `*.execute` trigger names
/// it, so echoing it is redundant. `resolve_actions` turns this into the internal
/// [`Action`] (id always present) at the transport boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WireAction {
    /// `id` omitted ⇒ the engine mints one; it becomes the assistant node's id.
    #[serde(rename = "llm.call")]
    CallLlm {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        request: LlmRequest,
        #[serde(default)]
        stream: bool,
        #[serde(default = "RetryPolicy::no_retry")]
        retry: RetryPolicy,
        handler: LlmHandler,
    },
    /// `id` omitted ⇒ the engine mints one (LLM-driven tools carry the model's id).
    #[serde(rename = "tool.call")]
    CallTool {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        name: String,
        arguments: String,
        handler: ToolHandler,
        #[serde(default = "RetryPolicy::no_retry")]
        retry: RetryPolicy,
    },
    /// `id` omitted ⇒ the effect named by the answering `tool.execute` trigger.
    #[serde(rename = "tool.result")]
    ToolResult {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        attempt: Option<u32>,
        result: String,
    },
    /// `id` omitted ⇒ the effect named by the answering `llm.execute` trigger.
    #[serde(rename = "llm.result")]
    LlmResult {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        attempt: Option<u32>,
        response: LlmResponse,
    },
    #[serde(rename = "tool.error")]
    ToolError {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        attempt: Option<u32>,
        error: String,
        retryable: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        code: Option<ErrorCode>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<serde_json::Value>,
    },
    #[serde(rename = "llm.error")]
    LlmError {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        attempt: Option<u32>,
        error: String,
        retryable: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        code: Option<ErrorCode>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<serde_json::Value>,
    },
    #[serde(rename = "sub_agent.spawn")]
    SpawnSubAgent {
        session_id: String,
        agent_id: String,
        /// The model tool-call this delegation answers — always required.
        tool_call_id: String,
        #[serde(default = "RetryPolicy::no_retry")]
        retry: RetryPolicy,
    },
    #[serde(rename = "message.send")]
    SendMessage {
        session_id: String,
        message: WireMessage,
    },
    /// `interrupt_id` omitted ⇒ the engine mints one to correlate the later resume.
    #[serde(rename = "interrupt")]
    Interrupt {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        interrupt_id: Option<String>,
        reason: String,
        #[serde(default)]
        payload: serde_json::Value,
    },
    #[serde(rename = "done")]
    Done {
        #[serde(default)]
        data: serde_json::Value,
    },
}

/// Which `*.execute` trigger can name an omitted settle id.
#[derive(Debug, Clone, Copy)]
enum SettleKind {
    Tool,
    Llm,
}

impl SettleKind {
    fn as_str(self) -> &'static str {
        match self {
            SettleKind::Tool => "tool",
            SettleKind::Llm => "llm",
        }
    }
}

/// A wire action could not be lowered to an internal [`Action`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveError {
    /// A settle omitted its effect id, but the answered decision was not the matching
    /// `*.execute` (or there was no trigger context), so the id can't be inferred.
    UnresolvableSettleId { kind: &'static str },
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::UnresolvableSettleId { kind } => write!(
                f,
                "{kind}.result/{kind}.error omitted its id, but this decision does not answer a {kind}.execute"
            ),
        }
    }
}

impl std::error::Error for ResolveError {}

fn resolve_settle_id(
    id: Option<String>,
    trigger: Option<&WireTrigger>,
    want: SettleKind,
) -> Result<String, ResolveError> {
    if let Some(id) = id {
        return Ok(id);
    }
    match (want, trigger) {
        (SettleKind::Tool, Some(WireTrigger::ToolExecute { id, .. })) => Ok(id.clone()),
        (SettleKind::Llm, Some(WireTrigger::LlmExecute { id, .. })) => Ok(id.clone()),
        (want, _) => Err(ResolveError::UnresolvableSettleId {
            kind: want.as_str(),
        }),
    }
}

/// Lower worker-authored wire actions to internal actions, filling any omitted settle
/// id from `trigger` (the `*.execute` this decision answers). `trigger` is `None` on
/// the out-of-band submit route, where an omitted settle id is an error.
pub fn resolve_actions(
    actions: Vec<WireAction>,
    trigger: Option<&WireTrigger>,
) -> Result<Vec<Action>, ResolveError> {
    actions
        .into_iter()
        .map(|action| {
            Ok(match action {
                WireAction::CallLlm {
                    id,
                    request,
                    stream,
                    retry,
                    handler,
                } => Action::CallLlm {
                    id: id.unwrap_or_else(new_call_id),
                    request,
                    stream,
                    retry,
                    handler,
                },
                WireAction::CallTool {
                    id,
                    name,
                    arguments,
                    handler,
                    retry,
                } => Action::CallTool {
                    id: id.unwrap_or_else(new_call_id),
                    name,
                    arguments,
                    handler,
                    retry,
                },
                WireAction::ToolResult {
                    id,
                    attempt,
                    result,
                } => Action::ToolResult {
                    id: resolve_settle_id(id, trigger, SettleKind::Tool)?,
                    attempt,
                    result,
                },
                WireAction::LlmResult {
                    id,
                    attempt,
                    response,
                } => Action::LlmResult {
                    id: resolve_settle_id(id, trigger, SettleKind::Llm)?,
                    attempt,
                    response,
                },
                WireAction::ToolError {
                    id,
                    attempt,
                    error,
                    retryable,
                    code,
                    detail,
                } => Action::ToolError {
                    id: resolve_settle_id(id, trigger, SettleKind::Tool)?,
                    attempt,
                    error,
                    retryable,
                    code,
                    detail,
                },
                WireAction::LlmError {
                    id,
                    attempt,
                    error,
                    retryable,
                    code,
                    detail,
                } => Action::LlmError {
                    id: resolve_settle_id(id, trigger, SettleKind::Llm)?,
                    attempt,
                    error,
                    retryable,
                    code,
                    detail,
                },
                WireAction::SpawnSubAgent {
                    session_id,
                    agent_id,
                    tool_call_id,
                    retry,
                } => Action::SpawnSubAgent {
                    session_id,
                    agent_id,
                    tool_call_id,
                    retry,
                },
                WireAction::SendMessage {
                    session_id,
                    message,
                } => Action::SendMessage {
                    session_id,
                    message,
                },
                WireAction::Interrupt {
                    interrupt_id,
                    reason,
                    payload,
                } => Action::Interrupt {
                    interrupt_id: interrupt_id.unwrap_or_else(new_call_id),
                    reason,
                    payload,
                },
                WireAction::Done { data } => Action::Done { data },
            })
        })
        .collect()
}

/// Project an internal [`DecisionTrigger`] to the [`WireTrigger`] a worker sees. A bare
/// `ClientMessage` becomes a full proposed transcript; a `ClientTranscript` has its
/// `new_from` recomputed against the frozen tree; every other trigger maps 1:1. The
/// tree is frozen while the decision is pending, so the result is stable across
/// redeliveries and matches what reconciling the echo will write.
pub fn to_wire_trigger(
    trigger: DecisionTrigger,
    active_path: &[Message],
    tree: &MessageTree,
) -> WireTrigger {
    match trigger {
        DecisionTrigger::ClientMessage { message } => {
            let mut messages: Vec<WireMessage> =
                active_path.iter().cloned().map(WireMessage::from).collect();
            let new_from = messages.len();
            messages.push(message);
            WireTrigger::ClientTranscript { messages, new_from }
        }
        DecisionTrigger::ClientTranscript { messages, .. } => {
            let known: std::collections::HashSet<&str> =
                tree.nodes.iter().map(|n| n.id()).collect();
            let new_from = news_start(&known, &messages);
            WireTrigger::ClientTranscript { messages, new_from }
        }
        DecisionTrigger::ClientAction { name, args } => WireTrigger::ClientAction { name, args },
        DecisionTrigger::ToolExecute {
            id,
            name,
            arguments,
            attempt,
            deadline,
        } => WireTrigger::ToolExecute {
            id,
            name,
            arguments,
            attempt,
            deadline,
        },
        DecisionTrigger::LlmExecute {
            id,
            request,
            stream,
            attempt,
            deadline,
        } => WireTrigger::LlmExecute {
            id,
            request,
            stream,
            attempt,
            deadline,
        },
        DecisionTrigger::ToolFinished {
            id,
            ok,
            name,
            result,
            error,
        } => WireTrigger::ToolFinished {
            id,
            ok,
            name,
            result,
            error,
        },
        DecisionTrigger::SubAgentFinished {
            id,
            ok,
            session_id,
            agent_id,
            result,
            error,
        } => WireTrigger::SubAgentFinished {
            id,
            ok,
            session_id,
            agent_id,
            result,
            error,
        },
        DecisionTrigger::LlmFinished {
            id,
            ok,
            message,
            truncated,
            usage,
            cost,
            error,
            code,
            detail,
        } => WireTrigger::LlmFinished {
            id,
            ok,
            message,
            truncated,
            usage,
            cost,
            error,
            code,
            detail,
        },
        DecisionTrigger::InterruptResumed {
            interrupt_id,
            payload,
        } => WireTrigger::InterruptResumed {
            interrupt_id,
            payload,
        },
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireDecisionResponse {
    #[serde(default)]
    pub messages: Vec<WireMessage>,
    pub actions: Vec<WireAction>,
    /// Omitted or `null` keeps the current state; clear with a non-null empty value.
    #[serde(default)]
    pub state: Option<WorkerState>,
}

#[derive(Debug, Serialize)]
pub struct WireDecisionRequest<'a> {
    pub session_id: &'a str,
    pub decision_id: &'a str,
    pub agent_id: &'a str,
    pub identity: &'a SessionOwner,
    pub trigger: &'a WireTrigger,
    pub state: &'a WorkerState,
    pub calls: &'a [Effect],
    /// Count of in-flight `tool_call`/`sub_agent` calls.
    pub pending_calls: usize,
    pub messages: &'a [Message],
    pub message_tree: &'a MessageTree,
    pub ancestry: &'a [String],
    pub attempts: u32,
    pub deadline: &'a Option<DateTime<Utc>>,
    pub turn_id: &'a Option<String>,
}

impl<'a> From<&'a WorkerDecisionRequest> for WireDecisionRequest<'a> {
    fn from(r: &'a WorkerDecisionRequest) -> Self {
        WireDecisionRequest {
            session_id: &r.session_id,
            decision_id: &r.decision_id,
            agent_id: &r.agent_id,
            identity: &r.identity,
            trigger: &r.trigger,
            state: &r.state,
            calls: &r.calls,
            pending_calls: r.pending_calls,
            messages: &r.transcript,
            message_tree: &r.message_tree,
            ancestry: &r.ancestry,
            attempts: r.attempts,
            deadline: &r.deadline,
            turn_id: &r.turn_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::session::events::{NewMessage, Node};
    use crate::runtime::session::message::{Content, Role};

    #[test]
    fn call_id_is_minted_when_omitted() {
        let actions = resolve_actions(
            vec![WireAction::CallTool {
                id: None,
                name: "do_thing".to_string(),
                arguments: "{}".to_string(),
                handler: ToolHandler::Worker,
                retry: RetryPolicy::no_retry(),
            }],
            None,
        )
        .expect("resolves");
        match &actions[0] {
            Action::CallTool { id, .. } => assert!(!id.is_empty(), "engine mints an id"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn omitted_settle_id_is_filled_from_the_answered_execute() {
        let trigger = WireTrigger::ToolExecute {
            id: "eff-1".to_string(),
            name: "do_thing".to_string(),
            arguments: "{}".to_string(),
            attempt: 0,
            deadline: None,
        };
        let actions = resolve_actions(
            vec![WireAction::ToolResult {
                id: None,
                attempt: None,
                result: "ok".to_string(),
            }],
            Some(&trigger),
        )
        .expect("resolves");
        match &actions[0] {
            Action::ToolResult { id, .. } => assert_eq!(id, "eff-1"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn omitted_settle_id_without_a_matching_execute_is_an_error() {
        let err = resolve_actions(
            vec![WireAction::ToolResult {
                id: None,
                attempt: None,
                result: "ok".to_string(),
            }],
            None,
        )
        .unwrap_err();
        assert_eq!(err, ResolveError::UnresolvableSettleId { kind: "tool" });
    }

    fn msg(id: &str, role: Role, text: &str) -> Message {
        Message {
            id: id.to_string(),
            role,
            content: Some(Content::Text(text.to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    fn linear_tree(messages: &[Message]) -> MessageTree {
        let nodes: Vec<Node> = messages
            .iter()
            .enumerate()
            .map(|(i, m)| {
                Node::Message(NewMessage {
                    message: m.clone(),
                    parent_id: (i > 0).then(|| messages[i - 1].id.clone()),
                })
            })
            .collect();
        MessageTree {
            head_id: messages.last().map(|m| m.id.clone()),
            nodes,
        }
    }

    fn transcript_of(trigger: WireTrigger) -> (Vec<WireMessage>, usize) {
        match trigger {
            WireTrigger::ClientTranscript { messages, new_from } => (messages, new_from),
            t => panic!("expected a client.messages trigger; got {t:?}"),
        }
    }

    fn wire_view(messages: &[Message]) -> Vec<WireMessage> {
        messages.iter().cloned().map(WireMessage::from).collect()
    }

    #[test]
    fn materializes_a_client_message_onto_the_active_path() {
        let path = vec![
            msg("u1", Role::User, "hi"),
            msg("a1", Role::Assistant, "yo"),
        ];
        let tree = linear_tree(&path);

        let (messages, new_from) = transcript_of(to_wire_trigger(
            DecisionTrigger::ClientMessage {
                message: msg("u2", Role::User, "more").into(),
            },
            &path,
            &tree,
        ));

        assert_eq!(
            messages.iter().map(|m| m.id.as_deref()).collect::<Vec<_>>(),
            vec![Some("u1"), Some("a1"), Some("u2")]
        );
        assert_eq!(new_from, 2);
    }

    #[test]
    fn annotates_an_appending_full_view() {
        let path = vec![
            msg("u1", Role::User, "hi"),
            msg("a1", Role::Assistant, "yo"),
        ];
        let tree = linear_tree(&path);
        let view = wire_view(&[
            msg("u1", Role::User, "hi"),
            msg("a1", Role::Assistant, "yo"),
            msg("u2", Role::User, "more"),
        ]);

        let (_, new_from) = transcript_of(to_wire_trigger(
            DecisionTrigger::ClientTranscript {
                messages: view,
                new_from: 0,
            },
            &path,
            &tree,
        ));
        assert_eq!(new_from, 2);
    }

    #[test]
    fn annotates_an_edit_at_its_divergence_point() {
        let path = vec![
            msg("u1", Role::User, "hi"),
            msg("a1", Role::Assistant, "yo"),
            msg("u2", Role::User, "more"),
        ];
        let tree = linear_tree(&path);
        // An edit of a1: fresh id, resent up to that point.
        let view = wire_view(&[msg("u1", Role::User, "hi"), msg("e1", Role::User, "edited")]);

        let (_, new_from) = transcript_of(to_wire_trigger(
            DecisionTrigger::ClientTranscript {
                messages: view,
                new_from: 0,
            },
            &path,
            &tree,
        ));
        assert_eq!(new_from, 1);
    }

    #[test]
    fn annotates_a_no_op_resend_as_all_recorded() {
        let path = vec![
            msg("u1", Role::User, "hi"),
            msg("a1", Role::Assistant, "yo"),
        ];
        let tree = linear_tree(&path);

        let (_, new_from) = transcript_of(to_wire_trigger(
            DecisionTrigger::ClientTranscript {
                messages: wire_view(&path),
                new_from: 0,
            },
            &path,
            &tree,
        ));
        assert_eq!(new_from, 2, "empty news: nothing to write");
    }

    #[test]
    fn annotates_an_idless_view_as_all_new() {
        let path = vec![msg("u1", Role::User, "hi")];
        let tree = linear_tree(&path);
        let idless = |text: &str| WireMessage {
            id: None,
            role: Role::User,
            content: Some(Content::Text(text.to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let view = vec![idless("hi"), idless("more")];

        let (_, new_from) = transcript_of(to_wire_trigger(
            DecisionTrigger::ClientTranscript {
                messages: view,
                new_from: 0,
            },
            &path,
            &tree,
        ));
        assert_eq!(new_from, 0);
    }

    #[test]
    fn passes_non_client_triggers_through() {
        let tree = MessageTree::default();
        let out = to_wire_trigger(
            DecisionTrigger::ToolFinished {
                id: "tc-1".to_string(),
                ok: true,
                name: "t".to_string(),
                result: Some("r".to_string()),
                error: None,
            },
            &[],
            &tree,
        );
        assert!(matches!(out, WireTrigger::ToolFinished { .. }));
    }
}
