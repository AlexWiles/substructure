//! The boundaries that convert the wire protocol ([`crate::protocol`]) to and from
//! the internal representation.
//!
//! - Inbound (client → engine): [`ClientInput`] is the authoritative set of everything a
//!   client can send — a submit, an interrupt resume, or a client tool settle. Each variant
//!   carries its own addressing (a submit's `agent_id`/`turn_id`; a settle's effect id), so
//!   `Runtime::handle_client_input` dispatches it directly, mirroring `resolve_response` on the
//!   worker side. A submit rebuilds a [`ClientPayload`] — still what the
//!   `SubmitClientPayload` command seam (`command.rs`) lowers to domain events, and still a
//!   live wire type on the machine and embedded surfaces.
//! - Inbound (worker → engine): [`DecisionAction`] is deserialized from untrusted worker
//!   output; `resolve_response` is the single seam where it becomes the strict internal
//!   [`Action`] the core consumes.
//! - Outbound (engine → worker): [`DecisionTrigger`] is the materialized projection of the
//!   internal [`Trigger`] a worker sees; `to_wire_trigger` is the single seam
//!   that produces it. It has no `ClientMessage` — that variant can never reach a worker.
//!
//! The outbound request also carries `proposed`, the engine-derived default
//! continuation for the trigger; its derivation lives in [`super::propose`].

use std::collections::HashMap;

use serde_json::Value;

use super::decision::{Action, Trigger};
use super::reconcile::news_start;
use super::state::{new_call_id, new_message_id, LlmCallState};
use super::tool_contract::{classify_arguments, declared_tool, DeclaredTool};
use crate::protocol::{
    AgentConfig, DecisionAction, DecisionRequest, DecisionResponse, DecisionTrigger, DraftMessage,
    LlmRequest, Message, MessageTree, RetryPolicy, WorkerState,
};
use crate::runtime::worker::WorkerDecisionRequest;

impl From<Message> for DraftMessage {
    fn from(m: Message) -> Self {
        DraftMessage {
            id: Some(m.id),
            role: m.role,
            content: m.content,
            tool_calls: m.tool_calls,
            tool_call_id: m.tool_call_id,
            name: m.name,
        }
    }
}

impl DraftMessage {
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

/// Canonicalize a lenient JSON wire value (a tool result, tool arguments) to the
/// strict internal string form: a string passes through, `null` becomes empty,
/// any other value becomes its JSON text. Lets workers and clients author these
/// fields as plain values instead of JSON-encoded-in-a-string.
pub fn result_to_string(value: Value) -> String {
    match value {
        Value::String(s) => s,
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

impl<'a> From<&'a WorkerDecisionRequest> for DecisionRequest<'a> {
    fn from(r: &'a WorkerDecisionRequest) -> Self {
        DecisionRequest {
            session_id: &r.session_id,
            decision_id: &r.decision_id,
            agent_id: &r.agent_id,
            identity: &r.identity,
            trigger: &r.trigger,
            proposed: &r.proposed,
            state: &r.state,
            agent: &r.agent,
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
    /// An `llm.call` omitted `model` and no agent config supplied one.
    MissingModel,
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::UnresolvableSettleId { kind } => write!(
                f,
                "{kind}.result/{kind}.error omitted its id, but this decision does not answer a {kind}.execute"
            ),
            ResolveError::MissingModel => write!(
                f,
                "llm.call omitted `model` and no agent config supplies one"
            ),
        }
    }
}

impl std::error::Error for ResolveError {}

fn resolve_settle_id(
    id: Option<String>,
    trigger: Option<&DecisionTrigger>,
    want: SettleKind,
) -> Result<String, ResolveError> {
    if let Some(id) = id {
        return Ok(id);
    }
    match (want, trigger) {
        (SettleKind::Tool, Some(DecisionTrigger::ToolExecute { id, .. })) => Ok(id.clone()),
        (SettleKind::Llm, Some(DecisionTrigger::LlmExecute { id, .. })) => Ok(id.clone()),
        (want, _) => Err(ResolveError::UnresolvableSettleId {
            kind: want.as_str(),
        }),
    }
}

/// The internal, fully-resolved form of a worker decision response: canonical
/// `Action`s (ids minted, `llm.call` merged into a full `LlmRequest`), plus the
/// transcript and the state/agent-config writes.
#[derive(Debug)]
pub struct ResolvedResponse {
    pub messages: Vec<DraftMessage>,
    pub actions: Vec<Action>,
    pub state: Option<WorkerState>,
    pub agent: Option<AgentConfig>,
}

/// Lower a worker-authored decision response to its canonical internal form.
///
/// Fills omitted settle ids from `trigger` (the `*.execute` this decision
/// answers; `None` on the out-of-band submit route, where an omitted id is an
/// error) and merges each flat `llm.call` into a full [`LlmRequest`] using
/// precedence explicit > merge-source config > engine default. The merge source
/// is the config the response itself sets, else `echoed_config` (the config
/// resolved for this decision). The declared view for an omitted-`messages`
/// `llm.call` is the response's `messages`.
pub fn resolve_response(
    response: DecisionResponse,
    echoed_config: Option<&AgentConfig>,
    trigger: Option<&DecisionTrigger>,
) -> Result<ResolvedResponse, ResolveError> {
    let DecisionResponse {
        messages,
        actions,
        state,
        agent,
    } = response;
    let merge_cfg = agent.as_ref().or(echoed_config);
    let resolved = lower_actions(actions, &messages, merge_cfg, trigger)?;
    Ok(ResolvedResponse {
        messages,
        actions: resolved,
        state,
        agent,
    })
}

/// Lower each wire action to its internal [`Action`]. `view` is the response's
/// declared transcript, used to fill an omitted-`messages` `llm.call`.
fn lower_actions(
    actions: Vec<DecisionAction>,
    view: &[DraftMessage],
    config: Option<&AgentConfig>,
    trigger: Option<&DecisionTrigger>,
) -> Result<Vec<Action>, ResolveError> {
    actions
        .into_iter()
        .map(|action| {
            Ok(match action {
                DecisionAction::CallLlm {
                    id,
                    model,
                    messages,
                    tools,
                    temperature,
                    max_completion_tokens,
                    reasoning,
                    stream,
                    retry,
                    handler,
                } => {
                    let model = model
                        .or_else(|| config.map(|c| c.model.clone()))
                        .ok_or(ResolveError::MissingModel)?;
                    // Explicit messages are used verbatim (no system injection);
                    // omitted ⇒ config.system prepended to the declared view.
                    let messages = messages.unwrap_or_else(|| match config {
                        Some(c) => c.prompt_for(view),
                        None => view.to_vec(),
                    });
                    let tools = tools.or_else(|| config.and_then(|c| c.tools_as_llm()));
                    let stream = stream.or_else(|| config.map(|c| c.stream)).unwrap_or(false);
                    let retry = retry
                        .or_else(|| config.and_then(|c| c.retry.clone()))
                        .unwrap_or_else(RetryPolicy::no_retry);
                    Action::CallLlm {
                        id: id.unwrap_or_else(new_call_id),
                        request: LlmRequest {
                            model,
                            messages,
                            tools,
                            temperature,
                            max_completion_tokens,
                            reasoning,
                        },
                        stream,
                        retry,
                        handler,
                    }
                }
                DecisionAction::CallTool {
                    id,
                    name,
                    arguments,
                    handler,
                    retry,
                } => Action::CallTool {
                    id: id.unwrap_or_else(new_call_id),
                    name,
                    arguments: result_to_string(arguments),
                    handler,
                    retry,
                },
                DecisionAction::ToolResult {
                    id,
                    attempt,
                    result,
                } => Action::ToolResult {
                    id: resolve_settle_id(id, trigger, SettleKind::Tool)?,
                    attempt,
                    result: result_to_string(result),
                },
                DecisionAction::LlmResult {
                    id,
                    attempt,
                    response,
                } => Action::LlmResult {
                    id: resolve_settle_id(id, trigger, SettleKind::Llm)?,
                    attempt,
                    response,
                },
                DecisionAction::ToolError {
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
                DecisionAction::LlmError {
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
                DecisionAction::SpawnSubAgent {
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
                DecisionAction::SendMessage {
                    session_id,
                    message,
                } => Action::SendMessage {
                    session_id,
                    message,
                },
                DecisionAction::Interrupt {
                    interrupt_id,
                    reason,
                    payload,
                } => Action::Interrupt {
                    interrupt_id: interrupt_id.unwrap_or_else(new_call_id),
                    reason,
                    payload,
                },
                DecisionAction::Done { data } => Action::Done { data },
            })
        })
        .collect()
}

/// Project an internal [`Trigger`] to the [`DecisionTrigger`] a worker sees. A bare
/// `ClientMessage` becomes a full proposed transcript; a `ClientTranscript` has its
/// `new_from` recomputed against the frozen tree; a `ToolExecute` has its arguments
/// classified against the tool's declared `input` schema (resolved from
/// `open_llm_calls`); every other trigger maps 1:1. The tree is frozen while the
/// decision is pending, so the result is stable across redeliveries and matches
/// what reconciling the echo will write.
pub fn to_wire_trigger(
    trigger: Trigger,
    active_path: &[Message],
    tree: &MessageTree,
    open_llm_calls: &HashMap<String, LlmCallState>,
) -> DecisionTrigger {
    match trigger {
        Trigger::SessionStart => DecisionTrigger::SessionStart,
        Trigger::ClientMessage { message } => {
            let mut messages: Vec<DraftMessage> = active_path
                .iter()
                .cloned()
                .map(DraftMessage::from)
                .collect();
            let new_from = messages.len();
            messages.push(message);
            DecisionTrigger::ClientTranscript { messages, new_from }
        }
        Trigger::ClientTranscript { messages, .. } => {
            let known: std::collections::HashSet<&str> =
                tree.nodes.iter().map(|n| n.id()).collect();
            let new_from = news_start(&known, &messages);
            DecisionTrigger::ClientTranscript { messages, new_from }
        }
        Trigger::ClientAction { name, args } => DecisionTrigger::ClientAction { name, args },
        Trigger::ToolExecute {
            id,
            name,
            arguments,
            attempt,
            deadline,
        } => {
            let schema = match declared_tool(&id, &name, active_path, open_llm_calls) {
                DeclaredTool::Declared(t) => t.input.as_ref(),
                _ => None,
            };
            let input = classify_arguments(&arguments, schema);
            DecisionTrigger::ToolExecute {
                id,
                name,
                arguments,
                input,
                attempt,
                deadline,
            }
        }
        Trigger::LlmExecute {
            id,
            request,
            stream,
            attempt,
            deadline,
        } => DecisionTrigger::LlmExecute {
            id,
            request,
            stream,
            attempt,
            deadline,
        },
        Trigger::ToolFinished {
            id,
            ok,
            name,
            result,
            error,
        } => DecisionTrigger::ToolFinished {
            id,
            ok,
            name,
            result,
            error,
        },
        Trigger::SubAgentFinished {
            id,
            ok,
            session_id,
            agent_id,
            result,
            error,
        } => DecisionTrigger::SubAgentFinished {
            id,
            ok,
            session_id,
            agent_id,
            result,
            error,
        },
        Trigger::LlmFinished {
            id,
            ok,
            message,
            truncated,
            usage,
            cost,
            error,
            code,
            detail,
        } => DecisionTrigger::LlmFinished {
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
        Trigger::InterruptResumed {
            interrupt_id,
            payload,
        } => DecisionTrigger::InterruptResumed {
            interrupt_id,
            payload,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{
        ClientInput, ClientPayload, Content, LlmHandler, NewMessage, Node, Role, ToolCall,
        ToolCallFunction, ToolHandler, ToolInput,
    };
    use crate::runtime::session::state::LlmCallSpec;

    /// Lower just `actions` (no messages/config) through the response seam.
    fn resolve_test_actions(
        actions: Vec<DecisionAction>,
        trigger: Option<&DecisionTrigger>,
    ) -> Result<Vec<Action>, ResolveError> {
        resolve_response(
            DecisionResponse {
                messages: vec![],
                actions,
                state: None,
                agent: None,
            },
            None,
            trigger,
        )
        .map(|r| r.actions)
    }

    #[test]
    fn call_id_is_minted_when_omitted() {
        let actions = resolve_test_actions(
            vec![DecisionAction::CallTool {
                id: None,
                name: "do_thing".to_string(),
                arguments: serde_json::json!({}),
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
        let trigger = DecisionTrigger::ToolExecute {
            id: "eff-1".to_string(),
            name: "do_thing".to_string(),
            arguments: "{}".to_string(),
            input: ToolInput::Valid {
                value: serde_json::json!({}),
            },
            attempt: 0,
            deadline: None,
        };
        let actions = resolve_test_actions(
            vec![DecisionAction::ToolResult {
                id: None,
                attempt: None,
                result: serde_json::json!("ok"),
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
        let err = resolve_test_actions(
            vec![DecisionAction::ToolResult {
                id: None,
                attempt: None,
                result: serde_json::json!("ok"),
            }],
            None,
        )
        .unwrap_err();
        assert_eq!(err, ResolveError::UnresolvableSettleId { kind: "tool" });
    }

    fn cfg(model: &str, system: Option<&str>) -> AgentConfig {
        AgentConfig {
            model: model.to_string(),
            system: system.map(str::to_string),
            stream: true,
            retry: None,
            tools: Vec::new(),
            sub_agents: Vec::new(),
        }
    }

    fn user_wire(text: &str) -> DraftMessage {
        DraftMessage {
            id: None,
            role: Role::User,
            content: Some(Content::Text(text.to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    fn bare_llm_call() -> DecisionAction {
        DecisionAction::CallLlm {
            id: None,
            model: None,
            messages: None,
            tools: None,
            temperature: None,
            max_completion_tokens: None,
            reasoning: None,
            stream: None,
            retry: None,
            handler: LlmHandler::Server,
        }
    }

    fn resolve_one_call(
        call: DecisionAction,
        view: Vec<DraftMessage>,
        echoed: Option<&AgentConfig>,
        response_agent: Option<AgentConfig>,
    ) -> Result<Action, ResolveError> {
        let r = resolve_response(
            DecisionResponse {
                messages: view,
                actions: vec![call],
                state: None,
                agent: response_agent,
            },
            echoed,
            None,
        )?;
        Ok(r.actions.into_iter().next().expect("one action"))
    }

    #[test]
    fn bare_llm_call_merges_model_stream_and_system_from_config() {
        let config = cfg("m1", Some("be nice"));
        let action =
            resolve_one_call(bare_llm_call(), vec![user_wire("hi")], Some(&config), None).unwrap();
        match action {
            Action::CallLlm {
                request, stream, ..
            } => {
                assert_eq!(request.model, "m1");
                assert!(stream, "stream comes from config");
                let roles: Vec<_> = request.messages.iter().map(|m| &m.role).collect();
                assert!(
                    matches!(roles[..], [Role::System, Role::User]),
                    "omitted messages ⇒ system + declared view; got {roles:?}"
                );
            }
            other => panic!("expected llm.call; got {other:?}"),
        }
    }

    #[test]
    fn explicit_messages_suppress_system_injection() {
        let config = cfg("m1", Some("be nice"));
        let call = DecisionAction::CallLlm {
            id: None,
            model: None,
            messages: Some(vec![user_wire("only me")]),
            tools: None,
            temperature: None,
            max_completion_tokens: None,
            reasoning: None,
            stream: None,
            retry: None,
            handler: LlmHandler::Server,
        };
        let action =
            resolve_one_call(call, vec![user_wire("the view")], Some(&config), None).unwrap();
        match action {
            Action::CallLlm { request, .. } => {
                let roles: Vec<_> = request.messages.iter().map(|m| &m.role).collect();
                assert!(
                    matches!(roles[..], [Role::User]),
                    "explicit messages are verbatim, no system; got {roles:?}"
                );
            }
            other => panic!("expected llm.call; got {other:?}"),
        }
    }

    #[test]
    fn explicit_model_overrides_config() {
        let config = cfg("base", None);
        let call = DecisionAction::CallLlm {
            id: None,
            model: Some("override".to_string()),
            messages: None,
            tools: None,
            temperature: None,
            max_completion_tokens: None,
            reasoning: None,
            stream: None,
            retry: None,
            handler: LlmHandler::Server,
        };
        let action = resolve_one_call(call, vec![], Some(&config), None).unwrap();
        match action {
            Action::CallLlm { request, .. } => assert_eq!(request.model, "override"),
            other => panic!("expected llm.call; got {other:?}"),
        }
    }

    #[test]
    fn bare_llm_call_without_a_model_source_is_an_error() {
        let err = resolve_one_call(bare_llm_call(), vec![], None, None).unwrap_err();
        assert_eq!(err, ResolveError::MissingModel);
    }

    #[test]
    fn the_response_config_is_the_merge_source_over_the_echoed_one() {
        let echoed = cfg("old", None);
        let action = resolve_one_call(
            bare_llm_call(),
            vec![user_wire("hi")],
            Some(&echoed),
            Some(cfg("new", None)),
        )
        .unwrap();
        match action {
            Action::CallLlm { request, .. } => assert_eq!(
                request.model, "new",
                "a config set in this response wins over the echoed one"
            ),
            other => panic!("expected llm.call; got {other:?}"),
        }
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

    fn transcript_of(trigger: DecisionTrigger) -> (Vec<DraftMessage>, usize) {
        match trigger {
            DecisionTrigger::ClientTranscript { messages, new_from } => (messages, new_from),
            t => panic!("expected a client.messages trigger; got {t:?}"),
        }
    }

    fn wire_view(messages: &[Message]) -> Vec<DraftMessage> {
        messages.iter().cloned().map(DraftMessage::from).collect()
    }

    #[test]
    fn materializes_a_client_message_onto_the_active_path() {
        let path = vec![
            msg("u1", Role::User, "hi"),
            msg("a1", Role::Assistant, "yo"),
        ];
        let tree = linear_tree(&path);

        let (messages, new_from) = transcript_of(to_wire_trigger(
            Trigger::ClientMessage {
                message: msg("u2", Role::User, "more").into(),
            },
            &path,
            &tree,
            &HashMap::new(),
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
            Trigger::ClientTranscript {
                messages: view,
                new_from: 0,
            },
            &path,
            &tree,
            &HashMap::new(),
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
            Trigger::ClientTranscript {
                messages: view,
                new_from: 0,
            },
            &path,
            &tree,
            &HashMap::new(),
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
            Trigger::ClientTranscript {
                messages: wire_view(&path),
                new_from: 0,
            },
            &path,
            &tree,
            &HashMap::new(),
        ));
        assert_eq!(new_from, 2, "empty news: nothing to write");
    }

    #[test]
    fn annotates_an_idless_view_as_all_new() {
        let path = vec![msg("u1", Role::User, "hi")];
        let tree = linear_tree(&path);
        let idless = |text: &str| DraftMessage {
            id: None,
            role: Role::User,
            content: Some(Content::Text(text.to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let view = vec![idless("hi"), idless("more")];

        let (_, new_from) = transcript_of(to_wire_trigger(
            Trigger::ClientTranscript {
                messages: view,
                new_from: 0,
            },
            &path,
            &tree,
            &HashMap::new(),
        ));
        assert_eq!(new_from, 0);
    }

    #[test]
    fn passes_non_client_triggers_through() {
        let tree = MessageTree::default();
        let out = to_wire_trigger(
            Trigger::ToolFinished {
                id: "tc-1".to_string(),
                ok: true,
                name: "t".to_string(),
                result: Some("r".to_string()),
                error: None,
            },
            &[],
            &tree,
            &HashMap::new(),
        );
        assert!(matches!(out, DecisionTrigger::ToolFinished { .. }));
    }

    fn tool_execute(
        name: &str,
        arguments: &str,
        active_path: &[Message],
        open_llm_calls: &HashMap<String, LlmCallState>,
    ) -> ToolInput {
        let trigger = to_wire_trigger(
            Trigger::ToolExecute {
                id: "tc-1".to_string(),
                name: name.to_string(),
                arguments: arguments.to_string(),
                attempt: 0,
                deadline: None,
            },
            active_path,
            &MessageTree::default(),
            open_llm_calls,
        );
        match trigger {
            DecisionTrigger::ToolExecute { input, .. } => input,
            t => panic!("expected a tool.execute trigger; got {t:?}"),
        }
    }

    fn weather_call(schema: serde_json::Value) -> (Vec<Message>, HashMap<String, LlmCallState>) {
        use crate::protocol::LlmTool;
        use crate::runtime::session::state::EffectTracking;

        let assistant = Message {
            id: "call-1".to_string(),
            role: Role::Assistant,
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "tc-1".to_string(),
                call_type: "function".to_string(),
                function: ToolCallFunction {
                    name: "get_weather".to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            tool_call_id: None,
            name: None,
        };
        let call = LlmCallState {
            call_id: "call-1".to_string(),
            tracking: EffectTracking::new(RetryPolicy::no_retry(), chrono::Utc::now()),
            prompt: vec![],
            spec: LlmCallSpec {
                model: "m".to_string(),
                tools: Some(vec![LlmTool {
                    name: "get_weather".to_string(),
                    description: "d".to_string(),
                    input: Some(schema),
                    output: None,
                }]),
                temperature: None,
                max_completion_tokens: None,
                reasoning: None,
            },
            stream: false,
            handler: LlmHandler::Server,
            anchor: None,
        };
        (
            vec![msg("u1", Role::User, "hi"), assistant],
            HashMap::from([("call-1".to_string(), call)]),
        )
    }

    #[test]
    fn tool_arguments_are_classified_against_the_declared_input() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
        });
        let (path, calls) = weather_call(schema);

        assert!(matches!(
            tool_execute("get_weather", r#"{"city":"NYC"}"#, &path, &calls),
            ToolInput::Valid { .. }
        ));
        match tool_execute("get_weather", r#"{"city":5}"#, &path, &calls) {
            ToolInput::Invalid { value, error } => {
                assert_eq!(
                    value,
                    serde_json::json!({"city":5}),
                    "the object is still delivered"
                );
                assert!(
                    error.contains("city"),
                    "the violation is reported; got {error}"
                );
            }
            other => panic!("expected invalid; got {other:?}"),
        }
        assert!(matches!(
            tool_execute("get_weather", "not json", &path, &calls),
            ToolInput::Malformed { .. }
        ));
    }

    #[test]
    fn the_input_classification_is_always_on_the_wire() {
        let trigger = to_wire_trigger(
            Trigger::ToolExecute {
                id: "tc-1".to_string(),
                name: "t".to_string(),
                arguments: "not json".to_string(),
                attempt: 0,
                deadline: None,
            },
            &[],
            &MessageTree::default(),
            &HashMap::new(),
        );
        let v = serde_json::to_value(&trigger).expect("serializes");
        assert_eq!(v["input"]["status"], "malformed");
        assert!(v["input"]["error"].is_string());
    }

    #[test]
    fn an_undeclared_tool_is_classified_by_parse_alone() {
        let (path, calls) = weather_call(serde_json::json!({"type": "object"}));
        assert!(
            matches!(
                tool_execute("not_declared", r#"{"anything": true}"#, &path, &calls),
                ToolInput::Valid { .. }
            ),
            "no declared schema to check against; the proposal carries the unknown-tool default"
        );
    }

    #[test]
    fn action_defaults_fill_handler_and_retryable() {
        let actions: Vec<DecisionAction> = serde_json::from_str(
            r#"[
                {"type":"llm.call","request":{"model":"m","messages":[]}},
                {"type":"tool.call","name":"t","arguments":{"city":"NYC"}},
                {"type":"tool.error","id":"tc-1","error":"boom"}
            ]"#,
        )
        .expect("defaults fill");
        assert!(matches!(
            &actions[0],
            DecisionAction::CallLlm {
                handler: LlmHandler::Server,
                ..
            }
        ));
        match &actions[1] {
            DecisionAction::CallTool {
                handler, arguments, ..
            } => {
                assert!(matches!(handler, ToolHandler::Worker));
                assert_eq!(
                    arguments,
                    &serde_json::json!({"city":"NYC"}),
                    "object arguments ride the wire as-is; the resolve seam canonicalizes"
                );
            }
            other => panic!("expected a tool.call; got {other:?}"),
        }
        assert!(matches!(
            &actions[2],
            DecisionAction::ToolError {
                retryable: false,
                ..
            }
        ));
    }

    #[test]
    fn a_tool_result_settles_with_any_json_value() {
        let settle = |json: &str| {
            let a: DecisionAction = serde_json::from_str(json).expect("parses");
            let actions = resolve_test_actions(vec![a], None).expect("resolves");
            match actions.into_iter().next().expect("one action") {
                Action::ToolResult { result, .. } => result,
                other => panic!("expected a tool.result; got {other:?}"),
            }
        };
        assert_eq!(
            settle(r#"{"type":"tool.result","id":"tc-1","result":{"temp":71}}"#),
            r#"{"temp":71}"#
        );
        assert_eq!(
            settle(r#"{"type":"tool.result","id":"tc-1","result":"plain"}"#),
            "plain"
        );
    }

    #[test]
    fn a_decision_may_omit_actions() {
        let r: DecisionResponse = serde_json::from_str(r#"{"messages":[]}"#).expect("parses");
        assert!(r.actions.is_empty());
    }

    #[test]
    fn client_input_parses_every_tag_to_its_variant() {
        let cases: [(&str, fn(&ClientInput) -> bool); 6] = [
            (
                r#"{"type":"client.message","agent_id":"bot","message":{"role":"user","content":"hi"}}"#,
                |i| matches!(i, ClientInput::Message { .. }),
            ),
            (
                r#"{"type":"client.messages","agent_id":"bot","messages":[]}"#,
                |i| matches!(i, ClientInput::Messages { .. }),
            ),
            (
                r#"{"type":"client.action","agent_id":"bot","name":"approve","args":{"ok":true}}"#,
                |i| matches!(i, ClientInput::Action { .. }),
            ),
            (r#"{"type":"interrupt.resume","interrupt_id":"iid"}"#, |i| {
                matches!(i, ClientInput::InterruptResume { .. })
            }),
            (
                r#"{"type":"tool.result","id":"c1","result":{"n":1}}"#,
                |i| matches!(i, ClientInput::ToolResult { .. }),
            ),
            (
                r#"{"type":"tool.error","id":"c1","error":"boom","retryable":true}"#,
                |i| matches!(i, ClientInput::ToolError { .. }),
            ),
        ];
        for (json, is_variant) in cases {
            let input: ClientInput = serde_json::from_str(json).expect("parses");
            assert!(is_variant(&input), "wrong variant for {json}");
        }
    }

    #[test]
    fn a_submit_without_an_agent_id_is_a_deserialize_error() {
        // `agent_id` is a required field of the submit variants, so a missing one is a
        // type error at the boundary — not a runtime check.
        let err = serde_json::from_str::<ClientInput>(
            r#"{"type":"client.message","message":{"role":"user","content":"hi"}}"#,
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("agent_id"),
            "error should name agent_id: {err}"
        );
    }

    #[test]
    fn a_settle_has_no_agent_or_turn_slot_to_misplace() {
        // A stray `agent_id`/`turn_id` on a settle has no field to land in, so it is
        // ignored rather than mistaken for addressing.
        let input: ClientInput = serde_json::from_str(
            r#"{"type":"tool.result","id":"c1","result":"ok","agent_id":"bot","turn_id":"t1"}"#,
        )
        .expect("parses; the stray addressing fields are ignored");
        assert!(matches!(input, ClientInput::ToolResult { .. }));
    }

    #[test]
    fn submit_payload_json_stays_flat_after_the_newtype_conversion() {
        // The `client.action` body was `#[serde(flatten)]`; the newtype variant must keep
        // `name`/`args` at the top level rather than nesting them under `action`.
        let payload = ClientPayload::Action(crate::protocol::ClientAction {
            name: "approve".to_string(),
            args: Some(serde_json::json!({"ok": true})),
        });
        assert_eq!(
            serde_json::to_value(&payload).unwrap(),
            serde_json::json!({"type": "client.action", "name": "approve", "args": {"ok": true}})
        );
    }

    #[test]
    fn unknown_client_input_tag_lists_all_six() {
        let err = serde_json::from_str::<ClientInput>(r#"{"type":"frob"}"#)
            .unwrap_err()
            .to_string();
        for tag in [
            "client.message",
            "client.messages",
            "client.action",
            "interrupt.resume",
            "tool.result",
            "tool.error",
        ] {
            assert!(err.contains(tag), "error missing {tag}: {err}");
        }
    }

    #[test]
    fn interrupt_resume_uses_the_interrupt_id_field() {
        let input: ClientInput =
            serde_json::from_str(r#"{"type":"interrupt.resume","interrupt_id":"iid"}"#)
                .expect("parses");
        match input {
            ClientInput::InterruptResume { interrupt_id, .. } => {
                assert_eq!(interrupt_id, "iid")
            }
            other => panic!("expected interrupt.resume, got {other:?}"),
        }
    }

    #[test]
    fn tool_result_canonicalizes_non_string_values() {
        assert_eq!(result_to_string(serde_json::json!({"n": 1})), r#"{"n":1}"#);
        assert_eq!(result_to_string(serde_json::json!("plain")), "plain");
        assert_eq!(
            result_to_string(Value::Null),
            "",
            "null canonicalizes to empty"
        );
    }
}
