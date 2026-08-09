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

use super::decision::{Action, ToolHandler, Trigger};
use super::reconcile::news_start;
use super::state::{new_call_id, new_message_id, EffectState};
use super::tool_contract::{classify_arguments, declared_tool, DeclaredTool};
use crate::protocol::{
    AgentConfig, DecisionAction, DecisionRequest, DecisionResponse, DecisionTrigger, DraftMessage,
    ErrorInfo, Handler, InterruptResumption, LlmFormat, LlmRequest, LlmResponse, Message,
    MessageTree, RetryPolicy, WorkerIdentity, WorkerState,
};
use crate::runtime::llm::LlmBlocks;
use crate::runtime::retry::RetryTarget;
use crate::runtime::worker::WorkerDecisionRequest;

impl From<Message> for DraftMessage {
    fn from(m: Message) -> Self {
        DraftMessage {
            id: Some(m.id),
            role: m.role,
            content: m.content,
            tool_calls: (!m.tool_calls.is_empty()).then_some(m.tool_calls),
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
            tool_calls: self.tool_calls.unwrap_or_default(),
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
            tool_calls: self.tool_calls.unwrap_or_default(),
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
            identity: WorkerIdentity {
                id: r.identity.id.clone(),
                metadata: r.identity.metadata.clone(),
            },
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
    /// An `llm.call` named no `[llm.*]` block and no agent config supplied one.
    MissingLlm { declared: String },
    /// An `llm.call` named a block this app does not declare.
    UnknownLlm { name: String, declared: String },
    /// A handler value the addressed surface does not support: `server` on a
    /// declared tool.
    InvalidHandler {
        surface: &'static str,
        handler: &'static str,
    },
    /// An `llm.result` response that doesn't parse in the call's format.
    InvalidLlmResponse { message: String },
}

impl ResolveError {
    /// The one field to go and fix, in the sense of Stripe's `param`.
    pub fn param(&self) -> Option<String> {
        Some(match self {
            ResolveError::UnresolvableSettleId { kind } => format!("{kind}.result.id"),
            ResolveError::MissingModel => "model".to_string(),
            ResolveError::MissingLlm { .. } | ResolveError::UnknownLlm { .. } => "llm".to_string(),
            ResolveError::InvalidHandler { .. } => "handler".to_string(),
            ResolveError::InvalidLlmResponse { .. } => "response".to_string(),
        })
    }

    /// The machine-readable form, for the error event's `detail`. A consumer
    /// that wants to say "you named `claud`, did you mean `claude`?" needs the
    /// names, not the sentence they were rendered into.
    pub fn detail(&self) -> Value {
        match self {
            ResolveError::UnresolvableSettleId { kind } => {
                serde_json::json!({ "reason": "unresolvable_settle_id", "kind": kind })
            }
            ResolveError::MissingModel => serde_json::json!({ "reason": "missing_model" }),
            ResolveError::MissingLlm { declared } => {
                serde_json::json!({ "reason": "missing_llm", "declared": declared })
            }
            ResolveError::UnknownLlm { name, declared } => {
                serde_json::json!({ "reason": "unknown_llm", "name": name, "declared": declared })
            }
            ResolveError::InvalidHandler { surface, handler } => serde_json::json!({
                "reason": "invalid_handler", "surface": surface, "handler": handler,
            }),
            ResolveError::InvalidLlmResponse { message } => {
                serde_json::json!({ "reason": "invalid_llm_response", "message": message })
            }
        }
    }
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
            ResolveError::MissingLlm { declared } => write!(
                f,
                "agent config names no llm — declared: {declared}"
            ),
            ResolveError::UnknownLlm { name, declared } => write!(
                f,
                "no llm block `{name}` is declared — declared: {declared}"
            ),
            ResolveError::InvalidHandler { surface, handler } => {
                write!(f, "`{handler}` is not a valid handler for {surface}")
            }
            ResolveError::InvalidLlmResponse { message } => {
                write!(f, "llm.result response does not parse: {message}")
            }
        }
    }
}

impl std::error::Error for ResolveError {}

/// A tool a worker declares runs on the worker or the client, never on the
/// engine. Engine-executed tools come from a connector, which the worker names
/// by id in `mcp`, so `server` here is always a mistake.
fn declared_tool_handler(h: Handler) -> Result<ToolHandler, ResolveError> {
    match h {
        Handler::Server => Err(ResolveError::InvalidHandler {
            surface: "a declared tool",
            handler: h.as_str(),
        }),
        _ => h
            .try_into()
            .map_err(|h: Handler| ResolveError::InvalidHandler {
                surface: "a declared tool",
                handler: h.as_str(),
            }),
    }
}

/// Resolve a settle's `id` and `attempt` against the `*.execute` trigger it
/// answers. An omitted `id` is filled from the trigger (an error out-of-band,
/// where there is none). An omitted `attempt` is fenced to the trigger's attempt
/// on the answer path, so a settle targets the exact attempt it ran — a stale
/// executor whose attempt has since been superseded no longer settles the wrong
/// one. Out-of-band (no trigger), an omitted attempt stays `None`: settle current.
fn resolve_settle(
    id: Option<String>,
    attempt: Option<u32>,
    trigger: Option<&DecisionTrigger>,
    want: SettleKind,
) -> Result<(String, Option<u32>), ResolveError> {
    let answered = match (want, trigger) {
        (SettleKind::Tool, Some(DecisionTrigger::ToolExecute { id, attempt, .. })) => {
            Some((id, *attempt))
        }
        (SettleKind::Llm, Some(DecisionTrigger::LlmExecute { id, attempt, .. })) => {
            Some((id, *attempt))
        }
        _ => None,
    };
    let id = match id {
        Some(id) => id,
        None => answered
            .map(|(id, _)| id.clone())
            .ok_or(ResolveError::UnresolvableSettleId {
                kind: want.as_str(),
            })?,
    };
    Ok((id, attempt.or(answered.map(|(_, attempt)| attempt))))
}

/// An `llm.result` response value → the neutral `LlmResponse`: parsed with the
/// answered call's provider format when one was declared, else deserialized as-is.
fn parse_llm_response(
    response: Value,
    format: Option<LlmFormat>,
) -> Result<LlmResponse, ResolveError> {
    match format {
        Some(f) => f.response_from_wire(response),
        // Path-aware: `llm.result` bodies are the deepest thing a worker
        // authors, so "response does not parse" without a field name is the
        // least actionable error the engine can give.
        None => crate::json::from_value("llm.result response", response).map_err(|e| e.to_string()),
    }
    .map_err(|message| ResolveError::InvalidLlmResponse { message })
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
    pub channels: std::collections::BTreeMap<String, Value>,
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
///
/// `blocks` is where a call's `llm` name becomes a venue and a wire shape: this
/// is the one seam that reads the declared `[llm.*]`, so nothing downstream has
/// to re-derive where a call runs.
pub fn resolve_response(
    response: DecisionResponse,
    echoed_config: Option<&AgentConfig>,
    trigger: Option<&DecisionTrigger>,
    blocks: &LlmBlocks,
) -> Result<ResolvedResponse, ResolveError> {
    let DecisionResponse {
        messages,
        actions,
        state,
        agent,
        channels,
    } = response;
    if let Some(cfg) = &agent {
        for t in &cfg.tools {
            if let Some(h) = t.handler {
                declared_tool_handler(h)?;
            }
        }
    }
    let merge_cfg = agent.as_ref().or(echoed_config);
    let resolved = lower_actions(actions, &messages, merge_cfg, trigger, blocks)?;
    Ok(ResolvedResponse {
        messages,
        actions: resolved,
        state,
        agent,
        channels,
    })
}

/// Lower each wire action to its internal [`Action`]. `view` is the response's
/// declared transcript, used to fill an omitted-`messages` `llm.call`.
fn lower_actions(
    actions: Vec<DecisionAction>,
    view: &[DraftMessage],
    config: Option<&AgentConfig>,
    trigger: Option<&DecisionTrigger>,
    blocks: &LlmBlocks,
) -> Result<Vec<Action>, ResolveError> {
    let config_retry = config.and_then(|c| c.retry.as_deref());
    actions
        .into_iter()
        .map(|action| {
            Ok(match action {
                DecisionAction::CallLlm {
                    id,
                    llm,
                    model,
                    messages,
                    tools,
                    temperature,
                    max_completion_tokens,
                    reasoning,
                    stream,
                    retry,
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
                    let stream = stream.unwrap_or(true);
                    let retry =
                        RetryPolicy::resolve(retry.as_ref(), config_retry, RetryTarget::Llm);
                    // The block settles the venue and, for a worker-run call,
                    // the wire shape; a server call is always neutral.
                    let llm = llm
                        .or_else(|| config.and_then(|c| c.llm.clone()))
                        .ok_or_else(|| ResolveError::MissingLlm {
                            declared: blocks.declared(),
                        })?;
                    let block = blocks.get(&llm).ok_or_else(|| ResolveError::UnknownLlm {
                        name: llm.clone(),
                        declared: blocks.declared(),
                    })?;
                    Action::CallLlm {
                        id: id.unwrap_or_else(new_call_id),
                        llm,
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
                        handler: block.handler,
                        format: block.format,
                    }
                }
                DecisionAction::CallTool {
                    id,
                    name,
                    arguments,
                    retry,
                } => Action::CallTool {
                    id: id.unwrap_or_else(new_call_id),
                    name,
                    arguments: result_to_string(arguments),
                    retry,
                },
                DecisionAction::ToolResult {
                    id,
                    attempt,
                    result,
                } => {
                    let (id, attempt) = resolve_settle(id, attempt, trigger, SettleKind::Tool)?;
                    Action::ToolResult {
                        id,
                        attempt,
                        result: result_to_string(result),
                    }
                }
                DecisionAction::LlmResult {
                    id,
                    attempt,
                    response,
                } => {
                    let (id, attempt) = resolve_settle(id, attempt, trigger, SettleKind::Llm)?;
                    let format = match trigger {
                        Some(DecisionTrigger::LlmExecute {
                            id: answered,
                            format,
                            ..
                        }) if *answered == id => *format,
                        _ => None,
                    };
                    Action::LlmResult {
                        id,
                        attempt,
                        response: parse_llm_response(response, format)?,
                    }
                }
                DecisionAction::ToolError {
                    id,
                    attempt,
                    error,
                    retryable,
                    code,
                    detail,
                } => {
                    let (id, attempt) = resolve_settle(id, attempt, trigger, SettleKind::Tool)?;
                    Action::ToolError {
                        id,
                        attempt,
                        error: ErrorInfo::handler(error).or_code(code).or_detail(detail),
                        retryable,
                    }
                }
                DecisionAction::LlmError {
                    id,
                    attempt,
                    error,
                    retryable,
                    code,
                    detail,
                } => {
                    let (id, attempt) = resolve_settle(id, attempt, trigger, SettleKind::Llm)?;
                    Action::LlmError {
                        id,
                        attempt,
                        error: ErrorInfo::handler(error).or_code(code).or_detail(detail),
                        retryable,
                    }
                }
                DecisionAction::SpawnSubAgent {
                    session_id,
                    agent_id,
                    tool_call_id,
                    message,
                    retry,
                } => Action::SpawnSubAgent {
                    session_id,
                    agent_id,
                    tool_call_id,
                    message,
                    retry: RetryPolicy::resolve(
                        retry.as_ref(),
                        config_retry,
                        RetryTarget::SubAgent,
                    ),
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
                DecisionAction::ResolveInterrupt {
                    interrupt_id,
                    payload,
                } => Action::ResolveInterrupt {
                    interrupt_id,
                    payload,
                },
                DecisionAction::SyncConnector { id } => Action::SyncConnector { id },
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
    open_llm_calls: &HashMap<String, EffectState>,
) -> DecisionTrigger {
    match trigger {
        Trigger::SessionStart => DecisionTrigger::SessionStart,
        // The worker wire has no notion of a deferred turn: by delivery the turn
        // is running, so `turn_id` is dropped here.
        Trigger::ClientMessage {
            messages, client, ..
        } => {
            let known: std::collections::HashSet<&str> =
                tree.nodes.iter().map(|n| n.message.id.as_str()).collect();
            let mut view: Vec<DraftMessage> = active_path
                .iter()
                .cloned()
                .map(DraftMessage::from)
                .collect();
            let new_from = view.len();
            // Already-recorded ids are dropped: an append is "ensure these
            // are at the head", and a duplicate would land the news on its
            // old node instead of the head.
            view.extend(
                messages
                    .into_iter()
                    .filter(|m| m.id.as_deref().is_none_or(|id| !known.contains(id))),
            );
            DecisionTrigger::ClientTranscript {
                messages: view,
                new_from,
                client,
            }
        }
        Trigger::ClientTranscript {
            messages, client, ..
        } => {
            let known: std::collections::HashSet<&str> =
                tree.nodes.iter().map(|n| n.message.id.as_str()).collect();
            let new_from = news_start(&known, &messages);
            DecisionTrigger::ClientTranscript {
                messages,
                new_from,
                client,
            }
        }
        Trigger::ClientAction { name, args } => DecisionTrigger::ClientAction { name, args },
        Trigger::ToolExecute {
            id,
            name,
            arguments,
            attempt,
            deadline,
        } => {
            let schema = match declared_tool(&id, &name, active_path, |id| {
                open_llm_calls.get(id).and_then(|e| e.llm())
            }) {
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
            format,
            stream,
            attempt,
            deadline,
        } => DecisionTrigger::LlmExecute {
            id,
            request: match format {
                Some(f) => f.request_to_wire(&request),
                None => serde_json::to_value(&request).unwrap_or_default(),
            },
            format,
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
        } => DecisionTrigger::LlmFinished {
            id,
            ok,
            message,
            truncated,
            usage,
            cost,
            error,
        },
        Trigger::InterruptResumed {
            interrupt_id,
            payload,
        } => DecisionTrigger::InterruptResumed {
            resumption: InterruptResumption {
                interrupt_id,
                payload,
            },
        },
        Trigger::TurnFinished {
            turn_id,
            data,
            cost,
            usage,
        } => DecisionTrigger::TurnFinished {
            turn_id,
            data,
            cost,
            usage,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{
        AgentTool, ClientContext, ClientInput, ClientPayload, Content, NewMessage, Role, ToolCall,
        ToolCallFunction, ToolInput,
    };
    use crate::runtime::llm::LlmBlock;
    use crate::runtime::session::decision::LlmHandler;
    use crate::runtime::session::state::LlmCallSpec;

    /// The two blocks these tests resolve against: one the engine runs, one the
    /// worker runs in Anthropic's own wire shape.
    fn blocks() -> LlmBlocks {
        LlmBlocks::from_iter([
            ("claude".to_string(), LlmBlock::engine()),
            (
                "byo".to_string(),
                LlmBlock::worker(Some(LlmFormat::Anthropic)),
            ),
        ])
    }

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
                channels: Default::default(),
            },
            None,
            trigger,
            &blocks(),
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
                retry: None,
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
    fn a_call_naming_no_block_is_rejected_at_the_seam() {
        let mut call = bare_llm_call();
        if let DecisionAction::CallLlm { model, .. } = &mut call {
            *model = Some("m".to_string());
        }
        let err = resolve_test_actions(vec![call], None).unwrap_err();
        assert_eq!(
            err,
            ResolveError::MissingLlm {
                declared: "byo, claude".to_string()
            },
            "the error says what could have been named"
        );
    }

    #[test]
    fn a_call_naming_an_undeclared_block_is_rejected_at_the_seam() {
        let mut call = bare_llm_call();
        if let DecisionAction::CallLlm { llm, model, .. } = &mut call {
            *llm = Some("clade".to_string());
            *model = Some("m".to_string());
        }
        let err = resolve_test_actions(vec![call], None).unwrap_err();
        assert_eq!(
            err,
            ResolveError::UnknownLlm {
                name: "clade".to_string(),
                declared: "byo, claude".to_string()
            }
        );
    }

    #[test]
    fn a_declared_server_tool_is_rejected_at_the_seam() {
        let mut config = cfg("m1", None);
        config.tools.push(AgentTool {
            name: "t".to_string(),
            description: String::new(),
            input: None,
            output: None,
            handler: Some(Handler::Server),
        });
        let err = resolve_response(
            DecisionResponse {
                messages: vec![],
                actions: vec![],
                state: None,
                agent: Some(config),
                channels: Default::default(),
            },
            None,
            None,
            &blocks(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            ResolveError::InvalidHandler {
                surface: "a declared tool",
                handler: "server"
            },
            "engine-executed tools come from a connector, never from a declaration"
        );
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
    fn omitted_settle_attempt_is_fenced_to_the_answered_execute() {
        let trigger = DecisionTrigger::ToolExecute {
            id: "eff-1".to_string(),
            name: "do_thing".to_string(),
            arguments: "{}".to_string(),
            input: ToolInput::Valid {
                value: serde_json::json!({}),
            },
            attempt: 2,
            deadline: None,
        };
        let resolved = |attempt| {
            resolve_test_actions(
                vec![DecisionAction::ToolResult {
                    id: None,
                    attempt,
                    result: serde_json::json!("ok"),
                }],
                Some(&trigger),
            )
            .expect("resolves")
        };
        // Omitted ⇒ fenced to the trigger's attempt, not left unfenced.
        match &resolved(None)[0] {
            Action::ToolResult { attempt, .. } => assert_eq!(*attempt, Some(2)),
            other => panic!("unexpected: {other:?}"),
        }
        // Explicit ⇒ preserved verbatim (fence a specific attempt).
        match &resolved(Some(0))[0] {
            Action::ToolResult { attempt, .. } => assert_eq!(*attempt, Some(0)),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn out_of_band_omitted_attempt_stays_current() {
        // No answering trigger: an omitted attempt settles the current one (None).
        let actions = resolve_test_actions(
            vec![DecisionAction::ToolResult {
                id: Some("eff-1".to_string()),
                attempt: None,
                result: serde_json::json!("ok"),
            }],
            None,
        )
        .expect("resolves");
        match &actions[0] {
            Action::ToolResult { attempt, .. } => assert_eq!(*attempt, None),
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

    #[test]
    fn the_block_settles_the_venue_and_the_wire_shape() {
        // A worker-run block carries its format onto the call; an engine-run
        // one is always neutral. Neither is stated on the call itself.
        let byo = cfg_on("byo", "m1", None);
        match resolve_one_call(bare_llm_call(), vec![user_wire("hi")], Some(&byo), None).unwrap() {
            Action::CallLlm {
                handler, format, ..
            } => {
                assert_eq!(handler, LlmHandler::Worker);
                assert_eq!(format, Some(LlmFormat::Anthropic));
            }
            other => panic!("expected llm.call; got {other:?}"),
        }

        let claude = cfg("m1", None);
        match resolve_one_call(bare_llm_call(), vec![user_wire("hi")], Some(&claude), None).unwrap()
        {
            Action::CallLlm {
                handler, format, ..
            } => {
                assert_eq!(handler, LlmHandler::Server);
                assert_eq!(format, None);
            }
            other => panic!("expected llm.call; got {other:?}"),
        }
    }

    #[test]
    fn a_call_may_name_a_different_block_than_the_config() {
        // Mixing venues per call: the call's own `llm` wins over the config's.
        let claude = cfg("m1", None);
        let mut call = bare_llm_call();
        if let DecisionAction::CallLlm { llm, .. } = &mut call {
            *llm = Some("byo".to_string());
        }
        match resolve_one_call(call, vec![user_wire("hi")], Some(&claude), None).unwrap() {
            Action::CallLlm { llm, handler, .. } => {
                assert_eq!(llm, "byo");
                assert_eq!(handler, LlmHandler::Worker);
            }
            other => panic!("expected llm.call; got {other:?}"),
        }
    }

    #[test]
    fn a_format_execute_carries_the_provider_native_request() {
        let request = LlmRequest {
            model: "claude-haiku-4-5".to_string(),
            messages: vec![
                DraftMessage {
                    id: None,
                    role: Role::System,
                    content: Some(Content::Text("be nice".to_string())),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
                user_wire("hi"),
            ],
            tools: None,
            temperature: None,
            max_completion_tokens: None,
            reasoning: None,
        };
        let wire = to_wire_trigger(
            Trigger::LlmExecute {
                id: "llm-1".to_string(),
                request: request.clone(),
                format: Some(LlmFormat::Anthropic),
                stream: true,
                attempt: 0,
                deadline: None,
            },
            &[],
            &MessageTree::default(),
            &HashMap::new(),
        );
        match wire {
            DecisionTrigger::LlmExecute {
                request, format, ..
            } => {
                assert_eq!(format, Some(LlmFormat::Anthropic));
                assert_eq!(request["system"][0]["text"], "be nice");
                assert_eq!(request["messages"][0]["content"][0]["text"], "hi");
                assert!(
                    request.get("stream").is_none(),
                    "the trigger's stream flag is authoritative"
                );
            }
            t => panic!("expected llm.execute; got {t:?}"),
        }

        // No format ⇒ the neutral LlmRequest JSON, verbatim.
        let wire = to_wire_trigger(
            Trigger::LlmExecute {
                format: None,
                id: "llm-1".to_string(),
                request: request.clone(),
                stream: true,
                attempt: 0,
                deadline: None,
            },
            &[],
            &MessageTree::default(),
            &HashMap::new(),
        );
        match wire {
            DecisionTrigger::LlmExecute {
                request: wired,
                format,
                ..
            } => {
                assert_eq!(format, None);
                assert_eq!(wired, serde_json::to_value(&request).unwrap());
            }
            t => panic!("expected llm.execute; got {t:?}"),
        }
    }

    fn format_execute_trigger(format: Option<LlmFormat>) -> DecisionTrigger {
        DecisionTrigger::LlmExecute {
            id: "llm-1".to_string(),
            request: serde_json::json!({}),
            format,
            stream: false,
            attempt: 0,
            deadline: None,
        }
    }

    #[test]
    fn a_raw_response_answering_a_format_execute_is_translated() {
        let trigger = format_execute_trigger(Some(LlmFormat::Anthropic));
        let actions = resolve_test_actions(
            vec![DecisionAction::LlmResult {
                id: None,
                attempt: None,
                response: serde_json::json!({
                    "model": "claude-haiku-4-5",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "NYC"}}
                    ],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 1, "output_tokens": 2}
                }),
            }],
            Some(&trigger),
        )
        .expect("resolves");
        match &actions[0] {
            Action::LlmResult { id, response, .. } => {
                assert_eq!(id, "llm-1");
                assert_eq!(response.content.as_deref(), Some("hello"));
                assert_eq!(response.tool_calls[0].function.name, "get_weather");
                assert_eq!(response.finish_reason.as_deref(), Some("tool_calls"));
            }
            other => panic!("expected llm.result; got {other:?}"),
        }
    }

    #[test]
    fn an_unparseable_llm_result_is_a_resolve_error() {
        for format in [Some(LlmFormat::Anthropic), None] {
            let trigger = format_execute_trigger(format);
            let err = resolve_test_actions(
                vec![DecisionAction::LlmResult {
                    id: None,
                    attempt: None,
                    response: serde_json::json!(42),
                }],
                Some(&trigger),
            )
            .unwrap_err();
            assert!(
                matches!(err, ResolveError::InvalidLlmResponse { .. }),
                "got {err:?}"
            );
        }
    }

    fn cfg(model: &str, system: Option<&str>) -> AgentConfig {
        cfg_on("claude", model, system)
    }

    fn cfg_on(llm: &str, model: &str, system: Option<&str>) -> AgentConfig {
        AgentConfig {
            llm: Some(llm.to_string()),
            model: model.to_string(),
            system: system.map(str::to_string),
            retry: None,
            tools: Vec::new(),
            sub_agents: Vec::new(),
            mcp: Vec::new(),
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
            llm: None,
            model: None,
            messages: None,
            tools: None,
            temperature: None,
            max_completion_tokens: None,
            reasoning: None,
            stream: None,
            retry: None,
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
                channels: Default::default(),
            },
            echoed,
            None,
            &blocks(),
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
            llm: None,
            model: None,
            messages: Some(vec![user_wire("only me")]),
            tools: None,
            temperature: None,
            max_completion_tokens: None,
            reasoning: None,
            stream: None,
            retry: None,
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
            llm: None,
            model: Some("override".to_string()),
            messages: None,
            tools: None,
            temperature: None,
            max_completion_tokens: None,
            reasoning: None,
            stream: None,
            retry: None,
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
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
        }
    }

    fn linear_tree(messages: &[Message]) -> MessageTree {
        let nodes: Vec<NewMessage> = messages
            .iter()
            .enumerate()
            .map(|(i, m)| NewMessage {
                message: m.clone(),
                parent_id: (i > 0).then(|| messages[i - 1].id.clone()),
            })
            .collect();
        MessageTree {
            head_id: messages.last().map(|m| m.id.clone()),
            nodes,
        }
    }

    fn transcript_of(trigger: DecisionTrigger) -> (Vec<DraftMessage>, usize) {
        match trigger {
            DecisionTrigger::ClientTranscript {
                messages, new_from, ..
            } => (messages, new_from),
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
                messages: vec![msg("u2", Role::User, "more").into()],
                client: ClientContext::default(),
                turn_id: None,
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
    fn materializes_an_append_batch_dropping_recorded_ids() {
        let path = vec![
            msg("u1", Role::User, "hi"),
            msg("a1", Role::Assistant, "yo"),
        ];
        let tree = linear_tree(&path);

        // "a1" is already recorded: dropped, so the news lands on the head
        // instead of folding back onto its old node.
        let (messages, new_from) = transcript_of(to_wire_trigger(
            Trigger::ClientMessage {
                messages: vec![
                    msg("a1", Role::Assistant, "yo").into(),
                    msg("u2", Role::User, "more").into(),
                    msg("u3", Role::User, "and more").into(),
                ],
                client: ClientContext::default(),
                turn_id: None,
            },
            &path,
            &tree,
            &HashMap::new(),
        ));

        assert_eq!(
            messages.iter().map(|m| m.id.as_deref()).collect::<Vec<_>>(),
            vec![Some("u1"), Some("a1"), Some("u2"), Some("u3")]
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
                client: ClientContext::default(),
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
                client: ClientContext::default(),
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
                client: ClientContext::default(),
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
                client: ClientContext::default(),
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
        open_llm_calls: &HashMap<String, EffectState>,
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

    fn weather_call(schema: serde_json::Value) -> (Vec<Message>, HashMap<String, EffectState>) {
        use crate::protocol::LlmTool;
        use crate::runtime::session::state::{EffectPayload, EffectTracking, LlmCallState};

        let assistant = Message {
            id: "call-1".to_string(),
            role: Role::Assistant,
            content: None,
            tool_calls: vec![ToolCall {
                id: "tc-1".to_string(),
                call_type: "function".to_string(),
                function: ToolCallFunction {
                    name: "get_weather".to_string(),
                    arguments: "{}".to_string(),
                },
            }],
            tool_call_id: None,
            name: None,
        };
        let call = EffectState::new(
            "call-1",
            EffectTracking::new(RetryPolicy::no_retry(), chrono::Utc::now()),
            EffectPayload::LlmCall(LlmCallState {
                format: None,
                llm: "claude".to_string(),
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
                handler: crate::runtime::session::decision::LlmHandler::Server,
            }),
        );
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
            DecisionAction::CallLlm { llm: None, .. }
        ));
        match &actions[1] {
            DecisionAction::CallTool { arguments, .. } => {
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

    type VariantCheck = fn(&ClientInput) -> bool;

    #[test]
    fn client_input_parses_every_tag_to_its_variant() {
        let cases: [(&str, VariantCheck); 7] = [
            (
                r#"{"type":"client.message","agent_id":"bot","message":{"role":"user","content":"hi"}}"#,
                |i| matches!(i, ClientInput::Message { .. }),
            ),
            (
                r#"{"type":"client.messages","agent_id":"bot","messages":[]}"#,
                |i| matches!(i, ClientInput::Messages { .. }),
            ),
            (
                r#"{"type":"client.append","agent_id":"bot","messages":[]}"#,
                |i| matches!(i, ClientInput::Append { .. }),
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
    fn unknown_client_input_tag_lists_all_seven() {
        let err = serde_json::from_str::<ClientInput>(r#"{"type":"frob"}"#)
            .unwrap_err()
            .to_string();
        for tag in [
            "client.message",
            "client.messages",
            "client.append",
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
            ClientInput::InterruptResume {
                resumption: InterruptResumption { interrupt_id, .. },
            } => {
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
