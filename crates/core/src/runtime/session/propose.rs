//! Proposed decisions: the engine's derivation of a trigger's obvious continuation.
//!
//! For triggers whose next decision is mechanical, the engine derives the
//! decision an SDK agent loop would author and sends it on the decision request
//! as `proposed`. A worker accepts by echoing it back, amended or verbatim, as
//! its decision. `llm.finished` records the assistant message and dispatches its
//! tool calls or finishes; `tool.finished` and `sub_agent.finished` record the
//! result as a tool message, then wait for in-flight siblings or re-issue the
//! parent LLM request; a terminal `llm.finished` failure (or truncation)
//! interrupts the session — pausing is recoverable in both directions, unlike
//! `done`; a `tool.execute` that fails its contract — an undeclared name, or
//! arguments the declared `input` schema rejects — answers with a
//! `tool.error`, so the failure flows back to a model that can repair the call.
//!
//! A proposal is advice, not authority: the engine never applies one, so the
//! worker remains the sole author of every decision. Triggers that need worker
//! knowledge (`client.messages` — the LLM request is the agent's identity;
//! `tool.execute` for a declared tool with valid arguments — the computation
//! itself) carry no proposal, so a worker that echoes blindly fails fast
//! instead of stalling the session.
//!
//! Like [`to_wire_trigger`](super::wire::to_wire_trigger), derivation is a pure
//! function of state frozen while the decision is pending, so redeliveries carry
//! the same proposal.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::events::ToolHandler;
use super::message::{Content, Message, Role};
use super::state::LlmCallState;
use super::tool_contract::{declared_tool, DeclaredTool};
use super::wire::{WireAction, WireMessage, WireTrigger};
use crate::runtime::llm::ErrorCode;
use crate::runtime::retry::RetryPolicy;

/// The decision an SDK agent loop would author for a trigger. `messages` is the
/// full conversation view with the trigger's implied message appended, so echoing
/// it reconciles as a plain append.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proposal {
    pub messages: Vec<WireMessage>,
    pub actions: Vec<WireAction>,
}

pub fn propose(
    trigger: &WireTrigger,
    transcript: &[Message],
    llm_calls: &HashMap<String, LlmCallState>,
    pending_calls: usize,
) -> Option<Proposal> {
    match trigger {
        WireTrigger::LlmFinished {
            ok: true,
            message: Some(message),
            truncated: false,
            ..
        } => Some(llm_finished(message, transcript)),
        WireTrigger::LlmFinished {
            id,
            ok,
            truncated,
            error,
            code,
            ..
        } if !*ok || *truncated => Some(llm_failed(
            id,
            error.as_deref(),
            code,
            *truncated,
            transcript,
        )),
        WireTrigger::ToolFinished {
            id,
            ok,
            name,
            result,
            error,
        } => tool_finished(
            id,
            name,
            if *ok { result } else { error }
                .as_deref()
                .unwrap_or_default(),
            transcript,
            llm_calls,
            pending_calls,
        ),
        WireTrigger::SubAgentFinished {
            id,
            ok,
            agent_id,
            result,
            error,
            ..
        } => tool_finished(
            id,
            agent_id,
            if *ok { result } else { error }
                .as_deref()
                .unwrap_or_default(),
            transcript,
            llm_calls,
            pending_calls,
        ),
        WireTrigger::ToolExecute {
            id, name, input, ..
        } => match declared_tool(id, name, transcript, llm_calls) {
            DeclaredTool::Undeclared => {
                Some(tool_error(id, format!("unknown tool: {name}"), transcript))
            }
            _ => input.error().map(|error| {
                tool_error(id, format!("invalid tool arguments: {error}"), transcript)
            }),
        },
        _ => None,
    }
}

/// Record the assistant message, then dispatch its tool calls — or, when the
/// model stopped calling tools, end the turn with the message content.
fn llm_finished(message: &WireMessage, transcript: &[Message]) -> Proposal {
    let tool_calls = message.tool_calls.as_deref().unwrap_or_default();
    let actions = if tool_calls.is_empty() {
        vec![WireAction::Done {
            data: serde_json::to_value(&message.content).unwrap_or_default(),
        }]
    } else {
        tool_calls
            .iter()
            .map(|call| WireAction::CallTool {
                id: Some(call.id.clone()),
                name: call.function.name.clone(),
                arguments: call.function.arguments.clone(),
                handler: ToolHandler::Worker,
                retry: RetryPolicy::no_retry(),
            })
            .collect()
    };
    Proposal {
        messages: appended(transcript, message.clone()),
        actions,
    }
}

/// Pause the session: a terminal model failure is the loop's own engine dying,
/// not news the model can react to, and `done` would unilaterally end a turn
/// that may be salvageable. Nothing is recorded — a truncation's partial
/// message lives durably on the finished event.
fn llm_failed(
    id: &str,
    error: Option<&str>,
    code: &Option<ErrorCode>,
    truncated: bool,
    transcript: &[Message],
) -> Proposal {
    let reason = match error {
        Some(error) => format!("llm call failed: {error}"),
        None if truncated => "llm call truncated".to_string(),
        None => "llm call failed".to_string(),
    };
    Proposal {
        messages: recorded(transcript),
        actions: vec![WireAction::Interrupt {
            interrupt_id: None,
            reason,
            payload: serde_json::json!({
                "type": "llm.failed",
                "id": id,
                "error": error,
                "code": code,
                "truncated": truncated,
            }),
        }],
    }
}

/// Answer a `tool.execute` that failed its contract with a `tool.error`, so
/// the failure flows back to the model, which can repair the call. A worker
/// that treats `arguments` as an arbitrary string channel, or serves names it
/// never declared, ignores the proposal and answers itself.
fn tool_error(id: &str, error: String, transcript: &[Message]) -> Proposal {
    Proposal {
        messages: recorded(transcript),
        actions: vec![WireAction::ToolError {
            id: Some(id.to_string()),
            attempt: None,
            error,
            retryable: false,
            code: None,
            detail: None,
        }],
    }
}

/// Record the tool message (the error text when the call failed, so the model
/// sees it), then wait for in-flight siblings or re-issue the parent request.
/// No proposal when the parent call can't be resolved: a half proposal that
/// records but never continues would stall the turn on echo.
fn tool_finished(
    id: &str,
    name: &str,
    content: &str,
    transcript: &[Message],
    llm_calls: &HashMap<String, LlmCallState>,
    pending_calls: usize,
) -> Option<Proposal> {
    let tool_message = WireMessage {
        id: None,
        role: Role::Tool,
        content: Some(Content::Text(content.to_string())),
        tool_calls: None,
        tool_call_id: Some(id.to_string()),
        name: Some(name.to_string()),
    };
    let actions = if pending_calls > 0 {
        Vec::new()
    } else {
        vec![reissue(id, &tool_message, transcript, llm_calls)?]
    };
    Some(Proposal {
        messages: appended(transcript, tool_message),
        actions,
    })
}

/// The parent llm.call re-issued: its verbatim prompt (preserving any prompt
/// shaping the worker did — system message, compaction) extended with the
/// recorded path from the assistant message on, plus this tool result. The
/// parent is named by lineage: the tool call id appears in exactly one
/// assistant message, recorded under its llm.call's id.
fn reissue(
    tool_call_id: &str,
    tool_message: &WireMessage,
    transcript: &[Message],
    llm_calls: &HashMap<String, LlmCallState>,
) -> Option<WireAction> {
    let assistant_at = transcript.iter().rposition(|m| {
        m.tool_calls
            .as_deref()
            .unwrap_or_default()
            .iter()
            .any(|c| c.id == tool_call_id)
    })?;
    let call = llm_calls.get(&transcript[assistant_at].id)?;

    let mut messages: Vec<WireMessage> =
        call.prompt.iter().cloned().map(WireMessage::from).collect();
    messages.extend(
        transcript[assistant_at..]
            .iter()
            .cloned()
            .map(WireMessage::from),
    );
    messages.push(tool_message.clone());

    Some(WireAction::CallLlm {
        id: None,
        request: call.spec.to_wire_request(messages),
        stream: call.stream,
        retry: call.tracking.retry_policy.clone(),
        handler: call.handler.clone(),
    })
}

fn recorded(transcript: &[Message]) -> Vec<WireMessage> {
    transcript.iter().cloned().map(WireMessage::from).collect()
}

fn appended(transcript: &[Message], message: WireMessage) -> Vec<WireMessage> {
    let mut messages = recorded(transcript);
    messages.push(message);
    messages
}

#[cfg(test)]
mod tests {
    use chrono::Utc;

    use super::*;
    use crate::runtime::llm::LlmRequest;
    use crate::runtime::session::events::LlmHandler;
    use crate::runtime::session::message::{ToolCall, ToolCallFunction};
    use crate::runtime::session::state::{EffectTracking, LlmCallSpec};
    use crate::runtime::session::tool_contract::ToolInput;

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

    fn assistant_with_calls(id: &str, calls: &[(&str, &str)]) -> Message {
        Message {
            id: id.to_string(),
            role: Role::Assistant,
            content: None,
            tool_calls: Some(
                calls
                    .iter()
                    .map(|(call_id, name)| ToolCall {
                        id: call_id.to_string(),
                        call_type: "function".to_string(),
                        function: ToolCallFunction {
                            name: name.to_string(),
                            arguments: "{}".to_string(),
                        },
                    })
                    .collect(),
            ),
            tool_call_id: None,
            name: None,
        }
    }

    fn call_state(call_id: &str, prompt: Vec<Message>) -> LlmCallState {
        LlmCallState {
            call_id: call_id.to_string(),
            tracking: EffectTracking::new(RetryPolicy::no_retry(), Utc::now()),
            prompt,
            spec: LlmCallSpec {
                model: "test-model".to_string(),
                tools: None,
                temperature: Some(0.5),
                max_completion_tokens: None,
                reasoning: None,
            },
            stream: true,
            handler: LlmHandler::Server,
            anchor: None,
        }
    }

    fn llm_finished_trigger(message: WireMessage, ok: bool, truncated: bool) -> WireTrigger {
        WireTrigger::LlmFinished {
            id: "call-1".to_string(),
            ok,
            message: Some(message),
            truncated,
            usage: None,
            cost: None,
            error: None,
            code: None,
            detail: None,
        }
    }

    fn tool_finished_trigger(id: &str, name: &str, outcome: Result<&str, &str>) -> WireTrigger {
        WireTrigger::ToolFinished {
            id: id.to_string(),
            ok: outcome.is_ok(),
            name: name.to_string(),
            result: outcome.ok().map(str::to_string),
            error: outcome.err().map(str::to_string),
        }
    }

    fn reissued_request(proposal: &Proposal) -> (&LlmRequest, bool, &LlmHandler) {
        match &proposal.actions[..] {
            [WireAction::CallLlm {
                id: None,
                request,
                stream,
                handler,
                ..
            }] => (request, *stream, handler),
            other => panic!("expected a single llm.call with no id; got {other:?}"),
        }
    }

    #[test]
    fn llm_finished_with_tool_calls_proposes_dispatch() {
        let transcript = vec![msg("u1", Role::User, "hi")];
        let assistant = WireMessage::from(assistant_with_calls(
            "call-1",
            &[("tc-1", "get_time"), ("tc-2", "get_weather")],
        ));

        let p = propose(
            &llm_finished_trigger(assistant, true, false),
            &transcript,
            &HashMap::new(),
            0,
        )
        .expect("proposes");

        assert_eq!(
            p.messages
                .iter()
                .map(|m| m.id.as_deref())
                .collect::<Vec<_>>(),
            vec![Some("u1"), Some("call-1")],
            "assistant message appended under its call id"
        );
        let dispatched: Vec<_> = p
            .actions
            .iter()
            .map(|a| match a {
                WireAction::CallTool {
                    id: Some(id),
                    name,
                    handler: ToolHandler::Worker,
                    ..
                } => (id.as_str(), name.as_str()),
                other => panic!("expected tool.call with the model's id; got {other:?}"),
            })
            .collect();
        assert_eq!(
            dispatched,
            vec![("tc-1", "get_time"), ("tc-2", "get_weather")]
        );
    }

    #[test]
    fn llm_finished_without_tool_calls_proposes_done() {
        let transcript = vec![msg("u1", Role::User, "hi")];
        let assistant = WireMessage::from(msg("call-1", Role::Assistant, "hello"));

        let p = propose(
            &llm_finished_trigger(assistant, true, false),
            &transcript,
            &HashMap::new(),
            0,
        )
        .expect("proposes");

        match &p.actions[..] {
            [WireAction::Done { data }] => assert_eq!(data, &serde_json::json!("hello")),
            other => panic!("expected done with the message content; got {other:?}"),
        }
    }

    #[test]
    fn failed_or_truncated_llm_finished_proposes_interrupt() {
        let transcript = vec![msg("u1", Role::User, "hi")];
        let assistant = WireMessage::from(msg("call-1", Role::Assistant, "partial"));
        for (trigger, reason) in [
            (
                llm_finished_trigger(assistant.clone(), false, false),
                "llm call failed",
            ),
            (
                llm_finished_trigger(assistant, true, true),
                "llm call truncated",
            ),
        ] {
            let p = propose(&trigger, &transcript, &HashMap::new(), 0).expect("proposes");
            assert_eq!(p.messages.len(), 1, "nothing recorded, not even a partial");
            match &p.actions[..] {
                [WireAction::Interrupt {
                    reason: r, payload, ..
                }] => {
                    assert!(r.starts_with(reason), "got reason {r:?}");
                    assert_eq!(payload["type"], serde_json::json!("llm.failed"));
                    assert_eq!(payload["id"], serde_json::json!("call-1"));
                }
                other => panic!("expected an interrupt; got {other:?}"),
            }
        }
    }

    #[test]
    fn a_terminal_llm_error_carries_its_context_in_the_interrupt() {
        let trigger = WireTrigger::LlmFinished {
            id: "call-1".to_string(),
            ok: false,
            message: None,
            truncated: false,
            usage: None,
            cost: None,
            error: Some("rate limited".to_string()),
            code: Some(ErrorCode::RateLimited),
            detail: None,
        };
        let p = propose(&trigger, &[], &HashMap::new(), 0).expect("proposes");
        match &p.actions[..] {
            [WireAction::Interrupt {
                reason, payload, ..
            }] => {
                assert_eq!(reason, "llm call failed: rate limited");
                assert_eq!(payload["error"], serde_json::json!("rate limited"));
                assert_eq!(payload["code"], serde_json::json!("rate_limited"));
            }
            other => panic!("expected an interrupt; got {other:?}"),
        }
    }

    #[test]
    fn sub_agent_finished_folds_like_a_tool_and_reissues() {
        let transcript = vec![
            msg("u1", Role::User, "hi"),
            assistant_with_calls("call-1", &[("tc-1", "researcher")]),
        ];
        let llm_calls = HashMap::from([("call-1".to_string(), call_state("call-1", vec![]))]);
        let trigger = WireTrigger::SubAgentFinished {
            id: "tc-1".to_string(),
            ok: true,
            session_id: "child".to_string(),
            agent_id: "researcher".to_string(),
            result: Some("findings".to_string()),
            error: None,
        };

        let p = propose(&trigger, &transcript, &llm_calls, 0).expect("proposes");

        let folded = p.messages.last().expect("tool message appended");
        assert!(matches!(folded.role, Role::Tool));
        assert_eq!(folded.tool_call_id.as_deref(), Some("tc-1"));
        assert_eq!(folded.name.as_deref(), Some("researcher"));
        assert_eq!(
            folded.content.as_ref().and_then(Content::text),
            Some("findings")
        );
        let (request, _, _) = reissued_request(&p);
        assert_eq!(request.model, "test-model");
    }

    fn expect_tool_error(p: &Proposal) -> (&str, &str) {
        match &p.actions[..] {
            [WireAction::ToolError {
                id: Some(id),
                error,
                retryable: false,
                ..
            }] => (id, error),
            other => panic!("expected a terminal tool.error; got {other:?}"),
        }
    }

    #[test]
    fn unusable_tool_arguments_propose_the_tool_error() {
        for input in [
            ToolInput::Malformed {
                error: "expected value at line 1 column 1".to_string(),
            },
            ToolInput::Invalid {
                value: serde_json::json!({"city": 5}),
                error: "expected value at line 1 column 1".to_string(),
            },
        ] {
            let trigger = WireTrigger::ToolExecute {
                id: "tc-1".to_string(),
                name: "get_time".to_string(),
                arguments: "not json".to_string(),
                input,
                attempt: 0,
                deadline: None,
            };
            let p = propose(&trigger, &[], &HashMap::new(), 0).expect("proposes");
            let (id, error) = expect_tool_error(&p);
            assert_eq!(id, "tc-1");
            assert_eq!(
                error,
                "invalid tool arguments: expected value at line 1 column 1"
            );
        }
    }

    #[test]
    fn an_undeclared_tool_proposes_the_unknown_tool_error() {
        let transcript = vec![
            msg("u1", Role::User, "hi"),
            assistant_with_calls("call-1", &[("tc-1", "hallucinated")]),
        ];
        let mut call = call_state("call-1", vec![]);
        call.spec.tools = Some(vec![crate::runtime::llm::LlmTool {
            name: "get_time".to_string(),
            description: "d".to_string(),
            input: None,
            output: None,
        }]);
        let llm_calls = HashMap::from([("call-1".to_string(), call)]);
        let trigger = WireTrigger::ToolExecute {
            id: "tc-1".to_string(),
            name: "hallucinated".to_string(),
            arguments: "{}".to_string(),
            input: ToolInput::Valid {
                value: serde_json::json!({}),
            },
            attempt: 0,
            deadline: None,
        };

        let p = propose(&trigger, &transcript, &llm_calls, 0).expect("proposes");
        let (id, error) = expect_tool_error(&p);
        assert_eq!(id, "tc-1");
        assert_eq!(error, "unknown tool: hallucinated");
    }

    #[test]
    fn tool_finished_with_pending_siblings_waits() {
        let transcript = vec![
            msg("u1", Role::User, "hi"),
            assistant_with_calls("call-1", &[("tc-1", "get_time"), ("tc-2", "get_weather")]),
        ];

        let p = propose(
            &tool_finished_trigger("tc-1", "get_time", Ok("3pm")),
            &transcript,
            &HashMap::new(),
            1,
        )
        .expect("proposes");

        assert!(p.actions.is_empty(), "waits for the sibling call");
        let tool = p.messages.last().expect("tool message appended");
        assert!(matches!(tool.role, Role::Tool));
        assert_eq!(tool.tool_call_id.as_deref(), Some("tc-1"));
        assert_eq!(tool.name.as_deref(), Some("get_time"));
        assert_eq!(tool.content.as_ref().and_then(Content::text), Some("3pm"));
    }

    #[test]
    fn last_tool_finished_reissues_the_parent_call() {
        // The system message lives only in the stored prompt, not the transcript:
        // the re-issued request must preserve the worker's prompt shaping.
        let transcript = vec![
            msg("u1", Role::User, "hi"),
            assistant_with_calls("call-1", &[("tc-1", "get_time")]),
        ];
        let llm_calls = HashMap::from([(
            "call-1".to_string(),
            call_state(
                "call-1",
                vec![
                    msg("s1", Role::System, "be brief"),
                    msg("u1", Role::User, "hi"),
                ],
            ),
        )]);

        let p = propose(
            &tool_finished_trigger("tc-1", "get_time", Ok("3pm")),
            &transcript,
            &llm_calls,
            0,
        )
        .expect("proposes");

        let (request, stream, handler) = reissued_request(&p);
        assert_eq!(request.model, "test-model");
        assert_eq!(request.temperature, Some(0.5));
        assert!(stream);
        assert!(matches!(handler, LlmHandler::Server));
        let roles: Vec<_> = request.messages.iter().map(|m| &m.role).collect();
        assert!(
            matches!(
                roles[..],
                [Role::System, Role::User, Role::Assistant, Role::Tool]
            ),
            "prompt + path from the assistant + tool result; got {roles:?}"
        );
    }

    #[test]
    fn tool_error_feeds_the_error_to_the_model() {
        let transcript = vec![
            msg("u1", Role::User, "hi"),
            assistant_with_calls("call-1", &[("tc-1", "get_time")]),
        ];
        let llm_calls = HashMap::from([("call-1".to_string(), call_state("call-1", vec![]))]);

        let p = propose(
            &tool_finished_trigger("tc-1", "get_time", Err("clock offline")),
            &transcript,
            &llm_calls,
            0,
        )
        .expect("proposes");

        let (request, _, _) = reissued_request(&p);
        let tool = request.messages.last().expect("tool message in prompt");
        assert_eq!(
            tool.content.as_ref().and_then(Content::text),
            Some("clock offline")
        );
    }

    #[test]
    fn unresolvable_parent_proposes_nothing() {
        let transcript = vec![msg("u1", Role::User, "hi")];
        assert!(propose(
            &tool_finished_trigger("tc-1", "get_time", Ok("3pm")),
            &transcript,
            &HashMap::new(),
            0,
        )
        .is_none());
    }

    #[test]
    fn must_answer_triggers_carry_no_proposal() {
        let triggers = [
            WireTrigger::ClientTranscript {
                messages: vec![],
                new_from: 0,
            },
            WireTrigger::ToolExecute {
                id: "tc-1".to_string(),
                name: "get_time".to_string(),
                arguments: "{}".to_string(),
                input: ToolInput::Valid {
                    value: serde_json::json!({}),
                },
                attempt: 0,
                deadline: None,
            },
        ];
        for trigger in triggers {
            assert!(propose(&trigger, &[], &HashMap::new(), 0).is_none());
        }
    }
}
