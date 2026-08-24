//! What Slack proposes for a decision, when the worker does not say.
//!
//! Everything is derived from the turn's events, so a redelivery proposes the
//! same thing. A worker can amend or replace any of it.

use serde_json::Value;

use crate::transport::channel::ChannelKind;

use super::activity::TurnActivity;
use super::render;
use super::{clip, display_of, with_footer, ButtonValue};
use crate::protocol::{
    Content, DecisionAction, DecisionResponse, DecisionTrigger, InterruptResolution,
    InterruptResponder, Message, ResumeStatus, Role,
};
use crate::runtime::session::state::SessionState;
use crate::runtime::worker::{ChannelProposer, Proposal};
use crate::session::events::EventPayload;
use crate::session::SessionEvent;

#[derive(Default)]
pub struct SlackProposer;

impl SlackProposer {
    pub fn new() -> Self {
        Self
    }
}

impl ChannelProposer for SlackProposer {
    fn channel(&self) -> ChannelKind {
        ChannelKind::SLACK
    }

    fn translate(&self, cx: &Proposal<'_>) -> Option<DecisionResponse> {
        match cx.trigger {
            DecisionTrigger::ClientAction {
                args: Some(args), ..
            } => click_proposal(cx.state, args),
            _ => None,
        }
    }

    fn render(&self, cx: &Proposal<'_>, proposed: &DecisionResponse) -> Option<Value> {
        match cx.trigger {
            DecisionTrigger::LlmFinished { .. }
            | DecisionTrigger::ToolFinished { .. }
            | DecisionTrigger::SubAgentFinished { .. }
            | DecisionTrigger::InterruptResumed { .. } => {
                // A proposal that interrupts owns the message; no view.
                let authors_interrupt = proposed
                    .actions
                    .iter()
                    .any(|a| matches!(a, DecisionAction::Interrupt { .. }));
                match authors_interrupt {
                    true => None,
                    false => streaming_view(cx.events, &title_of(cx.transcript)).map(view),
                }
            }
            DecisionTrigger::TurnFinished { data, .. } => {
                Some(view(final_view(cx.events, data, &title_of(cx.transcript))))
            }
            _ => None,
        }
    }
}

fn view(view: Value) -> Value {
    serde_json::json!({ "view": view })
}

/// The card's title: what the turn was asked to do.
fn title_of(transcript: &[Message]) -> String {
    transcript
        .iter()
        .rev()
        .filter(|m| m.role == Role::User)
        .find_map(|m| {
            let text = m.content.as_ref().map(Content::text_owned)?;
            let text = plain(&text);
            let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
            (!text.is_empty()).then(|| clip(&text, 60))
        })
        .unwrap_or_else(|| "Working…".to_string())
}

/// The text without Slack markup. A finished card shows its title raw, so
/// a `<@id>` mention would read as the id, not the name.
fn plain(text: &str) -> String {
    // The batch writes "<@author>: text"; the title wants the text.
    let text = match text.strip_prefix("<@").and_then(|r| r.split_once(">: ")) {
        Some((_, rest)) => rest,
        None => text,
    };
    // Any other token reads as its label. Without one it is a bare id — a
    // mention, a channel, a group — which names nobody once the markup is
    // gone, so it leaves the title rather than reading as the id.
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find('<') {
        out.push_str(&rest[..start]);
        match rest[start + 1..].split_once('>') {
            Some((token, after)) => {
                let label = match token.split_once('|') {
                    Some((_, label)) => label,
                    None if token.starts_with(['@', '#', '!']) => "",
                    None => token,
                };
                out.push_str(label);
                rest = after;
            }
            None => {
                out.push('<');
                rest = &rest[start + 1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// The message so far: one card for the turn. Only the events write the
/// log — the stream appends, so a line that later reads differently would
/// freeze it. A proposed call lands when its event does.
fn streaming_view(events: &[SessionEvent], title: &str) -> Option<Value> {
    let turn = TurnActivity::fold(events)?;
    Some(serde_json::json!({ "text": "", "blocks": [turn.card(title)] }))
}

/// The final message: the card at rest, the answer, and how long it took.
fn final_view(events: &[SessionEvent], data: &Value, title: &str) -> Value {
    let mut blocks = Vec::new();
    if let Some(turn) = TurnActivity::fold(events) {
        blocks.push(turn.settled_card(title));
    }
    let answer = match data {
        Value::Null => "(no result)".to_string(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    blocks.push(render::section_block(&answer));
    let footer = elapsed(events);
    if let Some(footer) = &footer {
        blocks.push(render::context_block(footer));
    }
    serde_json::json!({
        "text": with_footer(&answer, footer.as_deref()),
        "blocks": blocks,
    })
}

/// From `turn.started` to the last event given.
fn elapsed(events: &[SessionEvent]) -> Option<String> {
    let started = events.iter().rev().find_map(|e| match &e.payload {
        EventPayload::TurnStarted(_) => Some(e.occurred_at),
        _ => None,
    })?;
    let last = events.last()?.occurred_at;
    Some(super::activity::elapsed(started, last))
}

/// A click, as the bot puts it in the action's `args`.
struct ClickArgs<'a> {
    value: &'a str,
    user: &'a str,
    channel: &'a str,
    message_ts: &'a str,
    message_text: &'a str,
    message_blocks: Vec<Value>,
}

fn click_args(args: &Value) -> Option<ClickArgs<'_>> {
    Some(ClickArgs {
        value: args["value"].as_str()?,
        user: args["user"].as_str()?,
        channel: args["channel"].as_str()?,
        message_ts: args["message_ts"].as_str()?,
        message_text: args["message_text"].as_str().unwrap_or_default(),
        message_blocks: args["message_blocks"]
            .as_array()
            .cloned()
            .unwrap_or_default(),
    })
}

/// What a click on one of the bot's own buttons means. A value that is not a
/// [`ButtonValue`] belongs to the worker, which answers it.
fn click_proposal(state: &SessionState, args: &Value) -> Option<DecisionResponse> {
    let click = click_args(args)?;
    let ButtonValue::InterruptOption {
        interrupt_id,
        option: option_idx,
    } = serde_json::from_str(click.value).ok()?;

    let Some(open) = state.open_interrupt(&interrupt_id) else {
        return Some(stale_prompt(&click));
    };
    let option = display_of(&open.payload).and_then(|d| d.options.into_iter().nth(option_idx));
    let Some(option) = option else {
        return Some(stale_prompt(&click));
    };
    let resolution = InterruptResolution {
        status: ResumeStatus::Resolved,
        payload: option.value,
        responder: Some(InterruptResponder {
            channel: ChannelKind::SLACK.to_string(),
            user: Some(click.user.to_string()),
            label: Some(option.label),
            style: option.style,
        }),
    };
    Some(DecisionResponse {
        actions: vec![DecisionAction::ResolveInterrupt {
            interrupt_id,
            payload: serde_json::to_value(resolution).unwrap_or_default(),
        }],
        ..Default::default()
    })
}

/// A click on a prompt that is no longer open: remove its buttons.
fn stale_prompt(click: &ClickArgs<'_>) -> DecisionResponse {
    let note = "(no longer active)";
    let text = format!("{}\n\n{note}", click.message_text);
    let blocks = render::settled_prompt_blocks(&click.message_blocks, click.message_text, note);
    DecisionResponse {
        channels: [(
            ChannelKind::SLACK.to_string(),
            serde_json::json!({
                "update": {
                    "channel": click.channel,
                    "ts": click.message_ts,
                    "text": text,
                    "blocks": blocks,
                }
            }),
        )]
        .into(),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{Content, Role, StoredResult};
    use crate::protocol::{InterruptOrigin, Requester, RetryPolicy};
    use crate::protocol::{Issuer, Subject};
    use crate::runtime::session::state::OpenInterrupt;
    use crate::session::events::{ToolCallCompleted, ToolCallRequested, TurnStarted};
    use crate::session::state::EventMeta;

    const SESSION: &str = "slack:C1:1.0";

    fn cx<'a>(
        trigger: &'a DecisionTrigger,
        state: &'a SessionState,
        events: &'a [SessionEvent],
    ) -> Proposal<'a> {
        Proposal {
            trigger,
            state,
            events,
            transcript: &[],
        }
    }

    fn translated(trigger: &DecisionTrigger, state: &SessionState) -> Option<DecisionResponse> {
        SlackProposer::new().translate(&cx(trigger, state, &[]))
    }

    fn rendered(
        trigger: &DecisionTrigger,
        state: &SessionState,
        events: &[SessionEvent],
        proposed: &DecisionResponse,
    ) -> Option<Value> {
        SlackProposer::new()
            .render(&cx(trigger, state, events), proposed)
            .map(|v| v["view"].clone())
    }

    fn action(value: &str) -> DecisionTrigger {
        DecisionTrigger::ClientAction {
            name: "prompt_option_0".to_string(),
            args: Some(serde_json::json!({
                "action_id": "prompt_option_0",
                "value": value,
                "user": "U9",
                "channel": "C1",
                "message_ts": "8.0",
                "thread_ts": "1.0",
                "message_text": "Run it?",
                "message_blocks": [],
            })),
        }
    }

    fn state() -> SessionState {
        let mut s = SessionState::new(SESSION.to_string());
        s.owner = Some(crate::protocol::SessionOwner {
            tenant_id: "t".to_string(),
            requester: Requester::new(Subject::new(Issuer::slack(), "T1:U1"), Default::default()),
            metadata: std::collections::HashMap::from_iter([
                ("slack_channel".to_string(), "C1".to_string()),
                ("slack_thread_ts".to_string(), "1.0".to_string()),
            ]),
        });
        s
    }

    fn state_with_prompt() -> SessionState {
        let mut s = state();
        s.open_interrupts.push(OpenInterrupt {
            interrupt_id: "int-1".to_string(),
            origin: InterruptOrigin::Frontend,
            reason: "hold".to_string(),
            payload: serde_json::json!({
                "message": "Run it?",
                "metadata": { "options": [
                    { "label": "Approve", "value": { "decision": "approve" } },
                    { "label": "Deny", "value": { "decision": "deny" } },
                ]},
            }),
            anchor: None,
        });
        s
    }

    fn event(seq: u64, secs: i64, payload: EventPayload) -> SessionEvent {
        let at = chrono::DateTime::from_timestamp(1_700_000_000, 0).unwrap();
        SessionEvent {
            id: uuid::Uuid::nil(),
            tenant_id: "t".into(),
            session_id: SESSION.into(),
            seq,
            span: crate::span::SpanContext::root(),
            occurred_at: at + chrono::Duration::seconds(secs),
            payload,
            meta: EventMeta {
                status: crate::session::state::SessionStatus::Idle,
                wake_at: None,
                owner: None,
                agent_id: None,
                ancestry: Vec::new(),
                turn_id: None,
                cost: Default::default(),
                sub_agent_cost: Default::default(),
                head_id: None,
                calls: Vec::new(),
                decisions: Vec::new(),
            },
            start_time: at,
            end_time: at,
        }
    }

    fn turn_events() -> Vec<SessionEvent> {
        vec![
            event(
                1,
                0,
                EventPayload::TurnStarted(TurnStarted {
                    turn_id: "turn-1".into(),
                }),
            ),
            event(
                2,
                1,
                EventPayload::ToolCallRequested(ToolCallRequested {
                    id: "tc1".into(),
                    attempt: 0,
                    name: "search_web".into(),
                    arguments: "{}".into(),
                    handler: Default::default(),
                    target: None,
                    retry: RetryPolicy::no_retry(),
                }),
            ),
            event(
                3,
                2,
                EventPayload::ToolCallCompleted(ToolCallCompleted {
                    id: "tc1".into(),
                    name: "search_web".into(),
                    result: StoredResult::text("found"),
                }),
            ),
        ]
    }

    #[test]
    fn clicking_an_authorization_prompt_only_clears_it() {
        let mut state = state();
        state.open_interrupts.push(OpenInterrupt {
            interrupt_id: "mcp-auth:mcp.sentry".to_string(),
            origin: InterruptOrigin::Frontend,
            reason: "hold".to_string(),
            payload: serde_json::json!({
                "message": "authorize it",
                "metadata": { "options": [
                    { "label": "Retry", "value": { "connection": "mcp.sentry" } },
                ]},
            }),
            anchor: None,
        });
        let p = translated(
            &action(
                r#"{"type":"interrupt.option","interrupt_id":"mcp-auth:mcp.sentry","option":0}"#,
            ),
            &state,
        )
        .expect("a click it recognises");
        match &p.actions[..] {
            [DecisionAction::ResolveInterrupt { interrupt_id, .. }] => {
                assert_eq!(interrupt_id, "mcp-auth:mcp.sentry");
            }
            other => panic!("expected the resolve and nothing else; got {other:?}"),
        }
    }

    #[test]
    fn clicking_an_ordinary_prompt_asks_for_no_fetch() {
        let p = translated(
            &action(r#"{"type":"interrupt.option","interrupt_id":"int-1","option":1}"#),
            &state_with_prompt(),
        )
        .expect("a click it recognises");
        assert_eq!(p.actions.len(), 1, "got {:?}", p.actions);
    }

    #[test]
    fn turn_finished_proposes_the_final_view_with_footer() {
        let trigger = DecisionTrigger::TurnFinished {
            turn_id: "turn-1".to_string(),
            data: serde_json::json!("the answer"),
            cost: Default::default(),
            usage: Default::default(),
        };
        let view = &rendered(
            &trigger,
            &state(),
            &turn_events(),
            &DecisionResponse::default(),
        )
        .expect("a view");
        assert_eq!(view["text"], "the answer\n\n_2.0s_");
        let blocks = view["blocks"].as_array().unwrap();
        assert_eq!(blocks[0]["type"], "task_card", "the settled card leads");
        assert_eq!(blocks[0]["status"], "complete");
        assert_eq!(
            blocks[0]["details"]["elements"][0]["elements"][0]["text"], "• search_web `{}`",
            "the log rides behind the fold"
        );
        assert_eq!(blocks[1]["text"]["text"], "the answer");
        assert_eq!(blocks[2]["type"], "context", "the footer closes it");
    }

    #[test]
    fn the_card_is_titled_by_what_the_user_asked() {
        let transcript = vec![
            crate::protocol::Message {
                id: "m1".into(),
                role: Role::User,
                content: Some(Content::Text("fix the   login\nbutton".into())),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
                reasoning: None,
            },
            crate::protocol::Message {
                id: "m2".into(),
                role: Role::Assistant,
                content: Some(Content::Text("on it".into())),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
                reasoning: None,
            },
        ];
        assert_eq!(title_of(&transcript), "fix the login button");
        assert_eq!(title_of(&[]), "Working…");
        // A finished card shows its title raw: no markup survives.
        let mention = crate::protocol::Message {
            content: Some(Content::Text(
                "<@U0B8ECR818W>: check <#C1|general> and <https://a.test|the doc> for <@U5>".into(),
            )),
            ..transcript[0].clone()
        };
        assert_eq!(
            title_of(&[mention]),
            "check general and the doc for",
            "the author prefix goes; tokens read as their labels, ids as nothing"
        );
        // Addressing the bot is how a channel reaches it, and the mention
        // leads: an id there would be the first thing the card says.
        let addressed = crate::protocol::Message {
            content: Some(Content::Text("<@U0BNB0VAF8E> what is the push rate".into())),
            ..transcript[0].clone()
        };
        assert_eq!(title_of(&[addressed]), "what is the push rate");
        let long = crate::protocol::Message {
            content: Some(Content::Text("x".repeat(100))),
            ..transcript[0].clone()
        };
        assert_eq!(title_of(&[long]).chars().count(), 60);
    }

    #[test]
    fn a_finished_call_proposes_a_view_of_the_evented_work() {
        let trigger = DecisionTrigger::ToolFinished {
            id: "tc1".to_string(),
            ok: true,
            name: "search_web".to_string(),
            result: Some(StoredResult::text("found")),
            error: None,
        };
        let proposed = DecisionResponse {
            actions: vec![DecisionAction::CallTool {
                id: Some("tc2".to_string()),
                name: "send_email".to_string(),
                arguments: serde_json::json!({"to": "x"}),
                retry: None,
            }],
            ..Default::default()
        };
        let view = rendered(&trigger, &state(), &turn_events(), &proposed).expect("a view");
        let blocks = view["blocks"].as_array().unwrap();
        assert_eq!(blocks.len(), 1, "one card carries the turn");
        assert_eq!(blocks[0]["task_id"], "turn-1");
        assert_eq!(blocks[0]["status"], "in_progress");
        assert_eq!(
            blocks[0]["title"], "Working… · search_web · 1 action",
            "no transcript, no ask; the title still says where the turn is"
        );
        // The proposed call is not in the log: a proposal can render its
        // call another way than the event will, and the stream appends —
        // a line that changed would freeze the log. It lands one tick on.
        assert_eq!(
            blocks[0]["details"]["elements"][0]["elements"][0]["text"], "• search_web `{}`",
            "the events write the log; the proposal does not"
        );
    }

    #[test]
    fn a_proposal_that_interrupts_gets_no_view() {
        let trigger = DecisionTrigger::LlmFinished {
            id: "call-1".to_string(),
            ok: false,
            message: None,
            truncated: false,
            refused: false,
            usage: None,
            cost: None,
            error: None,
        };
        let proposed = DecisionResponse {
            actions: vec![DecisionAction::Interrupt {
                interrupt_id: None,
                reason: "llm call failed".to_string(),
                payload: Value::Null,
            }],
            ..Default::default()
        };
        let view = rendered(&trigger, &state(), &turn_events(), &proposed);
        assert!(
            view.is_none(),
            "the prompt owns the message; no view rides along"
        );
    }

    #[test]
    fn a_prompt_click_proposes_resolving_with_the_recorded_value() {
        let p = translated(
            &action(r#"{"type":"interrupt.option","interrupt_id":"int-1","option":1}"#),
            &state_with_prompt(),
        )
        .expect("a click it recognises");
        match &p.actions[..] {
            [DecisionAction::ResolveInterrupt {
                interrupt_id,
                payload,
            }] => {
                assert_eq!(interrupt_id, "int-1");
                assert_eq!(payload["status"], "resolved");
                assert_eq!(payload["payload"]["decision"], "deny");
                assert_eq!(payload["responder"]["user"], "U9");
                assert_eq!(payload["responder"]["label"], "Deny");
            }
            other => panic!("expected one interrupt.resolve; got {other:?}"),
        }
    }

    #[test]
    fn a_click_on_a_settled_prompt_proposes_clearing_it() {
        let p = translated(
            &action(r#"{"type":"interrupt.option","interrupt_id":"gone","option":0}"#),
            &state(),
        )
        .expect("a click it recognises");
        assert!(p.actions.is_empty());
        let slack = &p.channels["slack"];
        assert_eq!(slack["update"]["ts"], "8.0");
        assert!(slack["update"]["text"]
            .as_str()
            .unwrap()
            .contains("no longer active"));
    }

    #[test]
    fn a_workers_own_button_passes_through_untouched() {
        let p = translated(&action("summarize-thread"), &state_with_prompt());
        assert!(p.is_none(), "a worker's button is not Slack's to answer");
    }

    #[test]
    fn it_declares_the_one_channel_it_answers_for() {
        assert_eq!(SlackProposer::new().channel(), ChannelKind::SLACK);
    }
}
