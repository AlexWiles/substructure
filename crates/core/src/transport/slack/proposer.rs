//! What Slack proposes for a decision, when the worker does not say.
//!
//! Everything is derived from the turn's events, so a redelivery proposes the
//! same thing. A worker can amend or replace any of it.

use serde_json::Value;

use super::activity::TurnActivity;
use super::render::{self, StepStatus, StepView};
use super::{display_of, with_footer, ButtonValue};
use crate::connectors::{AuthNeed, Requester};
use crate::protocol::{
    DecisionAction, DecisionResponse, DecisionTrigger, InterruptResolution, InterruptResponder,
    Message, ResumeStatus,
};
use crate::runtime::session::interrupts::{self, auth};
use crate::runtime::session::state::SessionState;
use crate::runtime::worker::ChannelProposer;
use crate::session::events::EventPayload;
use crate::session::SessionEvent;

#[derive(Default)]
pub struct SlackProposer;

impl SlackProposer {
    pub fn new() -> Self {
        Self
    }
}

/// Whether Slack owns this session: its owner records a Slack channel.
fn slack_owned(state: &SessionState) -> bool {
    state
        .owner
        .as_ref()
        .is_some_and(|o| o.metadata.contains_key("slack_channel"))
}

impl ChannelProposer for SlackProposer {
    fn propose(
        &self,
        _session_id: &str,
        trigger: &DecisionTrigger,
        state: &SessionState,
        events: &[SessionEvent],
        _transcript: &[Message],
        proposed: DecisionResponse,
    ) -> DecisionResponse {
        if !slack_owned(state) {
            return proposed;
        }
        // First: the proposal below would run against tools the agent has lost.
        if let Some(prompt) = self.authorize_prompt(state) {
            return prompt;
        }
        match trigger {
            DecisionTrigger::ClientAction {
                args: Some(args), ..
            } => match click_proposal(state, args) {
                Some(p) => p,
                None => proposed,
            },
            DecisionTrigger::LlmFinished { .. }
            | DecisionTrigger::ToolFinished { .. }
            | DecisionTrigger::SubAgentFinished { .. }
            | DecisionTrigger::InterruptResumed { .. } => {
                // A proposal that interrupts owns the message; no view.
                let authors_interrupt = proposed
                    .actions
                    .iter()
                    .any(|a| matches!(a, DecisionAction::Interrupt { .. }));
                if authors_interrupt {
                    return proposed;
                }
                let view = streaming_view(events, &proposed);
                with_view(proposed, view)
            }
            DecisionTrigger::TurnFinished { data, .. } => {
                let view = final_view(events, data);
                with_view(proposed, Some(view))
            }
            _ => proposed,
        }
    }
}

impl SlackProposer {
    /// The prompt for the first connection that needs a person. It replaces
    /// the proposal, because the session stops here.
    ///
    /// It writes why, and the facts a way in is built from. How to authorize
    /// is the delivering channel's to add, so no link is written here.
    fn authorize_prompt(&self, state: &SessionState) -> Option<DecisionResponse> {
        let (connection, need) = auth::needing(state)?;
        let authorize = auth::Authorize {
            principal: Requester::of_owner(state.owner.as_ref()),
            connection: connection.clone(),
        };

        Some(DecisionResponse {
            actions: vec![DecisionAction::Interrupt {
                interrupt_id: Some(auth::interrupt_id(&connection)),
                reason: format!("connection `{connection}` needs authorizing"),
                payload: serde_json::json!({
                    "message": ask(&connection, need),
                    "authorize": authorize,
                    "metadata": { "options": [{
                        "label": "Retry",
                        "value": { "connection": connection },
                    }] },
                }),
            }],
            ..Default::default()
        })
    }
}

/// Why the session stopped. A rejected token names its fix here, because no
/// consent flow can replace a static token.
fn ask(connection: &str, need: AuthNeed) -> String {
    match need {
        AuthNeed::NeverAuthorized => {
            format!("*{connection}* is not authorized yet, so I cannot use it.")
        }
        AuthNeed::Reauthorize => {
            format!("*{connection}* needs to be authorized again. Its access expired.")
        }
        AuthNeed::TokenRejected => format!(
            "*{connection}* rejected its token. An operator must set a new one \
             with `subs mcp set-token {connection}`."
        ),
    }
}

fn with_view(mut proposed: DecisionResponse, view: Option<Value>) -> DecisionResponse {
    if let Some(view) = view {
        let slack = proposed
            .channels
            .entry("slack".to_string())
            .or_insert_with(|| serde_json::json!({}));
        slack["view"] = view;
    }
    proposed
}

/// The message so far: the turn's finished work, plus an in-progress card for
/// each call this proposal starts.
fn streaming_view(events: &[SessionEvent], proposed: &DecisionResponse) -> Option<Value> {
    let mut blocks = TurnActivity::fold(events)
        .map(|turn| turn.blocks())
        .unwrap_or_default();
    for action in &proposed.actions {
        let card = match action {
            DecisionAction::CallTool {
                id: Some(id),
                name,
                arguments,
                ..
            } => {
                let input = match arguments {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                render::step_block(&StepView {
                    id,
                    name,
                    status: StepStatus::InProgress,
                    took: None,
                    input: Some(&input),
                    output: None,
                })
            }
            DecisionAction::SpawnSubAgent {
                agent_id,
                tool_call_id,
                ..
            } => render::step_block(&StepView {
                id: tool_call_id,
                name: &format!("agent {agent_id}"),
                status: StepStatus::InProgress,
                took: None,
                input: None,
                output: None,
            }),
            _ => continue,
        };
        blocks.push(card);
    }
    (!blocks.is_empty()).then(|| serde_json::json!({ "text": "", "blocks": blocks }))
}

/// The final message: the turn's work, the answer, and how long it took.
fn final_view(events: &[SessionEvent], data: &Value) -> Value {
    let mut blocks = TurnActivity::fold(events)
        .map(|turn| turn.blocks())
        .unwrap_or_default();
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
            channel: "slack".to_string(),
            user: Some(click.user.to_string()),
            label: Some(option.label),
            style: option.style,
        }),
    };
    let followups = interrupts::resolve_followups(&interrupt_id);
    Some(DecisionResponse {
        actions: std::iter::once(DecisionAction::ResolveInterrupt {
            interrupt_id,
            payload: serde_json::to_value(resolution).unwrap_or_default(),
        })
        .chain(followups)
        .collect(),
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
            "slack".to_string(),
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
    use crate::protocol::StoredResult;
    use crate::protocol::{AgentConfig, AuthFailure, InterruptOrigin, McpServer, RetryPolicy};
    use crate::protocol::{Issuer, Subject};
    use crate::runtime::session::state::OpenInterrupt;
    use crate::session::events::{AgentConfigUpdated, ConnectorAuthFailed, ConnectorSyncRequested};
    use crate::session::events::{ToolCallCompleted, ToolCallRequested, TurnStarted};
    use crate::session::state::ApplyContext;
    use crate::session::state::EventMeta;

    const SESSION: &str = "slack:C1:1.0";

    fn propose(
        session_id: &str,
        trigger: &DecisionTrigger,
        state: &SessionState,
        events: &[SessionEvent],
        proposed: DecisionResponse,
    ) -> DecisionResponse {
        SlackProposer::new().propose(session_id, trigger, state, events, &[], proposed)
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

    fn ctx() -> ApplyContext {
        ApplyContext {
            occurred_at: chrono::Utc::now(),
            sequence: 1,
        }
    }

    fn state_needing_auth(need: AuthNeed, policy: AuthFailure) -> SessionState {
        let mut s = state();
        s.apply(
            &EventPayload::AgentConfigUpdated(AgentConfigUpdated {
                config: AgentConfig {
                    mcp: vec![McpServer {
                        id: "sentry".to_string(),
                        tools: None,
                        auth_failure: policy,
                        approve: Default::default(),
                    }],
                    ..AgentConfig {
                        llm: None,
                        model: "m1".to_string(),
                        system: None,
                        retry: None,
                        tools: vec![],
                        sub_agents: vec![],
                        mcp: vec![],
                        defer_tools: None,
                        announce_mcp: Default::default(),
                        plugins: Vec::new(),
                        effort: None,
                    }
                },
                anchor: None,
            }),
            &ctx(),
        );
        s.apply(
            &EventPayload::ConnectorSyncRequested(ConnectorSyncRequested {
                id: "sentry".to_string(),
                attempt: 0,
                retry: RetryPolicy::no_retry(),
            }),
            &ctx(),
        );
        s.apply(
            &EventPayload::ConnectorAuthFailed(ConnectorAuthFailed {
                id: "sentry".to_string(),
                auth: need,
            }),
            &ctx(),
        );
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
    fn a_connection_that_needs_authorizing_proposes_a_prompt_instead_of_the_turn() {
        let p = propose(
            SESSION,
            &DecisionTrigger::ToolFinished {
                id: "tc1".to_string(),
                ok: false,
                name: "sentry__search_issues".to_string(),
                result: None,
                error: None,
            },
            &state_needing_auth(AuthNeed::Reauthorize, AuthFailure::Interrupt),
            &[],
            DecisionResponse {
                actions: vec![DecisionAction::CallLlm {
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
                }],
                ..Default::default()
            },
        );
        match &p.actions[..] {
            [DecisionAction::Interrupt {
                interrupt_id,
                payload,
                ..
            }] => {
                assert_eq!(interrupt_id.as_deref(), Some("mcp-auth:sentry"));
                let message = payload["message"].as_str().unwrap();
                assert!(message.contains("authorized again"), "got {message}");
                // Why, and the facts a way in is built from. The link itself
                // is minted by whoever delivers this.
                assert!(!message.contains("http"), "got {message}");
                let authorize: auth::Authorize =
                    serde_json::from_value(payload["authorize"].clone()).unwrap();
                assert_eq!(authorize.connection, "sentry");
                assert_eq!(
                    authorize.principal,
                    Requester::new(Subject::new(Issuer::slack(), "T1:U1"), Default::default())
                );
            }
            other => panic!("expected one interrupt; got {other:?}"),
        }
    }

    #[test]
    fn a_connection_already_asked_about_is_not_asked_about_again() {
        let mut state = state_needing_auth(AuthNeed::Reauthorize, AuthFailure::Interrupt);
        state.open_interrupts.push(OpenInterrupt {
            interrupt_id: "mcp-auth:sentry".to_string(),
            origin: InterruptOrigin::Frontend,
            reason: "hold".to_string(),
            payload: Value::Null,
            anchor: None,
        });
        let p = propose(
            SESSION,
            &DecisionTrigger::TurnFinished {
                turn_id: "turn-1".to_string(),
                data: serde_json::json!("done"),
                cost: Default::default(),
                usage: Default::default(),
            },
            &state,
            &[],
            DecisionResponse::default(),
        );
        assert!(
            !p.actions
                .iter()
                .any(|a| matches!(a, DecisionAction::Interrupt { .. })),
            "got {:?}",
            p.actions
        );
    }

    #[test]
    fn a_connection_configured_to_degrade_never_stops_the_session() {
        let p = propose(
            SESSION,
            &DecisionTrigger::TurnFinished {
                turn_id: "turn-1".to_string(),
                data: serde_json::json!("done"),
                cost: Default::default(),
                usage: Default::default(),
            },
            &state_needing_auth(AuthNeed::Reauthorize, AuthFailure::Degrade),
            &[],
            DecisionResponse::default(),
        );
        assert!(
            !p.actions
                .iter()
                .any(|a| matches!(a, DecisionAction::Interrupt { .. })),
            "got {:?}",
            p.actions
        );
    }

    #[test]
    fn a_rejected_token_names_the_command_an_operator_runs() {
        let p = propose(
            SESSION,
            &DecisionTrigger::TurnFinished {
                turn_id: "turn-1".to_string(),
                data: Value::Null,
                cost: Default::default(),
                usage: Default::default(),
            },
            &state_needing_auth(AuthNeed::TokenRejected, AuthFailure::Interrupt),
            &[],
            DecisionResponse::default(),
        );
        let DecisionAction::Interrupt { payload, .. } = &p.actions[0] else {
            panic!("expected an interrupt; got {:?}", p.actions);
        };
        let message = payload["message"].as_str().unwrap();
        assert!(
            message.contains("subs mcp set-token sentry"),
            "got {message}"
        );
        assert!(
            !message.contains("authorize"),
            "no link to click; got {message}"
        );
    }

    #[test]
    fn clicking_an_authorization_prompt_also_asks_for_the_tools_again() {
        let mut state = state();
        state.open_interrupts.push(OpenInterrupt {
            interrupt_id: "mcp-auth:sentry".to_string(),
            origin: InterruptOrigin::Frontend,
            reason: "hold".to_string(),
            payload: serde_json::json!({
                "message": "authorize it",
                "metadata": { "options": [
                    { "label": "Retry", "value": { "connection": "sentry" } },
                ]},
            }),
            anchor: None,
        });
        let p = propose(
            SESSION,
            &action(r#"{"type":"interrupt.option","interrupt_id":"mcp-auth:sentry","option":0}"#),
            &state,
            &[],
            DecisionResponse::default(),
        );
        match &p.actions[..] {
            [DecisionAction::ResolveInterrupt { interrupt_id, .. }, DecisionAction::SyncConnector { id }] =>
            {
                assert_eq!(interrupt_id, "mcp-auth:sentry");
                assert_eq!(id, "sentry");
            }
            other => panic!("expected resolve then sync; got {other:?}"),
        }
    }

    #[test]
    fn clicking_an_ordinary_prompt_asks_for_no_fetch() {
        let p = propose(
            SESSION,
            &action(r#"{"type":"interrupt.option","interrupt_id":"int-1","option":1}"#),
            &state_with_prompt(),
            &[],
            DecisionResponse::default(),
        );
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
        let p = propose(
            SESSION,
            &trigger,
            &state(),
            &turn_events(),
            DecisionResponse::default(),
        );
        let view = &p.channels["slack"]["view"];
        assert_eq!(view["text"], "the answer\n\n_2.0s_");
        let blocks = view["blocks"].as_array().unwrap();
        assert_eq!(blocks[0]["type"], "task_card", "the settled work leads");
        assert_eq!(blocks[1]["text"]["text"], "the answer");
        assert_eq!(blocks[2]["type"], "context", "the footer closes it");
    }

    #[test]
    fn a_finished_call_proposes_a_view_with_the_next_calls_card() {
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
        let p = propose(SESSION, &trigger, &state(), &turn_events(), proposed);
        let blocks = p.channels["slack"]["view"]["blocks"].as_array().unwrap();
        assert_eq!(blocks[0]["task_id"], "tc1");
        assert_eq!(blocks[0]["status"], "complete");
        assert_eq!(blocks[1]["task_id"], "tc2", "the dispatched call's card");
        assert_eq!(blocks[1]["status"], "in_progress");
        assert!(!p.actions.is_empty(), "the core continuation is kept");
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
        let p = propose(SESSION, &trigger, &state(), &turn_events(), proposed);
        assert!(
            p.channels.is_empty(),
            "the prompt owns the message; no view rides along"
        );
    }

    #[test]
    fn a_prompt_click_proposes_resolving_with_the_recorded_value() {
        let p = propose(
            SESSION,
            &action(r#"{"type":"interrupt.option","interrupt_id":"int-1","option":1}"#),
            &state_with_prompt(),
            &[],
            DecisionResponse::default(),
        );
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
        let p = propose(
            SESSION,
            &action(r#"{"type":"interrupt.option","interrupt_id":"gone","option":0}"#),
            &state(),
            &[],
            DecisionResponse::default(),
        );
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
        let p = propose(
            SESSION,
            &action("summarize-thread"),
            &state_with_prompt(),
            &[],
            DecisionResponse::default(),
        );
        assert!(p.authors_nothing(), "no proposal for a worker's button");
    }

    #[test]
    fn sessions_slack_does_not_own_are_left_alone() {
        // No `slack_channel` on the owner.
        let mut foreign = SessionState::new("web:abc".to_string());
        foreign.open_interrupts = state_with_prompt().open_interrupts;
        let p = propose(
            "web:abc",
            &action(r#"{"type":"interrupt.option","interrupt_id":"int-1","option":0}"#),
            &foreign,
            &[],
            DecisionResponse::default(),
        );
        assert!(p.authors_nothing());
    }
}
