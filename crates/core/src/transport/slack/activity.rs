use serde_json::Value;

use super::clip;
use super::render;
use crate::session::events::EventPayload;
use crate::session::SessionEvent;

const MAX_SAID: usize = 180;
const MAX_STEP: usize = 60;
/// The stream freezes here; the settled card clips here too.
const MAX_LOG: usize = 3_000;

/// One thing the turn showed, in the order it happened. The stream can only
/// append, so the log only grows: an id seen again is dropped, not edited.
enum Item {
    /// Preamble text before a call.
    Said { id: String, text: String },
    /// A tool call or a sub-agent run.
    Step { id: String, preview: String },
}

impl Item {
    fn id(&self) -> &str {
        match self {
            Item::Said { id, .. } | Item::Step { id, .. } => id,
        }
    }
}

/// A turn's visible work, folded from the event log into one card. Every
/// render is derived, so a replay converges.
pub(super) struct TurnActivity {
    turn_id: String,
    items: Vec<Item>,
}

impl TurnActivity {
    /// The last turn in `events`, or `None` if there is no turn.
    pub(super) fn fold(events: &[SessionEvent]) -> Option<Self> {
        let mut turn: Option<Self> = None;
        for event in events {
            if let EventPayload::TurnStarted(t) = &event.payload {
                turn = Some(Self {
                    turn_id: t.turn_id.clone(),
                    items: Vec::new(),
                });
                continue;
            }
            if let Some(turn) = turn.as_mut() {
                turn.apply(event);
            }
        }
        turn
    }

    fn apply(&mut self, event: &SessionEvent) {
        match &event.payload {
            // A response with no calls is the answer, not a preamble.
            EventPayload::LlmCallCompleted(c) if !c.response.tool_calls.is_empty() => {
                if let Some(text) = c.response.content.as_deref().map(str::trim) {
                    if !text.is_empty() {
                        self.say(&c.id, text);
                    }
                }
            }
            EventPayload::ToolCallRequested(t) => {
                self.step(&t.id, &t.name, Some(&t.arguments));
            }
            EventPayload::SubAgentRequested(s) => {
                self.step(&s.id, &format!("agent {}", s.agent_id), None);
            }
            _ => {}
        }
    }

    fn say(&mut self, id: &str, text: &str) {
        if self.items.iter().any(|item| item.id() == id) {
            return;
        }
        self.items.push(Item::Said {
            id: id.to_string(),
            text: text.to_string(),
        });
    }

    /// Count a call once; a retry or redelivery is the same call.
    pub(super) fn step(&mut self, id: &str, name: &str, input: Option<&str>) {
        if self.items.iter().any(|item| item.id() == id) {
            return;
        }
        let preview = match input.map(flatten).filter(|i| !i.is_empty()) {
            Some(input) => clip(&format!("{name} {input}"), MAX_STEP),
            None => clip(name, MAX_STEP),
        };
        self.items.push(Item::Step {
            id: id.to_string(),
            preview,
        });
    }

    /// The turn's one card while it runs: the log so far. The log only
    /// grows, so the stream appends each render's delta.
    pub(super) fn card(&self, title: &str) -> Value {
        render::turn_card(&self.turn_id, title, "in_progress", self.log().as_deref())
    }

    /// The card at rest: done, with the whole log behind the fold.
    pub(super) fn settled_card(&self, title: &str) -> Value {
        render::turn_card(&self.turn_id, title, "complete", self.log().as_deref())
    }

    /// Everything the turn showed, oldest first.
    fn log(&self) -> Option<String> {
        let lines: Vec<String> = self
            .items
            .iter()
            .map(|item| match item {
                Item::Said { text, .. } => clip(text, MAX_SAID),
                Item::Step { preview, .. } => format!("• {preview}"),
            })
            .collect();
        (!lines.is_empty()).then(|| clip(&lines.join("\n"), MAX_LOG))
    }
}

pub(super) fn elapsed(
    from: chrono::DateTime<chrono::Utc>,
    to: chrono::DateTime<chrono::Utc>,
) -> String {
    let ms = (to - from).num_milliseconds().max(0);
    if ms < 60_000 {
        return format!("{:.1}s", ms as f64 / 1000.0);
    }
    format!("{}m {:02}s", ms / 60_000, (ms % 60_000) / 1000)
}

fn flatten(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::events::{SubAgentRequested, ToolCallRequested, TurnStarted};

    /// An LLM response saying `content`, with a call to make when `calling`.
    fn said(id: &str, content: &str, calling: bool, seq: u64) -> SessionEvent {
        use crate::protocol::{LlmResponse, ToolCall, ToolCallFunction};
        event(
            seq,
            EventPayload::LlmCallCompleted(crate::session::events::LlmCallCompleted {
                id: id.into(),
                attempt: 0,
                response: LlmResponse {
                    model: "m".into(),
                    content: Some(content.into()),
                    tool_calls: calling
                        .then(|| ToolCall {
                            id: "tc1".into(),
                            call_type: "function".into(),
                            function: ToolCallFunction {
                                name: "search_web".into(),
                                arguments: "{}".into(),
                            },
                        })
                        .into_iter()
                        .collect(),
                    finish_reason: None,
                    usage: None,
                    cost: None,
                    images: Vec::new(),
                    reasoning: None,
                },
            }),
        )
    }

    fn retry() -> crate::protocol::RetryPolicy {
        crate::protocol::RetryPolicy {
            attempt_timeout_secs: None,
            total_timeout_secs: None,
            max_attempts: 0,
            backoff_base_secs: 1,
            backoff_max_secs: 1,
        }
    }

    fn event(seq: u64, payload: EventPayload) -> SessionEvent {
        let at = chrono::DateTime::from_timestamp(1_700_000_000, 0).unwrap();
        SessionEvent {
            id: uuid::Uuid::nil(),
            tenant_id: "t".into(),
            session_id: "slack:C1:1.0".into(),
            seq,
            span: crate::span::SpanContext::root(),
            occurred_at: at,
            payload,
            meta: crate::session::state::EventMeta {
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

    fn started(turn: &str, seq: u64) -> SessionEvent {
        event(
            seq,
            EventPayload::TurnStarted(TurnStarted {
                turn_id: turn.into(),
            }),
        )
    }

    fn tool_requested(id: &str, name: &str, arguments: &str, seq: u64) -> SessionEvent {
        event(
            seq,
            EventPayload::ToolCallRequested(ToolCallRequested {
                id: id.into(),
                attempt: 0,
                name: name.into(),
                arguments: arguments.into(),
                handler: Default::default(),
                target: None,
                retry: retry(),
            }),
        )
    }

    fn details(events: &[SessionEvent]) -> String {
        let card = TurnActivity::fold(events).unwrap().card("t");
        card["details"]["elements"][0]["elements"][0]["text"]
            .as_str()
            .unwrap_or_default()
            .to_string()
    }

    #[test]
    fn the_card_carries_the_turn_as_a_growing_log() {
        let mut events = vec![
            started("turn-1", 1),
            said("llm1", "Let me look that up.", true, 2),
            tool_requested("tc1", "search_web", r#"{"q":"x"}"#, 3),
        ];
        let card = TurnActivity::fold(&events).unwrap().card("Find x");
        assert_eq!(card["type"], "task_card");
        assert_eq!(card["task_id"], "turn-1");
        assert_eq!(card["title"], "Find x");
        assert_eq!(card["status"], "in_progress");
        let first = details(&events);
        assert_eq!(first, "Let me look that up.\n• search_web {\"q\":\"x\"}");
        // The log only grows, so the stream can append each render's delta.
        events.push(said("llm2", "Now the details.", true, 4));
        events.push(tool_requested("tc2", "get_page", "{}", 5));
        let second = details(&events);
        assert!(second.starts_with(&first), "{second}");
        assert_eq!(
            second,
            "Let me look that up.\n• search_web {\"q\":\"x\"}\nNow the details.\n• get_page {}"
        );
    }

    #[test]
    fn a_redelivered_preamble_lands_once() {
        let events = vec![
            started("turn-1", 1),
            said("llm1", "First.", true, 2),
            tool_requested("tc1", "a", "{}", 3),
            said("llm1", "First.", true, 4),
        ];
        assert_eq!(details(&events), "First.\n• a {}");
    }

    #[test]
    fn a_retry_lands_once() {
        let events = vec![
            started("turn-1", 1),
            tool_requested("tc1", "a", "{}", 2),
            tool_requested("tc1", "a", r#"{"again":1}"#, 3),
        ];
        assert_eq!(details(&events), "• a {}");
    }

    #[test]
    fn a_response_with_nothing_left_to_call_is_the_answer_not_a_preamble() {
        // The reply already carries it; saying it here would say it twice.
        let events = vec![
            started("turn-1", 1),
            said("llm1", "Here is the answer.", false, 2),
        ];
        let card = TurnActivity::fold(&events).unwrap().card("t");
        assert!(card.get("details").is_none());
    }

    #[test]
    fn a_silent_call_says_nothing() {
        let events = vec![started("turn-1", 1), said("llm1", "   ", true, 2)];
        let card = TurnActivity::fold(&events).unwrap().card("t");
        assert!(card.get("details").is_none());
    }

    #[test]
    fn a_sub_agent_previews_by_name() {
        let events = vec![
            started("turn-1", 1),
            event(
                2,
                EventPayload::SubAgentRequested(SubAgentRequested {
                    id: "sub-1".into(),
                    agent_id: "researcher".into(),
                    tool_call_id: "tc1".into(),
                    message: None,
                    retry: retry(),
                }),
            ),
        ];
        assert_eq!(details(&events), "• agent researcher");
    }

    #[test]
    fn only_the_latest_turn_folds() {
        let events = vec![
            started("turn-1", 1),
            tool_requested("tc1", "old", "{}", 2),
            started("turn-2", 3),
            tool_requested("tc2", "new", "{}", 4),
        ];
        let turn = TurnActivity::fold(&events).unwrap();
        assert_eq!(turn.card("t")["task_id"], "turn-2");
        assert_eq!(details(&events), "• new {}");
        assert!(TurnActivity::fold(&[tool_requested("tc1", "t", "{}", 1)]).is_none());
    }

    #[test]
    fn a_chatty_item_is_clipped_per_line() {
        let long = "x".repeat(500);
        let events = vec![
            started("turn-1", 1),
            said("llm1", &long, true, 2),
            tool_requested("tc1", "tool", &long, 3),
        ];
        let text = details(&events);
        assert!(text.chars().count() <= MAX_SAID + MAX_STEP + 20, "{text}");
        assert!(text.contains('…'));
    }

    #[test]
    fn the_settled_card_carries_the_same_log_done() {
        let events = vec![
            started("turn-1", 1),
            said("llm1", "First.", true, 2),
            tool_requested("tc1", "a", "{}", 3),
            said("llm2", "Second.", true, 4),
            tool_requested("tc2", "b", "{}", 5),
        ];
        let card = TurnActivity::fold(&events).unwrap().settled_card("Find x");
        assert_eq!(card["task_id"], "turn-1");
        assert_eq!(card["status"], "complete");
        assert_eq!(
            card["details"]["elements"][0]["elements"][0]["text"],
            "First.\n• a {}\nSecond.\n• b {}",
            "every step, oldest first"
        );
    }

    #[test]
    fn a_long_log_is_clipped() {
        let arg = "x".repeat(100);
        let mut events = vec![started("turn-1", 1)];
        for i in 0..200 {
            events.push(tool_requested(&format!("tc{i}"), "tool", &arg, 2 + i));
        }
        let card = TurnActivity::fold(&events).unwrap().settled_card("t");
        let log = card["details"]["elements"][0]["elements"][0]["text"]
            .as_str()
            .unwrap();
        assert_eq!(log.chars().count(), MAX_LOG);
    }

    #[test]
    fn a_proposed_step_lands_before_its_event_does() {
        let events = vec![started("turn-1", 1), said("llm1", "Next.", true, 2)];
        let mut turn = TurnActivity::fold(&events).unwrap();
        turn.step("tc9", "send_email", Some(r#"{"to":"x"}"#));
        let card = turn.card("t");
        assert_eq!(
            card["details"]["elements"][0]["elements"][0]["text"],
            "Next.\n• send_email {\"to\":\"x\"}"
        );
    }

    #[test]
    fn the_footer_reports_how_long_the_turn_took() {
        let at = |s: i64| chrono::DateTime::from_timestamp(1_700_000_000 + s, 0).unwrap();
        assert_eq!(elapsed(at(0), at(4)), "4.0s");
        assert_eq!(elapsed(at(0), at(90)), "1m 30s");
    }
}
