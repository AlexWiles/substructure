use serde_json::Value;

use super::clip;
use super::render;
use crate::session::events::EventPayload;
use crate::session::SessionEvent;

// The stream caps a task_update chunk at 256 characters, and the title
// rides in the same chunk; these budgets keep the card inside it.
const MAX_SAID: usize = 120;
const MAX_STEP: usize = 60;

/// A turn's visible work, folded from the event log into one card: the
/// latest preamble, and the calls made since it. Every render is derived,
/// so a replay converges.
pub(super) struct TurnActivity {
    turn_id: String,
    /// The latest preamble, keyed so a redelivery does not reset the count.
    said: Option<(String, String)>,
    /// The calls since the preamble: id and preview. The id folds a retry
    /// into the call it retries.
    steps: Vec<(String, String)>,
}

impl TurnActivity {
    /// The last turn in `events`, or `None` if there is no turn.
    pub(super) fn fold(events: &[SessionEvent]) -> Option<Self> {
        let mut turn: Option<Self> = None;
        for event in events {
            if let EventPayload::TurnStarted(t) = &event.payload {
                turn = Some(Self {
                    turn_id: t.turn_id.clone(),
                    said: None,
                    steps: Vec::new(),
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

    /// A new preamble starts a new count; the same one redelivered does not.
    fn say(&mut self, id: &str, text: &str) {
        if let Some((said, existing)) = self.said.as_mut() {
            if said == id {
                *existing = text.to_string();
                return;
            }
        }
        self.said = Some((id.to_string(), text.to_string()));
        self.steps.clear();
    }

    /// Count a call once; a retry or redelivery updates it in place.
    pub(super) fn step(&mut self, id: &str, name: &str, input: Option<&str>) {
        let preview = match input.map(flatten).filter(|i| !i.is_empty()) {
            Some(input) => clip(&format!("{name} {input}"), MAX_STEP),
            None => clip(name, MAX_STEP),
        };
        match self.steps.iter_mut().find(|(step, _)| step == id) {
            Some((_, existing)) => *existing = preview,
            None => self.steps.push((id.to_string(), preview)),
        }
    }

    /// The turn's one card while it runs. The same id every render, so each
    /// update replaces the card in place.
    pub(super) fn card(&self, title: &str) -> Value {
        render::turn_card(
            &self.turn_id,
            title,
            "in_progress",
            self.details().as_deref(),
        )
    }

    /// The card at rest: done, and folded to its title.
    pub(super) fn settled_card(&self, title: &str) -> Value {
        render::turn_card(&self.turn_id, title, "complete", None)
    }

    /// What the card says now: the preamble, then how much ran since it and
    /// the newest call.
    fn details(&self) -> Option<String> {
        let said = self.said.as_ref().map(|(_, text)| clip(text, MAX_SAID));
        let ran = self.steps.last().map(|(_, latest)| {
            let n = self.steps.len();
            let s = if n == 1 { "" } else { "s" };
            format!("Ran {n} action{s} — {latest}")
        });
        match (said, ran) {
            (None, None) => None,
            (Some(said), None) => Some(said),
            (None, Some(ran)) => Some(ran),
            (Some(said), Some(ran)) => Some(format!("{said}\n{ran}")),
        }
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
    fn the_card_carries_the_turn_and_replaces_in_place() {
        let events = vec![
            started("turn-1", 1),
            said("llm1", "Let me look that up.", true, 2),
            tool_requested("tc1", "search_web", r#"{"q":"x"}"#, 3),
        ];
        let card = TurnActivity::fold(&events).unwrap().card("Find x");
        assert_eq!(card["type"], "task_card");
        assert_eq!(card["task_id"], "turn-1");
        assert_eq!(card["title"], "Find x");
        assert_eq!(card["status"], "in_progress");
        assert_eq!(
            details(&events),
            "Let me look that up.\nRan 1 action — search_web {\"q\":\"x\"}"
        );
    }

    #[test]
    fn a_new_preamble_starts_a_new_count() {
        let mut events = vec![
            started("turn-1", 1),
            said("llm1", "First.", true, 2),
            tool_requested("tc1", "a", "{}", 3),
            tool_requested("tc2", "b", "{}", 4),
        ];
        assert_eq!(details(&events), "First.\nRan 2 actions — b {}");
        events.push(said("llm2", "Second.", true, 5));
        assert_eq!(details(&events), "Second.");
        events.push(tool_requested("tc3", "c", "{}", 6));
        assert_eq!(details(&events), "Second.\nRan 1 action — c {}");
    }

    #[test]
    fn a_redelivered_preamble_keeps_the_count() {
        let events = vec![
            started("turn-1", 1),
            said("llm1", "First.", true, 2),
            tool_requested("tc1", "a", "{}", 3),
            said("llm1", "First.", true, 4),
        ];
        assert_eq!(details(&events), "First.\nRan 1 action — a {}");
    }

    #[test]
    fn a_retry_counts_once() {
        let events = vec![
            started("turn-1", 1),
            tool_requested("tc1", "a", "{}", 2),
            tool_requested("tc1", "a", r#"{"again":1}"#, 3),
        ];
        assert_eq!(details(&events), "Ran 1 action — a {\"again\":1}");
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
        assert_eq!(details(&events), "Ran 1 action — agent researcher");
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
        assert_eq!(details(&events), "Ran 1 action — new {}");
        assert!(TurnActivity::fold(&[tool_requested("tc1", "t", "{}", 1)]).is_none());
    }

    #[test]
    fn a_chatty_step_fits_the_chunk() {
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
    fn the_settled_card_folds_to_its_title() {
        let events = vec![
            started("turn-1", 1),
            said("llm1", "Working on it.", true, 2),
            tool_requested("tc1", "a", "{}", 3),
        ];
        let card = TurnActivity::fold(&events).unwrap().settled_card("Find x");
        assert_eq!(card["task_id"], "turn-1");
        assert_eq!(card["status"], "complete");
        assert!(card.get("details").is_none());
    }

    #[test]
    fn a_proposed_step_counts_before_its_event_lands() {
        let events = vec![started("turn-1", 1), said("llm1", "Next.", true, 2)];
        let mut turn = TurnActivity::fold(&events).unwrap();
        turn.step("tc9", "send_email", Some(r#"{"to":"x"}"#));
        let card = turn.card("t");
        assert_eq!(
            card["details"]["elements"][0]["elements"][0]["text"],
            "Next.\nRan 1 action — send_email {\"to\":\"x\"}"
        );
    }

    #[test]
    fn the_footer_reports_how_long_the_turn_took() {
        let at = |s: i64| chrono::DateTime::from_timestamp(1_700_000_000 + s, 0).unwrap();
        assert_eq!(elapsed(at(0), at(4)), "4.0s");
        assert_eq!(elapsed(at(0), at(90)), "1m 30s");
    }
}
