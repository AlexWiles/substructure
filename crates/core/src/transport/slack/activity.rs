use serde_json::Value;

use super::clip;
use super::controller::{SlackController, StepStatus, StepView};
use crate::session::events::EventPayload;
use crate::session::SessionEvent;

/// Keeps a chatty tool from bloating the message; not a Slack limit.
const MAX_TEXT: usize = 1500;
/// A message takes 50 blocks; the answer and footer need two of them.
const MAX_BLOCKS: usize = 48;
/// Of Slack's 40,000-character message cap; the rest is headroom.
const MAX_CHARS: usize = 36_000;

type Status = StepStatus;

/// One thing the turn showed, in the order it happened.
enum Item {
    /// Preamble text before a call.
    Said {
        id: String,
        text: String,
    },
    Step(Step),
}

impl Item {
    fn block(&self, controller: &dyn SlackController) -> Value {
        match self {
            Item::Said { text, .. } => controller.said_block(text),
            Item::Step(step) => controller.step_block(&step.view()),
        }
    }
}

/// One unit of visible work: a tool call, or a sub-agent run.
struct Step {
    id: String,
    name: String,
    status: Status,
    started_at: chrono::DateTime<chrono::Utc>,
    took: Option<String>,
    input: Option<String>,
    output: Option<String>,
}

impl Step {
    /// The folded facts, for the controller to render.
    fn view(&self) -> StepView<'_> {
        StepView {
            id: &self.id,
            name: &self.name,
            status: self.status,
            took: self.took.as_deref(),
            input: self.input.as_deref(),
            output: self.output.as_deref(),
        }
    }
}

/// A turn's visible work, folded from the event log. Every render is
/// derived, so a replay converges.
pub(super) struct TurnActivity {
    pub(super) turn_id: String,
    items: Vec<Item>,
}

impl TurnActivity {
    /// The last turn in `events`. `open` seeds a turn already under way.
    pub(super) fn fold(events: &[SessionEvent], open: Option<String>) -> Option<Self> {
        let mut turn = open.map(|turn_id| Self {
            turn_id,
            items: Vec::new(),
        });
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
            EventPayload::ToolCallRequested(t) => {
                self.start(&t.id, t.name.clone(), Some(&t.arguments), event)
            }
            EventPayload::ToolCallCompleted(t) => self.end(&t.id, event, Ok(Some(&t.result))),
            EventPayload::ToolCallErrored(t) => self.end(&t.id, event, Err(&t.error.message)),
            EventPayload::SubAgentRequested(s) => {
                self.start(&s.id, format!("agent {}", s.agent_id), None, event)
            }
            EventPayload::SubAgentTurnCompleted(s) => {
                let data = (!s.data.is_null()).then(|| s.data.to_string());
                self.end(&s.id, event, Ok(data.as_deref()))
            }
            EventPayload::SubAgentErrored(s) => self.end(&s.id, event, Err(&s.error.message)),
            // The engine dropped the call: an interrupt, a branch it left, a
            // state it could not settle against. It is over, so the card is.
            EventPayload::CallVoided(v) => self.cancel(&v.id, event),
            // Nothing outlives the turn that asked for it. A card left running
            // on a finished message spins for as long as anyone looks at it,
            // which reads as work still going on.
            EventPayload::TurnCompleted(_) | EventPayload::SessionCancelled => self.settle(event),
            // A response with no calls is the answer, not a preamble.
            EventPayload::LlmCallCompleted(c) if !c.response.tool_calls.is_empty() => {
                if let Some(text) = c.response.content.as_deref().map(str::trim) {
                    if !text.is_empty() {
                        self.say(&c.id, clip(text, MAX_TEXT));
                    }
                }
            }
            _ => {}
        }
    }

    /// The key keeps a re-fold from saying it twice.
    fn say(&mut self, id: &str, text: String) {
        let said = self.items.iter_mut().find_map(|item| match item {
            Item::Said { id: said, text } if said == id => Some(text),
            _ => None,
        });
        if let Some(said) = said {
            *said = text;
            return;
        }
        self.items.push(Item::Said {
            id: id.to_string(),
            text,
        });
    }

    fn step(&mut self, id: &str) -> Option<&mut Step> {
        self.items.iter_mut().find_map(|item| match item {
            Item::Step(step) if step.id == id => Some(step),
            _ => None,
        })
    }

    fn start(&mut self, id: &str, name: String, input: Option<&str>, event: &SessionEvent) {
        let input = input.map(|i| clip(&flatten(i), MAX_TEXT));
        // A retry reuses its call id: restart the card in place.
        if let Some(step) = self.step(id) {
            step.status = Status::InProgress;
            step.started_at = event.occurred_at;
            step.took = None;
            step.output = None;
            step.input = input;
            return;
        }
        self.items.push(Item::Step(Step {
            id: id.to_string(),
            name,
            status: Status::InProgress,
            started_at: event.occurred_at,
            took: None,
            input,
            output: None,
        }));
    }

    /// End every step still running, for the events that end all of them at
    /// once. Derived like the rest of the fold, so a replay converges on the
    /// same settled cards.
    fn settle(&mut self, event: &SessionEvent) {
        let running: Vec<String> = self
            .items
            .iter()
            .filter_map(|item| match item {
                Item::Step(step) if step.status == Status::InProgress => Some(step.id.clone()),
                _ => None,
            })
            .collect();
        for id in running {
            self.cancel(&id, event);
        }
    }

    /// End one step with no result. The card carries the fact; the wording is
    /// the controller's — see [`Status::Cancelled`].
    fn cancel(&mut self, id: &str, event: &SessionEvent) {
        let Some(step) = self.step(id) else {
            return;
        };
        if step.status != Status::InProgress {
            return;
        }
        step.status = Status::Cancelled;
        step.took = Some(elapsed(step.started_at, event.occurred_at));
    }

    fn end(&mut self, id: &str, event: &SessionEvent, result: Result<Option<&str>, &str>) {
        let Some(step) = self.step(id) else {
            return;
        };
        step.took = Some(elapsed(step.started_at, event.occurred_at));
        let output = match result {
            Ok(result) => {
                step.status = Status::Complete;
                result
            }
            Err(error) => {
                step.status = Status::Error;
                Some(error)
            }
        };
        step.output = output.map(|o| clip(&flatten(o), MAX_TEXT));
    }

    /// One chunk per item, in order. The sender dedupes on the key;
    /// appended text cannot be set again the way a card can.
    pub(super) fn chunks(&self, controller: &dyn SlackController) -> Vec<(String, Value)> {
        self.items
            .iter()
            .map(|item| match item {
                Item::Said { id, text } => (
                    format!("say:{id}"),
                    serde_json::json!({ "type": "markdown_text", "text": text }),
                ),
                Item::Step(step) => (step.id.clone(), controller.step_chunk(&step.view())),
            })
            .collect()
    }

    /// The finished turn as blocks. Replaces the streamed content: the turn's
    /// visible work; what surrounds it is the controller's business.
    pub(super) fn blocks(&self, controller: &dyn SlackController) -> Vec<Value> {
        self.item_blocks(controller)
    }

    /// Drop the oldest items until the message fits; one line stands for them.
    ///
    /// Measured on the rendered blocks, not estimated from the fields behind
    /// them: a controller may render a step any way it likes, and what Slack
    /// counts is what we send.
    fn item_blocks(&self, controller: &dyn SlackController) -> Vec<Value> {
        let rendered: Vec<Value> = self.items.iter().map(|i| i.block(controller)).collect();
        let mut earlier = 0;
        while earlier < rendered.len() && !fits(&rendered[earlier..], earlier > 0) {
            earlier += 1;
        }
        let shown = rendered[earlier..].iter().cloned();
        if earlier == 0 {
            return shown.collect();
        }
        std::iter::once(controller.elided_block(earlier))
            .chain(shown)
            .collect()
    }
}

/// True if `shown` fits Slack's block and character limits.
fn fits(shown: &[Value], elided: bool) -> bool {
    let blocks = usize::from(elided) + shown.len();
    let chars: usize = shown.iter().map(|b| b.to_string().chars().count()).sum();
    blocks <= MAX_BLOCKS && chars <= MAX_CHARS
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
    use crate::protocol::ErrorInfo;
    use crate::session::events::{
        SubAgentRequested, SubAgentTurnCompleted, ToolCallCompleted, ToolCallErrored,
        ToolCallRequested, TurnStarted,
    };

    /// An LLM response saying `content`, with a call to make when `calling`.
    fn said(id: &str, content: &str, calling: bool, seq: u64) -> SessionEvent {
        use crate::protocol::{LlmResponse, ToolCall, ToolCallFunction};
        event(
            seq,
            0,
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

    fn event(seq: u64, secs: i64, payload: EventPayload) -> SessionEvent {
        let at = chrono::DateTime::from_timestamp(1_700_000_000, 0).unwrap();
        SessionEvent {
            id: uuid::Uuid::nil(),
            tenant_id: "t".into(),
            session_id: "slack:C1:1.0".into(),
            seq,
            span: crate::span::SpanContext::root(),
            occurred_at: at + chrono::Duration::seconds(secs),
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
            0,
            EventPayload::TurnStarted(TurnStarted {
                turn_id: turn.into(),
            }),
        )
    }

    fn tool_requested(id: &str, name: &str, seq: u64, secs: i64) -> SessionEvent {
        event(
            seq,
            secs,
            EventPayload::ToolCallRequested(ToolCallRequested {
                id: id.into(),
                attempt: 0,
                name: name.into(),
                arguments: "{}".into(),
                handler: Default::default(),
                target: None,
                retry: retry(),
            }),
        )
    }

    fn tool_completed(id: &str, seq: u64, secs: i64) -> SessionEvent {
        event(
            seq,
            secs,
            EventPayload::ToolCallCompleted(ToolCallCompleted {
                id: id.into(),
                name: "tool".into(),
                result: "ok".into(),
            }),
        )
    }

    #[test]
    fn a_tool_call_becomes_one_card_that_settles() {
        let events = vec![
            started("turn-1", 1),
            tool_requested("tc1", "search_web", 2, 0),
            tool_completed("tc1", 3, 1),
        ];
        let turn = TurnActivity::fold(&events, None).unwrap();
        assert_eq!(turn.turn_id, "turn-1");
        let chunks = turn.chunks(&crate::transport::slack::controller::DefaultController);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].0, "tc1");
        assert_eq!(
            chunks[0].1,
            serde_json::json!({
                "type": "task_update",
                "id": "tc1",
                "title": "search_web · 1.0s",
                "status": "complete",
            })
        );
        // The call itself arrives with the blocks, not the live card.
        let card = &turn.blocks(&crate::transport::slack::controller::DefaultController)[0];
        assert_eq!(card["type"], "task_card");
        assert_eq!(card["details"]["elements"][0]["elements"][0]["text"], "{}");
        // The result reads under a bold heading of its own.
        let out = &card["output"]["elements"][0]["elements"];
        assert_eq!(out[0]["text"], "Result:\n");
        assert_eq!(out[0]["style"]["bold"], true);
        assert_eq!(out[1]["text"], "ok");
    }

    #[test]
    fn what_the_model_said_lands_before_the_call_it_said_it_before() {
        let events = vec![
            started("turn-1", 1),
            said("llm1", "Let me look that up.", true, 2),
            tool_requested("tc1", "search_web", 3, 0),
            tool_completed("tc1", 4, 1),
        ];
        let turn = TurnActivity::fold(&events, None).unwrap();
        let chunks = turn.chunks(&crate::transport::slack::controller::DefaultController);
        assert_eq!(chunks.len(), 2);
        // Its own text, ahead of the card, keyed apart from any call id.
        assert_eq!(chunks[0].0, "say:llm1");
        assert_eq!(
            chunks[0].1,
            serde_json::json!({
                "type": "markdown_text",
                "text": "Let me look that up.",
            })
        );
        assert_eq!(chunks[1].0, "tc1");
        assert_eq!(chunks[1].1["type"], "task_update");
        // The rebuild keeps it in place, above the card.
        let blocks = turn.blocks(&crate::transport::slack::controller::DefaultController);
        assert_eq!(blocks[0]["type"], "section");
        assert_eq!(blocks[0]["text"]["text"], "Let me look that up.");
        assert_eq!(blocks[1]["type"], "task_card");
    }

    #[test]
    fn a_response_with_nothing_left_to_call_is_the_answer_not_a_preamble() {
        // The reply already carries it; saying it here would say it twice.
        let events = vec![
            started("turn-1", 1),
            said("llm1", "Here is the answer.", false, 2),
        ];
        assert!(TurnActivity::fold(&events, None)
            .unwrap()
            .chunks(&crate::transport::slack::controller::DefaultController)
            .is_empty());
    }

    #[test]
    fn a_silent_call_says_nothing() {
        let events = vec![started("turn-1", 1), said("llm1", "   ", true, 2)];
        assert!(TurnActivity::fold(&events, None)
            .unwrap()
            .chunks(&crate::transport::slack::controller::DefaultController)
            .is_empty());
    }

    #[test]
    fn an_open_turn_seeds_a_read_that_starts_past_it() {
        // The warm path reads past `turn.started`, so the turn id is carried.
        let events = vec![tool_requested("tc1", "search_web", 5, 0)];
        let turn = TurnActivity::fold(&events, Some("turn-1".into())).unwrap();
        assert_eq!(turn.turn_id, "turn-1");
        // A call still running has no duration in its title yet.
        assert_eq!(
            turn.chunks(&crate::transport::slack::controller::DefaultController)[0].1["status"],
            "in_progress"
        );
        assert_eq!(
            turn.chunks(&crate::transport::slack::controller::DefaultController)[0].1["title"],
            "search_web"
        );
        assert!(
            turn.chunks(&crate::transport::slack::controller::DefaultController)[0]
                .1
                .get("output")
                .is_none()
        );
        // Without a seed there is no turn to attach the work to.
        assert!(TurnActivity::fold(&events, None).is_none());
    }

    #[test]
    fn only_the_latest_turn_renders() {
        let events = vec![
            started("turn-1", 1),
            tool_requested("tc1", "old", 2, 0),
            tool_completed("tc1", 3, 1),
            started("turn-2", 4),
            tool_requested("tc2", "new", 5, 2),
        ];
        let turn = TurnActivity::fold(&events, Some("stale".into())).unwrap();
        assert_eq!(turn.turn_id, "turn-2");
        assert_eq!(
            turn.chunks(&crate::transport::slack::controller::DefaultController)
                .len(),
            1
        );
        assert_eq!(
            turn.chunks(&crate::transport::slack::controller::DefaultController)[0].0,
            "tc2"
        );
    }

    #[test]
    fn an_error_card_carries_its_reason_on_one_line() {
        let events = vec![
            started("turn-1", 1),
            tool_requested("tc1", "send_email", 2, 0),
            event(
                3,
                1,
                EventPayload::ToolCallErrored(ToolCallErrored {
                    id: "tc1".into(),
                    name: "send_email".into(),
                    error: ErrorInfo::handler("permission\n  denied"),
                    retryable: false,
                }),
            ),
        ];
        let turn = TurnActivity::fold(&events, None).unwrap();
        let chunks = turn.chunks(&crate::transport::slack::controller::DefaultController);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].1["status"], "error");
        assert_eq!(chunks[0].1["title"], "send_email · 1.0s");
        // The reason is the finished card's output, on one line.
        let card = &turn.blocks(&crate::transport::slack::controller::DefaultController)[0];
        assert_eq!(card["status"], "error");
        assert_eq!(
            card["output"]["elements"][0]["elements"][1]["text"],
            "permission denied"
        );
    }

    fn turn_ended(seq: u64, secs: i64, error: Option<&str>) -> SessionEvent {
        event(
            seq,
            secs,
            EventPayload::TurnCompleted(crate::session::events::TurnCompleted {
                turn_id: "turn-1".into(),
                data: serde_json::Value::Null,
                turn_cost: Default::default(),
                turn_token_usage: Default::default(),
                error: error.map(ErrorInfo::handler),
            }),
        )
    }

    /// A card still running on a message nobody will touch again spins for as
    /// long as anyone looks at it — the turn is over, and so is the call.
    #[test]
    fn a_failed_turn_settles_the_calls_it_was_still_waiting_on() {
        let events = vec![
            started("turn-1", 1),
            tool_requested("tc1", "search_web", 2, 0),
            tool_completed("tc1", 3, 1),
            tool_requested("tc2", "send_email", 4, 1),
            turn_ended(5, 4, Some("the model gave up")),
        ];
        let turn = TurnActivity::fold(&events, None).unwrap();
        let chunks = turn.chunks(&crate::transport::slack::controller::DefaultController);
        // The one that landed keeps its own outcome.
        assert_eq!(chunks[0].1["status"], "complete");
        // The one that never did is ended and timed. The card says so in the
        // engine's default wording, which is the controller's to change.
        assert_eq!(chunks[1].1["status"], "error");
        assert_eq!(chunks[1].1["title"], "send_email · 3.0s");
        let card = &turn.blocks(&crate::transport::slack::controller::DefaultController)[1];
        assert_eq!(
            card["output"]["elements"][0]["elements"][1]["text"],
            "Cancelled."
        );
    }

    /// A turn that answered is no different: nothing outlives it.
    #[test]
    fn a_turn_that_ended_well_settles_them_too() {
        let events = vec![
            started("turn-1", 1),
            tool_requested("tc1", "search_web", 2, 0),
            turn_ended(3, 2, None),
        ];
        let chunks = TurnActivity::fold(&events, None)
            .unwrap()
            .chunks(&crate::transport::slack::controller::DefaultController);
        assert_eq!(chunks[0].1["status"], "error");
    }

    /// The engine dropped the call — an interrupt, a branch it left behind.
    /// The card settles where it stopped rather than on the turn's terminal.
    #[test]
    fn a_voided_call_settles_when_it_is_dropped() {
        let events = vec![
            started("turn-1", 1),
            tool_requested("tc1", "search_web", 2, 0),
            event(
                3,
                2,
                EventPayload::CallVoided(crate::session::events::CallVoided {
                    kind: crate::protocol::EffectKind::ToolCall,
                    id: "tc1".into(),
                }),
            ),
        ];
        let turn = TurnActivity::fold(&events, None).unwrap();
        let chunks = turn.chunks(&crate::transport::slack::controller::DefaultController);
        assert_eq!(chunks[0].1["status"], "error");
        assert_eq!(chunks[0].1["title"], "search_web · 2.0s");
        let card = &turn.blocks(&crate::transport::slack::controller::DefaultController)[0];
        assert_eq!(
            card["output"]["elements"][0]["elements"][1]["text"],
            "Cancelled."
        );
    }

    /// A cancel ends the turn without completing it, so it is the cancel that
    /// has to settle the cards.
    #[test]
    fn a_cancelled_session_settles_what_was_running() {
        let events = vec![
            started("turn-1", 1),
            tool_requested("tc1", "search_web", 2, 0),
            event(3, 5, EventPayload::SessionCancelled),
        ];
        let card = &TurnActivity::fold(&events, None)
            .unwrap()
            .blocks(&crate::transport::slack::controller::DefaultController)[0];
        assert_eq!(card["status"], "error");
        assert_eq!(
            card["output"]["elements"][0]["elements"][1]["text"],
            "Cancelled."
        );
    }

    #[test]
    fn a_call_that_returned_nothing_leaves_the_field_out() {
        // Slack rejects an empty rich-text element (invalid_blocks).
        let events = vec![
            started("turn-1", 1),
            tool_requested("tc1", "send_email", 2, 0),
            event(
                3,
                1,
                EventPayload::ToolCallCompleted(ToolCallCompleted {
                    id: "tc1".into(),
                    name: "send_email".into(),
                    result: "   ".into(),
                }),
            ),
        ];
        let card = &TurnActivity::fold(&events, None)
            .unwrap()
            .blocks(&crate::transport::slack::controller::DefaultController)[0];
        assert!(card.get("output").is_none());
        assert_eq!(card["details"]["elements"][0]["elements"][0]["text"], "{}");
    }

    #[test]
    fn a_retry_restarts_its_card_in_place() {
        let events = vec![
            started("turn-1", 1),
            tool_requested("tc1", "search_web", 2, 0),
            event(
                3,
                1,
                EventPayload::ToolCallErrored(ToolCallErrored {
                    id: "tc1".into(),
                    name: "search_web".into(),
                    error: ErrorInfo::handler("rate limited"),
                    retryable: true,
                }),
            ),
            tool_requested("tc1", "search_web", 4, 2),
            tool_completed("tc1", 5, 3),
        ];
        // The retry clears the first attempt's failure.
        let chunks = TurnActivity::fold(&events, None)
            .unwrap()
            .chunks(&crate::transport::slack::controller::DefaultController);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].1["status"], "complete");
        assert_eq!(chunks[0].1["title"], "search_web · 1.0s");
    }

    #[test]
    fn a_sub_agent_is_one_card_over_its_whole_run() {
        let events = vec![
            started("turn-1", 1),
            event(
                2,
                0,
                EventPayload::SubAgentRequested(SubAgentRequested {
                    id: "sub-1".into(),
                    agent_id: "researcher".into(),
                    tool_call_id: "tc1".into(),
                    message: None,
                    retry: retry(),
                }),
            ),
            event(
                3,
                90,
                EventPayload::SubAgentTurnCompleted(SubAgentTurnCompleted {
                    id: "sub-1".into(),
                    cost: Default::default(),
                    token_usage: Default::default(),
                    data: Value::Null,
                }),
            ),
        ];
        let chunks = TurnActivity::fold(&events, None)
            .unwrap()
            .chunks(&crate::transport::slack::controller::DefaultController);
        assert_eq!(chunks[0].1["title"], "agent researcher · 1m 30s");
        // A sub-agent has no arguments of its own to preview.
        assert!(chunks[0].1.get("details").is_none());
    }

    #[test]
    fn a_turn_longer_than_the_message_stands_its_oldest_on_a_count() {
        let mut events = vec![started("turn-1", 1)];
        for i in 0..300 {
            let id = format!("tc{i}");
            events.push(tool_requested(&id, &format!("tool_{i}"), 2 + i * 2, 0));
            events.push(tool_completed(&id, 3 + i * 2, 1));
        }
        let turn = TurnActivity::fold(&events, None).unwrap();
        // Every step still streams as its own card; only the rebuild packs.
        assert_eq!(
            turn.chunks(&crate::transport::slack::controller::DefaultController)
                .len(),
            300
        );

        // One line for what did not fit, then the newest cards.
        // The activity alone; what surrounds it is the controller's.
        let blocks = turn.blocks(&crate::transport::slack::controller::DefaultController);
        assert_eq!(blocks.len(), 48);
        assert_eq!(blocks[0]["elements"][0]["text"], "_… 253 earlier steps_");
        assert_eq!(blocks[1]["title"], "tool_253 · 1.0s");
        assert_eq!(blocks[47]["title"], "tool_299 · 1.0s");
    }

    #[test]
    fn chatty_cards_run_out_of_characters_before_they_run_out_of_blocks() {
        // Characters give up the oldest cards, the same as blocks do.
        let big = "x".repeat(MAX_TEXT);
        let mut events = vec![started("turn-1", 1)];
        for i in 0..30 {
            let id = format!("tc{i}");
            events.push(tool_requested(&id, "tool", 2 + i * 2, 0));
            events.push(event(
                3 + i * 2,
                1,
                EventPayload::ToolCallCompleted(ToolCallCompleted {
                    id,
                    name: "tool".into(),
                    result: big.clone(),
                }),
            ));
        }
        let blocks = TurnActivity::fold(&events, None)
            .unwrap()
            .blocks(&crate::transport::slack::controller::DefaultController);
        // The budget is measured on the rendered blocks, which carry the card's
        // JSON as well as its text, so fewer fit than the raw output alone
        // suggests. What Slack counts is what we send.
        let elided: usize = blocks[0]["elements"][0]["text"]
            .as_str()
            .unwrap()
            .chars()
            .filter(char::is_ascii_digit)
            .collect::<String>()
            .parse()
            .unwrap();
        let shown = blocks.len() - 1; // all but the line that stands for the rest
        assert_eq!(elided + shown, 30, "every step is either shown or counted");
        let chars: usize = blocks.iter().map(|b| b.to_string().chars().count()).sum();
        assert!(
            chars <= 36_000 + 200,
            "the message stays inside the cap: {chars}"
        );
        // What survives keeps everything it carried.
        let out = &blocks[1]["output"]["elements"][0]["elements"][1]["text"];
        assert_eq!(out.as_str().unwrap().chars().count(), MAX_TEXT);
    }

    #[test]
    fn an_unmatched_completion_is_ignored() {
        let events = vec![started("turn-1", 1), tool_completed("ghost", 2, 1)];
        assert!(TurnActivity::fold(&events, None)
            .unwrap()
            .chunks(&crate::transport::slack::controller::DefaultController)
            .is_empty());
    }

    #[test]
    fn the_footer_reports_how_long_the_turn_took() {
        let at = |s: i64| chrono::DateTime::from_timestamp(1_700_000_000 + s, 0).unwrap();
        assert_eq!(elapsed(at(0), at(4)), "4.0s");
        assert_eq!(elapsed(at(0), at(90)), "1m 30s");
    }

    #[test]
    fn a_card_stays_inside_the_chunk_budget() {
        assert_eq!(
            clip(
                "short",
                crate::transport::slack::controller::MAX_TITLE_FOR_TESTS
            ),
            "short"
        );
        let long = "x".repeat(MAX_TEXT + 100);
        assert_eq!(
            clip(
                &long,
                crate::transport::slack::controller::MAX_TITLE_FOR_TESTS
            )
            .chars()
            .count(),
            crate::transport::slack::controller::MAX_TITLE_FOR_TESTS
        );
        assert!(clip(&long, MAX_TEXT).ends_with('…'));

        // A live card carries no payload; it cannot pass Slack's 256.
        let events = vec![
            started("turn-1", 1),
            tool_requested(
                &format!("toolu_{}", "0".repeat(40)),
                &"deeply_namespaced_tool".repeat(4),
                2,
                0,
            ),
        ];
        let turn = TurnActivity::fold(&events, None).unwrap();
        let rendered = turn.chunks(&crate::transport::slack::controller::DefaultController)[0]
            .1
            .to_string();
        assert!(rendered.chars().count() <= 256, "{rendered}");
    }
}
