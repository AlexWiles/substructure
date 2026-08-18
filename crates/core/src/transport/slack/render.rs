//! Slack blocks. What a message says when no decision gave it a view.

use std::time::Duration;

use serde_json::Value;

use super::{clip, MAX_SECTION};
use crate::session::events::TurnCompleted;

/// Keeps a chatty tool from bloating a card; not a Slack limit.
const MAX_TITLE: usize = 60;
const CANCELLED: &str = "Cancelled.";

/// One Slack message: `text` is the notification, `blocks` the display.
#[derive(Debug, Clone, Default)]
pub struct Rendered {
    pub text: String,
    pub blocks: Vec<Value>,
}

/// One answer a user can click. The bot reads `action_id` and `value` back
/// out of the click, so render them unchanged.
#[derive(Debug, Clone)]
pub struct PromptOption {
    pub label: String,
    /// `primary` or `danger`, when the author asked for one.
    pub style: Option<String>,
    pub action_id: String,
    pub value: String,
}

impl PromptOption {
    pub fn button(&self) -> Value {
        let mut button = serde_json::json!({
            "type": "button",
            "action_id": self.action_id,
            "text": { "type": "plain_text", "text": self.label },
            "value": self.value,
        });
        if let Some(style @ ("primary" | "danger")) = self.style.as_deref() {
            button["style"] = style.into();
        }
        button
    }
}

/// A question the session is parked on.
#[derive(Debug, Clone)]
pub struct PromptView<'a> {
    pub message: &'a str,
    pub options: &'a [PromptOption],
    /// RFC 3339, as the author wrote it.
    pub expires_at: Option<&'a str>,
}

/// What a completed turn says. A failure reads as its error.
pub fn render_turn(turn: &TurnCompleted, elapsed: Option<Duration>) -> Rendered {
    let answer = match (&turn.error, &turn.data) {
        (Some(error), _) => format!("Error: {error}"),
        (None, Value::Null) => "(no result)".to_string(),
        (None, Value::String(s)) => s.clone(),
        (None, other) => other.to_string(),
    };
    let footer = elapsed.map(format_elapsed);
    let mut blocks = vec![section_block(&answer)];
    if let Some(footer) = &footer {
        blocks.push(context_block(footer));
    }
    Rendered {
        text: super::with_footer(&answer, footer.as_deref()),
        blocks,
    }
}

/// The message a cancelled run leaves: its work settled, and a line saying
/// it stopped.
pub fn render_cancelled(activity: &[Value]) -> Rendered {
    let mut blocks = settle_cards(activity);
    blocks.push(section_block(CANCELLED));
    Rendered {
        text: CANCELLED.to_string(),
        blocks,
    }
}

/// The blocks with every running card ended. A finished message must not
/// show a spinner: it would read as work still going on.
pub fn settle_cards(blocks: &[Value]) -> Vec<Value> {
    blocks
        .iter()
        .cloned()
        .map(|mut block| {
            if block["type"] == "task_card" && block["status"] == "in_progress" {
                block["status"] = "error".into();
            }
            block
        })
        .collect()
}

/// The turn's one card. The same id every render, so a stream replaces it
/// in place.
pub fn turn_card(id: &str, title: &str, status: &str, details: Option<&str>) -> Value {
    let mut card = serde_json::json!({
        "type": "task_card",
        "task_id": id,
        "title": clip(title, MAX_TITLE),
        "status": status,
    });
    // Slack rejects an empty text element.
    if let Some(details) = details.filter(|d| !d.trim().is_empty()) {
        card["details"] = rich_text(None, details);
    }
    card
}

/// A prompt the session is waiting on.
pub fn prompt_blocks(prompt: &PromptView<'_>) -> Vec<Value> {
    let mut blocks = vec![section_block(prompt.message)];
    if !prompt.options.is_empty() {
        let buttons: Vec<Value> = prompt.options.iter().map(PromptOption::button).collect();
        blocks.push(serde_json::json!({ "type": "actions", "elements": buttons }));
    }
    if let Some((raw, ts)) = prompt
        .expires_at
        .and_then(|e| Some((e, chrono::DateTime::parse_from_rfc3339(e).ok()?)))
    {
        blocks.push(context_block(&format!(
            "Expires <!date^{}^{{date_short_pretty}} {{time}}|{raw}>",
            ts.timestamp()
        )));
    }
    blocks
}

/// The same prompt, answered: no buttons, and the outcome written under it.
pub fn settled_prompt_blocks(posted: &[Value], text: &str, resolution: &str) -> Vec<Value> {
    let mut settled: Vec<Value> = posted
        .iter()
        .filter(|b| b["type"] != "actions")
        .cloned()
        .collect();
    if settled.is_empty() {
        settled.push(section_block(text));
    }
    settled.push(context_block(resolution));
    settled
}

/// The thread status while a turn runs.
pub const WORKING_STATUS: &str = "is typing…";

pub fn section_block(text: &str) -> Value {
    serde_json::json!({
        "type": "section",
        "text": { "type": "mrkdwn", "text": clip(text, MAX_SECTION) },
    })
}

pub fn context_block(text: &str) -> Value {
    serde_json::json!({
        "type": "context",
        "elements": [{ "type": "mrkdwn", "text": text }],
    })
}

pub fn rich_text(heading: Option<&str>, text: &str) -> Value {
    let mut elements = Vec::new();
    if let Some(heading) = heading {
        elements.push(serde_json::json!({
            "type": "text",
            "text": format!("{heading}\n"),
            "style": { "bold": true },
        }));
    }
    elements.push(serde_json::json!({ "type": "text", "text": text }));
    serde_json::json!({
        "type": "rich_text",
        "elements": [{ "type": "rich_text_section", "elements": elements }],
    })
}

pub fn format_elapsed(d: Duration) -> String {
    let ms = d.as_millis() as i64;
    if ms < 60_000 {
        return format!("{:.1}s", ms as f64 / 1000.0);
    }
    format!("{}m {:02}s", ms / 60_000, (ms % 60_000) / 1000)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{ErrorCode, ErrorInfo};

    fn completed(data: Value) -> TurnCompleted {
        TurnCompleted {
            turn_id: "turn-1".into(),
            data,
            turn_cost: Default::default(),
            turn_token_usage: Default::default(),
            error: None,
        }
    }

    #[test]
    fn a_turn_renders_its_output_with_the_elapsed_time() {
        let p = render_turn(
            &completed(Value::String("done".into())),
            Some(Duration::from_millis(1500)),
        );
        assert_eq!(p.text, "done\n\n_1.5s_");
        assert_eq!(p.blocks.len(), 2, "the answer and its footer");
        assert_eq!(p.blocks[0]["text"]["text"], "done");
        assert_eq!(p.blocks[1]["type"], "context");
    }

    #[test]
    fn a_non_string_result_renders_as_json() {
        let p = render_turn(&completed(serde_json::json!({"a": 1})), None);
        assert!(p.text.starts_with(r#"{"a":1}"#), "{}", p.text);
    }

    #[test]
    fn an_error_reads_as_its_sentence_not_its_code() {
        let mut t = completed(Value::Null);
        t.error = Some(
            ErrorInfo::new(ErrorCode::BudgetExceeded, "the monthly budget is spent")
                .with_param("agent.llm"),
        );
        let p = render_turn(&t, None);
        assert!(
            p.text.starts_with("Error: the monthly budget is spent"),
            "{}",
            p.text
        );
        assert!(
            !p.text.contains("agent.llm") && !p.text.contains("budget_exceeded"),
            "the machine-readable half stays out of the message; got {}",
            p.text
        );
    }

    #[test]
    fn a_turn_card_carries_its_title_and_details() {
        let card = turn_card(
            "turn-1",
            "Find x",
            "in_progress",
            Some("Ran 1 action — search"),
        );
        assert_eq!(card["type"], "task_card");
        assert_eq!(card["task_id"], "turn-1");
        assert_eq!(card["title"], "Find x");
        assert_eq!(card["status"], "in_progress");
        assert_eq!(card["details"]["type"], "rich_text");
        assert_eq!(
            card["details"]["elements"][0]["elements"][0]["text"],
            "Ran 1 action — search"
        );
    }

    #[test]
    fn a_card_with_nothing_to_say_leaves_details_out() {
        let card = turn_card("turn-1", "Find x", "complete", Some("   "));
        assert!(card.get("details").is_none());
        assert!(turn_card("turn-1", "t", "complete", None)
            .get("details")
            .is_none());
    }

    #[test]
    fn a_long_title_is_clipped() {
        let card = turn_card("turn-1", &"x".repeat(100), "in_progress", None);
        assert_eq!(card["title"].as_str().unwrap().chars().count(), 60);
    }

    #[test]
    fn settling_ends_only_the_running_cards() {
        let blocks = vec![
            turn_card("turn-1", "t", "in_progress", None),
            turn_card("turn-2", "t", "complete", None),
            section_block("text"),
        ];
        let settled = settle_cards(&blocks);
        assert_eq!(settled[0]["status"], "error");
        assert_eq!(settled[1]["status"], "complete");
        assert_eq!(settled[2]["type"], "section");
    }

    #[test]
    fn a_prompt_button_carries_the_bots_coordinates() {
        let options = vec![PromptOption {
            label: "Approve".into(),
            style: Some("primary".into()),
            action_id: "prompt_option_0".into(),
            value: r#"{"interrupt_id":"i-1","option":0}"#.into(),
        }];
        let blocks = prompt_blocks(&PromptView {
            message: "Run it?",
            options: &options,
            expires_at: None,
        });
        let button = &blocks[1]["elements"][0];
        assert_eq!(button["action_id"], "prompt_option_0");
        assert_eq!(button["value"], r#"{"interrupt_id":"i-1","option":0}"#);
        assert_eq!(button["style"], "primary");
    }

    #[test]
    fn settling_a_prompt_drops_its_buttons_and_says_the_outcome() {
        let posted = vec![
            section_block("Run it?"),
            serde_json::json!({ "type": "actions", "elements": [] }),
        ];
        let settled = settled_prompt_blocks(&posted, "Run it?", "Approved");
        assert!(settled.iter().all(|b| b["type"] != "actions"));
        assert_eq!(settled.last().unwrap()["elements"][0]["text"], "Approved");
    }

    #[test]
    fn a_cancel_settles_the_work_and_says_it_stopped() {
        let m = render_cancelled(&[turn_card("turn-1", "t", "in_progress", None)]);
        assert_eq!(m.text, "Cancelled.");
        assert_eq!(m.blocks.len(), 2);
        assert_eq!(m.blocks[0]["status"], "error", "no spinner at rest");
        assert_eq!(m.blocks[1]["text"]["text"], "Cancelled.");
    }

    #[test]
    fn elapsed_crosses_into_minutes() {
        assert_eq!(format_elapsed(Duration::from_millis(950)), "0.9s");
        assert_eq!(format_elapsed(Duration::from_secs(75)), "1m 15s");
    }
}
