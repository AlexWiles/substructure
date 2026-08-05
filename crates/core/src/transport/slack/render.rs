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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StepStatus {
    InProgress,
    Complete,
    Error,
    /// Ended with no result.
    Cancelled,
}

impl StepStatus {
    pub fn wire(self) -> &'static str {
        match self {
            StepStatus::InProgress => "in_progress",
            StepStatus::Complete => "complete",
            StepStatus::Error | StepStatus::Cancelled => "error",
        }
    }
}

/// One unit of visible work: a tool call or a sub-agent run.
#[derive(Debug, Clone)]
pub struct StepView<'a> {
    pub id: &'a str,
    pub name: &'a str,
    pub status: StepStatus,
    pub took: Option<&'a str>,
    pub input: Option<&'a str>,
    pub output: Option<&'a str>,
}

impl StepView<'_> {
    pub fn title(&self) -> String {
        match self.took {
            Some(took) => format!("{} · {took}", self.name),
            None => self.name.to_string(),
        }
    }
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

/// The message a cancelled run leaves: its work, and a line saying it stopped.
pub fn render_cancelled(activity: &[Value]) -> Rendered {
    let mut blocks = activity.to_vec();
    blocks.push(section_block(CANCELLED));
    Rendered {
        text: CANCELLED.to_string(),
        blocks,
    }
}

/// A settled unit of work, in the finished message.
pub fn step_block(step: &StepView<'_>) -> Value {
    let mut card = serde_json::json!({
        "type": "task_card",
        "task_id": step.id,
        "title": clip(&step.title(), MAX_TITLE),
        "status": step.status.wire(),
    });
    let outcome = step.output.or(match step.status {
        StepStatus::Cancelled => Some(CANCELLED),
        _ => None,
    });
    for (field, heading, text) in [
        ("details", None, step.input),
        ("output", Some("Result:"), outcome),
    ] {
        // Slack rejects an empty text element.
        if let Some(text) = text.filter(|t| !t.trim().is_empty()) {
            card[field] = rich_text(heading, text);
        }
    }
    card
}

/// What the model said between calls.
pub fn said_block(text: &str) -> Value {
    section_block(text)
}

/// Stands for the steps dropped to fit Slack's message limits.
pub fn elided_block(count: usize) -> Value {
    context_block(&format!("_… {count} earlier steps_"))
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
pub const WORKING_STATUS: &str = "is thinking…";

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

    fn step() -> StepView<'static> {
        StepView {
            id: "tc-1",
            name: "search",
            status: StepStatus::Complete,
            took: Some("1.2s"),
            input: Some("{\"q\":\"x\"}"),
            output: Some("found"),
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
    fn a_step_card_carries_its_title_and_both_fields() {
        let card = step_block(&step());
        assert_eq!(card["type"], "task_card");
        assert_eq!(card["title"], "search · 1.2s");
        assert_eq!(card["status"], "complete");
        assert!(card["details"]["type"] == "rich_text");
        assert!(card["output"]["type"] == "rich_text");
    }

    #[test]
    fn a_step_with_nothing_to_show_leaves_the_field_out() {
        let mut s = step();
        s.output = Some("   ");
        s.input = None;
        let card = step_block(&s);
        assert!(card.get("details").is_none());
        assert!(card.get("output").is_none(), "blank is not output");
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
    fn work_that_ended_with_no_result_is_a_status_not_a_sentence() {
        let ended = StepView {
            status: StepStatus::Cancelled,
            output: None,
            took: Some("2.0s"),
            ..step()
        };
        let card = step_block(&ended);
        assert_eq!(card["status"], "error");
        assert_eq!(
            card["output"]["elements"][0]["elements"][1]["text"],
            "Cancelled."
        );
    }

    #[test]
    fn a_cancel_keeps_the_work_and_says_it_stopped() {
        let m = render_cancelled(&[section_block("a card")]);
        assert_eq!(m.text, "Cancelled.");
        assert_eq!(m.blocks.len(), 2);
        assert_eq!(m.blocks[1]["text"]["text"], "Cancelled.");
    }

    #[test]
    fn elapsed_crosses_into_minutes() {
        assert_eq!(format_elapsed(Duration::from_millis(950)), "0.9s");
        assert_eq!(format_elapsed(Duration::from_secs(75)), "1m 15s");
    }
}
