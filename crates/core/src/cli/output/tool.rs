//! How a tool call and its result read.

use std::time::{Duration, Instant};

use super::term::{paint, BOLD, DIM};

pub(super) const RESULT_LINES: usize = 12;
const SLOW: Duration = Duration::from_millis(100);

pub(super) struct PendingTool {
    pub name: String,
    pub args: String,
    pub started: Instant,
    pub attempt: u32,
}

impl PendingTool {
    pub(super) fn new(name: String) -> Self {
        Self {
            name,
            args: String::new(),
            started: Instant::now(),
            attempt: 0,
        }
    }

    pub(super) fn about(&self) -> Option<String> {
        (self.attempt > 0).then(|| format!("attempt {}", self.attempt + 1))
    }
}

/// The name reads first and the arguments recede: what a tool was asked is
/// worth less of the eye than which tool answered.
pub(super) fn head_of(tool: &PendingTool, is_error: bool, again: bool, color: bool) -> String {
    let glyph = match (is_error, again) {
        (_, true) => "↻",
        (true, false) => "✗",
        (false, false) => "●",
    };
    let mut head = paint(BOLD, &format!("{glyph} {}", tool.name), color);

    let args = compact_args(&tool.args);
    if !args.is_empty() {
        head.push(' ');
        head.push_str(&paint(DIM, &args, color));
    }

    let mut about = Vec::new();
    if again || tool.attempt > 1 {
        about.push(format!("attempt {}", tool.attempt));
    }
    if let Some(took) = duration(tool.started.elapsed()) {
        about.push(took);
    }
    if !about.is_empty() {
        head.push(' ');
        head.push_str(&paint(DIM, &format!("({})", about.join(", ")), color));
    }
    head
}

fn duration(took: Duration) -> Option<String> {
    if took < SLOW {
        return None;
    }
    let secs = took.as_secs();
    Some(match secs {
        0 => format!("{}ms", took.as_millis()),
        1..=59 => format!("{:.1}s", took.as_secs_f64()),
        _ => format!("{}m {:02}s", secs / 60, secs % 60),
    })
}

const ARG_WIDTH: usize = 72;

/// Verbatim, only shorter. Nothing here knows what any tool's arguments mean,
/// so none of them are picked over the others.
fn compact_args(raw: &str) -> String {
    if raw.trim().is_empty() {
        return String::new();
    }
    let compact = match serde_json::from_str::<serde_json::Value>(raw) {
        Ok(serde_json::Value::Object(map)) if map.is_empty() => return String::new(),
        Ok(value) => serde_json::to_string(&value).unwrap_or_else(|_| raw.to_string()),
        Err(_) => raw.to_string(),
    };
    let compact = compact.replace('\n', " ");
    match compact.chars().count() > ARG_WIDTH {
        true => format!(
            "{}…",
            compact.chars().take(ARG_WIDTH - 1).collect::<String>()
        ),
        false => compact,
    }
}

pub(super) fn format_result(content: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(content) {
        Ok(value) => serde_json::to_string_pretty(&value).unwrap_or_else(|_| content.to_string()),
        Err(_) => content.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(name: &str, args: &str, attempt: u32) -> PendingTool {
        PendingTool {
            name: name.into(),
            args: args.into(),
            started: Instant::now(),
            attempt,
        }
    }

    #[test]
    fn a_quick_call_shows_no_duration_and_a_slow_one_does() {
        assert_eq!(duration(Duration::from_millis(9)), None);
        assert_eq!(duration(Duration::from_millis(247)).unwrap(), "247ms");
        assert_eq!(duration(Duration::from_millis(1250)).unwrap(), "1.2s");
        assert_eq!(duration(Duration::from_secs(75)).unwrap(), "1m 15s");
    }

    /// Nothing here knows what a tool's arguments mean, so they are shown as
    /// written rather than picked over.
    #[test]
    fn arguments_are_shown_as_written() {
        assert_eq!(compact_args(""), "");
        assert_eq!(compact_args("{}"), "");
        assert_eq!(
            compact_args(r#"{"path": "src/lib.rs"}"#),
            r#"{"path":"src/lib.rs"}"#
        );
        assert_eq!(compact_args("not json"), "not json");
    }

    #[test]
    fn arguments_are_read_on_one_line_and_clipped_when_long() {
        // JSON escapes a newline, so a serialised argument is one line already.
        assert_eq!(compact_args("{\"cmd\":\"a\\nb\"}"), r#"{"cmd":"a\nb"}"#);
        assert_eq!(compact_args("raw\nlines"), "raw lines");
        let long = compact_args(&format!(r#"{{"q":"{}"}}"#, "x".repeat(200)));
        assert_eq!(long.chars().count(), ARG_WIDTH);
        assert!(long.ends_with('…'), "got {long}");
    }

    #[test]
    fn a_settled_call_is_marked_by_how_it_ended() {
        assert_eq!(
            head_of(&tool("read_file", "", 1), false, false, false),
            "● read_file"
        );
        assert_eq!(
            head_of(&tool("read_file", "", 1), true, false, false),
            "✗ read_file"
        );
    }

    #[test]
    fn attempts_are_counted_in_brackets() {
        assert_eq!(
            head_of(&tool("fetch_url", "", 1), true, true, false),
            "↻ fetch_url (attempt 1)"
        );
        assert_eq!(
            head_of(&tool("fetch_url", "", 3), false, false, false),
            "● fetch_url (attempt 3)"
        );
    }

    #[test]
    fn args_tell_same_named_calls_apart() {
        assert_eq!(
            head_of(
                &tool("get_weather", r#"{"city":"Paris"}"#, 1),
                false,
                false,
                false
            ),
            r#"● get_weather {"city":"Paris"}"#
        );
    }

    #[test]
    fn a_running_call_says_which_try_it_is_on() {
        assert_eq!(tool("fetch_url", "", 0).about(), None);
        assert_eq!(tool("fetch_url", "", 1).about().unwrap(), "attempt 2");
    }
}
