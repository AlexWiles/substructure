//! How a tool call and its result read.

use std::time::{Duration, Instant};

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

pub(super) fn head_of(tool: &PendingTool, is_error: bool, again: bool) -> String {
    let glyph = match (is_error, again) {
        (_, true) => "↻",
        (true, false) => "✗",
        (false, false) => "●",
    };
    let args = compact_args(&tool.args);
    let head = match args.as_str() {
        "{}" => format!("{glyph} {}", tool.name),
        _ => format!("{glyph} {} {args}", tool.name),
    };
    let mut about = Vec::new();
    if again || tool.attempt > 1 {
        about.push(format!("attempt {}", tool.attempt));
    }
    if let Some(took) = duration(tool.started.elapsed()) {
        about.push(took);
    }
    match about.is_empty() {
        true => head,
        false => format!("{head} ({})", about.join(", ")),
    }
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

fn compact_args(raw: &str) -> String {
    if raw.trim().is_empty() {
        return "{}".to_string();
    }
    match serde_json::from_str::<serde_json::Value>(raw) {
        Ok(value) => serde_json::to_string(&value).unwrap_or_else(|_| raw.to_string()),
        Err(_) => raw.to_string(),
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

    #[test]
    fn empty_args_read_as_an_object() {
        assert_eq!(compact_args(""), "{}");
        assert_eq!(compact_args(r#"{"city": "SF"}"#), r#"{"city":"SF"}"#);
        assert_eq!(compact_args("not json"), "not json");
    }

    #[test]
    fn a_settled_call_is_marked_by_how_it_ended() {
        assert_eq!(
            head_of(&tool("read_file", "", 1), false, false),
            "● read_file"
        );
        assert_eq!(
            head_of(&tool("read_file", "", 1), true, false),
            "✗ read_file"
        );
    }

    #[test]
    fn attempts_are_counted_in_brackets() {
        assert_eq!(
            head_of(&tool("fetch_url", "", 1), true, true),
            "↻ fetch_url (attempt 1)"
        );
        assert_eq!(
            head_of(&tool("fetch_url", "", 3), false, false),
            "● fetch_url (attempt 3)"
        );
    }

    #[test]
    fn args_tell_same_named_calls_apart() {
        assert_eq!(
            head_of(&tool("get_weather", r#"{"city":"Paris"}"#, 1), false, false),
            r#"● get_weather {"city":"Paris"}"#
        );
    }

    #[test]
    fn a_running_call_says_which_try_it_is_on() {
        assert_eq!(tool("fetch_url", "", 0).about(), None);
        assert_eq!(tool("fetch_url", "", 1).about().unwrap(), "attempt 2");
    }
}
