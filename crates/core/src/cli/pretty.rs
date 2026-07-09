use std::collections::HashMap;
use std::io::{self, Write};

use crate::transport::ag_ui::events::{AgUiEvent, RunOutcome};

const RESET: &str = "\x1b[0m";
const DIM: &str = "\x1b[2m";
const CYAN: &str = "\x1b[36m";
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";

struct PendingTool {
    name: String,
    args: String,
}

/// Renders the AG-UI event stream as human-readable terminal text. Assistant
/// text streams inline; reasoning, tool calls, results, and the final outcome
/// each get their own marked block. ANSI styling is applied only when `color`
/// is set (i.e. stdout is a terminal).
pub struct PrettyPrinter {
    color: bool,
    at_line_start: bool,
    tools: HashMap<String, PendingTool>,
}

impl PrettyPrinter {
    pub fn new(color: bool) -> Self {
        Self {
            color,
            at_line_start: true,
            tools: HashMap::new(),
        }
    }

    pub fn render(&mut self, w: &mut impl Write, event: &AgUiEvent) -> io::Result<()> {
        match event {
            AgUiEvent::TextMessageContent { delta, .. } => self.write_body(w, delta)?,
            AgUiEvent::TextMessageEnd { .. } => self.break_line(w)?,

            AgUiEvent::ReasoningMessageStart { .. } => {
                self.break_line(w)?;
                self.write_styled(w, DIM, "thinking")?;
                self.newline(w)?;
                self.open(w, DIM)?;
            }
            AgUiEvent::ReasoningMessageContent { delta, .. } => self.write_body(w, delta)?,
            AgUiEvent::ReasoningEnd { .. } => {
                self.close(w)?;
                self.break_line(w)?;
            }

            AgUiEvent::ToolCallStart {
                tool_call_id,
                tool_call_name,
                ..
            } => {
                self.tools.insert(
                    tool_call_id.clone(),
                    PendingTool {
                        name: tool_call_name.clone(),
                        args: String::new(),
                    },
                );
            }
            AgUiEvent::ToolCallArgs {
                tool_call_id,
                delta,
            } => {
                if let Some(tool) = self.tools.get_mut(tool_call_id) {
                    tool.args.push_str(delta);
                }
            }
            AgUiEvent::ToolCallEnd { tool_call_id } => {
                if let Some(tool) = self.tools.remove(tool_call_id) {
                    self.break_line(w)?;
                    let args = compact_args(&tool.args);
                    self.write_styled(w, CYAN, &format!("→ {}", tool.name))?;
                    self.write_body(w, &format!(" {args}"))?;
                    self.newline(w)?;
                }
            }
            AgUiEvent::ToolCallResult { content, .. } => {
                self.break_line(w)?;
                self.open(w, DIM)?;
                self.write_body(w, &indent(&format_result(content)))?;
                self.close(w)?;
                self.break_line(w)?;
            }

            AgUiEvent::RunFinished {
                outcome: Some(RunOutcome::Interrupt { interrupts }),
                ..
            } => {
                for it in interrupts {
                    self.break_line(w)?;
                    let msg = it.message.as_deref().unwrap_or("(no message)");
                    self.write_styled(w, YELLOW, &format!("⚠ interrupt [{}]: {msg}", it.reason))?;
                    self.newline(w)?;
                }
            }
            AgUiEvent::RunError { message } => {
                self.break_line(w)?;
                self.write_styled(w, RED, &format!("✗ error: {message}"))?;
                self.newline(w)?;
            }

            _ => {}
        }
        w.flush()
    }

    /// Body text that streams verbatim; tracks whether the cursor is at the
    /// start of a line so block markers land on their own line.
    fn write_body(&mut self, w: &mut impl Write, text: &str) -> io::Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        w.write_all(text.as_bytes())?;
        self.at_line_start = text.ends_with('\n');
        Ok(())
    }

    fn write_styled(&mut self, w: &mut impl Write, style: &str, text: &str) -> io::Result<()> {
        if self.color {
            self.write_body(w, &format!("{style}{text}{RESET}"))
        } else {
            self.write_body(w, text)
        }
    }

    fn open(&mut self, w: &mut impl Write, style: &str) -> io::Result<()> {
        if self.color {
            w.write_all(style.as_bytes())?;
        }
        Ok(())
    }

    fn close(&mut self, w: &mut impl Write) -> io::Result<()> {
        if self.color {
            w.write_all(RESET.as_bytes())?;
        }
        Ok(())
    }

    fn newline(&mut self, w: &mut impl Write) -> io::Result<()> {
        w.write_all(b"\n")?;
        self.at_line_start = true;
        Ok(())
    }

    /// A single newline only if we're mid-line, so blocks separate without piling
    /// up blank lines.
    fn break_line(&mut self, w: &mut impl Write) -> io::Result<()> {
        if !self.at_line_start {
            self.newline(w)?;
        }
        Ok(())
    }
}

/// Collapse streamed tool arguments to one line; empty args read as `{}`.
fn compact_args(raw: &str) -> String {
    if raw.trim().is_empty() {
        return "{}".to_string();
    }
    match serde_json::from_str::<serde_json::Value>(raw) {
        Ok(value) => serde_json::to_string(&value).unwrap_or_else(|_| raw.to_string()),
        Err(_) => raw.to_string(),
    }
}

/// Tool results are usually JSON; pretty-print when they parse, else pass through.
fn format_result(content: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(content) {
        Ok(value) => serde_json::to_string_pretty(&value).unwrap_or_else(|_| content.to_string()),
        Err(_) => content.to_string(),
    }
}

fn indent(text: &str) -> String {
    text.lines()
        .map(|line| format!("  {line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render_all(events: Vec<AgUiEvent>) -> String {
        let mut printer = PrettyPrinter::new(false);
        let mut buf: Vec<u8> = Vec::new();
        for ev in &events {
            printer.render(&mut buf, ev).unwrap();
        }
        String::from_utf8(buf).unwrap()
    }

    fn text(id: &str, delta: &str) -> AgUiEvent {
        AgUiEvent::TextMessageContent {
            message_id: id.into(),
            delta: delta.into(),
        }
    }

    #[test]
    fn streams_text_then_breaks_line() {
        let out = render_all(vec![
            AgUiEvent::TextMessageStart {
                message_id: "c1".into(),
                role: "assistant",
            },
            text("c1", "Hel"),
            text("c1", "lo"),
            AgUiEvent::TextMessageEnd {
                message_id: "c1".into(),
            },
        ]);
        assert_eq!(out, "Hello\n");
    }

    #[test]
    fn tool_call_accumulates_args_onto_one_line() {
        let out = render_all(vec![
            AgUiEvent::ToolCallStart {
                tool_call_id: "x".into(),
                tool_call_name: "get_weather".into(),
                parent_message_id: None,
            },
            AgUiEvent::ToolCallArgs {
                tool_call_id: "x".into(),
                delta: "{\"city\":".into(),
            },
            AgUiEvent::ToolCallArgs {
                tool_call_id: "x".into(),
                delta: "\"SF\"}".into(),
            },
            AgUiEvent::ToolCallEnd {
                tool_call_id: "x".into(),
            },
        ]);
        assert_eq!(out, "→ get_weather {\"city\":\"SF\"}\n");
    }

    #[test]
    fn empty_args_render_as_object() {
        let out = render_all(vec![
            AgUiEvent::ToolCallStart {
                tool_call_id: "x".into(),
                tool_call_name: "now".into(),
                parent_message_id: None,
            },
            AgUiEvent::ToolCallEnd {
                tool_call_id: "x".into(),
            },
        ]);
        assert_eq!(out, "→ now {}\n");
    }

    #[test]
    fn reasoning_gets_a_header_and_breaks_after() {
        let out = render_all(vec![
            AgUiEvent::ReasoningMessageStart {
                message_id: "c1-reasoning".into(),
                role: "reasoning",
            },
            AgUiEvent::ReasoningMessageContent {
                message_id: "c1-reasoning".into(),
                delta: "hmm".into(),
            },
            AgUiEvent::ReasoningEnd {
                message_id: "c1-reasoning".into(),
            },
        ]);
        assert_eq!(out, "thinking\nhmm\n");
    }

    #[test]
    fn tool_result_is_pretty_printed_and_indented() {
        let out = render_all(vec![AgUiEvent::ToolCallResult {
            message_id: "x".into(),
            tool_call_id: "x".into(),
            content: r#"{"temp":62}"#.into(),
            role: "tool",
        }]);
        assert_eq!(out, "  {\n    \"temp\": 62\n  }\n");
    }

    #[test]
    fn run_finished_does_not_print_the_result() {
        let out = render_all(vec![AgUiEvent::RunFinished {
            thread_id: "t".into(),
            run_id: "r".into(),
            result: Some(serde_json::json!({"answer": 42})),
            outcome: None,
        }]);
        assert_eq!(out, "");
    }

    #[test]
    fn interrupt_outcome_is_surfaced() {
        let out = render_all(vec![AgUiEvent::RunFinished {
            thread_id: "t".into(),
            run_id: "r".into(),
            result: None,
            outcome: Some(RunOutcome::Interrupt {
                interrupts: vec![crate::transport::ag_ui::events::AgUiInterrupt {
                    id: "int-1".into(),
                    reason: "confirmation".into(),
                    message: Some("Send the email?".into()),
                    tool_call_id: None,
                    response_schema: None,
                    expires_at: None,
                    metadata: None,
                }],
            }),
        }]);
        assert_eq!(out, "⚠ interrupt [confirmation]: Send the email?\n");
    }

    #[test]
    fn error_is_rendered() {
        let out = render_all(vec![AgUiEvent::RunError {
            message: "boom".into(),
        }]);
        assert_eq!(out, "✗ error: boom\n");
    }

    #[test]
    fn text_then_tool_call_separates_cleanly() {
        let out = render_all(vec![
            AgUiEvent::TextMessageStart {
                message_id: "c1".into(),
                role: "assistant",
            },
            text("c1", "Let me check."),
            AgUiEvent::TextMessageEnd {
                message_id: "c1".into(),
            },
            AgUiEvent::ToolCallStart {
                tool_call_id: "x".into(),
                tool_call_name: "get_weather".into(),
                parent_message_id: None,
            },
            AgUiEvent::ToolCallEnd {
                tool_call_id: "x".into(),
            },
        ]);
        assert_eq!(out, "Let me check.\n→ get_weather {}\n");
    }
}
