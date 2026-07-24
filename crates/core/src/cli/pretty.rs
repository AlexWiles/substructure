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
                if let Some(tool) = self.tools.get(tool_call_id) {
                    let line = format!("→ {} {}", tool.name, compact_args(&tool.args));
                    self.break_line(w)?;
                    self.write_styled(w, CYAN, &line)?;
                    self.newline(w)?;
                }
            }
            AgUiEvent::ToolCallResult {
                tool_call_id,
                content,
                ..
            } => {
                self.break_line(w)?;
                // Name the call this result answers so parallel calls stay legible;
                // echo non-empty args to tell same-named calls apart.
                if let Some(tool) = self.tools.remove(tool_call_id) {
                    let args = compact_args(&tool.args);
                    let head = if args == "{}" {
                        format!("← {}", tool.name)
                    } else {
                        format!("← {} {args}", tool.name)
                    };
                    self.write_styled(w, CYAN, &head)?;
                    self.newline(w)?;
                }
                self.open(w, DIM)?;
                self.write_body(w, &indent(&format_result(content)))?;
                self.close(w)?;
                self.break_line(w)?;
            }

            AgUiEvent::RunFinished { outcome, .. } => {
                if let Some(RunOutcome::Interrupt { interrupts }) = outcome {
                    for it in interrupts {
                        self.break_line(w)?;
                        let msg = it.message.as_deref().unwrap_or("(no message)");
                        // The id is what an `interrupt.resume` input needs.
                        self.write_styled(
                            w,
                            YELLOW,
                            &format!("⚠ interrupt {} [{}]: {msg}", it.id, it.reason),
                        )?;
                        self.newline(w)?;
                    }
                }
                self.render_pending_settlements(w)?;
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

    /// A client tool yields the run with no result event, so its call stays in
    /// `tools` after every worker call and sub-agent has been removed by its
    /// result. On finish, surface each leftover call's id and a ready-to-edit
    /// settle input — `tool.result` requires the id, which the streamed `→`
    /// line alone doesn't show.
    fn render_pending_settlements(&mut self, w: &mut impl Write) -> io::Result<()> {
        let mut pending: Vec<(String, String)> = self
            .tools
            .drain()
            .map(|(id, tool)| (id, tool.name))
            .collect();
        pending.sort();
        for (id, name) in pending {
            self.break_line(w)?;
            self.write_styled(
                w,
                YELLOW,
                &format!("⧗ {name} awaiting result — settle with:"),
            )?;
            self.newline(w)?;
            let hint = format!(
                "    --input '{{\"type\":\"tool.result\",\"id\":\"{id}\",\"result\":\"...\"}}'"
            );
            self.write_styled(w, DIM, &hint)?;
            self.newline(w)?;
        }
        Ok(())
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

    fn call(id: &str, name: &str) -> Vec<AgUiEvent> {
        vec![
            AgUiEvent::ToolCallStart {
                tool_call_id: id.into(),
                tool_call_name: name.into(),
                parent_message_id: None,
            },
            AgUiEvent::ToolCallEnd {
                tool_call_id: id.into(),
            },
        ]
    }

    fn result(id: &str, content: &str) -> AgUiEvent {
        AgUiEvent::ToolCallResult {
            message_id: id.into(),
            tool_call_id: id.into(),
            content: content.into(),
            role: "tool",
        }
    }

    #[test]
    fn tool_result_is_labeled_with_its_call() {
        let mut evs = call("x", "get_current_time");
        evs.push(result("x", "2026-07-10T04:14:56.322Z"));
        let out = render_all(evs);
        assert_eq!(
            out,
            "→ get_current_time {}\n← get_current_time\n  2026-07-10T04:14:56.322Z\n"
        );
    }

    #[test]
    fn parallel_tool_results_are_each_labeled() {
        let mut evs = call("a", "get_current_time_zone");
        evs.extend(call("b", "get_current_time"));
        evs.push(result("a", "Asia/Bangkok"));
        evs.push(result("b", "2026-07-10T04:14:56.322Z"));
        let out = render_all(evs);
        assert_eq!(
            out,
            "→ get_current_time_zone {}\n→ get_current_time {}\n\
             ← get_current_time_zone\n  Asia/Bangkok\n\
             ← get_current_time\n  2026-07-10T04:14:56.322Z\n"
        );
    }

    #[test]
    fn results_pair_by_id_regardless_of_completion_order() {
        let mut evs = call("a", "first");
        evs.extend(call("b", "second"));
        // Results arrive in reverse dispatch order.
        evs.push(result("b", "B"));
        evs.push(result("a", "A"));
        let out = render_all(evs);
        assert_eq!(
            out,
            "→ first {}\n→ second {}\n← second\n  B\n← first\n  A\n"
        );
    }

    #[test]
    fn same_named_calls_are_disambiguated_by_args() {
        let out = render_all(vec![
            AgUiEvent::ToolCallStart {
                tool_call_id: "a".into(),
                tool_call_name: "get_weather".into(),
                parent_message_id: None,
            },
            AgUiEvent::ToolCallArgs {
                tool_call_id: "a".into(),
                delta: r#"{"city":"Paris"}"#.into(),
            },
            AgUiEvent::ToolCallEnd {
                tool_call_id: "a".into(),
            },
            result("a", "68"),
        ]);
        assert_eq!(
            out,
            "→ get_weather {\"city\":\"Paris\"}\n← get_weather {\"city\":\"Paris\"}\n  68\n"
        );
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
        assert_eq!(out, "⚠ interrupt int-1 [confirmation]: Send the email?\n");
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

    fn run_finished() -> AgUiEvent {
        AgUiEvent::RunFinished {
            thread_id: "t".into(),
            run_id: "r".into(),
            result: None,
            outcome: None,
        }
    }

    #[test]
    fn client_tool_yield_surfaces_id_to_settle() {
        let out = render_all(vec![
            AgUiEvent::ToolCallStart {
                tool_call_id: "call_x".into(),
                tool_call_name: "get_weather".into(),
                parent_message_id: None,
            },
            AgUiEvent::ToolCallArgs {
                tool_call_id: "call_x".into(),
                delta: r#"{"city":"SF"}"#.into(),
            },
            AgUiEvent::ToolCallEnd {
                tool_call_id: "call_x".into(),
            },
            run_finished(),
        ]);
        assert_eq!(
            out,
            "→ get_weather {\"city\":\"SF\"}\n⧗ get_weather awaiting result — settle with:\n    --input '{\"type\":\"tool.result\",\"id\":\"call_x\",\"result\":\"...\"}'\n"
        );
    }

    #[test]
    fn parallel_client_tools_each_surface_their_id() {
        let mut evs = call("call_a", "get_weather");
        evs.extend(call("call_b", "get_time"));
        evs.push(run_finished());
        let out = render_all(evs);
        assert_eq!(
            out,
            "→ get_weather {}\n→ get_time {}\n⧗ get_weather awaiting result — settle with:\n    --input '{\"type\":\"tool.result\",\"id\":\"call_a\",\"result\":\"...\"}'\n⧗ get_time awaiting result — settle with:\n    --input '{\"type\":\"tool.result\",\"id\":\"call_b\",\"result\":\"...\"}'\n"
        );
    }

    #[test]
    fn settled_tools_leave_no_pending_hint() {
        let mut evs = call("a", "get_current_time");
        evs.push(result("a", "2026-07-10T04:14:56.322Z"));
        evs.push(run_finished());
        let out = render_all(evs);
        assert_eq!(
            out,
            "→ get_current_time {}\n← get_current_time\n  2026-07-10T04:14:56.322Z\n"
        );
    }
}
