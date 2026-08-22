//! The AG-UI event stream as the text a reader sees.

use std::collections::HashMap;
use std::io::{self, Write};

use super::markdown::Markdown;
use super::reasoning::Reasoning;
use super::status::{Phase, Status};
use super::term::{fold, held_lines, indent, Theme, DARK};
use super::tool::{format_result, head_of, PendingTool, RESULT_LINES};
use crate::transport::ag_ui::events::{AgUiEvent, RunOutcome};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Reader {
    Shell,
    Prompt,
}

pub struct PrettyPrinter {
    color: bool,
    at_line_start: bool,
    tools: HashMap<String, PendingTool>,
    reader: Reader,
    status: Status,
    reasoning: Reasoning,
    markdown: Markdown,
    theme: &'static Theme,
}

impl PrettyPrinter {
    pub fn new(color: bool) -> Self {
        Self {
            color,
            at_line_start: true,
            tools: HashMap::new(),
            reader: Reader::Shell,
            status: Status::disabled(),
            reasoning: Reasoning::default(),
            markdown: Markdown::default(),
            theme: &DARK,
        }
    }

    pub(super) fn at_a_prompt(&mut self) {
        self.reader = Reader::Prompt;
    }

    pub(super) fn with_status(&mut self, status: Status) {
        self.status = status;
    }

    pub fn render(&mut self, w: &mut impl Write, event: &AgUiEvent) -> io::Result<()> {
        self.status.writing();
        let rendered = self.render_event(w, event);
        self.status.wrote(self.at_line_start);
        rendered?;
        w.flush()
    }

    fn render_event(&mut self, w: &mut impl Write, event: &AgUiEvent) -> io::Result<()> {
        match event {
            AgUiEvent::RunStarted { .. } => self.status.set(Phase::Thinking),

            AgUiEvent::TextMessageContent { delta, .. } => {
                let shown = self.markdown.take(delta, self.theme, self.color);
                self.write_body(w, &shown)?;
            }
            AgUiEvent::TextMessageEnd { .. } => {
                let rest = self.markdown.flush(self.theme, self.color);
                self.write_body(w, &rest)?;
                self.break_line(w)?;
            }

            AgUiEvent::ReasoningMessageStart { .. } => {
                self.break_line(w)?;
                let dim = self.theme.dim;
                self.write_styled(w, dim, "thinking")?;
                self.newline(w)?;
                let dim = self.theme.dim;
                self.open(w, dim)?;
                self.reasoning.start();
            }
            AgUiEvent::ReasoningMessageContent { delta, .. } => {
                let kept = self.reasoning.take(delta);
                self.write_body(w, &kept)?;
            }
            AgUiEvent::ReasoningEnd { .. } => {
                self.close(w)?;
                let held = self.reasoning.held();
                if held > 0 {
                    let dim = self.theme.dim;
                    self.break_line(w)?;
                    self.write_styled(w, dim, &held_lines(held))?;
                    self.newline(w)?;
                }
                self.break_line(w)?;
            }

            AgUiEvent::ToolCallStart {
                tool_call_id,
                tool_call_name,
                ..
            } => {
                self.tools.insert(
                    tool_call_id.clone(),
                    PendingTool::new(tool_call_name.clone()),
                );
                self.status.set(self.tool_phase());
            }
            AgUiEvent::ToolCallArgs {
                tool_call_id,
                delta,
            } => {
                if let Some(tool) = self.tools.get_mut(tool_call_id) {
                    tool.args.push_str(delta);
                }
            }
            AgUiEvent::ToolCallEnd { .. } => {}
            AgUiEvent::ToolCallResult {
                tool_call_id,
                content,
                is_error,
                retryable,
                ..
            } => {
                self.break_line(w)?;
                let again = *is_error && *retryable;
                let outcome = match again {
                    true => self.tools.get_mut(tool_call_id).map(|tool| {
                        tool.attempt += 1;
                        let head = head_of(tool, *is_error, again);
                        tool.started = std::time::Instant::now();
                        head
                    }),
                    false => self.tools.remove(tool_call_id).map(|mut tool| {
                        tool.attempt += 1;
                        head_of(&tool, *is_error, again)
                    }),
                };
                if let Some(head) = outcome {
                    let style = match *is_error {
                        true => self.theme.error,
                        false => self.theme.tool,
                    };
                    self.write_styled(w, style, &head)?;
                    self.newline(w)?;
                }
                let dim = self.theme.dim;
                self.open(w, dim)?;
                let (body, held) = fold(&format_result(content), RESULT_LINES);
                self.write_body(w, &indent(&body))?;
                self.close(w)?;
                self.break_line(w)?;
                if held > 0 {
                    self.write_styled(w, dim, &indent(&held_lines(held)))?;
                    self.newline(w)?;
                }
                self.status.set(self.tool_phase());
            }

            AgUiEvent::RunFinished { outcome, .. } => {
                self.status.set(Phase::Idle);
                if let Some(RunOutcome::Interrupt { interrupts }) = outcome {
                    for it in interrupts {
                        if let Some(id) = &it.tool_call_id {
                            self.tools.remove(id);
                        }
                        if self.reader == Reader::Shell {
                            self.break_line(w)?;
                            let msg = it.message.as_deref().unwrap_or("(no message)");
                            let warn = self.theme.warn;
                            self.write_styled(
                                w,
                                warn,
                                &format!("⚠ interrupt {} [{}]: {msg}", it.id, it.reason),
                            )?;
                            self.newline(w)?;
                        }
                    }
                }
                self.render_pending_settlements(w)?;
            }
            AgUiEvent::RunError { message } => {
                self.status.set(Phase::Idle);
                self.break_line(w)?;
                let error = self.theme.error;
                self.write_styled(w, error, &format!("✗ error: {message}"))?;
                self.newline(w)?;
            }

            _ => {}
        }
        w.flush()
    }

    fn render_pending_settlements(&mut self, w: &mut impl Write) -> io::Result<()> {
        let mut pending: Vec<(String, String)> = self
            .tools
            .drain()
            .map(|(id, tool)| (id, tool.name))
            .collect();
        pending.sort();
        for (id, name) in pending {
            self.break_line(w)?;
            if self.reader == Reader::Prompt {
                self.write_styled(
                    w,
                    self.theme.warn,
                    &format!("⧗ {name} is waiting on a client-side result, which this chat cannot settle."),
                )?;
                self.newline(w)?;
                continue;
            }
            self.write_styled(
                w,
                self.theme.warn,
                &format!("⧗ {name} awaiting result — settle with:"),
            )?;
            self.newline(w)?;
            let hint = format!(
                "    --input '{{\"type\":\"tool.result\",\"id\":\"{id}\",\"result\":\"...\"}}'"
            );
            self.write_styled(w, self.theme.dim, &hint)?;
            self.newline(w)?;
        }
        Ok(())
    }

    fn tool_phase(&self) -> Phase {
        let mut pending = self.tools.values();
        match (pending.next(), pending.next()) {
            (None, _) => Phase::Thinking,
            (Some(tool), None) => Phase::Tool {
                name: tool.name.clone(),
                about: tool.about(),
            },
            (Some(_), Some(_)) => Phase::Tool {
                name: format!("{} tools", self.tools.len()),
                about: None,
            },
        }
    }

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
            self.write_body(w, &format!("{style}{text}{}", self.theme.reset))
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
            w.write_all(self.theme.reset.as_bytes())?;
        }
        Ok(())
    }

    fn newline(&mut self, w: &mut impl Write) -> io::Result<()> {
        w.write_all(b"\n")?;
        self.at_line_start = true;
        Ok(())
    }

    fn break_line(&mut self, w: &mut impl Write) -> io::Result<()> {
        if !self.at_line_start {
            self.newline(w)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::output::status::Phase;
    use crate::transport::ag_ui::events::RunOutcome;

    fn render_all(events: Vec<AgUiEvent>) -> String {
        render_with(PrettyPrinter::new(false), events)
    }

    fn render_with(mut printer: PrettyPrinter, events: Vec<AgUiEvent>) -> String {
        let mut buf: Vec<u8> = Vec::new();
        for ev in &events {
            printer.render(&mut buf, ev).unwrap();
        }
        String::from_utf8(buf).unwrap()
    }

    fn at_a_prompt() -> PrettyPrinter {
        let mut printer = PrettyPrinter::new(false);
        printer.reader = Reader::Prompt;
        printer
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
                metadata: None,
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
        assert_eq!(out, "");
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
        assert_eq!(out, "");
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
            is_error: false,
            retryable: false,
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
            is_error: false,
            retryable: false,
            role: "tool",
        }
    }

    fn failed(id: &str, content: &str, retryable: bool) -> AgUiEvent {
        AgUiEvent::ToolCallResult {
            message_id: id.into(),
            tool_call_id: id.into(),
            content: content.into(),
            is_error: true,
            retryable,
            role: "tool",
        }
    }

    #[test]
    fn tool_result_is_labeled_with_its_call() {
        let mut evs = call("x", "get_current_time");
        evs.push(result("x", "2026-07-10T04:14:56.322Z"));
        let out = render_all(evs);
        assert_eq!(out, "● get_current_time\n  2026-07-10T04:14:56.322Z\n");
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
            "● get_current_time_zone\n  Asia/Bangkok\n\
             ● get_current_time\n  2026-07-10T04:14:56.322Z\n"
        );
    }

    #[test]
    fn results_pair_by_id_regardless_of_completion_order() {
        let mut evs = call("a", "first");
        evs.extend(call("b", "second"));
        evs.push(result("b", "B"));
        evs.push(result("a", "A"));
        let out = render_all(evs);
        assert_eq!(out, "● second\n  B\n● first\n  A\n");
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
        assert_eq!(out, "● get_weather {\"city\":\"Paris\"}\n  68\n");
    }

    #[test]
    fn run_finished_does_not_print_the_result() {
        let out = render_all(vec![AgUiEvent::RunFinished {
            thread_id: "t".into(),
            run_id: "r".into(),
            result: Some(serde_json::json!({"answer": 42})),
            outcome: None,
            metadata: None,
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
            metadata: None,
        }]);
        assert_eq!(out, "⚠ interrupt int-1 [confirmation]: Send the email?\n");
    }

    fn parked_on(tool_call_id: Option<&str>) -> AgUiEvent {
        AgUiEvent::RunFinished {
            thread_id: "t".into(),
            run_id: "r".into(),
            result: None,
            outcome: Some(RunOutcome::Interrupt {
                interrupts: vec![crate::transport::ag_ui::events::AgUiInterrupt {
                    id: "int-1".into(),
                    reason: "confirmation".into(),
                    message: Some("Send the email?".into()),
                    tool_call_id: tool_call_id.map(str::to_string),
                    response_schema: None,
                    expires_at: None,
                    metadata: None,
                }],
            }),
            metadata: None,
        }
    }

    fn interrupted() -> AgUiEvent {
        parked_on(None)
    }

    #[test]
    fn at_a_prompt_the_interrupt_line_is_left_to_the_prompt_that_answers_it() {
        assert_eq!(render_with(at_a_prompt(), vec![interrupted()]), "");
    }

    #[test]
    fn a_call_the_interrupt_parked_is_not_reported_as_awaiting_a_settlement() {
        let mut evs = call("toolu_1", "issues__delete_issue");
        evs.push(parked_on(Some("toolu_1")));
        let out = render_all(evs);
        assert_eq!(out, "⚠ interrupt int-1 [confirmation]: Send the email?\n");
        assert!(!out.contains("awaiting result"), "got {out}");

        let chatting = render_with(at_a_prompt(), {
            let mut evs = call("toolu_1", "issues__delete_issue");
            evs.push(parked_on(Some("toolu_1")));
            evs
        });
        assert_eq!(chatting, "");
    }

    #[test]
    fn a_pending_call_the_interrupt_did_not_park_is_still_reported() {
        let mut evs = call("call_a", "get_weather");
        evs.push(parked_on(Some("toolu_other")));
        let out = render_all(evs);
        assert!(out.contains("get_weather awaiting result"), "got {out}");
    }

    #[test]
    fn at_a_prompt_a_pending_client_tool_says_it_cannot_be_settled() {
        let mut evs = call("call_a", "get_weather");
        evs.push(run_finished());
        let out = render_with(at_a_prompt(), evs);
        assert_eq!(
            out,
            "⧗ get_weather is waiting on a client-side result, \
             which this chat cannot settle.\n"
        );
        assert!(!out.contains("--input"), "got {out}");
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
                metadata: None,
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
        assert_eq!(out, "Let me check.\n");
    }

    fn run_finished() -> AgUiEvent {
        AgUiEvent::RunFinished {
            thread_id: "t".into(),
            run_id: "r".into(),
            result: None,
            outcome: None,
            metadata: None,
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
            "⧗ get_weather awaiting result — settle with:\n    --input '{\"type\":\"tool.result\",\"id\":\"call_x\",\"result\":\"...\"}'\n"
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
            "⧗ get_weather awaiting result — settle with:\n    --input '{\"type\":\"tool.result\",\"id\":\"call_a\",\"result\":\"...\"}'\n⧗ get_time awaiting result — settle with:\n    --input '{\"type\":\"tool.result\",\"id\":\"call_b\",\"result\":\"...\"}'\n"
        );
    }

    #[test]
    fn a_failed_call_is_marked_as_one() {
        let mut evs = call("a", "read_file");
        evs.push(failed("a", "no such file", false));
        let out = render_all(evs);
        assert_eq!(out, "✗ read_file\n  no such file\n");
    }

    #[test]
    fn a_retried_call_counts_its_attempts() {
        let mut evs = call("a", "fetch_url");
        evs.push(failed("a", "503", true));
        evs.push(failed("a", "503", true));
        evs.push(result("a", "ok"));
        let out = render_all(evs);
        assert!(out.contains("↻ fetch_url (attempt 1)\n"), "got {out}");
        assert!(out.contains("↻ fetch_url (attempt 2)\n"), "got {out}");
        assert!(out.contains("● fetch_url (attempt 3)\n"), "got {out}");
    }

    #[test]
    fn a_retried_call_that_succeeds_leaves_no_pending_hint() {
        let mut evs = call("a", "fetch_url");
        evs.push(failed("a", "503", true));
        evs.push(result("a", "ok"));
        evs.push(run_finished());
        let out = render_all(evs);
        assert!(!out.contains("awaiting result"), "got {out}");
    }

    #[test]
    fn a_long_result_is_folded_and_the_rest_counted() {
        let body: String = (1..=20)
            .map(|n| format!("line {n}"))
            .collect::<Vec<_>>()
            .join("\n");
        let mut evs = call("a", "read_file");
        evs.push(result("a", &body));
        let out = render_all(evs);
        assert!(out.contains("  line 12\n"), "got {out}");
        assert!(!out.contains("line 13"), "got {out}");
        assert!(out.contains("  … +8 lines\n"), "got {out}");
    }

    #[test]
    fn a_short_result_is_not_folded() {
        let mut evs = call("a", "read_file");
        evs.push(result("a", "one\ntwo"));
        let out = render_all(evs);
        assert!(!out.contains("…"), "got {out}");
    }

    #[test]
    fn the_live_line_names_the_call_and_counts_its_attempts() {
        let mut printer = PrettyPrinter::new(false);
        let mut buf: Vec<u8> = Vec::new();
        let mut phase = |printer: &mut PrettyPrinter, event: &AgUiEvent| {
            printer.render(&mut buf, event).unwrap();
            printer.tool_phase()
        };

        assert_eq!(
            phase(&mut printer, &call("a", "fetch_url")[0]),
            Phase::Tool {
                name: "fetch_url".into(),
                about: None
            }
        );
        assert_eq!(
            phase(&mut printer, &failed("a", "503", true)),
            Phase::Tool {
                name: "fetch_url".into(),
                about: Some("attempt 2".into())
            }
        );
        assert_eq!(
            phase(&mut printer, &failed("a", "503", true)),
            Phase::Tool {
                name: "fetch_url".into(),
                about: Some("attempt 3".into())
            }
        );
        assert_eq!(phase(&mut printer, &result("a", "ok")), Phase::Thinking);
    }

    #[test]
    fn the_live_line_counts_a_batch() {
        let mut printer = PrettyPrinter::new(false);
        let mut buf: Vec<u8> = Vec::new();
        for id in ["a", "b", "c"] {
            printer.render(&mut buf, &call(id, "read_file")[0]).unwrap();
        }
        assert_eq!(
            printer.tool_phase(),
            Phase::Tool {
                name: "3 tools".into(),
                about: None
            }
        );

        printer.render(&mut buf, &result("a", "ok")).unwrap();
        printer.render(&mut buf, &result("b", "ok")).unwrap();
        assert_eq!(
            printer.tool_phase(),
            Phase::Tool {
                name: "read_file".into(),
                about: None
            }
        );

        printer.render(&mut buf, &result("c", "ok")).unwrap();
        assert_eq!(printer.tool_phase(), Phase::Thinking);
    }

    #[test]
    fn settled_tools_leave_no_pending_hint() {
        let mut evs = call("a", "get_current_time");
        evs.push(result("a", "2026-07-10T04:14:56.322Z"));
        evs.push(run_finished());
        let out = render_all(evs);
        assert_eq!(out, "● get_current_time\n  2026-07-10T04:14:56.322Z\n");
    }
}
