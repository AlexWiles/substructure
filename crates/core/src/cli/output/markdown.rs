//! Markdown as it streams, for a transcript that cannot be repainted.
//!
//! A line is rendered once it is whole, and a fenced block once it closes.
//! Nothing already written is revised, so the unit of work is the line rather
//! than the message.

use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};

use super::term::{paint, width, BOLD, DIM, ITALIC, RESET, UNDERLINE};

#[derive(Default)]
pub(super) struct Markdown {
    pending: String,
    fence: Option<Fence>,
}

struct Fence {
    /// The run that opened it, so a longer run inside the body does not close
    /// it early.
    ticks: usize,
    /// `md` and `markdown` fences hold markdown, not code. Models wrap tables
    /// and whole answers in them.
    literal: bool,
}

impl Markdown {
    /// What the deltas so far can be shown as. Text that is not yet a whole
    /// line is held back.
    pub(super) fn take(&mut self, delta: &str, color: bool) -> String {
        self.pending.push_str(delta);
        let mut out = String::new();
        while let Some(at) = self.pending.find('\n') {
            let line: String = self.pending.drain(..=at).collect();
            out.push_str(&self.line(line.trim_end_matches('\n'), color));
            out.push('\n');
        }
        out
    }

    /// Whatever is left when the message ends, whole or not.
    pub(super) fn flush(&mut self, color: bool) -> String {
        if self.pending.is_empty() {
            self.fence = None;
            return String::new();
        }
        let rest = std::mem::take(&mut self.pending);
        let out = self.line(&rest, color);
        self.fence = None;
        out
    }

    fn line(&mut self, line: &str, color: bool) -> String {
        if let Some(fence) = &self.fence {
            if closes(line, fence.ticks) {
                let literal = fence.literal;
                self.fence = None;
                return match literal {
                    true => String::new(),
                    false => paint(DIM, "", color),
                };
            }
            if fence.literal {
                return render(line, color);
            }
            return paint(DIM, line, color);
        }
        if let Some(ticks) = opens(line) {
            let info = line
                .trim()
                .trim_start_matches('`')
                .trim()
                .to_ascii_lowercase();
            let literal = info == "md" || info == "markdown";
            self.fence = Some(Fence { ticks, literal });
            return match literal {
                true => String::new(),
                false => paint(DIM, line.trim_end(), color),
            };
        }
        render(line, color)
    }
}

/// The tick count of a fence this line opens, if it opens one.
fn opens(line: &str) -> Option<usize> {
    let trimmed = line.trim_start();
    let ticks = trimmed.chars().take_while(|c| *c == '`').count();
    (ticks >= 3).then_some(ticks)
}

fn closes(line: &str, ticks: usize) -> bool {
    let trimmed = line.trim();
    trimmed.chars().all(|c| c == '`') && trimmed.len() >= ticks
}

/// One line of markdown as styled text. Parsed alone, so a construct that
/// needs its neighbours — a table, a lazy continuation — reads as its own
/// text rather than wrongly.
fn render(line: &str, color: bool) -> String {
    if line.trim().is_empty() {
        return String::new();
    }
    let mut out = String::new();
    let mut styles: Vec<&str> = Vec::new();
    let mut link: Option<String> = None;

    let push = |out: &mut String, styles: &Vec<&str>, text: &str| {
        if !color || styles.is_empty() {
            out.push_str(text);
            return;
        }
        for style in styles {
            out.push_str(style);
        }
        out.push_str(text);
        out.push_str(RESET);
    };

    for event in Parser::new_ext(line, Options::ENABLE_STRIKETHROUGH) {
        match event {
            Event::Start(Tag::Heading { .. }) => styles.push(BOLD),
            Event::End(TagEnd::Heading(_)) => {
                styles.pop();
            }
            Event::Start(Tag::Strong) => styles.push(BOLD),
            Event::End(TagEnd::Strong) => {
                styles.pop();
            }
            Event::Start(Tag::Emphasis) => styles.push(ITALIC),
            Event::End(TagEnd::Emphasis) => {
                styles.pop();
            }
            Event::Start(Tag::Strikethrough) => styles.push(DIM),
            Event::End(TagEnd::Strikethrough) => {
                styles.pop();
            }
            Event::Start(Tag::BlockQuote(_)) => {
                out.push_str(&paint(DIM, "│ ", color));
                styles.push(DIM);
            }
            Event::End(TagEnd::BlockQuote(_)) => {
                styles.pop();
            }
            Event::Start(Tag::Item) => out.push_str("• "),
            Event::Start(Tag::Link { dest_url, .. }) => {
                if color {
                    out.push_str(&format!("\x1b]8;;{dest_url}\x1b\\"));
                    link = Some(String::new());
                }
                styles.push(UNDERLINE);
            }
            Event::End(TagEnd::Link) => {
                styles.pop();
                if link.take().is_some() {
                    out.push_str("\x1b]8;;\x1b\\");
                }
            }
            Event::Code(text) => out.push_str(&paint(DIM, &text, color)),
            Event::Text(text) => push(&mut out, &styles, &text),
            Event::SoftBreak | Event::HardBreak => out.push(' '),
            Event::Rule => out.push_str(&paint(DIM, &"─".repeat(width()), color)),
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(delta: &str) -> String {
        Markdown::default().take(delta, false)
    }

    fn styled(delta: &str) -> String {
        Markdown::default().take(delta, true)
    }

    #[test]
    fn a_line_is_held_until_it_is_whole() {
        let mut md = Markdown::default();
        assert_eq!(md.take("half", false), "");
        assert_eq!(md.take(" a line\n", false), "half a line\n");
    }

    #[test]
    fn what_is_left_at_the_end_is_still_shown() {
        let mut md = Markdown::default();
        assert_eq!(md.take("no newline", false), "");
        assert_eq!(md.flush(false), "no newline");
    }

    #[test]
    fn markers_are_dropped_and_the_words_kept() {
        assert_eq!(plain("**bold** and *thin*\n"), "bold and thin\n");
        assert_eq!(plain("# A heading\n"), "A heading\n");
        assert_eq!(plain("- one\n"), "• one\n");
        assert_eq!(plain("> quoted\n"), "│ quoted\n");
    }

    #[test]
    fn styling_is_applied_only_on_a_terminal() {
        let out = styled("**bold**\n");
        assert!(out.contains(BOLD), "got {out:?}");
        assert!(!plain("**bold**\n").contains('\x1b'));
    }

    #[test]
    fn a_heading_and_a_link_get_their_own_style() {
        assert!(styled("## Title\n").contains(BOLD));
        let link = styled("see [docs](https://example.com)\n");
        assert!(link.contains(UNDERLINE), "got {link:?}");
        assert!(
            link.contains("\x1b]8;;https://example.com\x1b\\"),
            "got {link:?}"
        );
        assert_eq!(plain("see [docs](https://example.com)\n"), "see docs\n");
    }

    #[test]
    fn code_keeps_its_text_and_loses_its_backticks() {
        assert_eq!(plain("run `cargo test` now\n"), "run cargo test now\n");
        assert!(styled("run `cargo test`\n").contains(DIM));
    }

    /// A fence is code, so its body is shown as written rather than parsed.
    #[test]
    fn a_fenced_block_is_left_as_written() {
        let mut md = Markdown::default();
        let out: String = ["```rust\n", "let x = **not bold**;\n", "```\n"]
            .iter()
            .map(|d| md.take(d, false))
            .collect();
        assert!(out.contains("let x = **not bold**;"), "got {out:?}");
    }

    /// Models wrap whole answers in a `markdown` fence. Reading it as code
    /// would show the markers instead of the text.
    #[test]
    fn a_markdown_fence_is_unwrapped_rather_than_shown_as_code() {
        let mut md = Markdown::default();
        let out: String = ["```markdown\n", "**bold**\n", "```\n"]
            .iter()
            .map(|d| md.take(d, false))
            .collect();
        assert_eq!(out, "\nbold\n\n");
    }

    #[test]
    fn a_longer_run_inside_a_fence_does_not_close_it() {
        let mut md = Markdown::default();
        let out: String = ["````\n", "```\n", "still code\n", "````\n"]
            .iter()
            .map(|d| md.take(d, false))
            .collect();
        assert!(out.contains("still code"), "got {out:?}");
    }

    #[test]
    fn a_blank_line_stays_blank() {
        assert_eq!(plain("one\n\ntwo\n"), "one\n\ntwo\n");
    }
}
