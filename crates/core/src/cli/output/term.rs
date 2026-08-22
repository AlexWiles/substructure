//! Colour, and shaping text to fit. Knows nothing about agents.

use std::io::IsTerminal;

pub(super) const RESET: &str = "\x1b[0m";
pub(super) const DIM: &str = "\x1b[2m";
pub(super) const CYAN: &str = "\x1b[36m";
pub(super) const YELLOW: &str = "\x1b[33m";
pub(super) const RED: &str = "\x1b[31m";

/// What each part of the transcript is coloured with. One place to change,
/// and the shape a loaded theme would fill.
pub(super) struct Theme {
    pub reset: &'static str,
    pub dim: &'static str,
    pub bold: &'static str,
    pub italic: &'static str,
    pub tool: &'static str,
    pub warn: &'static str,
    pub error: &'static str,
    pub heading: &'static str,
    pub code: &'static str,
    pub code_block: &'static str,
    pub link: &'static str,
    pub quote: &'static str,
    pub rule: &'static str,
    pub bullet: &'static str,
}

pub(super) const DARK: Theme = Theme {
    reset: RESET,
    dim: DIM,
    bold: "\x1b[1m",
    italic: "\x1b[3m",
    tool: CYAN,
    warn: YELLOW,
    error: RED,
    heading: "\x1b[1m\x1b[38;5;222m",
    code: "\x1b[38;5;115m",
    code_block: "\x1b[38;5;143m",
    link: "\x1b[38;5;110m",
    quote: "\x1b[38;5;245m",
    rule: "\x1b[38;5;240m",
    bullet: "\x1b[38;5;115m",
};

pub(crate) fn color() -> bool {
    std::io::stdout().is_terminal()
}

pub(crate) fn note(text: &str) {
    if std::io::stderr().is_terminal() {
        eprintln!("{DIM}{text}{RESET}");
    } else {
        eprintln!("{text}");
    }
}

pub(super) fn fold(text: &str, cap: usize) -> (String, usize) {
    let total = text.lines().count();
    if total <= cap {
        return (text.to_string(), 0);
    }
    let kept: Vec<&str> = text.lines().take(cap).collect();
    (kept.join("\n"), total - cap)
}

pub(super) fn held_lines(held: usize) -> String {
    match held {
        1 => "… +1 line".to_string(),
        _ => format!("… +{held} lines"),
    }
}

pub(super) fn indent(text: &str) -> String {
    text.lines()
        .map(|line| format!("  {line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_within_the_cap_is_left_whole() {
        assert_eq!(fold("one\ntwo", 12), ("one\ntwo".to_string(), 0));
    }

    #[test]
    fn text_over_the_cap_keeps_the_first_lines_and_counts_the_rest() {
        let text = (1..=20)
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let (kept, held) = fold(&text, 12);
        assert_eq!(kept.lines().count(), 12);
        assert_eq!(kept.lines().next_back(), Some("12"));
        assert_eq!(held, 8);
    }

    #[test]
    fn one_held_line_is_not_pluralized() {
        assert_eq!(held_lines(1), "… +1 line");
        assert_eq!(held_lines(2), "… +2 lines");
    }

    #[test]
    fn every_line_is_indented() {
        assert_eq!(indent("a\nb"), "  a\n  b");
    }
}
