//! Colour, and shaping text to fit. Knows nothing about agents.

use std::io::IsTerminal;

pub const RESET: &str = "\x1b[0m";
pub const BOLD: &str = "\x1b[1m";
pub const DIM: &str = "\x1b[2m";
pub const ITALIC: &str = "\x1b[3m";
pub const UNDERLINE: &str = "\x1b[4m";

/// The terminal's width, for what cannot be drawn without one. Committed at
/// the width it was written: a resize reflows what the terminal wrapped, not
/// what we did.
pub fn width() -> usize {
    console::Term::stdout()
        .size_checked()
        .map(|(_, cols)| cols as usize)
        .unwrap_or(80)
        .clamp(20, 200)
}

pub fn color() -> bool {
    std::io::stdout().is_terminal()
}

pub fn note(text: &str) {
    if std::io::stderr().is_terminal() {
        eprintln!("{DIM}{text}{RESET}");
    } else {
        eprintln!("{text}");
    }
}

pub fn fold(text: &str, cap: usize) -> (String, usize) {
    let total = text.lines().count();
    if total <= cap {
        return (text.to_string(), 0);
    }
    let kept: Vec<&str> = text.lines().take(cap).collect();
    (kept.join("\n"), total - cap)
}

pub fn held_lines(held: usize) -> String {
    match held {
        1 => "… +1 line".to_string(),
        _ => format!("… +{held} lines"),
    }
}

pub fn paint(style: &str, text: &str, color: bool) -> String {
    match color {
        true => format!("{style}{text}{RESET}"),
        false => text.to_string(),
    }
}

pub fn indent(text: &str) -> String {
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
