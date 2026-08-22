//! The line editor, and what counts as a finished line.

use rustyline::completion::Completer;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::history::FileHistory;
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::{Cmd, Editor, EventHandler, Helper, KeyCode, KeyEvent, Modifiers};

pub(super) type ChatEditor = Editor<Continuation, FileHistory>;

/// A line ending in `\` is not finished, so `Enter` opens the next one.
pub(super) struct Continuation;

impl Validator for Continuation {
    fn validate(&self, ctx: &mut ValidationContext) -> rustyline::Result<ValidationResult> {
        match ctx.input().ends_with('\\') {
            true => Ok(ValidationResult::Incomplete),
            false => Ok(ValidationResult::Valid(None)),
        }
    }
}

impl Completer for Continuation {
    type Candidate = String;
}
impl Hinter for Continuation {
    type Hint = String;
}
impl Highlighter for Continuation {}
impl Helper for Continuation {}

pub(super) fn build() -> rustyline::Result<ChatEditor> {
    let mut editor = ChatEditor::new()?;
    editor.set_helper(Some(Continuation));
    for key in [
        KeyEvent(KeyCode::Enter, Modifiers::ALT),
        KeyEvent::ctrl('J'),
    ] {
        editor.bind_sequence(key, EventHandler::Simple(Cmd::Newline));
    }
    Ok(editor)
}

/// `\` at the end of a line held it open, so it stands for the newline the
/// reader typed rather than staying in the message.
pub(super) fn joined(line: &str) -> String {
    line.replace("\\\n", "\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn incomplete(input: &str) -> bool {
        input.ends_with('\\')
    }

    #[test]
    fn a_trailing_backslash_holds_the_line_open() {
        assert!(incomplete("first \\"));
        assert!(!incomplete("first"));
    }

    #[test]
    fn a_held_line_joins_as_the_newline_it_stood_for() {
        assert_eq!(joined("one \\\ntwo"), "one \ntwo");
        assert_eq!(joined("one\ntwo"), "one\ntwo");
        assert_eq!(joined("plain"), "plain");
    }

    #[test]
    fn a_backslash_inside_a_line_is_left_alone() {
        assert_eq!(joined(r"C:\path"), r"C:\path");
    }
}
