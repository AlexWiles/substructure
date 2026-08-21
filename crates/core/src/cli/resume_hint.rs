use super::pretty;

/// `payload` is the message or `--input` this invocation sent. A command that
/// sends none — `subs chat` prompts for its own — gets no placeholder.
pub(crate) fn print_resume_hint(session_id: &str, payload: Option<&str>) {
    let argv: Vec<String> = std::env::args().collect();
    pretty::note(&format!(
        "\ncontinue this session with:\n  {}",
        resume_command(&argv, session_id, payload)
    ));
}

/// This invocation's argv, with `--session` pinned and the message replaced by
/// a placeholder at the end.
fn resume_command(argv: &[String], session: &str, payload: Option<&str>) -> String {
    let mut out = vec![argv
        .first()
        .map(|p| file_name(p))
        .unwrap_or_else(|| "subs".into())];

    let mut moved_input = false;
    let mut elided = false;
    // The flag the previous argument was, when it takes a value.
    let mut pending: Option<&str> = None;
    for arg in argv.iter().skip(1) {
        let flag = pending.take();
        if flag == Some("--session") {
            continue;
        }
        if arg == "--session" {
            pending = Some("--session");
            continue;
        }
        if arg.starts_with("--session=") {
            continue;
        }
        let is_payload = match flag {
            Some(f) => f == "--input",
            None => !arg.starts_with('-') && payload == Some(arg.as_str()),
        };
        if is_payload && !elided {
            elided = true;
            // `--input` follows its value to the end.
            if flag == Some("--input") {
                out.pop();
                moved_input = true;
            }
            continue;
        }
        if takes_value(arg) {
            pending = Some(arg);
        }
        out.push(quote(arg));
    }
    out.push("--session".into());
    out.push(quote(session));
    if payload.is_some() {
        if moved_input {
            out.push("--input".into());
        }
        out.push("'...'".into());
    }
    out.join(" ")
}

/// The flags whose value is the next argument.
fn takes_value(arg: &str) -> bool {
    matches!(
        arg,
        "--agent" | "--input" | "--config" | "-c" | "--url" | "--db" | "--output" | "-o"
    )
}

fn file_name(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string())
}

/// Single-quotes anything a shell would read as more than one word.
fn quote(arg: &str) -> String {
    let plain = !arg.is_empty()
        && arg
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "_@%+=:,./-".contains(c));
    if plain {
        arg.to_string()
    } else {
        format!("'{}'", arg.replace('\'', r"'\''"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(args: &[&str]) -> Vec<String> {
        args.iter().map(|a| a.to_string()).collect()
    }

    #[test]
    fn resume_command_echoes_the_flags_the_caller_gave() {
        let cmd = resume_command(
            &argv(&[
                "../subs/target/debug/subs",
                "run",
                "-c",
                "subs.toml",
                "--output",
                "pretty",
                "--agent",
                "coder",
                "update the readme",
            ]),
            "sess-1",
            Some("update the readme"),
        );
        assert_eq!(
            cmd,
            "subs run -c subs.toml --output pretty --agent coder --session sess-1 '...'"
        );
    }

    #[test]
    fn resume_command_replaces_an_earlier_session() {
        let cmd = resume_command(
            &argv(&["subs", "run", "--session", "old", "hi"]),
            "new",
            Some("hi"),
        );
        assert_eq!(cmd, "subs run --session new '...'");

        let joined = resume_command(
            &argv(&["subs", "run", "--session=old", "hi"]),
            "new",
            Some("hi"),
        );
        assert_eq!(joined, "subs run --session new '...'");
    }

    /// `subs chat` prompts for the next message, so its hint ends at the
    /// session it resumes.
    #[test]
    fn a_command_that_sends_no_payload_gets_no_placeholder() {
        let cmd = resume_command(&argv(&["subs", "chat", "--agent", "coder"]), "sess-1", None);
        assert_eq!(cmd, "subs chat --agent coder --session sess-1");
    }

    #[test]
    fn resume_command_quotes_what_a_shell_would_split() {
        let cmd = resume_command(
            &argv(&["subs", "run", "--db", "my db.db", "hi there"]),
            "s",
            Some("hi there"),
        );
        assert_eq!(cmd, "subs run --db 'my db.db' --session s '...'");
    }

    #[test]
    fn resume_command_does_not_read_a_flags_value_as_the_message() {
        let cmd = resume_command(
            &argv(&["subs", "run", "--agent", "coder", "coder"]),
            "s",
            Some("coder"),
        );
        assert_eq!(cmd, "subs run --agent coder --session s '...'");
    }

    #[test]
    fn resume_command_moves_input_to_the_end_with_its_flag() {
        let cmd = resume_command(
            &argv(&[
                "subs",
                "run",
                "--input",
                r#"{"type":"client.message"}"#,
                "-o",
                "pretty",
            ]),
            "s",
            Some(r#"{"type":"client.message"}"#),
        );
        assert_eq!(cmd, "subs run -o pretty --session s --input '...'");
    }
}
