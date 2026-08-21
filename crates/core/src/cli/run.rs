use std::io::IsTerminal;
use std::sync::Arc;

use clap::Args;
use uuid::Uuid;

use super::cloud::project_config;
use super::env::{EnvVars, OutputFormat};
use super::pretty::{self, write_json, Renderer};
use super::run_remote;
use super::target::target;
use super::{local, DEFAULT_TENANT};
use crate::event_store::Seq;
use crate::protocol::{ClientInput, Content, DraftMessage, Role, SessionOwner};
use crate::providers::sqlite::{SqliteBlobStore, SqliteDb};
use crate::session::events::EventPayload;
use crate::session::index::SessionFilter;
use crate::session::subscriptions::{SessionSubscriptionSpec, SubscriptionScope};
use crate::span::SpanContext;
use crate::transport::ag_ui::translator::AgUiTranslator;
use crate::{Caller, HandleClientInput, Runtime};

#[derive(Args)]
pub struct RunArgs {
    /// The message to send. Shorthand for
    /// `--input '{"type":"client.message","message":{"role":"user","content":"..."}}'`.
    #[arg(value_name = "MESSAGE")]
    message: Option<String>,
    /// Agent id to run, naming an `[agent.<id>]` section. Falls back to
    /// `[run].agent`.
    #[arg(long)]
    agent: Option<String>,
    /// JSON input, for anything a plain message cannot say. Its `type` selects
    /// the path:
    ///   `client.message` / `client.messages` / `client.action`  submit a client payload;
    ///   `interrupt.resume`                                       resume a parked interrupt;
    ///   `tool.result` / `tool.error`                             settle a client-side tool call.
    ///
    /// A plain user turn:
    /// `{"type":"client.message","message":{"role":"user","content":"hi"}}`
    #[arg(long, conflicts_with = "message")]
    input: Option<String>,
    /// Resume/continue an existing session. Omit to start a new one (its id is
    /// printed). Required for `interrupt.resume` and `tool.result`/`tool.error`.
    #[arg(long)]
    session: Option<String>,
    /// Environment file (default: walks up from cwd looking for
    /// `substructure.toml`).
    #[arg(short = 'c', long)]
    config: Option<std::path::PathBuf>,
    /// Run the turn on the deployment at this URL. Point it at a `subs serve`
    /// to use the engine that runs there.
    #[arg(long)]
    url: Option<String>,
    /// SQLite dev database path. [default: substructure.db]
    #[arg(long)]
    db: Option<String>,
    /// Output mode. (Engine logs go to stderr at error level; set RUST_LOG=info for more.)
    #[arg(long, short = 'o', value_enum)]
    output: Option<OutputFormat>,
}

/// Parse `--input`, splicing `--agent` in as `agent_id` when the JSON didn't carry one.
/// A submit needs `agent_id`; the other tags have no such field, so a stray key from
/// `--agent` is simply ignored by serde.
fn parse_input(input: &str, agent: Option<String>) -> anyhow::Result<ClientInput> {
    let mut value: serde_json::Value =
        serde_json::from_str(input).map_err(|e| anyhow::anyhow!("invalid --input: {e}"))?;
    if let (Some(agent), Some(obj)) = (agent, value.as_object_mut()) {
        obj.entry("agent_id")
            .or_insert_with(|| serde_json::Value::String(agent));
    }
    serde_json::from_value(value).map_err(|e| anyhow::anyhow!("invalid --input: {e}"))
}

/// One user message, the shape a positional argument means.
fn message_input(message: String, agent_id: String) -> ClientInput {
    ClientInput::Message {
        agent_id,
        turn_id: None,
        message: DraftMessage {
            id: None,
            role: Role::User,
            content: Some(Content::Text(message)),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning: None,
        },
        stream: true,
        queue: false,
    }
}

/// Which agent this run drives. `--agent`, else `[run].agent` when the file
/// pins one; nothing is inferred from what happens to be declared, because an
/// engine that picks for you picks differently the day a second agent is added.
fn select_agent(
    flag: Option<String>,
    pinned: Option<String>,
    declared: &[String],
) -> anyhow::Result<String> {
    let Some(agent_id) = flag.or(pinned) else {
        anyhow::bail!(
            "no agent given. Pass `--agent <id>` or set `[run].agent`. Declared: {}",
            crate::worker::directory::declared(declared)
        );
    };
    // A typo here would otherwise surface as a failed decision one turn later.
    if !declared.contains(&agent_id) {
        anyhow::bail!(
            "no [agent.{agent_id}] in substructure.toml. Declared: {}",
            crate::worker::directory::declared(declared)
        );
    }
    Ok(agent_id)
}

impl RunArgs {
    pub fn config_path(&self) -> Option<&std::path::Path> {
        self.config.as_deref()
    }
}

pub async fn run(args: RunArgs) -> anyhow::Result<()> {
    // Anything argv omits can be pinned in the project file. Precedence is
    // flag > environment > file > default, applied one field at a time.
    let cfg = project_config::load(args.config.as_deref())?;

    let run = cfg.run.clone().unwrap_or_default();
    let output_mode = args.output.or(run.output).unwrap_or(OutputFormat::AgUi);
    let db_path = args.db.unwrap_or_else(|| cfg.db_path());

    // Preflight: the agent is checked against the file before a session exists,
    // so a typo costs nothing and leaves nothing behind.
    let agent_id = select_agent(args.agent, run.agent, &cfg.agent_ids())?;

    // Captured for the resume hint printed at the end, before the args are consumed.
    let payload = args.message.clone().or_else(|| args.input.clone());

    let input = match (args.message, args.input) {
        (Some(message), _) => message_input(message, agent_id),
        (None, Some(input)) => parse_input(&input, Some(agent_id))?,
        (None, None) => {
            anyhow::bail!("nothing to send. Pass a message, or `--input` for a non-message input.")
        }
    };

    // With a `[remote]`, no engine, database, or key is necessary here.
    let globals = super::cloud::CloudGlobals {
        config: args.config.clone(),
        url: args.url.clone(),
        ..Default::default()
    };
    if target(&globals)?.here().is_none() {
        return run_remote::run(run_remote::Run {
            globals,
            session_id: args.session,
            input,
            output: output_mode,
            payload,
        })
        .await;
    }

    // `run` exposes no server, so no client/worker auth env is required.
    let env = match EnvVars::load(cfg.provider_bindings(), false) {
        Some(e) => e,
        None => std::process::exit(2),
    };

    if let Some(parent) = std::path::Path::new(&db_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let db = SqliteDb::open(&db_path, std::time::Duration::from_secs(5))?;
    let blobs = Arc::new(SqliteBlobStore::new(db.clone()));
    let (rt, _adapter, _mcp_auth) = local::start_engine(db, blobs, env.providers, &cfg).await?;

    let session_id = args.session.unwrap_or_else(|| Uuid::now_v7().to_string());

    let caller = Caller::System {
        tenant_id: DEFAULT_TENANT.to_string(),
    };
    let owner = SessionOwner {
        tenant_id: DEFAULT_TENANT.to_string(),
        // Only this path names an unauthenticated person. A `[remote]` run
        // is the caller's account, and a served engine is the token's
        // subject.
        requester: crate::protocol::Requester::private(crate::protocol::Subject::new(
            crate::protocol::Issuer::cli(),
            super::LOCAL_SUBJECT,
        )),
        metadata: Default::default(),
    };

    // Token deltas are transient; subscribe before acting so none are missed.
    let mut deltas = rt.subscribe_token_deltas(&caller, &session_id).await;

    // Snapshot before acting: `base_seq` (the session's current stream version)
    // bounds the replay to this invocation's events. The active-turn lookup a
    // resume/settle needs lives in the router.
    let base_seq = match rt.get_session(DEFAULT_TENANT, &session_id).await {
        Ok(session) => Seq(session.seq),
        Err(_) => Seq(0),
    };

    let turn_id = rt
        .handle_client_input(HandleClientInput {
            session_id: session_id.clone(),
            caller: caller.clone(),
            owner,
            input,
            span: SpanContext::root().child("cli_client_input"),
        })
        .await?
        .turn_id;

    let mut events = rt
        .stream(
            SessionSubscriptionSpec {
                session_id: session_id.clone(),
                caller,
                scope: SubscriptionScope::Turn {
                    turn_id: turn_id.clone(),
                },
            },
            Some(base_seq),
        )
        .await?;

    // The translator is both the AG-UI event source and the terminal oracle:
    // `terminated` flips on RUN_FINISHED — normal completion, a client-tool
    // yield, or an interrupt — which is exactly when this invocation should stop.
    let mut translator = AgUiTranslator::new(session_id.clone(), turn_id);
    let mut stdout = std::io::stdout();
    let mut renderer = Renderer::new(output_mode, pretty::color());
    let raw = renderer.is_raw();

    let evs = translator.start();
    renderer.emit(&mut stdout, evs)?;

    let mut deltas_open = true;
    loop {
        tokio::select! {
            maybe_event = events.recv() => {
                let Some(event) = maybe_event else { break };
                if raw {
                    write_json(&mut stdout, &event)?;
                }
                let ends_run = event.ends_run();
                let payload = event.payload;
                // Drain queued deltas before `llm.call.completed` so no closing
                // event outruns its last streamed fragment.
                if matches!(payload, EventPayload::LlmCallCompleted(_)) {
                    while let Ok(d) = deltas.try_recv() {
                        let evs = translator.on_delta(d);
                        renderer.emit(&mut stdout, evs)?;
                    }
                }
                let evs = translator.on_event(payload, ends_run);
                renderer.emit(&mut stdout, evs)?;
                if translator.terminated() {
                    break;
                }
            }
            maybe_delta = deltas.recv(), if deltas_open => {
                match maybe_delta {
                    Some(d) => {
                        let evs = translator.on_delta(d);
                        renderer.emit(&mut stdout, evs)?;
                    }
                    None => deltas_open = false,
                }
            }
        }
    }

    if translator.terminated() {
        await_indexed(&rt, &session_id).await;
        print_resume_hint(&session_id, payload.as_deref());
        Ok(())
    } else {
        anyhow::bail!("event stream ended before the run finished")
    }
}

/// `payload` is the message or `--input` this run sent.
pub(crate) fn print_resume_hint(session_id: &str, payload: Option<&str>) {
    let argv: Vec<String> = std::env::args().collect();
    let hint = format!(
        "continue this session with:\n  {}",
        resume_command(&argv, session_id, payload)
    );
    // Faint so the hint reads as secondary; plain when piped.
    if std::io::stderr().is_terminal() {
        eprintln!("\n\x1b[2m{hint}\x1b[0m");
    } else {
        eprintln!("\n{hint}");
    }
}

/// Waits for the session index to read this turn, so `subs sessions list`
/// shows it. The index polls, and nothing runs after this process stops.
///
/// Best effort. The next run reads what this one leaves.
async fn await_indexed(rt: &Runtime, session_id: &str) {
    const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);
    const POLL: std::time::Duration = std::time::Duration::from_millis(20);

    let Ok(session) = rt.get_session(DEFAULT_TENANT, session_id).await else {
        return;
    };
    let filter = SessionFilter {
        tenant_id: Some(DEFAULT_TENANT.to_string()),
        session_id: Some(session_id.to_string()),
        ..Default::default()
    };

    let deadline = tokio::time::Instant::now() + TIMEOUT;
    while tokio::time::Instant::now() < deadline {
        if let Ok(page) = rt.list_sessions(&filter).await {
            if page.items.iter().any(|s| s.seq >= session.seq) {
                return;
            }
        }
        tokio::time::sleep(POLL).await;
    }
    tracing::debug!(session_id, "session index did not catch up before exit");
}

/// This run's argv, with `--session` pinned and the message replaced by a
/// placeholder at the end.
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
    if moved_input {
        out.push("--input".into());
    }
    out.push("'...'".into());
    out.join(" ")
}

/// `run`'s flags whose value is the next argument.
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

    fn declared() -> Vec<String> {
        vec!["assistant".to_string(), "researcher".to_string()]
    }

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

        let joined = resume_command(&argv(&["subs", "run", "--session=old", "hi"]), "new", None);
        assert_eq!(joined, "subs run hi --session new '...'");
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

    #[test]
    fn the_flag_wins_over_the_pinned_agent() {
        let agent = select_agent(
            Some("researcher".to_string()),
            Some("assistant".to_string()),
            &declared(),
        )
        .unwrap();
        assert_eq!(agent, "researcher");
        assert_eq!(
            select_agent(None, Some("assistant".to_string()), &declared()).unwrap(),
            "assistant"
        );
    }

    /// Nothing is inferred from what happens to be declared: an engine that
    /// picks for you picks differently the day a second agent is added.
    #[test]
    fn no_agent_anywhere_is_an_error_listing_the_declared_ones() {
        let err = select_agent(None, None, &declared())
            .unwrap_err()
            .to_string();
        assert!(err.contains("no agent given"), "got {err}");
        assert!(err.contains("assistant, researcher"), "got {err}");
    }

    /// The preflight is what makes a typo cost nothing: without it the run
    /// creates a session and fails one decision later.
    #[test]
    fn an_undeclared_agent_fails_before_the_session_exists() {
        let err = select_agent(Some("assistnat".to_string()), None, &declared())
            .unwrap_err()
            .to_string();
        assert!(err.contains("no [agent.assistnat]"), "got {err}");
        assert!(err.contains("assistant, researcher"), "got {err}");
    }

    #[test]
    fn a_positional_message_is_a_user_turn_for_the_selected_agent() {
        match message_input("hi".to_string(), "assistant".to_string()) {
            ClientInput::Message {
                agent_id, message, ..
            } => {
                assert_eq!(agent_id, "assistant");
                assert_eq!(message.role, Role::User);
                assert_eq!(message.content.as_ref().and_then(Content::text), Some("hi"));
            }
            other => panic!("expected client.message, got {other:?}"),
        }
    }

    #[test]
    fn agent_flag_fills_agent_id_on_a_submit() {
        let input = parse_input(
            r#"{"type":"client.message","message":{"role":"user","content":"hi"}}"#,
            Some("bot".to_string()),
        )
        .unwrap();
        match input {
            ClientInput::Message { agent_id, .. } => assert_eq!(agent_id, "bot"),
            other => panic!("expected client.message, got {other:?}"),
        }
    }

    #[test]
    fn input_agent_id_wins_over_the_agent_flag() {
        let input = parse_input(
            r#"{"type":"client.message","agent_id":"in-json","message":{"role":"user","content":"hi"}}"#,
            Some("flag".to_string()),
        )
        .unwrap();
        match input {
            ClientInput::Message { agent_id, .. } => assert_eq!(agent_id, "in-json"),
            other => panic!("expected client.message, got {other:?}"),
        }
    }

    #[test]
    fn invalid_json_is_a_clear_error() {
        let err = parse_input("not json", None).unwrap_err().to_string();
        assert!(err.contains("invalid --input"), "{err}");
    }
}
