use std::io::{IsTerminal, Write};

use clap::{Args, ValueEnum};
use uuid::Uuid;

use super::env::{EnvVars, LlmProviderArg};
use super::pretty::PrettyPrinter;
use super::{local, register_startup_worker, DEFAULT_TENANT};
use crate::event_store::Seq;
use crate::protocol::{ClientInput, SessionOwner};
use crate::session::events::EventPayload;
use crate::session::subscriptions::{SessionSubscriptionSpec, SubscriptionScope};
use crate::span::SpanContext;
use crate::transport::ag_ui::events::AgUiEvent;
use crate::transport::ag_ui::translator::AgUiTranslator;
use crate::{Caller, HandleClientInput};

#[derive(Args)]
pub struct RunArgs {
    /// Agent id the worker routes on. Required for submit inputs
    /// (`client.message`/`client.messages`/`client.action`); ignored otherwise.
    #[arg(long)]
    agent: Option<String>,
    /// JSON input. Its `type` selects the path:
    ///   `client.message` / `client.messages` / `client.action`  submit a client payload;
    ///   `interrupt.resume`                                       resume a parked interrupt;
    ///   `tool.result` / `tool.error`                             settle a client-side tool call.
    ///
    /// A plain user turn:
    /// `{"type":"client.message","message":{"role":"user","content":"hi"}}`
    #[arg(long)]
    input: String,
    /// Worker endpoint the engine POSTs decisions to.
    /// Falls back to $SUBSTRUCTURE_WORKER_URL, then http://localhost:3000/agent.
    #[arg(long)]
    worker_url: Option<String>,
    /// Resume/continue an existing session. Omit to start a new one (its id is
    /// printed). Required for `interrupt.resume` and `tool.result`/`tool.error`.
    #[arg(long)]
    session: Option<String>,
    /// LLM provider for engine-executed `llm.call` actions. Omit when the worker
    /// makes its own LLM calls.
    #[arg(long, value_enum)]
    provider: Option<LlmProviderArg>,
    /// SQLite dev database path.
    #[arg(long, default_value = "data.db")]
    db: String,
    /// Signing secret if the worker verifies webhook signatures.
    #[arg(long)]
    signing_secret: Option<String>,
    /// Output mode. (Engine logs go to stderr at error level; set RUST_LOG=info for more.)
    #[arg(long, short = 'o', value_enum, default_value_t = OutputMode::AgUi)]
    output: OutputMode,
}

#[derive(Copy, Clone, ValueEnum)]
enum OutputMode {
    /// Stream AG-UI protocol events, one JSON object per line.
    AgUi,
    /// Stream raw persisted engine events, one JSON object per line.
    Jsonl,
    /// Human-readable text: streamed replies, tool calls, and results.
    Pretty,
}

/// Where translated AG-UI events go. `Jsonl` renders nothing here — its raw
/// engine events are written straight to stdout in the run loop instead.
enum Renderer {
    AgUi,
    Jsonl,
    Pretty(PrettyPrinter),
}

impl Renderer {
    fn emit(&mut self, stdout: &mut std::io::Stdout, events: Vec<AgUiEvent>) -> anyhow::Result<()> {
        match self {
            Renderer::AgUi => {
                for ev in events {
                    write_json(stdout, &ev)?;
                }
            }
            Renderer::Pretty(printer) => {
                for ev in &events {
                    printer.render(stdout, ev)?;
                }
            }
            Renderer::Jsonl => {}
        }
        Ok(())
    }
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

fn write_json<T: serde::Serialize>(stdout: &mut std::io::Stdout, value: &T) -> anyhow::Result<()> {
    serde_json::to_writer(&mut *stdout, value)?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
}

pub async fn run(args: RunArgs) -> anyhow::Result<()> {
    // Captured for the resume hint printed at the end, before the args are consumed.
    let agent = args.agent.clone();
    let provider = args
        .provider
        .and_then(|p| p.to_possible_value())
        .map(|v| v.get_name().to_string());
    let output = args
        .output
        .to_possible_value()
        .map(|v| v.get_name().to_string());

    let input = parse_input(&args.input, args.agent)?;

    // dev=true: `run` exposes no server, so no client/worker auth env is required.
    let env = match EnvVars::load(args.provider, true) {
        Ok(e) => e,
        Err(_) => std::process::exit(2),
    };

    if let Some(parent) = std::path::Path::new(&args.db).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let (rt, adapter) = local::start_engine(&args.db, env.provider).await?;

    let worker_url = args
        .worker_url
        .or_else(|| std::env::var("SUBSTRUCTURE_WORKER_URL").ok())
        .unwrap_or_else(|| "http://localhost:3000/agent".to_string());
    register_startup_worker(&adapter, &worker_url, args.signing_secret).await?;

    let session_id = args.session.unwrap_or_else(|| Uuid::now_v7().to_string());

    let caller = Caller::System {
        tenant_id: DEFAULT_TENANT.to_string(),
    };
    let owner = SessionOwner {
        tenant_id: DEFAULT_TENANT.to_string(),
        id: Some("dev".to_string()),
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
                root_session_id: session_id.clone(),
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
    let raw = matches!(args.output, OutputMode::Jsonl);
    let mut renderer = match args.output {
        OutputMode::AgUi => Renderer::AgUi,
        OutputMode::Jsonl => Renderer::Jsonl,
        OutputMode::Pretty => Renderer::Pretty(PrettyPrinter::new(stdout.is_terminal())),
    };

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
                let payload = event.payload;
                // Drain queued deltas before `llm.call.completed` so no closing
                // event outruns its last streamed fragment.
                if matches!(payload, EventPayload::LlmCallCompleted(_)) {
                    while let Ok(d) = deltas.try_recv() {
                        let evs = translator.on_delta(d);
                        renderer.emit(&mut stdout, evs)?;
                    }
                }
                let evs = translator.on_event(payload);
                renderer.emit(&mut stdout, evs)?;
                if translator.terminated {
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

    if translator.terminated {
        let hint = format!(
            "continue this session with:\n  {}",
            resume_command(
                &program_name(),
                &session_id,
                &worker_url,
                agent.as_deref(),
                provider.as_deref(),
                output.as_deref(),
                &args.db,
            )
        );
        // Faint so the hint reads as secondary; plain when piped.
        if std::io::stderr().is_terminal() {
            eprintln!("\n\x1b[2m{hint}\x1b[0m");
        } else {
            eprintln!("\n{hint}");
        }
        Ok(())
    } else {
        anyhow::bail!("event stream ended before the run finished")
    }
}

/// The running binary's name, read from the executable itself so a future rename of
/// the `[[bin]]` flows through without touching this code.
fn program_name() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "subs".into())
}

/// The invocation that resumes this session with the next turn. `--output` and
/// `--db` are echoed only when non-default so the common case stays short.
fn resume_command(
    program: &str,
    session: &str,
    worker_url: &str,
    agent: Option<&str>,
    provider: Option<&str>,
    output: Option<&str>,
    db: &str,
) -> String {
    let mut cmd = format!("{program} run --session {session} --worker-url {worker_url}");
    if let Some(agent) = agent {
        cmd.push_str(&format!(" --agent {agent}"));
    }
    if let Some(provider) = provider {
        cmd.push_str(&format!(" --provider {provider}"));
    }
    if let Some(output) = output.filter(|o| *o != "ag-ui") {
        cmd.push_str(&format!(" --output {output}"));
    }
    if db != "data.db" {
        cmd.push_str(&format!(" --db {db}"));
    }
    cmd.push_str(
        r#" --input '{"type":"client.message","message":{"role":"user","content":"..."}}'"#,
    );
    cmd
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resume_command_leads_with_program_and_omits_default_db() {
        let cmd = resume_command(
            "subs",
            "sess-1",
            "http://localhost:4444",
            Some("my-agent"),
            Some("anthropic"),
            Some("ag-ui"),
            "data.db",
        );
        assert_eq!(
            cmd,
            r#"subs run --session sess-1 --worker-url http://localhost:4444 --agent my-agent --provider anthropic --input '{"type":"client.message","message":{"role":"user","content":"..."}}'"#
        );
    }

    #[test]
    fn resume_command_uses_the_given_program_name() {
        let cmd = resume_command("renamed-cli", "s", "w", None, None, None, "data.db");
        assert!(cmd.starts_with("renamed-cli run "), "{cmd}");
    }

    #[test]
    fn resume_command_echoes_non_default_db_and_skips_absent_flags() {
        let cmd = resume_command(
            "subs",
            "sess-2",
            "http://localhost:4444",
            None,
            None,
            Some("ag-ui"),
            "other.db",
        );
        assert!(cmd.contains(" --db other.db"), "{cmd}");
        assert!(!cmd.contains("--agent"), "{cmd}");
        assert!(!cmd.contains("--provider"), "{cmd}");
        assert!(!cmd.contains("--output"), "{cmd}");
    }

    #[test]
    fn resume_command_echoes_non_default_output_and_omits_the_default() {
        let pretty = resume_command("subs", "s", "w", None, None, Some("pretty"), "data.db");
        assert!(pretty.contains(" --output pretty"), "{pretty}");

        let default = resume_command("subs", "s", "w", None, None, Some("ag-ui"), "data.db");
        assert!(!default.contains("--output"), "{default}");
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
