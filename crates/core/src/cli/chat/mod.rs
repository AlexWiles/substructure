use std::io::IsTerminal;

use anyhow::{Context as _, Result};
use clap::Args;
use dialoguer::{theme::ColorfulTheme, Input, Select};
use rustyline::error::ReadlineError;
use serde_json::Value;
use uuid::Uuid;

use crate::protocol::{
    ClientInput, InterruptOption, InterruptResolution, InterruptResponder, InterruptResumption,
    ResumeStatus,
};
use crate::session::tool_contract;
use crate::transport::ag_ui::events::AgUiInterrupt;
use crate::transport::channel::ChannelKind;

use crate::manifest::McpRef;

use super::cloud::project_config::ProjectConfig;
use super::cloud::{project_config, CloudGlobals};
use super::env::OutputFormat;
use super::output::Status;
use super::output::{self, Renderer};
use super::resume_hint::print_resume_hint;
use super::turns::{self, declared_agent, message_input, Open, Turns};

mod editor;
use editor::ChatEditor;

const PROMPT: &str = "> ";

#[derive(Args)]
pub struct ChatArgs {
    /// Agent id to chat with, naming an `[agent.<id>]` section.
    #[arg(value_name = "AGENT")]
    agent: String,
    /// Resume an existing session. Omit to start a new one (its id is
    /// printed).
    #[arg(long)]
    session: Option<String>,
    /// Environment file (default: walks up from cwd looking for
    /// `subs.toml`).
    #[arg(short = 'c', long)]
    config: Option<std::path::PathBuf>,
    /// Chat with the deployment at this URL. Point it at a `subs serve` to use
    /// the engine that runs there.
    #[arg(long)]
    url: Option<String>,
    /// SQLite dev database path. [default: `db` in `subs.toml`, else
    /// `~/.config/subs/subs.db`]
    #[arg(long)]
    db: Option<String>,
}

impl ChatArgs {
    pub fn config_path(&self) -> Option<&std::path::Path> {
        self.config.as_deref()
    }
}

pub async fn chat(args: ChatArgs) -> Result<()> {
    let cfg = project_config::load(args.config.as_deref())?;
    let agent_id = declared_agent(args.agent, &cfg)?;

    if !std::io::stdin().is_terminal() {
        anyhow::bail!("chat needs a terminal. Send one message with `subs run` instead.");
    }

    let globals = CloudGlobals {
        config: args.config.clone(),
        url: args.url.clone(),
        ..Default::default()
    };

    let session_id = args.session.unwrap_or_else(|| Uuid::now_v7().to_string());
    let mut turns = turns::open(
        &cfg,
        Open {
            globals,
            session: session_id.clone(),
            db: args.db,
            output: OutputFormat::Pretty,
        },
    )
    .await?;

    banner(&cfg, &agent_id, &session_id);
    repl(turns.as_mut(), &agent_id, &session_id).await
}

async fn repl(turns: &mut dyn Turns, agent_id: &str, session_id: &str) -> Result<()> {
    let mut editor = editor::build().context("terminal input")?;
    let history = history_path();
    if let Some(path) = &history {
        if let Err(e) = editor.load_history(path) {
            tracing::debug!(path = %path.display(), error = %e, "no chat history read");
        }
    }

    let status = Status::start();
    let mut renderer = Renderer::new(OutputFormat::Pretty, output::color())
        .at_a_prompt()
        .with_status(status.clone());

    let parked = turns.parked().await?;
    let mut open = settle(turns, parked, &mut renderer).await?;
    status.idle();

    while open {
        let (returned, line) = read_line(editor).await?;
        editor = returned;
        let Some(line) = line else { break };
        let line = editor::joined(line.trim());
        if line.is_empty() {
            continue;
        }
        let kept = editor.add_history_entry(&line).and_then(|_| {
            history
                .as_deref()
                .map_or(Ok(()), |path| editor.append_history(path))
        });
        if let Err(e) = kept {
            tracing::debug!(error = %e, "line not kept in history");
        }

        let input = message_input(line, agent_id.to_string());
        let end = tokio::select! {
            driven = turns.drive(input, &mut renderer) => driven,
            _ = tokio::signal::ctrl_c() => {
                status.stop();
                output::note("\nstopped watching. The turn keeps running — resume the session to read it.");
                break;
            }
        };
        status.idle();
        open = settle(turns, end?.interrupts, &mut renderer).await?;
        status.idle();
        println!();
    }

    status.stop();
    turns.wait_for_index().await;
    print_resume_hint(session_id, None);
    Ok(())
}

async fn settle(
    turns: &mut dyn Turns,
    mut interrupts: Vec<AgUiInterrupt>,
    renderer: &mut Renderer,
) -> Result<bool> {
    while let Some(interrupt) = interrupts.first().cloned() {
        let Some(input) = answer(interrupt).await? else {
            return Ok(false);
        };
        interrupts = turns.drive(input, renderer).await?.interrupts;
    }
    Ok(true)
}

async fn read_line(mut editor: ChatEditor) -> Result<(ChatEditor, Option<String>)> {
    tokio::task::spawn_blocking(move || match editor.readline(PROMPT) {
        Ok(line) => Ok((editor, Some(line))),
        Err(ReadlineError::Interrupted | ReadlineError::Eof) => Ok((editor, None)),
        Err(e) => Err(anyhow::Error::new(e).context("terminal input")),
    })
    .await?
}

async fn answer(interrupt: AgUiInterrupt) -> Result<Option<ClientInput>> {
    let message = interrupt
        .message
        .clone()
        .unwrap_or_else(|| interrupt.reason.clone());
    let options = options_of(&interrupt);
    let schema = interrupt.response_schema.clone();
    let id = interrupt.id.clone();

    tokio::task::spawn_blocking(move || {
        let (payload, label, style) = if options.is_empty() {
            let Some(typed) = typed_answer(&message, schema.as_ref())? else {
                return Ok(None);
            };
            (typed, None, None)
        } else {
            let labels: Vec<&str> = options.iter().map(|o| o.label.as_str()).collect();
            let picked = Select::with_theme(&ColorfulTheme::default())
                .with_prompt(message.as_str())
                .items(&labels)
                .default(0)
                .interact();
            let Some(pick) = quit_or(picked)? else {
                return Ok(None);
            };
            let chosen = &options[pick];
            (
                chosen.value.clone(),
                Some(chosen.label.clone()),
                chosen.style.clone(),
            )
        };
        resume(id, payload, label, style).map(Some)
    })
    .await?
}

fn typed_answer(message: &str, schema: Option<&Value>) -> Result<Option<Value>> {
    loop {
        let typed = Input::<String>::with_theme(&ColorfulTheme::default())
            .with_prompt(message)
            .interact_text();
        let Some(typed) = quit_or(typed)? else {
            return Ok(None);
        };

        let Some(schema) = schema else {
            return Ok(Some(Value::String(typed)));
        };
        match tool_contract::output_violation(schema, &typed) {
            None => return Ok(Some(as_payload(typed))),
            Some(violation) => output::note(&format!("that answer does not fit: {violation}")),
        }
    }
}

fn quit_or<T>(result: dialoguer::Result<T>) -> Result<Option<T>> {
    match result {
        Ok(v) => Ok(Some(v)),
        Err(dialoguer::Error::IO(e)) if e.kind() == std::io::ErrorKind::Interrupted => Ok(None),
        Err(e) => Err(anyhow::Error::new(e).context("interrupt prompt")),
    }
}

fn as_payload(typed: String) -> Value {
    serde_json::from_str(&typed).unwrap_or(Value::String(typed))
}

fn options_of(interrupt: &AgUiInterrupt) -> Vec<InterruptOption> {
    let Some(raw) = interrupt.metadata.as_ref().and_then(|m| m.get("options")) else {
        return Vec::new();
    };
    match serde_json::from_value(raw.clone()) {
        Ok(options) => options,
        Err(e) => {
            tracing::warn!(interrupt = %interrupt.id, error = %e, "unreadable interrupt options");
            Vec::new()
        }
    }
}

fn resume(
    interrupt_id: String,
    payload: Value,
    label: Option<String>,
    style: Option<String>,
) -> Result<ClientInput> {
    let resolution = InterruptResolution {
        status: ResumeStatus::Resolved,
        payload,
        responder: Some(InterruptResponder {
            channel: ChannelKind::CLI.to_string(),
            user: None,
            label,
            style,
        }),
    };
    Ok(ClientInput::InterruptResume {
        resumption: InterruptResumption {
            interrupt_id,
            payload: serde_json::to_value(resolution).context("interrupt resolution")?,
        },
    })
}

fn banner(cfg: &ProjectConfig, agent_id: &str, session_id: &str) {
    output::note(agent_id);
    for (label, value) in agent_rows(cfg, agent_id, session_id) {
        output::note(&format!("  {label:<8}{value}"));
    }
    eprintln!();
}

fn agent_rows(
    cfg: &ProjectConfig,
    agent_id: &str,
    session_id: &str,
) -> Vec<(&'static str, String)> {
    let mut rows = Vec::new();
    let Some(agent) = cfg.agent.get(agent_id) else {
        rows.push(("session", session_id.to_string()));
        return rows;
    };

    if let Some(model) = &agent.model {
        rows.push(("model", model.clone()));
    }
    if let Some(llm) = &agent.llm {
        let named = match cfg.llm.get(llm) {
            Some(spec) => format!("{llm} ({})", spec.kind.name()),
            None => llm.clone(),
        };
        rows.push(("llm", named));
    }
    if let Some(effort) = &agent.effort {
        rows.push(("effort", format!("{effort:?}").to_lowercase()));
    }
    if let Some(worker) = &agent.worker {
        let shown = match cfg.worker.get(worker).and_then(|w| w.url.clone()) {
            Some(url) => format!("{worker} ({url})"),
            None => worker.clone(),
        };
        rows.push(("worker", shown));
    }
    let mcp: Vec<&str> = agent.mcp.iter().map(McpRef::id).collect();
    if !mcp.is_empty() {
        rows.push(("mcp", mcp.join(", ")));
    }
    if !agent.subagents.is_empty() {
        let subagents: Vec<&str> = agent.subagents.iter().map(|s| s.id()).collect();
        rows.push(("agents", subagents.join(", ")));
    }
    rows.push(("session", session_id.to_string()));
    rows
}

fn history_path() -> Option<std::path::PathBuf> {
    let dir = super::cloud::credentials::config_dir().ok()?;
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::debug!(dir = %dir.display(), error = %e, "no chat history dir");
        return None;
    }
    Some(dir.join("chat_history"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn interrupt(metadata: Option<Value>) -> AgUiInterrupt {
        AgUiInterrupt {
            id: "int-1".into(),
            reason: "confirmation".into(),
            message: Some("Send the email?".into()),
            tool_call_id: None,
            response_schema: None,
            expires_at: None,
            metadata,
        }
    }

    fn config(toml: &str) -> ProjectConfig {
        toml::from_str(toml).expect("a readable file")
    }

    #[test]
    fn the_banner_reads_the_agent_out_of_the_file() {
        let cfg = config(
            r#"
            name = "example"
            [llm.claude]
            type = "anthropic"
            [worker.local]
            url = "http://localhost:4444"
            [agent.my-agent]
            llm = "claude"
            model = "claude-haiku-4-5-20251001"
            worker = "local"
            mcp = ["bash", { id = "files" }]
            "#,
        );
        assert_eq!(
            agent_rows(&cfg, "my-agent", "sess-1"),
            vec![
                ("model", "claude-haiku-4-5-20251001".to_string()),
                ("llm", "claude (anthropic)".to_string()),
                ("worker", "local (http://localhost:4444)".to_string()),
                ("mcp", "bash, files".to_string()),
                ("session", "sess-1".to_string()),
            ]
        );
    }

    #[test]
    fn what_the_file_leaves_out_gets_no_row() {
        let cfg = config(
            r#"
            name = "example"
            [worker.local]
            url = "http://localhost:4444"
            [agent.bare]
            worker = "local"
            "#,
        );
        assert_eq!(
            agent_rows(&cfg, "bare", "sess-1"),
            vec![
                ("worker", "local (http://localhost:4444)".to_string()),
                ("session", "sess-1".to_string()),
            ]
        );
    }

    #[test]
    fn an_undeclared_agent_still_names_its_session() {
        let cfg = config(r#"name = "example""#);
        assert_eq!(
            agent_rows(&cfg, "ghost", "sess-1"),
            vec![("session", "sess-1".to_string())]
        );
    }

    #[test]
    fn options_are_read_from_the_ag_ui_metadata() {
        let options = options_of(&interrupt(Some(serde_json::json!({
            "options": [
                { "label": "Approve", "value": { "approved": true }, "style": "primary" },
                { "label": "Deny", "value": { "approved": false } },
            ]
        }))));
        assert_eq!(options.len(), 2);
        assert_eq!(options[0].label, "Approve");
        assert_eq!(options[0].style.as_deref(), Some("primary"));
        assert_eq!(options[1].value, serde_json::json!({ "approved": false }));
    }

    #[test]
    fn malformed_options_are_no_options() {
        assert!(options_of(&interrupt(None)).is_empty());
        assert!(options_of(&interrupt(Some(serde_json::json!({ "options": "bad" })))).is_empty());
        assert!(options_of(&interrupt(Some(serde_json::json!({})))).is_empty());
    }

    #[test]
    fn a_pick_resumes_with_the_options_value_and_this_channels_stamp() {
        let input = resume(
            "int-1".into(),
            serde_json::json!({ "approved": true }),
            Some("Approve".into()),
            Some("primary".into()),
        )
        .unwrap();
        let ClientInput::InterruptResume { resumption } = input else {
            panic!("expected interrupt.resume");
        };
        assert_eq!(resumption.interrupt_id, "int-1");

        let resolution: InterruptResolution = serde_json::from_value(resumption.payload).unwrap();
        assert_eq!(resolution.status, ResumeStatus::Resolved);
        assert_eq!(resolution.payload, serde_json::json!({ "approved": true }));
        let responder = resolution.responder.expect("responder");
        assert_eq!(responder.channel, "cli");
        assert_eq!(responder.label.as_deref(), Some("Approve"));
        assert_eq!(responder.style.as_deref(), Some("primary"));
    }

    #[test]
    fn an_answer_a_schema_asked_for_is_sent_as_data() {
        assert_eq!(as_payload("true".into()), serde_json::json!(true));
        assert_eq!(as_payload("7".into()), serde_json::json!(7));
        assert_eq!(
            as_payload(r#"{"approved":true}"#.into()),
            serde_json::json!({ "approved": true })
        );
        assert_eq!(as_payload("ship it".into()), serde_json::json!("ship it"));
    }

    #[test]
    fn a_schema_rejects_the_wrong_shape() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "approved": { "type": "boolean" } },
            "required": ["approved"],
        });
        assert!(tool_contract::output_violation(&schema, "yes").is_some());
        assert!(tool_contract::output_violation(&schema, r#"{"approved":true}"#).is_none());
    }

    #[test]
    fn a_resume_names_no_user() {
        let input = resume("int-1".into(), Value::Null, None, None).unwrap();
        let ClientInput::InterruptResume { resumption } = input else {
            panic!("expected interrupt.resume");
        };
        let resolution: InterruptResolution = serde_json::from_value(resumption.payload).unwrap();
        assert!(resolution.responder.expect("responder").user.is_none());
    }

    #[test]
    fn ctrl_c_at_a_prompt_is_a_quit_not_a_failure() {
        let interrupted: dialoguer::Result<usize> = Err(dialoguer::Error::IO(std::io::Error::new(
            std::io::ErrorKind::Interrupted,
            "read interrupted",
        )));
        assert!(quit_or(interrupted).unwrap().is_none());

        assert_eq!(quit_or(Ok(3)).unwrap(), Some(3));

        let broken: dialoguer::Result<usize> =
            Err(dialoguer::Error::IO(std::io::Error::other("boom")));
        assert!(quit_or(broken).is_err());
    }

    #[test]
    fn typed_text_resumes_as_the_payload_with_no_pick_recorded() {
        let input = resume("int-1".into(), Value::String("ship it".into()), None, None).unwrap();
        let ClientInput::InterruptResume { resumption } = input else {
            panic!("expected interrupt.resume");
        };
        let resolution: InterruptResolution = serde_json::from_value(resumption.payload).unwrap();
        assert_eq!(resolution.payload, serde_json::json!("ship it"));
        let responder = resolution.responder.expect("responder");
        assert!(responder.label.is_none());
        assert!(responder.style.is_none());
    }
}
