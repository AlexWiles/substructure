use clap::Args;
use uuid::Uuid;

use super::cloud::project_config;
use super::env::OutputFormat;
use super::pretty::{self, Renderer};
use super::resume_hint::print_resume_hint;
use super::turns::{self, message_input, select_agent, Open};
use crate::protocol::ClientInput;

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

    let globals = super::cloud::CloudGlobals {
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
            output: output_mode,
        },
    )
    .await?;

    let mut renderer = Renderer::new(output_mode, pretty::color());
    turns.drive(input, &mut renderer).await?;

    turns.wait_for_index().await;
    print_resume_hint(&session_id, payload.as_deref());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
