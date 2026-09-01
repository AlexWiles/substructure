use clap::Args;
use uuid::Uuid;

use super::cloud::project_config;
use super::env::OutputFormat;
use super::output::{self, Renderer};
use super::resume_hint::print_resume_hint;
use super::turns::{self, declared_agent, message_input, Open};
use crate::protocol::ClientInput;

#[derive(Args)]
pub struct RunArgs {
    /// Agent id to run, naming an `[agent.<id>]` section.
    #[arg(value_name = "AGENT")]
    agent: String,
    #[arg(value_name = "MESSAGE")]
    message: Option<String>,
    #[arg(long, conflicts_with = "message")]
    input: Option<String>,
    #[arg(long)]
    session: Option<String>,
    #[arg(short = 'c', long)]
    config: Option<std::path::PathBuf>,
    #[arg(long)]
    url: Option<String>,
    #[arg(long)]
    db: Option<String>,
    #[arg(long, short = 'o', value_enum)]
    output: Option<OutputFormat>,
}

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
    let cfg = project_config::load(args.config.as_deref())?;

    let output_mode = args.output.unwrap_or_else(OutputFormat::for_stdout);

    let agent_id = declared_agent(args.agent, &cfg)?;

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

    let status = super::output::Status::start();
    let mut renderer = Renderer::new(output_mode, output::color()).with_status(status.clone());
    let driven = turns.drive(input, &mut renderer).await;
    status.stop();
    driven?;

    turns.wait_for_index().await;
    print_resume_hint(&session_id, payload.as_deref());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_named_agent_fills_agent_id_on_a_submit() {
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
    fn input_agent_id_wins_over_the_named_agent() {
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
