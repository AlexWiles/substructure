//! `subs run` against the deployment the file names.
//!
//! One request submits the input and streams the turn. A client that submits
//! first and subscribes second loses the start of its turn.

use anyhow::{Context as _, Result};

use crate::api::v1::{RunFormat, RunRequest};
use crate::protocol::ClientInput;
use crate::session::SessionEvent;
use crate::transport::ag_ui::events::AgUiEvent;
use crate::transport::ag_ui::translator::AgUiTranslator;

use super::cloud::context::Context;
use super::cloud::{CloudGlobals, ProjectScope};
use super::env::OutputFormat;
use super::pretty::{self, write_json, Renderer};

pub struct Run {
    pub globals: CloudGlobals,
    pub session_id: Option<String>,
    pub input: ClientInput,
    pub output: OutputFormat,
    pub agent: String,
}

fn format_for(output: OutputFormat) -> RunFormat {
    match output {
        OutputFormat::Jsonl => RunFormat::Events,
        _ => RunFormat::AgUi,
    }
}

pub async fn run(cmd: Run) -> Result<()> {
    let scope = ProjectScope {
        org: None,
        project: None,
        globals: cmd.globals,
    };
    let (ctx, project) = Context::from_project(&scope).await?;

    let format = format_for(cmd.output);

    let body = RunRequest {
        session_id: cmd.session_id.clone(),
        input: cmd.input,
    };

    let mut stdout = std::io::stdout();
    let mut renderer = Renderer::new(cmd.output, pretty::color());
    let mut session_id = cmd.session_id;
    let mut failed: Option<anyhow::Error> = None;
    let mut finished = Finished::new(format);

    ctx.client
        .post_sse(
            &format!(
                "/api/v1/projects/{project}/run?format={}",
                format.as_query()
            ),
            &body,
            |line| {
                let Some(data) = line.strip_prefix("data:") else {
                    return;
                };
                let data = data.trim();
                if failed.is_some() || data.is_empty() {
                    return;
                }
                let rendered = match format {
                    RunFormat::Events => raw(&mut stdout, data, &mut session_id, &mut finished),
                    RunFormat::AgUi => translated(
                        &mut stdout,
                        &mut renderer,
                        data,
                        &mut session_id,
                        &mut finished,
                    ),
                };
                if let Err(e) = rendered {
                    failed = Some(e);
                }
            },
        )
        .await?;

    if let Some(e) = failed {
        return Err(e);
    }
    if !finished.yes() {
        anyhow::bail!("event stream ended before the run finished");
    }

    if let Some(session_id) = session_id {
        super::run::print_resume_hint(&session_id, &cmd.agent, cmd.output, None);
    }
    Ok(())
}

/// Whether the turn ended. A deployment can close a stream before it does.
enum Finished {
    Translated(bool),
    Raw(Box<AgUiTranslator>),
}

impl Finished {
    fn new(format: RunFormat) -> Self {
        match format {
            RunFormat::AgUi => Finished::Translated(false),
            RunFormat::Events => {
                Finished::Raw(Box::new(AgUiTranslator::new(String::new(), String::new())))
            }
        }
    }

    fn saw(&mut self, event: &AgUiEvent) {
        if let Finished::Translated(done) = self {
            *done |= matches!(
                event,
                AgUiEvent::RunFinished { .. } | AgUiEvent::RunError { .. }
            );
        }
    }

    fn saw_raw(&mut self, event: SessionEvent) {
        if let Finished::Raw(oracle) = self {
            oracle.on_event(event.payload);
        }
    }

    fn yes(&self) -> bool {
        match self {
            Finished::Translated(done) => *done,
            Finished::Raw(oracle) => oracle.terminated,
        }
    }
}

fn raw(
    stdout: &mut std::io::Stdout,
    data: &str,
    session_id: &mut Option<String>,
    finished: &mut Finished,
) -> Result<()> {
    let event: SessionEvent =
        serde_json::from_str(data).context("the deployment sent an event this CLI cannot read")?;
    if session_id.is_none() {
        *session_id = Some(event.session_id.clone());
    }
    write_json(stdout, &event)?;
    finished.saw_raw(event);
    Ok(())
}

fn translated(
    stdout: &mut std::io::Stdout,
    renderer: &mut Renderer,
    data: &str,
    session_id: &mut Option<String>,
    finished: &mut Finished,
) -> Result<()> {
    let event: AgUiEvent = serde_json::from_str(data)
        .context("the deployment sent an AG-UI event this CLI cannot read")?;
    if let (None, AgUiEvent::RunStarted { thread_id, .. }) = (&session_id, &event) {
        *session_id = Some(thread_id.clone());
    }
    finished.saw(&event);
    renderer.emit(stdout, vec![event])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_query_value_is_the_name_the_route_parses() {
        for format in [RunFormat::AgUi, RunFormat::Events] {
            let parsed: RunFormat =
                serde_json::from_value(serde_json::json!(format.as_query())).unwrap();
            assert_eq!(parsed, format, "{} did not parse back", format.as_query());
        }
    }

    #[test]
    fn the_output_mode_picks_what_the_deployment_streams() {
        assert_eq!(format_for(OutputFormat::Jsonl), RunFormat::Events);
        assert_eq!(format_for(OutputFormat::Pretty), RunFormat::AgUi);
        assert_eq!(format_for(OutputFormat::AgUi), RunFormat::AgUi);
    }

    #[test]
    fn a_new_session_is_learned_from_the_first_event() {
        let mut session_id = None;
        let mut stdout = std::io::stdout();
        let mut renderer = Renderer::new(OutputFormat::AgUi, false);
        let mut finished = Finished::new(RunFormat::AgUi);
        let started = serde_json::to_string(&AgUiEvent::RunStarted {
            thread_id: "sess-1".into(),
            run_id: "turn-1".into(),
        })
        .unwrap();

        translated(
            &mut stdout,
            &mut renderer,
            &started,
            &mut session_id,
            &mut finished,
        )
        .unwrap();
        assert_eq!(session_id.as_deref(), Some("sess-1"));
    }

    #[test]
    fn a_translated_run_is_unfinished_until_it_says_otherwise() {
        let mut finished = Finished::new(RunFormat::AgUi);
        assert!(!finished.yes());

        finished.saw(&AgUiEvent::TextMessageContent {
            message_id: "m".into(),
            delta: "hi".into(),
        });
        assert!(!finished.yes(), "content is not an ending");

        finished.saw(&AgUiEvent::RunFinished {
            thread_id: "s".into(),
            run_id: "r".into(),
            result: None,
            outcome: None,
        });
        assert!(finished.yes());
    }

    #[test]
    fn a_run_error_ends_the_run() {
        let mut finished = Finished::new(RunFormat::AgUi);
        finished.saw(&AgUiEvent::RunError {
            message: "no".into(),
        });
        assert!(finished.yes());
    }
}
