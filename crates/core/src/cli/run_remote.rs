//! `subs run` against the deployment the file names.
//!
//! The same turn, somewhere else. What differs is only where the events come
//! from: an engine here hands them over in process, and a deployment streams
//! them over one request that both submits the input and watches it — a client
//! that submitted and then subscribed would have missed the start of its own
//! turn.
//!
//! No credential of its own. The operator surface is the one the CLI already
//! authenticates against, so a turn run from a terminal needs what `subs login`
//! stored and nothing else.
//!
//! The rendering is `run`'s, unchanged: `pretty` and `ag-ui` read the AG-UI
//! events the deployment translates (token deltas included, which is what makes
//! a reply arrive as it is written), and `jsonl` reads the engine's own events.

use anyhow::{Context as _, Result};

use crate::api::v1::{RunFormat, RunRequest};
use crate::protocol::ClientInput;
use crate::session::SessionEvent;
use crate::transport::ag_ui::events::AgUiEvent;

use super::cloud::context::Context;
use super::cloud::{CloudGlobals, ProjectScope};
use super::env::OutputFormat;
use super::pretty::{self, write_json, Renderer};

pub struct Run {
    pub config: Option<std::path::PathBuf>,
    pub session_id: Option<String>,
    pub input: ClientInput,
    pub output: OutputFormat,
    pub agent: String,
}

/// What the deployment should stream for a given output mode.
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
        globals: CloudGlobals {
            config: cmd.config,
            ..Default::default()
        },
    };
    let (ctx, project) = Context::from_project(&scope).await?;

    // `jsonl` prints what the engine stored, so it asks for that; the rendered
    // modes take the deployment's translation, which carries the token deltas
    // no stored event has.
    let format = format_for(cmd.output);
    let query = match format {
        RunFormat::Events => "?format=events",
        RunFormat::AgUi => "?format=ag-ui",
    };

    let body = RunRequest {
        session_id: cmd.session_id.clone(),
        input: cmd.input,
    };

    let mut stdout = std::io::stdout();
    let mut renderer = Renderer::new(cmd.output, pretty::color());
    let mut session_id = cmd.session_id;
    let mut failed: Option<anyhow::Error> = None;

    ctx.client
        .post_sse(
            &format!("/api/v1/projects/{project}/run{query}"),
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
                    RunFormat::Events => raw(&mut stdout, data, &mut session_id),
                    RunFormat::AgUi => {
                        translated(&mut stdout, &mut renderer, data, &mut session_id)
                    }
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

    if let Some(session_id) = session_id {
        super::run::print_resume_hint(&session_id, &cmd.agent, cmd.output, None);
    }
    Ok(())
}

/// A stored event, printed as it came. Also where a `jsonl` run learns the
/// session it opened, since nothing translated is there to say it.
fn raw(stdout: &mut std::io::Stdout, data: &str, session_id: &mut Option<String>) -> Result<()> {
    let event: SessionEvent =
        serde_json::from_str(data).context("the deployment sent an event this CLI cannot read")?;
    if session_id.is_none() {
        *session_id = Some(event.session_id.clone());
    }
    write_json(stdout, &event)
}

/// An AG-UI event the deployment translated, rendered by the same printer a
/// local run uses. A run's first event names the thread it opened, which is the
/// session to continue.
fn translated(
    stdout: &mut std::io::Stdout,
    renderer: &mut Renderer,
    data: &str,
    session_id: &mut Option<String>,
) -> Result<()> {
    let event: AgUiEvent = serde_json::from_str(data)
        .context("the deployment sent an AG-UI event this CLI cannot read")?;
    if let (None, AgUiEvent::RunStarted { thread_id, .. }) = (&session_id, &event) {
        *session_id = Some(thread_id.clone());
    }
    renderer.emit(stdout, vec![event])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `jsonl` means the engine's own events, so it asks the deployment for
    /// those; the rendered modes take the translation, which carries the token
    /// deltas no stored event has.
    #[test]
    fn the_output_mode_picks_what_the_deployment_streams() {
        assert_eq!(format_for(OutputFormat::Jsonl), RunFormat::Events);
        assert_eq!(format_for(OutputFormat::Pretty), RunFormat::AgUi);
        assert_eq!(format_for(OutputFormat::AgUi), RunFormat::AgUi);
    }

    /// A run that opened a session learns which one from the stream, so the
    /// hint can offer to continue it.
    #[test]
    fn a_new_session_is_learned_from_the_first_event() {
        let mut session_id = None;
        let mut stdout = std::io::stdout();
        let mut renderer = Renderer::new(OutputFormat::AgUi, false);
        let started = serde_json::to_string(&AgUiEvent::RunStarted {
            thread_id: "sess-1".into(),
            run_id: "turn-1".into(),
        })
        .unwrap();

        translated(&mut stdout, &mut renderer, &started, &mut session_id).unwrap();
        assert_eq!(session_id.as_deref(), Some("sess-1"));
    }
}
