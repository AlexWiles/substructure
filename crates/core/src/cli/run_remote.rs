//! Driving one turn on the deployment the file names.
//!
//! One request submits the input and streams the turn. A client that submits
//! first and subscribes second loses the start of its turn.

use anyhow::{Context as _, Result};

use crate::api::v1::{RunFormat, RunRequest, RUN_DONE_EVENT};
use crate::protocol::ClientInput;
use crate::session::SessionEvent;
use crate::transport::ag_ui::events::AgUiEvent;

use super::cloud::context::Context;
use super::env::OutputFormat;
use super::output::Renderer;
use super::output::{unfinished, TurnEnd, TurnRender};

pub(crate) fn format_for(output: OutputFormat) -> RunFormat {
    match output {
        OutputFormat::Jsonl => RunFormat::Events,
        _ => RunFormat::AgUi,
    }
}

/// Submit one input to the deployment and render the turn it opens, to its end.
pub(crate) async fn drive(
    ctx: &Context,
    project: &str,
    session_id: &str,
    input: ClientInput,
    renderer: &mut Renderer,
    format: RunFormat,
) -> Result<TurnEnd> {
    let body = RunRequest {
        session_id: Some(session_id.to_string()),
        input,
    };

    let mut failed: Option<anyhow::Error> = None;
    let mut done_event = false;
    let mut render = TurnRender::new(renderer);

    ctx.client
        .post_sse(
            &format!(
                "/api/v1/projects/{project}/run?format={}",
                format.as_query()
            ),
            &body,
            |line| {
                if line.trim_end() == format!("event: {RUN_DONE_EVENT}") {
                    done_event = true;
                    return;
                }
                let Some(data) = line.strip_prefix("data:") else {
                    return;
                };
                let data = data.trim();
                if failed.is_some() || data.is_empty() {
                    return;
                }
                let rendered = match format {
                    RunFormat::Events => raw(&mut render, data),
                    RunFormat::AgUi => translated(&mut render, data),
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
    if !ended(format, done_event, &render) {
        return Err(unfinished());
    }
    Ok(render.into_end())
}

/// `format=events` carries no AG-UI events, so the route marks its end.
/// `format=ag-ui` ends on the AG-UI terminal event.
fn ended(format: RunFormat, done_event: bool, render: &TurnRender<'_>) -> bool {
    match format {
        RunFormat::Events => done_event,
        RunFormat::AgUi => render.terminated(),
    }
}

fn raw(render: &mut TurnRender<'_>, data: &str) -> Result<()> {
    let event: SessionEvent =
        serde_json::from_str(data).context("the deployment sent an event this CLI cannot read")?;
    render.raw(&event)
}

fn translated(render: &mut TurnRender<'_>, data: &str) -> Result<()> {
    let event: AgUiEvent = serde_json::from_str(data)
        .context("the deployment sent an AG-UI event this CLI cannot read")?;
    render.accept(vec![event])
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
        assert!(
            serde_json::from_value::<RunFormat>(serde_json::json!("ag_ui")).is_err(),
            "a near miss is a miss"
        );
    }

    #[test]
    fn an_unknown_format_is_refused() {
        let parsed: Result<RunFormat, _> =
            serde_json::from_value(serde_json::json!("something-new"));
        assert!(parsed.is_err());
    }

    #[test]
    fn the_output_mode_picks_what_the_deployment_streams() {
        assert_eq!(format_for(OutputFormat::Jsonl), RunFormat::Events);
        assert_eq!(format_for(OutputFormat::Pretty), RunFormat::AgUi);
        assert_eq!(format_for(OutputFormat::AgUi), RunFormat::AgUi);
    }

    fn saw(event: &AgUiEvent) -> bool {
        let json = serde_json::to_string(event).unwrap();
        let mut renderer = Renderer::new(OutputFormat::AgUi, false);
        let mut render = TurnRender::new(&mut renderer);
        translated(&mut render, &json).unwrap();
        render.terminated()
    }

    #[test]
    fn a_format_ends_where_that_format_ends() {
        let mut renderer = Renderer::new(OutputFormat::AgUi, false);
        let render = TurnRender::new(&mut renderer);
        assert!(ended(RunFormat::Events, true, &render));
        assert!(!ended(RunFormat::Events, false, &render));
        assert!(!ended(RunFormat::AgUi, true, &render));
    }

    #[test]
    fn only_a_terminal_event_ends_a_translated_run() {
        assert!(!saw(&AgUiEvent::TextMessageContent {
            message_id: "m".into(),
            delta: "hi".into(),
        }));
        assert!(saw(&AgUiEvent::RunFinished {
            thread_id: "s".into(),
            run_id: "r".into(),
            result: None,
            outcome: None,
            metadata: None,
        }));
        // An error is an ending too: the reader has been told what happened.
        assert!(saw(&AgUiEvent::RunError {
            message: "no".into(),
        }));
    }
}
