//! Turning a turn's events into what a reader sees.

mod markdown;
mod reasoning;
mod status;
mod term;
mod tool;
mod transcript;
mod turn;

pub(crate) use status::Status;
pub(crate) use term::{color, note};
pub(crate) use transcript::PrettyPrinter;
pub(crate) use turn::{unfinished, TurnEnd, TurnRender};

use std::io::Write;

use super::env::OutputFormat;
use crate::transport::ag_ui::events::AgUiEvent;

pub(crate) enum Renderer {
    AgUi,
    Jsonl,
    Pretty(PrettyPrinter),
}

impl Renderer {
    pub(crate) fn new(output: OutputFormat, color: bool) -> Self {
        match output {
            OutputFormat::AgUi => Renderer::AgUi,
            OutputFormat::Jsonl => Renderer::Jsonl,
            OutputFormat::Pretty => Renderer::Pretty(PrettyPrinter::new(color)),
        }
    }

    pub(crate) fn is_raw(&self) -> bool {
        matches!(self, Renderer::Jsonl)
    }

    pub(crate) fn at_a_prompt(mut self) -> Self {
        if let Renderer::Pretty(printer) = &mut self {
            printer.at_a_prompt();
        }
        self
    }

    pub(crate) fn with_status(mut self, status: Status) -> Self {
        if let Renderer::Pretty(printer) = &mut self {
            printer.with_status(status);
        }
        self
    }

    pub(crate) fn emit(
        &mut self,
        stdout: &mut std::io::Stdout,
        events: Vec<AgUiEvent>,
    ) -> anyhow::Result<()> {
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

pub(crate) fn write_json<T: serde::Serialize>(
    stdout: &mut std::io::Stdout,
    value: &T,
) -> anyhow::Result<()> {
    let mut line = serde_json::to_vec(value)?;
    line.push(b'\n');
    stdout.write_all(&line)?;
    stdout.flush()?;
    Ok(())
}
