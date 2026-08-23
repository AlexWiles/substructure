//! Rendering a turn's events and recording how the turn ended. Shared by the
//! local and the remote path.

use crate::transport::ag_ui::events::{AgUiEvent, AgUiInterrupt, RunOutcome};

use super::{write_json, Renderer};

/// What a finished turn left for its caller to answer.
#[derive(Debug, Default)]
pub struct TurnEnd {
    /// The engine parks at most one interrupt per path.
    pub interrupts: Vec<AgUiInterrupt>,
    pub error: Option<String>,
}

impl TurnEnd {
    pub fn note(&mut self, events: &[AgUiEvent]) {
        for event in events {
            match event {
                AgUiEvent::RunFinished {
                    outcome: Some(RunOutcome::Interrupt { interrupts }),
                    ..
                } => self.interrupts = interrupts.clone(),
                AgUiEvent::RunError { message } => self.error = Some(message.clone()),
                _ => {}
            }
        }
    }
}

/// The events AG-UI defines as the end of a run.
fn is_terminal(event: &AgUiEvent) -> bool {
    matches!(
        event,
        AgUiEvent::RunFinished { .. } | AgUiEvent::RunError { .. }
    )
}

pub fn unfinished() -> anyhow::Error {
    anyhow::anyhow!("event stream ended before the run finished")
}

pub struct TurnRender<'a> {
    renderer: &'a mut Renderer,
    stdout: std::io::Stdout,
    end: TurnEnd,
    terminated: bool,
}

impl<'a> TurnRender<'a> {
    pub fn new(renderer: &'a mut Renderer) -> Self {
        Self {
            renderer,
            stdout: std::io::stdout(),
            end: TurnEnd::default(),
            terminated: false,
        }
    }

    pub fn accept(&mut self, events: Vec<AgUiEvent>) -> anyhow::Result<()> {
        self.end.note(&events);
        self.terminated |= events.iter().any(is_terminal);
        self.renderer.emit(&mut self.stdout, events)
    }

    /// The engine's own event, written only when `--output jsonl` asked for it.
    pub fn raw<T: serde::Serialize>(&mut self, value: &T) -> anyhow::Result<()> {
        if !self.renderer.is_raw() {
            return Ok(());
        }
        write_json(&mut self.stdout, value)
    }

    pub fn terminated(&self) -> bool {
        self.terminated
    }

    pub fn into_end(self) -> TurnEnd {
        self.end
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn interrupt(id: &str) -> AgUiInterrupt {
        AgUiInterrupt {
            id: id.into(),
            reason: "confirmation".into(),
            message: Some("Send the email?".into()),
            tool_call_id: None,
            response_schema: None,
            expires_at: None,
            metadata: None,
        }
    }

    #[test]
    fn a_parked_run_reports_what_it_parked_on() {
        let mut end = TurnEnd::default();
        end.note(&[AgUiEvent::RunFinished {
            thread_id: "t".into(),
            run_id: "r".into(),
            result: None,
            outcome: Some(RunOutcome::Interrupt {
                interrupts: vec![interrupt("int-1")],
            }),
            metadata: None,
        }]);
        assert_eq!(end.interrupts.len(), 1);
        assert_eq!(end.interrupts[0].id, "int-1");
        assert!(end.error.is_none());
    }

    #[test]
    fn a_clean_run_leaves_nothing_to_answer() {
        let mut end = TurnEnd::default();
        end.note(&[AgUiEvent::RunFinished {
            thread_id: "t".into(),
            run_id: "r".into(),
            result: None,
            outcome: Some(RunOutcome::Success),
            metadata: None,
        }]);
        assert!(end.interrupts.is_empty());
        assert!(end.error.is_none());
    }

    #[test]
    fn a_failed_run_reports_its_message() {
        let mut end = TurnEnd::default();
        end.note(&[AgUiEvent::RunError {
            message: "boom".into(),
        }]);
        assert_eq!(end.error.as_deref(), Some("boom"));
    }
}
