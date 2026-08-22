//! The line that says a turn is still running.
//!
//! A turn is silent between the model call and its first token, and again
//! while a tool runs. This draws one transient line on stderr through those
//! gaps and erases it before anything real is written, so the transcript keeps
//! the shape it has when nothing is watching.

use std::io::{IsTerminal, Write};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const TICK: Duration = Duration::from_millis(100);
const QUIET: Duration = Duration::from_millis(200);

const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";
const ERASE: &str = "\r\x1b[2K";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Phase {
    Idle,
    Thinking,
    Tool { name: String, about: Option<String> },
}

impl Phase {
    fn label(&self) -> Option<&str> {
        match self {
            Phase::Idle => None,
            Phase::Thinking => Some("thinking"),
            Phase::Tool { name, .. } => Some(name),
        }
    }

    fn about(&self) -> Option<&str> {
        match self {
            Phase::Tool { about, .. } => about.as_deref(),
            _ => None,
        }
    }
}

struct Inner {
    phase: Phase,
    since: Instant,
    last_write: Instant,
    at_line_start: bool,
    frame: usize,
    drawn: bool,
    stopped: bool,
}

#[derive(Clone)]
pub(crate) struct Status {
    inner: Option<Arc<Mutex<Inner>>>,
}

impl Status {
    pub(crate) fn disabled() -> Self {
        Self { inner: None }
    }

    pub(crate) fn start() -> Self {
        if !std::io::stderr().is_terminal() || tokio::runtime::Handle::try_current().is_err() {
            return Self::disabled();
        }
        let now = Instant::now();
        let inner = Arc::new(Mutex::new(Inner {
            phase: Phase::Idle,
            since: now,
            last_write: now,
            at_line_start: true,
            frame: 0,
            drawn: false,
            stopped: false,
        }));
        spawn_ticker(inner.clone());
        Self { inner: Some(inner) }
    }

    pub(crate) fn set(&self, phase: Phase) {
        let Some(inner) = &self.inner else { return };
        let mut state = inner.lock().unwrap();
        if state.phase == phase {
            return;
        }
        state.phase = phase;
        state.since = Instant::now();
        state.frame = 0;
        erase(&mut state);
    }

    pub(crate) fn writing(&self) {
        let Some(inner) = &self.inner else { return };
        let mut state = inner.lock().unwrap();
        state.last_write = Instant::now();
        erase(&mut state);
    }

    pub(crate) fn wrote(&self, at_line_start: bool) {
        let Some(inner) = &self.inner else { return };
        inner.lock().unwrap().at_line_start = at_line_start;
    }

    pub(crate) fn idle(&self) {
        self.set(Phase::Idle);
    }

    pub(crate) fn stop(&self) {
        let Some(inner) = &self.inner else { return };
        let mut state = inner.lock().unwrap();
        state.stopped = true;
        state.phase = Phase::Idle;
        erase(&mut state);
    }
}

fn spawn_ticker(inner: Arc<Mutex<Inner>>) {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(TICK);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            let line = {
                let mut state = inner.lock().unwrap();
                if state.stopped {
                    break;
                }
                match paint(&mut state) {
                    Some(line) => line,
                    None => continue,
                }
            };
            let mut err = std::io::stderr();
            let _ = err.write_all(line.as_bytes());
            let _ = err.flush();
        }
    });
}

fn paint(state: &mut Inner) -> Option<String> {
    let label = state.phase.label()?.to_string();
    if state.last_write.elapsed() < QUIET || !state.at_line_start {
        return None;
    }
    let frame = FRAMES[state.frame % FRAMES.len()];
    state.frame = state.frame.wrapping_add(1);
    state.drawn = true;
    let about = match state.phase.about() {
        Some(about) => format!("{about}, {}", elapsed(state.since)),
        None => elapsed(state.since),
    };
    Some(format!("{ERASE}{DIM}{frame} {label} ({about}){RESET}"))
}

fn erase(state: &mut Inner) {
    if !state.drawn {
        return;
    }
    state.drawn = false;
    let mut err = std::io::stderr();
    let _ = err.write_all(ERASE.as_bytes());
    let _ = err.flush();
}

fn elapsed(since: Instant) -> String {
    let secs = since.elapsed().as_secs();
    if secs < 60 {
        format!("{secs}s")
    } else {
        format!("{}m {:02}s", secs / 60, secs % 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idle() -> Inner {
        let now = Instant::now();
        Inner {
            phase: Phase::Idle,
            since: now,
            last_write: now - QUIET * 2,
            at_line_start: true,
            frame: 0,
            drawn: false,
            stopped: false,
        }
    }

    #[test]
    fn an_idle_turn_draws_nothing() {
        assert!(paint(&mut idle()).is_none());
    }

    #[test]
    fn a_running_turn_draws_a_frame_a_label_and_a_clock() {
        let mut state = idle();
        state.phase = Phase::Thinking;
        let line = paint(&mut state).expect("a line");
        assert!(line.contains("thinking"), "got {line}");
        assert!(line.contains("(0s)"), "got {line}");
        assert!(line.starts_with(ERASE), "got {line}");
        assert!(state.drawn);
    }

    #[test]
    fn a_retried_tool_says_which_try_it_is_on() {
        let mut state = idle();
        state.phase = Phase::Tool {
            name: "fetch_url".into(),
            about: Some("attempt 2".into()),
        };
        let line = paint(&mut state).expect("a line");
        assert!(line.contains("fetch_url (attempt 2, 0s)"), "got {line}");
    }

    #[test]
    fn a_running_tool_is_named() {
        let mut state = idle();
        state.phase = Phase::Tool {
            name: "read_file".into(),
            about: None,
        };
        let line = paint(&mut state).expect("a line");
        assert!(line.contains("read_file"), "got {line}");
    }

    #[test]
    fn a_recent_write_holds_the_ticker_off() {
        let mut state = idle();
        state.phase = Phase::Thinking;
        state.last_write = Instant::now();
        assert!(paint(&mut state).is_none());
    }

    #[test]
    fn a_half_written_line_is_never_painted_over() {
        let mut state = idle();
        state.phase = Phase::Thinking;
        state.at_line_start = false;
        assert!(paint(&mut state).is_none());

        state.at_line_start = true;
        assert!(paint(&mut state).is_some());
    }

    #[test]
    fn the_frame_advances_each_tick() {
        let mut state = idle();
        state.phase = Phase::Thinking;
        let first = paint(&mut state).expect("a line");
        state.last_write = Instant::now() - QUIET * 2;
        let second = paint(&mut state).expect("a line");
        assert_ne!(first, second);
    }

    #[test]
    fn a_disabled_status_never_panics() {
        let status = Status::disabled();
        status.set(Phase::Thinking);
        status.writing();
        status.stop();
    }

    #[test]
    fn a_long_step_reads_in_minutes() {
        assert_eq!(elapsed(Instant::now()), "0s");
        let old = Instant::now() - Duration::from_secs(75);
        assert_eq!(elapsed(old), "1m 15s");
    }
}
