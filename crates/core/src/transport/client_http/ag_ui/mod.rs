//! Native AG-UI protocol endpoint: stock AG-UI clients (CopilotKit, assistant-ui)
//! post a [`RunAgentInput`] and read back the AG-UI SSE event sequence. A pure
//! translation layer over the turn-submission and event-streaming runtime APIs.
//!
//! [`RunAgentInput`]: types::RunAgentInput
mod events;
mod route;
mod translator;
mod types;

pub use route::ag_ui_run;
