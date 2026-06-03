//! Native AG-UI protocol endpoint.
//!
//! Lets stock AG-UI clients (CopilotKit's `HttpAgent`, assistant-ui) drive a
//! substructure agent directly: a client posts a [`RunAgentInput`] and reads
//! back the AG-UI SSE event sequence. The implementation is a pure translation
//! layer over the existing turn-submission and event-streaming runtime APIs.
//!
//! [`RunAgentInput`]: types::RunAgentInput
mod events;
mod route;
mod translator;
mod types;

pub use route::ag_ui_run;
