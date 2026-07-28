//! Engine-side execution against connections: fetching what a connection
//! offers, and running the calls the model makes against it.
//!
//! The same queue → projection → executor shape as `llm` and `sub_agent`, for
//! the same reason: both are network calls the session aggregate cannot make
//! itself, so they are requested as events and settled as commands.

mod executor;
mod projection;
mod queue;

pub use executor::spawn_connector_task_executor;
pub use projection::spawn_connector_dispatch_processor;
pub use queue::ConnectorTask;
