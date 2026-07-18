pub mod agent_config;
mod aggregate;
pub mod command;
pub mod decision;
pub mod events;
pub mod index;
pub mod message;
pub mod propose;
pub mod reconcile;
pub mod state;
pub mod subscriptions;
pub mod tool_contract;
pub mod wire;

#[cfg(test)]
pub use aggregate::CommitContext;
pub use aggregate::{
    execute, ConflictRetry, ExecuteError, ExecuteInput, NewSessionEvent, SessionAggregate,
    SessionEvent,
};
