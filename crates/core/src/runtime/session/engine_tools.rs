//! The engine's own tools, one module each: a constant definition (in
//! [`filter`]) and an answer read from state, so a replay answers the same.
//!
//! The dispatch is an exhaustive match on [`ConnectorToolKind`] — a new engine
//! tool is a new kind, a new definition, and a new arm here, and the compiler
//! holds the three together.

use super::state::{LocalAnswer, SessionState};
use crate::connectors::filter;
use crate::protocol::ConnectorToolKind;

/// The engine's answer to one of its own tools. `None` hands the call to its
/// target.
pub fn answer(
    state: &SessionState,
    kind: ConnectorToolKind,
    leaf: Option<&str>,
    arguments: &str,
) -> Option<LocalAnswer> {
    match kind {
        ConnectorToolKind::Remote => None,
        ConnectorToolKind::Find => Some(find(state, leaf, arguments)),
        ConnectorToolKind::Call => Some(call(state, leaf, arguments)),
    }
}

/// `tool_search`: BM25 over every tool the agent can reach, from state alone.
fn find(state: &SessionState, leaf: Option<&str>, arguments: &str) -> LocalAnswer {
    LocalAnswer::Result(filter::find_answer(
        &state.searchable_tools(leaf),
        &argument(arguments, "query"),
        state
            .resolve_agent_for(leaf)
            .map(|c| c.defer_settings())
            .unwrap_or_default()
            .max_matches,
    ))
}

/// `call_tool` reaching here could not be routed; the fault says why.
fn call(state: &SessionState, leaf: Option<&str>, arguments: &str) -> LocalAnswer {
    LocalAnswer::Error(
        state
            .call_tool_fault(arguments, leaf)
            .unwrap_or_else(|| "the call could not be routed".to_string()),
    )
}

fn argument(arguments: &str, key: &str) -> String {
    serde_json::from_str::<serde_json::Value>(arguments)
        .ok()
        .and_then(|v| v.get(key)?.as_str().map(str::to_string))
        .unwrap_or_default()
}
