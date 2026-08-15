//! Auth: a connection lost its access and a person must restore it. The
//! prompt's channel renders the message; resolving it re-fetches the tools,
//! because a replaced credential reaches the agent only through a new fetch.

use super::InterruptKind;
use crate::connectors::AuthNeed;
use crate::protocol::{AuthFailure, DecisionAction};
use crate::runtime::session::state::SessionState;

pub const PREFIX: &str = "mcp-auth:";

pub struct Auth;

impl InterruptKind for Auth {
    fn prefix(&self) -> &'static str {
        PREFIX
    }

    fn on_resolved(&self, tail: &str) -> Vec<DecisionAction> {
        vec![DecisionAction::SyncConnector {
            id: tail.to_string(),
        }]
    }
}

/// Derived from the connection, so a redelivery proposes the same id and the
/// engine keeps one prompt.
pub fn interrupt_id(connection: &str) -> String {
    format!("{PREFIX}{connection}")
}

/// The first connection that needs a person and has not been asked about.
/// A connection configured to degrade never stops the session.
pub fn needing(state: &SessionState) -> Option<(String, AuthNeed)> {
    let config = state.resolve_agent_for(state.head_id.as_deref())?;
    config.mcp.iter().find_map(|server| {
        if server.auth_failure == AuthFailure::Degrade {
            return None;
        }
        let need = state.connector_sync(&server.id)?.auth?;
        state
            .open_interrupt(&interrupt_id(&server.id))
            .is_none()
            .then(|| (server.id.clone(), need))
    })
}
