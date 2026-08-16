//! Auth: a connection lost its access and a person must restore it. Resolving
//! the interrupt re-fetches the tools with the new credential.

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

/// Derived from the connection, so a redelivery keeps one prompt.
pub fn interrupt_id(connection: &str) -> String {
    format!("{PREFIX}{connection}")
}

/// The first connection that needs a person and has not been asked about.
pub fn needing(state: &SessionState) -> Option<(String, AuthNeed)> {
    let leaf = state.head_id.clone();
    let config = state.resolve_agent_for(leaf.as_deref())?;
    state.servers_for(&config).into_iter().find_map(|server| {
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
