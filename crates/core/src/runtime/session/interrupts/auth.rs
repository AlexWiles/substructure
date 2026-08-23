//! Auth: a connection lost its access and a person must restore it. Resolving
//! the interrupt re-fetches the tools with the new credential.

use super::InterruptKind;
use crate::connectors::registry::ConnectionPath;
use crate::connectors::{AuthNeed, Requester};
use crate::protocol::{AuthFailure, DecisionAction};
use crate::runtime::session::state::SessionState;

pub const PREFIX: &str = "mcp-auth:";

/// What a channel builds a way in from: which connection, and who is
/// asking. Never a link. One is minted when the prompt is delivered, so the
/// log holds none.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Authorize {
    pub connection: ConnectionPath,
    pub requester: Requester,
}

pub struct Auth;

impl InterruptKind for Auth {
    fn prefix(&self) -> &'static str {
        PREFIX
    }

    fn on_resolved(&self, tail: &str) -> Vec<DecisionAction> {
        ConnectionPath::parse(tail)
            .map(|path| vec![DecisionAction::SyncConnector { path }])
            .unwrap_or_default()
    }
}

/// Derived from the connection, so a redelivery keeps one prompt.
pub fn interrupt_id(connection: &ConnectionPath) -> String {
    format!("{PREFIX}{connection}")
}

/// The first connection that needs a person and has not been asked about.
pub fn needing(state: &SessionState) -> Option<(ConnectionPath, AuthNeed)> {
    let leaf = state.head_id.clone();
    let config = state.resolve_agent_for(leaf.as_deref())?;
    state.servers_for(&config).into_iter().find_map(|server| {
        if server.auth_failure == AuthFailure::Degrade {
            return None;
        }
        let need = state.connector_sync(&server.path)?.auth?;
        state
            .open_interrupt(&interrupt_id(&server.path))
            .is_none()
            .then(|| (server.path.clone(), need))
    })
}
