//! Auth: a connection lost its access and a person must restore it. Resolving
//! the interrupt re-fetches the tools with the new credential.

use super::InterruptKind;
use crate::connectors::registry::ConnectionPath;
use crate::connectors::{AuthNeed, Requester};
use crate::protocol::{AuthFailure, DecisionAction};
use crate::runtime::session::state::SessionStateAtNode;

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
fn interrupt_id(connection: &ConnectionPath) -> String {
    format!("{PREFIX}{connection}")
}

/// The first connection that needs a person and has not been asked about.
fn needing(at: SessionStateAtNode) -> Option<(ConnectionPath, AuthNeed)> {
    let state = at.state();
    let config = at.resolve_agent_for()?;
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

pub fn prompt(at: SessionStateAtNode) -> Option<DecisionAction> {
    let (connection, need) = needing(at)?;

    let authorize = Authorize {
        requester: Requester::of_owner(at.state().owner.as_ref()),
        connection: connection.clone(),
    };

    Some(DecisionAction::Interrupt {
        interrupt_id: Some(interrupt_id(&connection)),
        reason: format!("connection `{connection}` needs authorizing"),
        payload: serde_json::json!({
            "message": ask(&connection, need),
            "authorize": authorize,
            "metadata": { "options": [{
                "label": "Retry",
                "value": { "connection": connection },
            }] },
        }),
    })
}

fn ask(connection: &ConnectionPath, need: AuthNeed) -> String {
    match need {
        AuthNeed::NeverAuthorized => {
            format!("`{connection}` is not authorized yet, so I cannot use it.")
        }
        AuthNeed::Reauthorize => {
            format!("`{connection}` needs to be authorized again. Its access expired.")
        }
        AuthNeed::TokenRejected => format!(
            "`{connection}` rejected its token. An operator must set a new one \
             with `subs auth {connection}`."
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::InterruptOrigin;
    use crate::protocol::{
        AgentConfig, AuthFailure, DecisionAction, Issuer, McpServer, RetryPolicy, SessionOwner,
        Subject,
    };
    use crate::runtime::session::events::{
        AgentConfigUpdated, ConnectorAuthFailed, ConnectorSyncRequested, EventPayload,
    };
    use crate::runtime::session::state::{ApplyContext, OpenInterrupt, SessionState};

    fn ctx() -> ApplyContext {
        ApplyContext {
            occurred_at: chrono::Utc::now(),
            sequence: 1,
        }
    }

    fn config(policy: AuthFailure) -> AgentConfig {
        AgentConfig {
            llm: None,
            model: "m1".to_string(),
            system: None,
            retry: None,
            tools: vec![],
            sub_agents: vec![],
            mcp: vec![McpServer {
                path: ConnectionPath::Mcp("sentry".into()),
                tools: None,
                auth_failure: policy,
                approve: Default::default(),
            }],
            defer_tools: None,
            announce_mcp: Default::default(),
            plugins: Vec::new(),
            effort: None,
        }
    }

    fn needing_auth(need: AuthNeed, policy: AuthFailure) -> SessionState {
        let mut s = SessionState::new("s1".to_string());
        s.owner = Some(SessionOwner {
            tenant_id: "t".to_string(),
            requester: crate::protocol::Requester::new(
                Subject::new(Issuer::slack(), "T1:U1"),
                Default::default(),
            ),
            metadata: Default::default(),
        });
        s.apply(
            &EventPayload::AgentConfigUpdated(AgentConfigUpdated {
                config: config(policy),
                anchor: None,
            }),
            &ctx(),
        );
        s.apply(
            &EventPayload::ConnectorSyncRequested(ConnectorSyncRequested {
                path: ConnectionPath::Mcp("sentry".into()),
                attempt: 0,
                retry: RetryPolicy::no_retry(),
            }),
            &ctx(),
        );
        s.apply(
            &EventPayload::ConnectorAuthFailed(ConnectorAuthFailed {
                path: ConnectionPath::Mcp("sentry".into()),
                auth: need,
            }),
            &ctx(),
        );
        s
    }

    #[test]
    fn a_connection_that_needs_a_person_is_asked_about_by_name() {
        let action = prompt(needing_auth(AuthNeed::Reauthorize, AuthFailure::Interrupt).at_head())
            .expect("it asks");
        let DecisionAction::Interrupt {
            interrupt_id,
            payload,
            ..
        } = action
        else {
            panic!("expected an interrupt");
        };
        assert_eq!(interrupt_id.as_deref(), Some("mcp-auth:mcp.sentry"));
        let message = payload["message"].as_str().expect("a message");
        assert!(message.contains("authorized again"), "got {message}");
        assert_eq!(payload["authorize"]["connection"], "mcp.sentry");
        assert_eq!(payload["metadata"]["options"][0]["label"], "Retry");
    }

    #[test]
    fn a_rejected_token_names_the_command_an_operator_runs() {
        let action =
            prompt(needing_auth(AuthNeed::TokenRejected, AuthFailure::Interrupt).at_head())
                .expect("it asks");
        let DecisionAction::Interrupt { payload, .. } = action else {
            panic!("expected an interrupt");
        };
        let message = payload["message"].as_str().expect("a message");
        assert!(message.contains("subs auth mcp.sentry"), "got {message}");
    }

    #[test]
    fn a_connection_configured_to_degrade_is_never_asked_about() {
        assert!(
            prompt(needing_auth(AuthNeed::Reauthorize, AuthFailure::Degrade).at_head()).is_none()
        );
    }

    #[test]
    fn a_connection_already_asked_about_is_not_asked_about_again() {
        let mut state = needing_auth(AuthNeed::Reauthorize, AuthFailure::Interrupt);
        state.open_interrupts.push(OpenInterrupt {
            interrupt_id: interrupt_id(&ConnectionPath::Mcp("sentry".into())),
            origin: InterruptOrigin::Frontend,
            reason: "hold".to_string(),
            payload: serde_json::Value::Null,
            anchor: None,
        });
        assert!(prompt(state.at_head()).is_none());
    }

    #[test]
    fn a_session_with_nothing_wrong_is_not_asked_about() {
        assert!(prompt(SessionState::new("s1".to_string()).at_head()).is_none());
    }
}
