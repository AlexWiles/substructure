use super::{fail, mismatched, void_events, KindSpec, Outcome, SettleError};
use crate::connectors::registry::ConnectionPath;
use crate::connectors::{filter, AuthNeed, RemoteTool};
use crate::protocol::{AgentConfig, RetryPolicy};
use crate::runtime::retry::RetryTarget;
use crate::runtime::session::events::*;
use crate::runtime::session::schedule::Dep;
use crate::runtime::session::state::EffectTracking;
use crate::runtime::session::state::{EffectKind, SessionState, SessionStateAtNode};

pub struct ConnectorSpec;

impl KindSpec for ConnectorSpec {
    fn kind(&self) -> EffectKind {
        EffectKind::ConnectorSync
    }

    fn settle(&self, state: &SessionState, id: &str, outcome: Outcome) -> Vec<EventPayload> {
        let Some(path) = ConnectionPath::parse(id) else {
            return Vec::new();
        };
        match outcome {
            Outcome::Connector {
                prefix,
                server,
                tools,
                instructions,
            } => {
                state.report_filter(&path, &tools, prefix.as_deref());
                vec![EventPayload::ConnectorSyncCompleted(Box::new(
                    ConnectorSyncCompleted {
                        path,
                        prefix,
                        server,
                        tools,
                        instructions,
                    },
                ))]
            }
            Outcome::Error(e) => fail(self, state, id, &e),
            other => mismatched(self.kind(), &other),
        }
    }

    fn errored(&self, state: &SessionState, id: &str, e: &SettleError) -> Option<EventPayload> {
        let path = ConnectionPath::parse(id)?;
        if let Some(t) = state.tracking(EffectKind::ConnectorSync, id) {
            report_failure(
                id,
                &e.error.message,
                e.auth,
                t.is_terminal_failure(e.retryable),
                t.retry.attempts,
            );
        }
        Some(EventPayload::ConnectorSyncErrored(ConnectorSyncErrored {
            path,
            error: e.error.clone(),
            retryable: e.retryable,
            auth: e.auth,
        }))
    }

    fn voids_when_missing(&self) -> bool {
        true
    }

    fn dispatch(&self, _state: &SessionState, id: &str) -> Vec<EventPayload> {
        void_events(EffectKind::ConnectorSync, id.to_string())
    }

    fn retry(&self, state: &SessionState, id: &str) -> Vec<EventPayload> {
        let (Some(path), Some(t)) = (
            ConnectionPath::parse(id),
            state.tracking(EffectKind::ConnectorSync, id),
        ) else {
            return Vec::new();
        };
        vec![EventPayload::ConnectorSyncRequested(
            ConnectorSyncRequested {
                path,
                attempt: t.retry.attempts,
                retry: t.retry_policy.clone(),
            },
        )]
    }
}

pub(in crate::runtime::session) fn owed(at: SessionStateAtNode) -> Vec<Dep> {
    let mut deps: Vec<Dep> = at
        .unsynced_connectors()
        .into_iter()
        .map(|connection_id| Dep::ConnectorSettled { connection_id })
        .collect();
    if let Some(config) = at.resolve_agent_for() {
        for c in at.state().servers_for(&config) {
            if at
                .state()
                .tracking(EffectKind::ConnectorSync, &c.path.to_string())
                .is_some_and(EffectTracking::is_in_flight)
            {
                deps.push(Dep::ConnectorSettled {
                    connection_id: c.path.clone(),
                });
            }
        }
    }
    deps.sort_by_key(Dep::label);
    deps.dedup();
    deps
}

pub(in crate::runtime::session) fn sync(
    state: &SessionState,
    config: &AgentConfig,
) -> Vec<EventPayload> {
    let retry = RetryPolicy::resolve(None, config.retry.as_deref(), RetryTarget::ConnectorSync);
    state
        .servers_for(config)
        .iter()
        .filter(|c| !state.has_effect(EffectKind::ConnectorSync, &c.path.to_string()))
        .map(|c| {
            EventPayload::ConnectorSyncRequested(ConnectorSyncRequested {
                path: c.path.clone(),
                attempt: 0,
                retry: retry.clone(),
            })
        })
        .collect()
}

fn report_failure(
    connection_id: &str,
    error: &str,
    auth: Option<AuthNeed>,
    terminal: bool,
    attempt: u32,
) {
    if terminal {
        tracing::error!(
            connection = %connection_id,
            error = %error,
            auth = ?auth,
            "connector unreachable; its tools are not offered to the model"
        );
    } else {
        tracing::warn!(
            connection = %connection_id,
            error = %error,
            attempt,
            "connector fetch failed; retrying"
        );
    }
}

impl SessionState {
    fn report_filter(
        &self,
        connection_id: &ConnectionPath,
        offered: &[RemoteTool],
        prefix: Option<&str>,
    ) {
        let Some(config) = self.at_head().resolve_agent_for() else {
            return;
        };
        let Some(connector) = self
            .servers_for(&config)
            .into_iter()
            .find(|c| c.path == *connection_id)
        else {
            return;
        };
        let defers = filter::defers(&connector, config.defers_tools());
        let r = filter::resolve(&connector, offered, prefix, defers);
        tracing::info!(
            connection = %connection_id,
            offered = r.offered,
            resolved = r.tools.len(),
            defer = defers,
            "fetched connector tools"
        );
        if !r.unmatched_include.is_empty() {
            tracing::warn!(
                connection = %connection_id,
                patterns = ?r.unmatched_include,
                "connector include patterns matched no tool"
            );
        }
        if !r.oversized.is_empty() && !defers {
            tracing::warn!(
                connection = %connection_id,
                tools = ?r.oversized,
                "connector tool names too long to offer; shorten the connection id or turn off prefixing"
            );
        }
        let collisions = self.connector_tools_for_config(&config).collisions;
        for name in collisions
            .iter()
            .filter(|name| r.tools.iter().any(|t| &&t.name == name))
        {
            tracing::warn!(
                connection = %connection_id,
                tool = %name,
                "tool name claimed twice; it is offered to the model by neither connector"
            );
        }
        if defers {
            for name in collisions
                .iter()
                .filter(|n| [filter::TOOL_SEARCH, filter::CALL_TOOL].contains(&n.as_str()))
            {
                tracing::warn!(
                    connection = %connection_id,
                    tool = %name,
                    "the config declares `{name}`, so the engine's own is not offered; \
                     this connection's tools cannot be found"
                );
            }
        }
        if !config.plugins.is_empty() && collisions.iter().any(|n| n == filter::SKILL) {
            tracing::warn!(
                "the config declares `skill`, so the engine's own is not offered; \
                 this agent's plugins cannot be used"
            );
        }
        if r.unannotated > 0 {
            tracing::warn!(
                connection = %connection_id,
                count = r.unannotated,
                "connector tools dropped for carrying no annotation to test"
            );
        }
    }
}
