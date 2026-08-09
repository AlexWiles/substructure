//! One connection's tool list.
//!
//! ```text
//! Pending ─complete────────────→ Completed  ⇒ its tools are offered
//!    │ ─error[retries left]────→ RetryScheduled ─due→ Pending
//!    └ ─error[exhausted]───────→ Failed     ⇒ the turn goes ahead without them
//! ```
//!
//! A fetch is a prerequisite, not queued work: it is requested and in flight at
//! once, and the planner offers owed fetches first and alone so one can never
//! queue behind its dependent. Decisions and LLM calls wait on it
//! ([`Dep::ConnectorSettled`](super::super::schedule::Dep)), which is why a
//! terminal failure still settles: a connection the engine cannot reach must
//! release the gate rather than park the session forever.
//!
//! Keyed on the connection, not the branch: a fetch belongs to the connection,
//! so a fork never abandons one and a config rewritten for unrelated reasons
//! costs nothing.

use super::{fail, mismatched, void_events, KindSpec, Outcome, SettleError};
use crate::connectors::{filter, AuthNeed, RemoteTool};
use crate::protocol::{AgentConfig, RetryPolicy};
use crate::runtime::retry::RetryTarget;
use crate::runtime::session::events::*;
use crate::runtime::session::schedule::Dep;
use crate::runtime::session::state::EffectTracking;
use crate::runtime::session::state::{EffectKind, SessionState};

pub struct ConnectorSpec;

impl KindSpec for ConnectorSpec {
    fn kind(&self) -> EffectKind {
        EffectKind::ConnectorSync
    }

    fn settle(&self, state: &SessionState, id: &str, outcome: Outcome) -> Vec<EventPayload> {
        match outcome {
            Outcome::Connector { prefix, tools } => {
                state.report_filter(id, &tools, prefix.as_deref());
                // The run tail promotes whatever this fetch was parking.
                vec![EventPayload::ConnectorSyncCompleted(Box::new(
                    ConnectorSyncCompleted {
                        id: id.to_string(),
                        prefix,
                        tools,
                    },
                ))]
            }
            Outcome::Error(e) => fail(self, state, id, &e),
            other => mismatched(self.kind(), &other),
        }
    }

    fn errored(&self, state: &SessionState, id: &str, e: &SettleError) -> Option<EventPayload> {
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
            id: id.to_string(),
            error: e.error.clone(),
            retryable: e.retryable,
            auth: e.auth,
        }))
    }

    // A terminal fetch failure folds back as nothing: it answers no call, and
    // the worker learns of it from the tools it is not offered. A fetch is
    // itself a prerequisite, so it is dep-free by construction and no
    // dependency cycle can form.

    /// A fetch never queues, so an entry for one is always inconsistent state.
    fn voids_when_missing(&self) -> bool {
        true
    }

    fn dispatch(&self, _state: &SessionState, id: &str) -> Vec<EventPayload> {
        // Fetches never queue; the planner routes an entry for one to
        // `VoidPhantom`, so this is unreachable.
        void_events(EffectKind::ConnectorSync, id.to_string())
    }

    fn retry(&self, state: &SessionState, id: &str) -> Vec<EventPayload> {
        let Some(t) = state.tracking(EffectKind::ConnectorSync, id) else {
            return Vec::new();
        };
        vec![EventPayload::ConnectorSyncRequested(
            ConnectorSyncRequested {
                id: id.to_string(),
                attempt: t.retry.attempts,
                retry: t.retry_policy.clone(),
            },
        )]
    }
}

/// A [`Dep`] for every fetch the config in force at `leaf` still owes:
/// connections never fetched, or fetched but unsettled. The edge other kinds
/// wait on — a call cannot be authored against tools the engine has not
/// fetched, and a decision cannot route them.
pub(in crate::runtime::session) fn owed(state: &SessionState, leaf: Option<&str>) -> Vec<Dep> {
    let mut deps: Vec<Dep> = state
        .unsynced_connectors(leaf)
        .into_iter()
        .map(|connection_id| Dep::ConnectorSettled { connection_id })
        .collect();
    if let Some(config) = state.resolve_agent_for(leaf) {
        for c in &config.mcp {
            if state
                .tracking(EffectKind::ConnectorSync, &c.id)
                .is_some_and(EffectTracking::is_in_flight)
            {
                deps.push(Dep::ConnectorSettled {
                    connection_id: c.id.clone(),
                });
            }
        }
    }
    deps.sort_by_key(Dep::label);
    deps.dedup();
    deps
}

/// Fetch every connection `config` names that this session has never fetched.
pub(in crate::runtime::session) fn sync(
    state: &SessionState,
    config: &AgentConfig,
) -> Vec<EventPayload> {
    let retry = RetryPolicy::resolve(None, config.retry.as_deref(), RetryTarget::ConnectorSync);
    config
        .mcp
        .iter()
        .filter(|c| !state.has_effect(EffectKind::ConnectorSync, &c.id))
        .map(|c| {
            EventPayload::ConnectorSyncRequested(ConnectorSyncRequested {
                id: c.id.clone(),
                attempt: 0,
                retry: retry.clone(),
            })
        })
        .collect()
}

/// Announce a failed fetch. A terminal failure is at ERROR because the turn goes
/// ahead without these tools, and a model answering confidently with a connector
/// missing looks exactly like one answering with it. `subs run` shows ERROR by
/// default, so this is the only notice a human gets.
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
    /// Report what the filter did to a fresh offer.
    ///
    /// Logged once per fetch rather than carried on every decision: these are
    /// facts about the config, identical on every turn, and a worker cannot act
    /// on them anyway — an `include` that matches nothing is a typo for whoever
    /// wrote the config, not a runtime condition for the agent to handle.
    fn report_filter(&self, connection_id: &str, offered: &[RemoteTool], prefix: Option<&str>) {
        let Some(connector) = self
            .resolve_agent_for(self.head_id.as_deref())
            .and_then(|c| c.mcp.into_iter().find(|c| c.id == connection_id))
        else {
            return;
        };
        let r = filter::resolve(&connector, offered, prefix);
        tracing::info!(
            connection = %connection_id,
            offered = r.offered,
            resolved = r.tools.len(),
            "fetched connector tools"
        );
        if !r.unmatched_include.is_empty() {
            tracing::warn!(
                connection = %connection_id,
                patterns = ?r.unmatched_include,
                "connector include patterns matched no tool"
            );
        }
        if !r.oversized.is_empty() {
            tracing::warn!(
                connection = %connection_id,
                tools = ?r.oversized,
                "connector tool names too long to offer; shorten the connection id or turn off prefixing"
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
