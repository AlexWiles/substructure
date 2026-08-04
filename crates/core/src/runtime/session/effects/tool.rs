//! A tool call.
//!
//! ```text
//! Queued ─dispatch→ Pending ─complete────────────→ Completed ⇒ queue tool.finished(ok)
//!    ↑                 │ ─error[retries left]────→ RetryScheduled ─due→ Queued
//!    │                 │ ─error[exhausted]───────→ Failed    ⇒ queue tool.finished(err)
//!    └──── requeue ────┘ ─void──────────────────→ Failed
//! ```
//!
//! Where a call runs follows from its name against the config in force,
//! resolved once at request and frozen onto the call: a `Client` call is
//! answered by the frontend, a `Worker` one is handed over as `tool.execute`,
//! and a `Server` one the engine runs against its connection.
//!
//! A result that violates the tool's own declared `output` schema settles as a
//! terminal tool error instead — the contract is the tool's, so breaking it is
//! the tool failing, not the model.

use super::{decision_queued, fail, mismatched, void_events, KindSpec, Outcome, SettleError};
use crate::protocol::{RetryOverride, RetryPolicy};
use crate::runtime::session::command::SessionError;
use crate::runtime::session::decision::{ToolHandler, Trigger};
use crate::runtime::session::events::*;
use crate::runtime::session::state::{EffectKind, SessionState};
use crate::runtime::session::tool_contract::{declared_tool, output_violation, DeclaredTool};
use crate::runtime::Caller;

pub struct ToolSpec;

impl KindSpec for ToolSpec {
    fn kind(&self) -> EffectKind {
        EffectKind::ToolCall
    }

    fn authorize(
        &self,
        state: &SessionState,
        id: &str,
        caller: &Caller,
    ) -> Result<(), SessionError> {
        let tc = state.tool_call(id).ok_or(SessionError::EffectNotFound)?;
        state.check_tool_call_caller(tc, caller)
    }

    fn settle(&self, state: &SessionState, id: &str, outcome: Outcome) -> Vec<EventPayload> {
        match outcome {
            Outcome::Tool { result } => {
                let name = state
                    .tool_call(id)
                    .map(|t| t.name.clone())
                    .unwrap_or_default();
                match state
                    .declared_output_schema(id, &name)
                    .and_then(|schema| output_violation(&schema, &result))
                {
                    Some(v) => fail(
                        self,
                        state,
                        id,
                        &SettleError::new(
                            format!("tool output violated its declared schema: {v}"),
                            false,
                        ),
                    ),
                    None => complete(id.to_string(), name, result),
                }
            }
            Outcome::Error(e) => fail(self, state, id, &e),
            other => mismatched(self.kind(), &other),
        }
    }

    fn errored(&self, state: &SessionState, id: &str, e: &SettleError) -> Option<EventPayload> {
        Some(EventPayload::ToolCallErrored(ToolCallErrored {
            id: id.to_string(),
            name: name_of(state, id),
            error: e.error.clone(),
            retryable: e.retryable,
        }))
    }

    /// When the retry policy is exhausted the error is the result: the model
    /// sees it as the tool's answer and decides what to do.
    fn terminal(&self, state: &SessionState, id: &str, e: &SettleError) -> Vec<EventPayload> {
        vec![decision_queued(Trigger::ToolFinished {
            id: id.to_string(),
            ok: false,
            name: name_of(state, id),
            result: None,
            error: Some(e.error.clone()),
        })]
    }

    fn dispatch(&self, state: &SessionState, id: &str) -> Vec<EventPayload> {
        let Some(effect) = state.effect(EffectKind::ToolCall, id) else {
            return void_events(EffectKind::ToolCall, id.to_string());
        };
        vec![EventPayload::ToolCallDispatched(ToolCallDispatched {
            id: id.to_string(),
            attempt: effect.tracking.retry.attempts,
        })]
    }

    fn execute_trigger(&self, state: &SessionState, id: &str) -> Option<Trigger> {
        let effect = state.effect(EffectKind::ToolCall, id)?;
        let tc = effect.tool().filter(|t| t.handler == ToolHandler::Worker)?;
        Some(Trigger::ToolExecute {
            id: id.to_string(),
            name: tc.name.clone(),
            arguments: tc.arguments.clone(),
            attempt: effect.tracking.retry.attempts,
            deadline: effect.tracking.deadline,
        })
    }

    fn retry(&self, state: &SessionState, id: &str) -> Vec<EventPayload> {
        let Some(effect) = state.effect(EffectKind::ToolCall, id) else {
            return Vec::new();
        };
        let (t, Some(tc)) = (&effect.tracking, effect.tool()) else {
            return Vec::new();
        };
        vec![EventPayload::ToolCallRequested(ToolCallRequested {
            id: id.to_string(),
            attempt: t.retry.attempts,
            name: tc.name.clone(),
            arguments: tc.arguments.clone(),
            handler: tc.handler,
            // A retry keeps the target it was routed to.
            target: tc.target.clone(),
            retry: t.retry_policy.clone(),
        })]
    }
}

fn name_of(state: &SessionState, id: &str) -> String {
    state
        .tool_call(id)
        .map(|t| t.name.clone())
        .unwrap_or_default()
}

/// A tool result: record it and queue `tool.finished`. Also the client-view
/// path's settle, which answers a pending `Client` call from a submitted
/// transcript rather than from the settle endpoint.
pub(in crate::runtime::session) fn complete(
    id: String,
    name: String,
    result: String,
) -> Vec<EventPayload> {
    vec![
        EventPayload::ToolCallCompleted(ToolCallCompleted {
            id: id.clone(),
            name: name.clone(),
            result: result.clone(),
        }),
        decision_queued(Trigger::ToolFinished {
            id,
            ok: true,
            name,
            result: Some(result),
            error: None,
        }),
    ]
}

/// Issue a call. Where it runs follows from its name against the config in
/// force, resolved once here and frozen onto the call — a later config change
/// must not reroute a call already in flight. Idempotent by id.
pub(in crate::runtime::session) fn request(
    state: &SessionState,
    tool_call_id: String,
    name: String,
    arguments: String,
    retry: Option<RetryOverride>,
    caller: &Caller,
) -> Result<Vec<EventPayload>, SessionError> {
    SessionState::ensure_internal(caller)?;
    let handler = state.tool_handler_for(&name);
    let target = state.connector_tool_for(&name).map(|t| ConnectorTarget {
        connector: t.connector,
        remote_name: t.remote_name,
    });
    // Resolved here, not at the seam: the default follows the handler, and a
    // client tool must stay unbounded — a deferred call waits for a human.
    let config = state.retry_config();
    let retry = RetryPolicy::resolve(retry.as_ref(), config.as_ref(), handler.retry_target());
    if state.has_effect(EffectKind::ToolCall, &tool_call_id) {
        return Ok(Vec::new());
    }
    // The execute decision for a worker-handled call queues at dispatch,
    // alongside the deadline clock.
    Ok(vec![EventPayload::ToolCallRequested(ToolCallRequested {
        id: tool_call_id,
        attempt: 0,
        name,
        arguments,
        handler,
        target,
        retry,
    })])
}

impl SessionState {
    /// The `output` schema the settling tool was declared with, resolved by
    /// lineage on the active path. `None` when the tool declared no output
    /// contract or the call has no resolvable spec.
    fn declared_output_schema(&self, tool_call_id: &str, name: &str) -> Option<serde_json::Value> {
        let tree = self.message_tree();
        let path = tree.path_to(tree.head_id.as_deref()?);
        match declared_tool(tool_call_id, name, &path, |id| self.llm_call(id)) {
            DeclaredTool::Declared(tool) => tool.output.clone(),
            _ => None,
        }
    }
}
