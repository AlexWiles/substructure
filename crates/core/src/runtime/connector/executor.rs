use std::sync::Arc;

use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::connectors::registry::Connections;
use crate::connectors::ConnectorError;
use crate::plugins::PluginResolver;
use crate::protocol::StoredResult;
use crate::providers::memory_queue::TaskQueue;
use crate::runtime::event_store::EventStore;
use crate::runtime::session::command::{CommandPayload, Outcome, SettleError};
use crate::runtime::session::engine_tools;
use crate::runtime::session::state::EffectKind;
use crate::runtime::session::{execute, ConflictRetry, ExecuteInput};
use crate::runtime::span::SpanContext;
use crate::runtime::Caller;

use super::ConnectorTask;
use crate::protocol::{ErrorCode, ErrorInfo};

pub fn spawn_connector_task_executor(
    store: Arc<dyn EventStore>,
    connections: Option<Arc<Connections>>,
    plugins: Arc<dyn PluginResolver>,
    queue: Arc<dyn TaskQueue<ConnectorTask>>,
    worker_count: usize,
    cancel: CancellationToken,
) -> Vec<JoinHandle<()>> {
    let worker_count = worker_count.max(1);
    let mut handles = Vec::with_capacity(worker_count);
    for _ in 0..worker_count {
        let store = store.clone();
        let connections = connections.clone();
        let plugins = plugins.clone();
        let mut rx = queue.subscribe();
        let cancel = cancel.clone();
        handles.push(tokio::spawn(async move {
            loop {
                let task = tokio::select! {
                    t = rx.recv() => match t {
                        Some(t) => t,
                        None => break,
                    },
                    _ = cancel.cancelled() => break,
                };
                handle_task(
                    store.as_ref(),
                    connections.as_deref(),
                    plugins.as_ref(),
                    task,
                )
                .await;
            }
        }));
    }
    handles
}

/// A network task with no connections settles as a terminal error, so the
/// session does not park.
fn unreachable() -> ConnectorError {
    ConnectorError::permanent("no connections are configured on this engine")
}

async fn handle_task(
    store: &dyn EventStore,
    connections: Option<&Connections>,
    plugins: &dyn PluginResolver,
    task: ConnectorTask,
) {
    match task {
        ConnectorTask::Sync {
            session_id,
            tenant_id,
            connection_id,
            requester,
            attempt,
            span,
            ..
        } => {
            let listed = match connections {
                Some(c) => c.list_tools(&tenant_id, &connection_id, &requester).await,
                None => Err(unreachable()),
            };
            let command = match listed {
                Ok(offer) => CommandPayload::settle(
                    EffectKind::ConnectorSync,
                    connection_id.to_string(),
                    Some(attempt),
                    Outcome::Connector {
                        server: offer.server,
                        prefix: offer.prefix,
                        tools: offer.tools,
                        instructions: offer.instructions,
                    },
                ),
                Err(err) => CommandPayload::settle(
                    EffectKind::ConnectorSync,
                    connection_id.to_string(),
                    Some(attempt),
                    SettleError::new(
                        ErrorInfo::new(ErrorCode::HandlerError, err.message.clone()),
                        err.retryable,
                    )
                    .auth(err.auth),
                ),
            };
            submit(
                store,
                &session_id,
                &tenant_id,
                command,
                span,
                "connector_sync",
            )
            .await;
        }

        ConnectorTask::CallTool {
            session_id,
            tenant_id,
            tool_call_id,
            attempt,
            connection_id,
            requester,
            remote_name,
            arguments,
            span,
            ..
        } => {
            let result = match connections {
                Some(c) => {
                    c.call_tool(
                        &tenant_id,
                        &connection_id,
                        &requester,
                        &remote_name,
                        &arguments,
                    )
                    .await
                }
                None => Err(unreachable()),
            };
            let command = settle_call(tool_call_id, attempt, result);
            submit(
                store,
                &session_id,
                &tenant_id,
                command,
                span,
                "connector_call",
            )
            .await;
        }

        // Answered here, not at dispatch, so that a dispatch stays the marker
        // it is for every other kind and one seam settles every tool call.
        ConnectorTask::Answer {
            session_id,
            tenant_id,
            tool_call_id,
            attempt,
            span,
            ..
        } => {
            let session = match store.load(&tenant_id, &session_id).await {
                Ok(session) => session,
                Err(err) => {
                    tracing::error!(
                        session_id = %session_id,
                        error = %err,
                        "failed to load session to answer a connector tool"
                    );
                    return;
                }
            };
            // A skill call reads the bundle, which is here, not in state.
            let answer = match session.state.skill_call(&tool_call_id) {
                Some(call) => {
                    let bundle = match call.plugin_id.is_empty() {
                        true => None,
                        false => plugins.resolve(&tenant_id, &call.plugin_id).await,
                    };
                    Some(engine_tools::skill_answer(
                        session.state.at(call.node.as_deref()),
                        bundle.as_ref(),
                        &call.arguments,
                    ))
                }
                None => session.state.local_connector_answer(&tool_call_id),
            };
            let Some(answer) = answer else {
                tracing::error!(
                    session_id = %session_id,
                    tool_call_id = %tool_call_id,
                    "no local answer for a call routed to the engine"
                );
                return;
            };
            // Settled like a connection's call: one reading of `is_error`.
            submit(
                store,
                &session_id,
                &tenant_id,
                settle_call(tool_call_id, attempt, Ok(answer)),
                span,
                "connector_answer",
            )
            .await;
        }
    }
}

/// A connection can fail a call two ways, and they settle differently: a tool
/// that ran and reported failure is a terminal `tool.error` the model should
/// read, while a transport fault is the engine's problem and retries under the
/// call's policy. A refused credential is different. It goes in `auth`.
fn settle_call(
    tool_call_id: String,
    attempt: u32,
    result: Result<StoredResult, ConnectorError>,
) -> CommandPayload {
    let outcome = match result {
        Ok(result) if result.is_error => SettleError::new(
            ErrorInfo::new(ErrorCode::HandlerError, result.rendered()),
            false,
        )
        .into(),
        Ok(result) => Outcome::Tool { result },
        Err(err) => SettleError::new(
            ErrorInfo::new(ErrorCode::HandlerError, err.message),
            err.retryable,
        )
        .auth(err.auth)
        .into(),
    };
    CommandPayload::settle(EffectKind::ToolCall, tool_call_id, Some(attempt), outcome)
}

async fn submit(
    store: &dyn EventStore,
    session_id: &str,
    tenant_id: &str,
    command: CommandPayload,
    span: SpanContext,
    name: &'static str,
) {
    let result = execute(
        store,
        ExecuteInput {
            session_id: session_id.to_string(),
            caller: Caller::System {
                tenant_id: tenant_id.to_string(),
            },
            command,
            span: span.child(name),
        },
        &ConflictRetry::default(),
    )
    .await;

    if let Err(err) = result {
        // A lost settle is recovered by the call's own deadline, not by
        // retrying here — the session may have moved on.
        tracing::error!(
            session_id = %session_id,
            error = %err,
            "failed to submit connector result"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(
        content: &str,
        structured: Option<serde_json::Value>,
        is_error: bool,
    ) -> StoredResult {
        StoredResult {
            content: vec![crate::protocol::StoredContent::Text {
                text: content.to_string(),
            }],
            structured_content: structured,
            is_error,
        }
    }

    #[test]
    fn a_tool_that_ran_and_failed_is_terminal_not_a_retry() {
        let cmd = settle_call("tc-1".into(), 0, Ok(outcome("no such issue", None, true)));
        match cmd {
            CommandPayload::SettleEffect {
                kind: EffectKind::ToolCall,
                outcome: Outcome::Error(e),
                ..
            } => {
                assert_eq!(e.error.message, "no such issue");
                assert!(!e.retryable, "the tool ran; running it again says the same");
            }
            other => panic!("expected a terminal failure; got {other:?}"),
        }
    }

    #[test]
    fn a_transport_fault_keeps_the_calls_own_retry_policy() {
        let cmd = settle_call(
            "tc-1".into(),
            2,
            Err(ConnectorError::retryable("connection reset")),
        );
        match cmd {
            CommandPayload::SettleEffect {
                kind: EffectKind::ToolCall,
                outcome: Outcome::Error(e),
                attempt,
                ..
            } => {
                assert!(e.retryable, "the engine's problem, not the model's");
                assert_eq!(attempt, Some(2), "fenced against a stale executor");
            }
            other => panic!("expected a retryable failure; got {other:?}"),
        }
    }

    #[test]
    fn structured_output_wins_over_its_rendering() {
        let structured = serde_json::json!({ "id": 7 });
        let cmd = settle_call(
            "tc-1".into(),
            0,
            Ok(outcome("Issue 7", Some(structured.clone()), false)),
        );
        match cmd {
            CommandPayload::SettleEffect {
                kind: EffectKind::ToolCall,
                outcome: Outcome::Tool { result, .. },
                ..
            } => {
                assert_eq!(
                    result.rendered(),
                    structured.to_string(),
                    "only the structured form can satisfy a declared output schema"
                );
            }
            other => panic!("expected a result; got {other:?}"),
        }
    }

    #[test]
    fn rendered_content_settles_a_connection_that_sent_no_structure() {
        let cmd = settle_call("tc-1".into(), 0, Ok(outcome("Issue 7", None, false)));
        match cmd {
            CommandPayload::SettleEffect {
                kind: EffectKind::ToolCall,
                outcome: Outcome::Tool { result, .. },
                ..
            } => assert_eq!(result.rendered(), "Issue 7"),
            other => panic!("expected a result; got {other:?}"),
        }
    }
}
