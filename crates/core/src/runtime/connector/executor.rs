use std::sync::Arc;

use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::connectors::registry::Connections;
use crate::connectors::{ConnectorError, ToolOutcome};
use crate::providers::memory_queue::TaskQueue;
use crate::runtime::event_store::EventStore;
use crate::runtime::session::command::{CommandPayload, Outcome, SettleError};
use crate::runtime::session::state::{EffectKind, LocalAnswer};
use crate::runtime::session::{execute, ConflictRetry, ExecuteInput};
use crate::runtime::span::SpanContext;
use crate::runtime::Caller;

use super::ConnectorTask;
use crate::protocol::{ErrorCode, ErrorInfo};

pub fn spawn_connector_task_executor(
    store: Arc<dyn EventStore>,
    connections: Arc<Connections>,
    queue: Arc<dyn TaskQueue<ConnectorTask>>,
    worker_count: usize,
    cancel: CancellationToken,
) -> Vec<JoinHandle<()>> {
    let worker_count = worker_count.max(1);
    let mut handles = Vec::with_capacity(worker_count);
    for _ in 0..worker_count {
        let store = store.clone();
        let connections = connections.clone();
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
                handle_task(store.as_ref(), connections.as_ref(), task).await;
            }
        }));
    }
    handles
}

async fn handle_task(store: &dyn EventStore, connections: &Connections, task: ConnectorTask) {
    match task {
        ConnectorTask::Sync {
            session_id,
            tenant_id,
            connection_id,
            attempt,
            span,
            ..
        } => {
            let command = match connections.list_tools(&tenant_id, &connection_id).await {
                Ok(offer) => CommandPayload::settle(
                    EffectKind::ConnectorSync,
                    connection_id.clone(),
                    Some(attempt),
                    Outcome::Connector {
                        prefix: offer.prefix,
                        tools: offer.tools,
                    },
                ),
                Err(err) => CommandPayload::settle(
                    EffectKind::ConnectorSync,
                    connection_id.clone(),
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
            remote_name,
            arguments,
            span,
            ..
        } => {
            let result = connections
                .call_tool(&tenant_id, &connection_id, &remote_name, &arguments)
                .await;
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
            let Some(answer) = session.state.local_connector_answer(&tool_call_id) else {
                tracing::error!(
                    session_id = %session_id,
                    tool_call_id = %tool_call_id,
                    "no local answer for a call routed to the engine"
                );
                return;
            };
            let outcome = match answer {
                LocalAnswer::Result(result) => Outcome::Tool { result },
                LocalAnswer::Error(message) => {
                    SettleError::new(ErrorInfo::new(ErrorCode::HandlerError, message), false).into()
                }
            };
            submit(
                store,
                &session_id,
                &tenant_id,
                CommandPayload::settle(EffectKind::ToolCall, tool_call_id, Some(attempt), outcome),
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
    result: Result<ToolOutcome, ConnectorError>,
) -> CommandPayload {
    let outcome = match result {
        Ok(outcome) if outcome.is_error => SettleError::new(
            ErrorInfo::new(ErrorCode::HandlerError, outcome.content),
            false,
        )
        .into(),
        // Prefer the structured form when the connection sent one: it round
        // trips through a declared `output` schema, where rendered text would not.
        Ok(outcome) => Outcome::Tool {
            result: match outcome.structured {
                Some(value) => value.to_string(),
                None => outcome.content,
            },
        },
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
    ) -> ToolOutcome {
        ToolOutcome {
            content: content.to_string(),
            structured,
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
                outcome: Outcome::Tool { result },
                ..
            } => {
                assert_eq!(
                    result,
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
                outcome: Outcome::Tool { result },
                ..
            } => assert_eq!(result, "Issue 7"),
            other => panic!("expected a result; got {other:?}"),
        }
    }
}
