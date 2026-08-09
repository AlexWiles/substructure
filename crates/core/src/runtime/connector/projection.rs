use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::protocol::ConnectorToolKind;
use crate::providers::memory_queue::TaskQueue;
use crate::runtime::event_store::EventStore;
use crate::runtime::processor::{
    EventProcessor, EventProcessorRunner, EventProcessorRunnerConfig, ProcessorCursorStore,
    ProcessorError,
};
use crate::runtime::session::decision::ToolHandler;
use crate::runtime::session::events::{ConnectorTarget, EventPayload};
use crate::runtime::session::SessionEvent;

use super::ConnectorTask;

struct ConnectorDispatchProjection {
    store: Arc<dyn EventStore>,
    queue: Arc<dyn TaskQueue<ConnectorTask>>,
}

#[async_trait::async_trait]
impl EventProcessor for ConnectorDispatchProjection {
    fn name(&self) -> &'static str {
        "connector_dispatch_v1"
    }

    async fn apply(&self, event: SessionEvent) -> Result<(), ProcessorError> {
        let task = match &event.payload {
            // Fetches are prerequisites, never queued: requested is dispatched.
            EventPayload::ConnectorSyncRequested(req) => ConnectorTask::Sync {
                source_event_id: event.id,
                session_id: event.session_id.clone(),
                tenant_id: event.tenant_id.clone(),
                connection_id: req.id.clone(),
                attempt: req.attempt,
                span: event.span,
            },
            // Executors key off the dispatch marker; the call is read from
            // state. Only calls the engine owns run here — a worker- or
            // client-handled call is answered by its owner instead.
            EventPayload::ToolCallDispatched(d) => {
                let session = self
                    .store
                    .load(&event.tenant_id, &event.session_id)
                    .await
                    .map_err(|e| {
                        ProcessorError::Apply(format!("load session for tool dispatch: {e}"))
                    })?;
                let Some(tc) = session.state.tool_call(&d.id) else {
                    return Ok(());
                };
                if tc.handler != ToolHandler::Server {
                    return Ok(());
                }
                let Some(target) = &tc.target else {
                    // `handler: server` is only ever set alongside a target;
                    // one without the other is a bug, not a runtime condition.
                    return Err(ProcessorError::Apply(format!(
                        "server-handled tool call `{}` has no connector target",
                        tc.name
                    )));
                };
                if answered_locally(target) {
                    ConnectorTask::Answer {
                        source_event_id: event.id,
                        session_id: event.session_id.clone(),
                        tenant_id: event.tenant_id.clone(),
                        tool_call_id: d.id.clone(),
                        attempt: d.attempt,
                        span: event.span,
                    }
                } else {
                    ConnectorTask::CallTool {
                        source_event_id: event.id,
                        session_id: event.session_id.clone(),
                        tenant_id: event.tenant_id.clone(),
                        tool_call_id: d.id.clone(),
                        attempt: d.attempt,
                        connection_id: target.connector.clone(),
                        remote_name: target.remote_name.clone(),
                        arguments: call_arguments(&tc.arguments, target.kind),
                        span: event.span,
                    }
                }
            }
            _ => return Ok(()),
        };

        tracing::debug!(
            session_id = %task.session_id(),
            dedupe_key = %task.dedupe_key(),
            "enqueuing connector task"
        );

        let shard_key = event.session_id;
        self.queue
            .enqueue(&shard_key, task)
            .await
            .map_err(ProcessorError::Apply)
    }
}

/// Whether the engine answers this call itself. A refused `call_tool` has no
/// remote name, and must not reach the connection.
fn answered_locally(target: &ConnectorTarget) -> bool {
    !target.kind.is_remote() && target.remote_name.is_empty()
}

/// What goes on the wire: the model's arguments, or the `arguments` object
/// inside a `call_tool`.
fn call_arguments(recorded: &str, kind: ConnectorToolKind) -> serde_json::Value {
    let value: serde_json::Value =
        serde_json::from_str(recorded).unwrap_or_else(|_| serde_json::json!({}));
    match kind {
        ConnectorToolKind::Call => value
            .get("arguments")
            .filter(|a| a.is_object())
            .cloned()
            .unwrap_or_else(|| serde_json::json!({})),
        _ => value,
    }
}

pub fn spawn_connector_dispatch_processor(
    store: Arc<dyn EventStore>,
    cursor_store: Arc<dyn ProcessorCursorStore>,
    queue: Arc<dyn TaskQueue<ConnectorTask>>,
    cancel: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    let projection = Arc::new(ConnectorDispatchProjection {
        store: store.clone(),
        queue,
    });
    let config = EventProcessorRunnerConfig {
        owner_id: Some("connector_dispatch".to_string()),
        ..Default::default()
    };
    EventProcessorRunner::new(store, cursor_store, projection, config, cancel).spawn()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(kind: ConnectorToolKind, remote_name: &str) -> ConnectorTarget {
        ConnectorTarget {
            connector: "sentry".to_string(),
            remote_name: remote_name.to_string(),
            kind,
        }
    }

    #[test]
    fn a_tool_the_connection_offers_goes_to_the_connection() {
        assert!(!answered_locally(&target(
            ConnectorToolKind::Remote,
            "search_issues"
        )));
    }

    #[test]
    fn a_find_never_goes_to_the_connection() {
        assert!(answered_locally(&target(ConnectorToolKind::Find, "")));
    }

    #[test]
    fn a_call_goes_to_the_connection_only_once_it_resolved() {
        assert!(
            answered_locally(&target(ConnectorToolKind::Call, "")),
            "the filter refused the name, so nothing may be dialled"
        );
        assert!(
            !answered_locally(&target(ConnectorToolKind::Call, "search_issues")),
            "a resolved name is an ordinary remote call"
        );
    }

    #[test]
    fn a_remote_call_sends_the_arguments_as_the_model_wrote_them() {
        assert_eq!(
            call_arguments(r#"{"q":"boom"}"#, ConnectorToolKind::Remote),
            serde_json::json!({ "q": "boom" })
        );
    }

    #[test]
    fn a_call_tool_sends_the_inner_arguments_without_the_wrapper() {
        assert_eq!(
            call_arguments(
                r#"{"tool":"search_issues","arguments":{"q":"boom"}}"#,
                ConnectorToolKind::Call
            ),
            serde_json::json!({ "q": "boom" }),
            "the connection never sees the engine's wrapper"
        );
    }

    #[test]
    fn a_call_tool_with_no_arguments_sends_an_empty_object() {
        for recorded in [
            r#"{"tool":"list_projects"}"#,
            r#"{"tool":"list_projects","arguments":null}"#,
            r#"{"tool":"list_projects","arguments":"nonsense"}"#,
            "not json at all",
        ] {
            assert_eq!(
                call_arguments(recorded, ConnectorToolKind::Call),
                serde_json::json!({}),
                "the connection's own schema rejects it better than we could: {recorded}"
            );
        }
    }
}
