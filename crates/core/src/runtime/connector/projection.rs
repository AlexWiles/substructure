use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::connectors::Requester;
use crate::providers::memory_queue::TaskQueue;
use crate::runtime::event_store::EventStore;
use crate::runtime::processor::{
    EventProcessor, EventProcessorRunner, EventProcessorRunnerConfig, ProcessorCursorStore,
    ProcessorError,
};
use crate::runtime::session::decision::ToolHandler;
use crate::runtime::session::events::EventPayload;
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
            // The principal is derived from the owner here, at dispatch, and
            // rides the task — it never reaches an event.
            EventPayload::ConnectorSyncRequested(req) => {
                let session = self
                    .store
                    .load(&event.tenant_id, &event.session_id)
                    .await
                    .map_err(|e| {
                        ProcessorError::Apply(format!("load session for connector sync: {e}"))
                    })?;
                ConnectorTask::Sync {
                    source_event_id: event.id,
                    session_id: event.session_id.clone(),
                    tenant_id: event.tenant_id.clone(),
                    connection_id: req.id.clone(),
                    principal: Requester::of_owner(session.state.owner.as_ref()),
                    attempt: req.attempt,
                    span: event.span,
                }
            }
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
                if target.kind.is_remote() {
                    ConnectorTask::CallTool {
                        source_event_id: event.id,
                        session_id: event.session_id.clone(),
                        tenant_id: event.tenant_id.clone(),
                        tool_call_id: d.id.clone(),
                        attempt: d.attempt,
                        connection_id: target.connector.clone(),
                        principal: Requester::of_owner(session.state.owner.as_ref()),
                        remote_name: target.remote_name.clone(),
                        // The recorded arguments are the tool's own: a `call_tool`
                        // was unwrapped into the real call before it was
                        // recorded.
                        arguments: serde_json::from_str(&tc.arguments)
                            .unwrap_or_else(|_| serde_json::json!({})),
                        span: event.span,
                    }
                } else {
                    ConnectorTask::Answer {
                        source_event_id: event.id,
                        session_id: event.session_id.clone(),
                        tenant_id: event.tenant_id.clone(),
                        tool_call_id: d.id.clone(),
                        attempt: d.attempt,
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
    use crate::protocol::ConnectorToolKind;

    /// Only a `Remote` carries a name to dial. Every other kind is one of the
    /// engine's own tools, answered from the session.
    #[test]
    fn only_a_remote_tool_reaches_the_connection() {
        assert!(ConnectorToolKind::Remote.is_remote());
        for kind in [ConnectorToolKind::Find, ConnectorToolKind::Call] {
            assert!(!kind.is_remote(), "{kind:?} is answered by the engine");
        }
    }
}
