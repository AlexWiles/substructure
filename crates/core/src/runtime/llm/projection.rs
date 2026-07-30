use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::providers::memory_queue::TaskQueue;
use crate::runtime::event_store::EventStore;
use crate::runtime::processor::{
    EventProcessor, EventProcessorRunner, EventProcessorRunnerConfig, ProcessorCheckpointStore,
    ProcessorError,
};
use crate::runtime::session::decision::LlmHandler;
use crate::runtime::session::events::EventPayload;
use crate::runtime::session::SessionEvent;

use super::LlmTask;

struct LlmDispatchProjection {
    store: Arc<dyn EventStore>,
    queue: Arc<dyn TaskQueue<LlmTask>>,
}

impl LlmDispatchProjection {
    fn new(store: Arc<dyn EventStore>, queue: Arc<dyn TaskQueue<LlmTask>>) -> Self {
        Self { store, queue }
    }
}

#[async_trait::async_trait]
impl EventProcessor for LlmDispatchProjection {
    fn name(&self) -> &'static str {
        "llm_dispatch_v1"
    }

    async fn apply(&self, event: SessionEvent) -> Result<(), ProcessorError> {
        // Executors key off the dispatch marker: a requested call is queued,
        // not running. The payload is read from state — post-dispatch, the
        // stored spec carries the connector tools in force.
        let dispatched = match &event.payload {
            EventPayload::LlmCallDispatched(d) => d,
            _ => return Ok(()),
        };

        let session = self
            .store
            .load(&event.tenant_id, &event.session_id)
            .await
            .map_err(|e| ProcessorError::Apply(format!("load session for llm dispatch: {e}")))?;
        let Some(call) = session.state.llm_call(&dispatched.id) else {
            // Settled and gone — nothing left to execute.
            return Ok(());
        };

        // Worker-handled calls run on the worker, not the server-side executor.
        if call.handler == LlmHandler::Worker {
            return Ok(());
        }

        let owner = event
            .meta
            .owner
            .clone()
            .ok_or_else(|| ProcessorError::Apply("missing owner in event meta".to_string()))?;
        let agent_id =
            event.meta.agent_id.clone().ok_or_else(|| {
                ProcessorError::Apply("missing agent_id in event meta".to_string())
            })?;
        let ancestry = event.meta.ancestry.clone();
        let turn_id = event.meta.turn_id.clone();

        let shard_key = event.session_id.clone();

        let task = LlmTask {
            session_id: event.session_id,
            tenant_id: event.tenant_id,
            agent_id,
            call_id: dispatched.id.clone(),
            attempt: dispatched.attempt,
            request: call.spec.to_request(call.prompt.clone()),
            stream: call.stream,
            owner,
            ancestry,
            turn_id,
            span: event.span,
        };

        tracing::debug!(
            session_id = %task.session_id,
            call_id = %task.call_id,
            dedupe_key = %task.dedupe_key(),
            "enqueuing llm task"
        );

        self.queue
            .enqueue(&shard_key, task)
            .await
            .map_err(ProcessorError::Apply)
    }
}

pub fn spawn_llm_dispatch_processor(
    store: Arc<dyn EventStore>,
    checkpoint_store: Arc<dyn ProcessorCheckpointStore>,
    queue: Arc<dyn TaskQueue<LlmTask>>,
    cancel: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    let projection = Arc::new(LlmDispatchProjection::new(store.clone(), queue));
    let config = EventProcessorRunnerConfig {
        owner_id: Some("llm_dispatch".to_string()),
        ..Default::default()
    };
    EventProcessorRunner::new(store, checkpoint_store, projection, config, cancel).spawn()
}
