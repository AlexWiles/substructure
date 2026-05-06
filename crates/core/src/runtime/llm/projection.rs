use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::providers::memory_queue::TaskQueue;
use crate::runtime::aggregate::{AggregateState, DomainEvent};
use crate::runtime::event_store::{Event, EventStore};
use crate::runtime::processor::{
    EventProcessor, EventProcessorRunner, EventProcessorRunnerConfig, ProcessorCheckpointStore,
    ProcessorError,
};
use crate::runtime::session::events::EventPayload;
use crate::runtime::session::state::SessionState;

use super::LlmTask;

struct LlmDispatchProjection {
    queue: Arc<dyn TaskQueue<LlmTask>>,
}

impl LlmDispatchProjection {
    fn new(queue: Arc<dyn TaskQueue<LlmTask>>) -> Self {
        Self { queue }
    }
}

#[async_trait::async_trait]
impl EventProcessor for LlmDispatchProjection {
    fn name(&self) -> &'static str {
        "llm_dispatch_v1"
    }

    fn shard_key(&self, event: &Event) -> Option<String> {
        if event.aggregate_type != SessionState::AGGREGATE_TYPE {
            return None;
        }
        Some(event.aggregate_id.clone())
    }

    async fn apply(&self, raw: &Event) -> Result<(), ProcessorError> {
        if raw.aggregate_type != SessionState::AGGREGATE_TYPE {
            return Ok(());
        }

        let event = DomainEvent::<SessionState>::from_raw(raw)
            .map_err(|e| ProcessorError::Apply(e.to_string()))?;

        let req = match &event.payload {
            EventPayload::LlmCallRequested(req) => req,
            _ => return Ok(()),
        };

        let derived = event.derived.as_ref().ok_or_else(|| {
            ProcessorError::Apply("missing derived state on llm event".to_string())
        })?;
        let identity = derived.identity.clone().ok_or_else(|| {
            ProcessorError::Apply("missing identity in derived state".to_string())
        })?;
        let agent_id = derived.agent_id.clone().ok_or_else(|| {
            ProcessorError::Apply("missing agent_id in derived state".to_string())
        })?;
        let ancestry = derived.ancestry.clone();

        let shard_key = raw.aggregate_id.clone();

        let task = LlmTask {
            session_id: event.aggregate_id,
            tenant_id: event.tenant_id,
            agent_id,
            call_id: req.call_id.clone(),
            llm_client: req.llm_client.clone(),
            request: req.request.clone(),
            identity,
            ancestry,
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
    let projection = Arc::new(LlmDispatchProjection::new(queue));
    let mut config = EventProcessorRunnerConfig::default();
    config.owner_id = Some("llm_dispatch".to_string());
    EventProcessorRunner::new(store, checkpoint_store, projection, config, cancel).spawn()
}
