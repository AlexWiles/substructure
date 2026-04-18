use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::runtime::aggregate::{AggregateState, DomainEvent};
use crate::runtime::event_store::{Event, EventStore};
use crate::runtime::processor::{
    EventProcessor, EventProcessorRunner, EventProcessorRunnerConfig, ProcessorCheckpointStore,
    ProcessorError,
};
use crate::runtime::session::events::EventPayload;
use crate::runtime::session::state::SessionState;

use super::{WorkerDecisionRequest, WorkerQueue};

struct WorkerDecisionProjection {
    queue: Arc<dyn WorkerQueue>,
}

impl WorkerDecisionProjection {
    fn new(queue: Arc<dyn WorkerQueue>) -> Self {
        Self { queue }
    }
}

#[async_trait::async_trait]
impl EventProcessor for WorkerDecisionProjection {
    fn name(&self) -> &'static str {
        "worker_decision_enqueue"
    }

    fn shard_key(&self, event: &Event) -> Option<String> {
        if event.aggregate_type != SessionState::AGGREGATE_TYPE {
            return None;
        }
        Some(event.aggregate_id.clone())
    }

    async fn apply(&self, event: &Event) -> Result<(), ProcessorError> {
        if let Some(decision) = try_extract(event) {
            tracing::debug!(
                session_id = %decision.session_id,
                decision_id = %decision.decision_id,
                agent_id = %decision.agent_id,
                trigger_type = ?decision.trigger,
                "enqueuing worker decision"
            );
            self.queue.enqueue(decision).await;
        }
        Ok(())
    }
}

pub fn spawn_worker_processor(
    store: Arc<dyn EventStore>,
    checkpoint_store: Arc<dyn ProcessorCheckpointStore>,
    queue: Arc<dyn WorkerQueue>,
    cancel: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    let projection = Arc::new(WorkerDecisionProjection::new(queue));
    EventProcessorRunner::new(
        store,
        checkpoint_store,
        projection,
        EventProcessorRunnerConfig::default(),
        cancel,
    )
    .spawn()
}

fn try_extract(raw: &Event) -> Option<WorkerDecisionRequest> {
    if raw.aggregate_type != SessionState::AGGREGATE_TYPE {
        return None;
    }
    let event = DomainEvent::<SessionState>::from_raw(raw).ok()?;
    let req = match &event.payload {
        EventPayload::WorkerDecisionRequested(req) => req,
        _ => return None,
    };
    let derived = event.derived.as_ref()?;
    let agent_id = derived.agent_id.as_ref()?;
    let identity = derived.identity.as_ref()?;
    let wd = derived.worker_decisions.get(&req.decision_id)?;

    Some(WorkerDecisionRequest {
        session_id: event.aggregate_id.clone(),
        tenant_id: event.tenant_id.clone(),
        decision_id: req.decision_id.clone(),
        agent_id: agent_id.clone(),
        identity: identity.clone(),
        trigger: req.trigger.clone(),
        worker_state: derived.worker_state.clone(),
        ancestry: derived.ancestry.clone(),
        span: event.span.clone(),
        attempts: wd.tracking.retry.attempts,
        deadline: wd.tracking.deadline,
    })
}
