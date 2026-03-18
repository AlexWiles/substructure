use std::sync::Arc;

use crate::runtime::aggregate::{AggregateState, DomainEvent};
use crate::runtime::event_store::{Event, EventStore};
use crate::runtime::projection::{Projection, ProjectionError, ProjectionRunner, ProjectionRunnerConfig, ProjectionCheckpointStore};
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
impl Projection for WorkerDecisionProjection {
    fn name(&self) -> &'static str {
        "worker_decision_enqueue"
    }

    fn shard_key(&self, event: &Event) -> Option<String> {
        if event.aggregate_type != SessionState::AGGREGATE_TYPE {
            return None;
        }
        Some(event.aggregate_id.clone())
    }

    async fn apply(&self, event: &Event) -> Result<(), ProjectionError> {
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

pub fn spawn_worker_projection(
    store: Arc<dyn EventStore>,
    checkpoint_store: Arc<dyn ProjectionCheckpointStore>,
    queue: Arc<dyn WorkerQueue>,
) -> tokio::task::JoinHandle<()> {
    let projection = Arc::new(WorkerDecisionProjection::new(queue));
    ProjectionRunner::new(
        store,
        checkpoint_store,
        projection,
        ProjectionRunnerConfig::default(),
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
    let auth = derived.auth.as_ref()?;
    let wd = derived.worker_decisions.get(&req.decision_id)?;

    Some(WorkerDecisionRequest {
        session_id: event.aggregate_id.clone(),
        tenant_id: event.tenant_id.clone(),
        decision_id: req.decision_id.clone(),
        agent_id: agent_id.clone(),
        auth: auth.clone(),
        trigger: req.trigger.clone(),
        worker_state: derived.worker_state.clone(),
        ancestry: derived.ancestry.clone(),
        span: event.span.clone(),
        attempts: wd.tracking.retry.attempts,
        deadline: wd.tracking.deadline,
    })
}
