use std::sync::Arc;

use tokio::task::JoinHandle;

use crate::runtime::aggregate::{AggregateState, DomainEvent};
use crate::runtime::event_store::{Event, EventStore};
use crate::runtime::session::events::EventPayload;
use crate::runtime::session::state::SessionState;

use super::{WorkerDecisionRequest, WorkerQueue};

pub fn spawn_worker_enqueue(
    store: Arc<dyn EventStore>,
    queue: Arc<dyn WorkerQueue>,
) -> JoinHandle<()> {
    let mut rx = store.subscribe();
    tokio::spawn(async move {
        while let Ok(batch) = rx.recv().await {
            for raw in batch.iter() {
                if let Some(decision) = try_extract(raw) {
                    queue.enqueue(decision).await;
                }
            }
        }
    })
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
        session_id: event.aggregate_id,
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
