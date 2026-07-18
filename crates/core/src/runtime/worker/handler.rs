use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::runtime::event_store::EventStore;
use crate::runtime::processor::{
    EventProcessor, EventProcessorRunner, EventProcessorRunnerConfig, ProcessorCheckpointStore,
    ProcessorError,
};
use crate::runtime::session::events::EventPayload;
use crate::runtime::session::propose::propose;
use crate::runtime::session::wire::to_wire_trigger;
use crate::runtime::session::SessionEvent;

use super::{WorkerDecisionRequest, WorkerQueue};

struct WorkerDecisionProjection {
    store: Arc<dyn EventStore>,
    queue: Arc<dyn WorkerQueue>,
}

impl WorkerDecisionProjection {
    fn new(store: Arc<dyn EventStore>, queue: Arc<dyn WorkerQueue>) -> Self {
        Self { store, queue }
    }
}

#[async_trait::async_trait]
impl EventProcessor for WorkerDecisionProjection {
    fn name(&self) -> &'static str {
        "worker_decision_enqueue"
    }

    async fn apply(&self, event: SessionEvent) -> Result<(), ProcessorError> {
        if let Some(decision) = extract(self.store.as_ref(), event).await? {
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
    let projection = Arc::new(WorkerDecisionProjection::new(store.clone(), queue));
    EventProcessorRunner::new(
        store,
        checkpoint_store,
        projection,
        EventProcessorRunnerConfig::default(),
        cancel,
    )
    .spawn()
}

/// Builds the wire request for a decision event: the tree comes rewound from
/// `load_as_of`, statuses come stamped on the event's meta. A load error must
/// propagate (the runner retries the batch) — the queue never redelivers a
/// dropped decision.
async fn extract(
    store: &dyn EventStore,
    event: SessionEvent,
) -> Result<Option<WorkerDecisionRequest>, ProcessorError> {
    let req = match &event.payload {
        EventPayload::WorkerDecisionRequested(req) => req,
        _ => return Ok(None),
    };
    let meta = &event.meta;

    let (Some(agent_id), Some(owner)) = (meta.agent_id.as_ref(), meta.owner.as_ref()) else {
        return Ok(None);
    };

    let Some(wd) = meta
        .decisions
        .iter()
        .find(|d| d.decision_id == req.decision_id)
    else {
        return Ok(None);
    };

    let session = store
        .load(&event.tenant_id, &event.session_id)
        .await
        .map_err(|e| ProcessorError::Apply(format!("load session for decision: {e}")))?;

    // The trigger's single stored copy rode the decision's queued event into
    // state; it is immutable per decision. Absent means the decision settled
    // after this event — nothing left to deliver.
    let Some(trigger) = session
        .state
        .worker_decisions
        .get(&req.decision_id)
        .map(|d| d.trigger.clone())
    else {
        return Ok(None);
    };

    let state = session.state.rewind(event.seq, meta.head_id.as_deref());

    let message_tree = state.message_tree();

    let transcript = message_tree
        .head_id
        .as_deref()
        .map(|h| message_tree.path_to(h))
        .unwrap_or_default();
    // Call entries are keyed by immutable prompt/spec; the as-of path picks
    // the as-of subset even from the current map.
    let open_llm_calls = state.open_llm_calls(&message_tree);

    let pending_calls = meta.pending_work(&req.decision_id);
    let worker_state = state.resolve_state_for(message_tree.head_id.as_deref());
    let agent_config = state.resolve_agent_for(message_tree.head_id.as_deref());

    let trigger = to_wire_trigger(trigger, &transcript, &message_tree, &open_llm_calls);
    let proposed = propose(
        &trigger,
        &transcript,
        &open_llm_calls,
        pending_calls,
        agent_config.as_ref(),
        &req.decision_id,
    )
    .unwrap_or_default();

    Ok(Some(WorkerDecisionRequest {
        session_id: event.session_id.clone(),
        decision_id: req.decision_id.clone(),
        agent_id: agent_id.clone(),
        identity: owner.clone(),
        trigger,
        proposed,
        state: worker_state,
        agent: agent_config,
        calls: meta.calls.clone(),
        pending_calls,
        transcript,
        message_tree,
        ancestry: meta.ancestry.clone(),
        span: event.span.clone(),
        attempts: wd.attempts,
        deadline: wd.deadline,
        turn_id: meta.turn_id.clone(),
    }))
}
