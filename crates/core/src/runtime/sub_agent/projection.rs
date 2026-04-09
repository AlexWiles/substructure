use std::sync::Arc;

use crate::providers::memory_queue::TaskQueue;
use crate::runtime::aggregate::{AggregateState, DomainEvent};
use crate::runtime::event_store::{Event, EventStore};
use crate::runtime::processor::{
    EventProcessor, EventProcessorRunner, EventProcessorRunnerConfig, ProcessorCheckpointStore,
    ProcessorError,
};
use crate::runtime::session::events::EventPayload;
use crate::runtime::session::state::SessionState;

use super::SubAgentTask;

struct SubAgentDispatchProjection {
    queue: Arc<dyn TaskQueue<SubAgentTask>>,
}

impl SubAgentDispatchProjection {
    fn new(queue: Arc<dyn TaskQueue<SubAgentTask>>) -> Self {
        Self { queue }
    }
}

#[async_trait::async_trait]
impl EventProcessor for SubAgentDispatchProjection {
    fn name(&self) -> &'static str {
        "sub_agent_dispatch_v1"
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

        let event =
            DomainEvent::<SessionState>::from_raw(raw).map_err(|e| ProcessorError::Apply(e.to_string()))?;

        let shard_key = raw.aggregate_id.clone();

        let task = match &event.payload {
            EventPayload::SubAgentRequested(req) => {
                let auth = event
                    .derived
                    .as_ref()
                    .and_then(|d| d.auth.as_ref())
                    .cloned()
                    .ok_or_else(|| {
                        ProcessorError::Apply("missing auth in derived state".to_string())
                    })?;

                let mut ancestry = event
                    .derived
                    .as_ref()
                    .map(|d| d.ancestry.clone())
                    .unwrap_or_default();
                ancestry.push(event.aggregate_id.clone());

                Some(SubAgentTask::SpawnSubAgent {
                    source_event_id: event.id,
                    parent_session_id: event.aggregate_id,
                    tenant_id: event.tenant_id,
                    child_session_id: req.session_id.clone(),
                    agent_id: req.agent_id.clone(),
                    auth,
                    ancestry,
                    retry: req.retry.clone(),
                    span: event.span,
                })
            }
            EventPayload::SessionMessageRequested(req) => Some(SubAgentTask::SendSessionMessage {
                source_event_id: event.id,
                tenant_id: event.tenant_id,
                target_session_id: req.target_session_id.clone(),
                message: req.message.clone(),
                span: event.span,
            }),
            EventPayload::TurnCompleted(tc) => {
                let derived = event
                    .derived
                    .as_ref()
                    .ok_or_else(|| ProcessorError::Apply("missing derived state".to_string()))?;
                let parent_session_id = match derived.ancestry.last() {
                    Some(id) => id.clone(),
                    None => return Ok(()),
                };
                let agent_id = derived.agent_id.clone().unwrap_or_default();

                Some(SubAgentTask::CompleteSubAgentTurn {
                    source_event_id: event.id,
                    parent_session_id,
                    tenant_id: event.tenant_id,
                    child_session_id: event.aggregate_id,
                    agent_id,
                    turn_id: tc.turn_id.clone(),
                    data: tc.data.clone(),
                    cost: tc.turn_cost,
                    token_usage: tc.turn_token_usage.clone(),
                    span: event.span,
                })
            }
            _ => None,
        };

        let Some(task) = task else {
            return Ok(());
        };

        tracing::debug!(
            dedupe_key = %task.dedupe_key(),
            "enqueuing sub-agent task"
        );

        self.queue
            .enqueue(&shard_key, task)
            .await
            .map_err(ProcessorError::Apply)
    }
}

pub fn spawn_sub_agent_dispatch_processor(
    store: Arc<dyn EventStore>,
    checkpoint_store: Arc<dyn ProcessorCheckpointStore>,
    queue: Arc<dyn TaskQueue<SubAgentTask>>,
) -> tokio::task::JoinHandle<()> {
    let projection = Arc::new(SubAgentDispatchProjection::new(queue));
    let mut config = EventProcessorRunnerConfig::default();
    config.owner_id = Some("sub_agent_dispatch".to_string());
    EventProcessorRunner::new(store, checkpoint_store, projection, config).spawn()
}
