use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::providers::memory_queue::TaskQueue;
use crate::runtime::event_store::EventStore;
use crate::runtime::processor::{
    EventProcessor, EventProcessorRunner, EventProcessorRunnerConfig, ProcessorCheckpointStore,
    ProcessorError,
};
use crate::runtime::session::events::{EffectKind, EventPayload};
use crate::runtime::session::SessionEvent;

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

    async fn apply(&self, event: SessionEvent) -> Result<(), ProcessorError> {
        let shard_key = event.session_id.clone();

        let task = match &event.payload {
            EventPayload::SubAgentRequested(req) => {
                let owner = event.meta.owner.clone().ok_or_else(|| {
                    ProcessorError::Apply("missing owner in event meta".to_string())
                })?;

                let mut ancestry = event.meta.ancestry.clone();
                ancestry.push(event.session_id.clone());

                Some(SubAgentTask::SpawnSubAgent {
                    source_event_id: event.id,
                    parent_session_id: event.session_id,
                    tenant_id: event.tenant_id,
                    child_session_id: req.session_id.clone(),
                    agent_id: req.agent_id.clone(),
                    owner,
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
            // A voided sub-agent delegation cancels its child session.
            EventPayload::CallVoided(v) if v.kind == EffectKind::SubAgent => v
                .session_id
                .as_ref()
                .map(|child_session_id| SubAgentTask::CancelSubAgent {
                    source_event_id: event.id,
                    tenant_id: event.tenant_id.clone(),
                    child_session_id: child_session_id.clone(),
                    span: event.span,
                }),
            EventPayload::TurnCompleted(tc) => {
                let parent_session_id = match event.meta.ancestry.last() {
                    Some(id) => id.clone(),
                    None => return Ok(()),
                };
                let agent_id = event.meta.agent_id.clone().unwrap_or_default();

                Some(SubAgentTask::CompleteSubAgentTurn {
                    source_event_id: event.id,
                    parent_session_id,
                    tenant_id: event.tenant_id,
                    child_session_id: event.session_id,
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
    cancel: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    let projection = Arc::new(SubAgentDispatchProjection::new(queue));
    let mut config = EventProcessorRunnerConfig::default();
    config.owner_id = Some("sub_agent_dispatch".to_string());
    EventProcessorRunner::new(store, checkpoint_store, projection, config, cancel).spawn()
}
