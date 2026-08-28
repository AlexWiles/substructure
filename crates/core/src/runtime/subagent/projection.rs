use std::sync::Arc;

use chrono::Utc;
use tokio_util::sync::CancellationToken;

use crate::providers::memory_queue::TaskQueue;
use crate::runtime::event_store::EventStore;
use crate::runtime::processor::{
    EventProcessor, EventProcessorRunner, EventProcessorRunnerConfig, ProcessorCursorStore,
    ProcessorError,
};
use crate::runtime::session::events::{EffectKind, EventPayload};
use crate::runtime::session::SessionEvent;

use super::SubagentTask;

struct SubagentDispatchProjection {
    store: Arc<dyn EventStore>,
    queue: Arc<dyn TaskQueue<SubagentTask>>,
}

impl SubagentDispatchProjection {
    fn new(store: Arc<dyn EventStore>, queue: Arc<dyn TaskQueue<SubagentTask>>) -> Self {
        Self { store, queue }
    }

    async fn parent(
        &self,
        tenant_id: &str,
        session_id: &str,
        what: &str,
    ) -> Result<crate::runtime::session::SessionAggregate, ProcessorError> {
        self.store
            .load(tenant_id, session_id)
            .await
            .map_err(|e| ProcessorError::Apply(format!("load session for {what}: {e}")))
    }
}

#[async_trait::async_trait]
impl EventProcessor for SubagentDispatchProjection {
    fn name(&self) -> &'static str {
        "sub_agent_dispatch_v1"
    }

    async fn apply(&self, event: SessionEvent) -> Result<(), ProcessorError> {
        let shard_key = event.session_id.clone();

        let task = match &event.payload {
            EventPayload::SubagentDispatched(d) => {
                let owner = event.meta.owner.clone().ok_or_else(|| {
                    ProcessorError::Apply("missing owner in event meta".to_string())
                })?;

                let session = self
                    .parent(&event.tenant_id, &event.session_id, "subagent dispatch")
                    .await?;
                let Some(effect) = session.state.effect(EffectKind::Subagent, &d.id) else {
                    return Ok(());
                };
                let Some(sa) = effect.subagent() else {
                    return Ok(());
                };

                let mut ancestry = event.meta.ancestry.clone();
                ancestry.push(event.session_id.clone());

                Some(SubagentTask::SpawnSubagent {
                    source_event_id: event.id,
                    parent_session_id: event.session_id,
                    tenant_id: event.tenant_id,
                    tool_call_id: d.id.clone(),
                    child_session_id: sa.session_id.clone(),
                    agent_id: sa.agent_id.clone(),
                    owner,
                    ancestry,
                    message: sa.message.clone(),
                    retry: effect.tracking.retry_policy.clone(),
                    enqueued_at: Utc::now(),
                    span: event.span,
                })
            }
            EventPayload::SessionMessageRequested(req) => Some(SubagentTask::SendSessionMessage {
                source_event_id: event.id,
                tenant_id: event.tenant_id,
                target_session_id: req.target_session_id.clone(),
                message: req.message.clone(),
                span: event.span,
            }),
            EventPayload::CallVoided(v) if v.kind == EffectKind::Subagent => {
                let session = self
                    .parent(&event.tenant_id, &event.session_id, "subagent cancel")
                    .await?;
                let Some(sa) = session.state.subagent(&v.id) else {
                    return Ok(());
                };
                Some(SubagentTask::CancelSubagent {
                    source_event_id: event.id,
                    tenant_id: event.tenant_id.clone(),
                    child_session_id: sa.session_id.clone(),
                    span: event.span,
                })
            }
            EventPayload::TurnCompleted(tc) => {
                let parent_session_id = match event.meta.ancestry.last() {
                    Some(id) => id.clone(),
                    None => return Ok(()),
                };
                let agent_id = event.meta.agent_id.clone().unwrap_or_default();

                Some(SubagentTask::CompleteSubagentTurn {
                    source_event_id: event.id,
                    parent_session_id,
                    tenant_id: event.tenant_id,
                    child_session_id: event.session_id,
                    agent_id,
                    turn_id: tc.turn_id.clone(),
                    data: tc.data.clone(),
                    cost: tc.turn_cost,
                    token_usage: tc.turn_token_usage.clone(),
                    error: tc.error.clone(),
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
            "enqueuing subagent task"
        );

        self.queue
            .enqueue(&shard_key, task)
            .await
            .map_err(ProcessorError::Apply)
    }
}

pub fn spawn_subagent_dispatch_processor(
    store: Arc<dyn EventStore>,
    cursor_store: Arc<dyn ProcessorCursorStore>,
    queue: Arc<dyn TaskQueue<SubagentTask>>,
    cancel: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    let projection = Arc::new(SubagentDispatchProjection::new(store.clone(), queue));
    let config = EventProcessorRunnerConfig {
        owner_id: Some("subagent_dispatch".to_string()),
        ..Default::default()
    };
    EventProcessorRunner::new(store, cursor_store, projection, config, cancel).spawn()
}
