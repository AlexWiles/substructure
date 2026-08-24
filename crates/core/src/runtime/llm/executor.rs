use std::sync::Arc;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::protocol::{ErrorCode, ErrorInfo, StreamDelta, TokenDelta};
use crate::providers::memory_queue::TaskQueue;
use crate::runtime::blob::BlobStore;
use crate::runtime::event_store::EventStore;
use crate::runtime::executor::{spawn_bounded_executors, ExecutorPool};
use crate::runtime::session::command::{CommandPayload, Outcome, SettleError};
use crate::runtime::session::state::EffectKind;
use crate::runtime::session::{execute, ConflictRetry, ExecuteInput};
use crate::runtime::Caller;

use super::{CallContext, LlmResolver, LlmTask, TokenDeltaTransport};

pub fn spawn_llm_task_executor(
    store: Arc<dyn EventStore>,
    providers: Arc<dyn LlmResolver>,
    queue: Arc<dyn TaskQueue<LlmTask>>,
    token_delta_transport: Arc<dyn TokenDeltaTransport>,
    blobs: Arc<dyn BlobStore>,
    pool: ExecutorPool,
    cancel: CancellationToken,
) -> Vec<JoinHandle<()>> {
    let inner = store.clone();
    spawn_bounded_executors(store, queue, pool, cancel, move |task| {
        let store = inner.clone();
        let providers = providers.clone();
        let blobs = blobs.clone();
        let token_delta_transport = token_delta_transport.clone();
        async move {
            handle_task(
                store.as_ref(),
                providers.as_ref(),
                blobs.as_ref(),
                token_delta_transport,
                task,
            )
            .await
        }
    })
}

async fn handle_task(
    store: &dyn EventStore,
    providers: &dyn LlmResolver,
    blobs: &dyn BlobStore,
    token_delta_transport: Arc<dyn TokenDeltaTransport>,
    task: LlmTask,
) {
    let resolved = providers.resolve(&task.llm, &task.owner).await;

    let command = match resolved {
        Ok(client) => {
            let ctx = CallContext {
                session_id: &task.session_id,
                tenant_id: &task.tenant_id,
                agent_id: &task.agent_id,
                call_id: &task.call_id,
                attempt: task.attempt,
                owner: &task.owner,
                ancestry: &task.ancestry,
                defer_tools_strategy: task.defer_tools_strategy,
            };
            let prompt = crate::runtime::blob::resolve(&task.request, blobs, &task.tenant_id).await;
            let result = match prompt {
                Err(err) => Err(err),
                Ok(prompt) if task.stream => {
                    let (tx, rx) = mpsc::unbounded_channel();
                    let pump = spawn_delta_pump(&task, token_delta_transport.clone(), rx);
                    let result = client.call_streaming(&prompt, &ctx, tx).await;
                    let _ = pump.await;
                    result
                }
                Ok(prompt) => client.call(&prompt, &ctx).await,
            };
            let outcome = match result {
                Ok(response) => Outcome::Llm(Box::new(response)),
                Err(err) => SettleError::new(err.error, err.retryable).into(),
            };
            CommandPayload::settle(
                EffectKind::LlmCall,
                task.call_id.clone(),
                Some(task.attempt),
                outcome,
            )
        }
        Err(err) => CommandPayload::settle(
            EffectKind::LlmCall,
            task.call_id.clone(),
            Some(task.attempt),
            SettleError::new(ErrorInfo::new(ErrorCode::ProviderError, err), false),
        ),
    };

    let result = execute(
        store,
        ExecuteInput {
            session_id: task.session_id.clone(),
            caller: Caller::System {
                tenant_id: task.tenant_id.clone(),
            },
            command,
            span: task.span.child("llm_call"),
        },
        &ConflictRetry::default(),
    )
    .await;

    if let Err(err) = result {
        tracing::error!(
            session_id = %task.session_id,
            call_id = %task.call_id,
            error = %err,
            "failed to submit llm completion command"
        );
    }
}

fn spawn_delta_pump(
    task: &LlmTask,
    transport: Arc<dyn TokenDeltaTransport>,
    mut rx: mpsc::UnboundedReceiver<StreamDelta>,
) -> JoinHandle<()> {
    let template = TokenDelta {
        root_session_id: task
            .ancestry
            .first()
            .cloned()
            .unwrap_or_else(|| task.session_id.clone()),
        session_id: task.session_id.clone(),
        tenant_id: task.tenant_id.clone(),
        agent_id: task.agent_id.clone(),
        turn_id: task.turn_id.clone(),
        call_id: task.call_id.clone(),
        attempt: task.attempt,
        seq: 0,
        text: None,
        reasoning: None,
        tool_calls: Vec::new(),
        finish_reason: None,
    };
    tokio::spawn(async move {
        let mut seq: u32 = 0;
        while let Some(delta) = rx.recv().await {
            transport
                .publish(TokenDelta {
                    seq,
                    text: delta.text,
                    reasoning: delta.reasoning,
                    tool_calls: delta.tool_calls,
                    finish_reason: delta.finish_reason,
                    ..template.clone()
                })
                .await;
            seq = seq.saturating_add(1);
        }
    })
}
