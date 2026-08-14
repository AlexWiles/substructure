use std::sync::Arc;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::protocol::{ErrorCode, ErrorInfo, StreamDelta, TokenDelta};
use crate::providers::memory_queue::TaskQueue;
use crate::runtime::blob::BlobStore;
use crate::runtime::event_store::EventStore;
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
    worker_count: usize,
    cancel: CancellationToken,
) -> Vec<JoinHandle<()>> {
    let worker_count = worker_count.max(1);
    let mut handles = Vec::with_capacity(worker_count);
    for _ in 0..worker_count {
        let store = store.clone();
        let providers = providers.clone();
        let blobs = blobs.clone();
        let token_delta_transport = token_delta_transport.clone();
        let mut rx = queue.subscribe();
        let cancel = cancel.clone();
        handles.push(tokio::spawn(async move {
            loop {
                let task = tokio::select! {
                    t = rx.recv() => match t {
                        Some(t) => t,
                        None => break,
                    },
                    _ = cancel.cancelled() => break,
                };

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
                        // A provider reads the resolved shape, and resolving
                        // is the only way to build one — so no call can carry
                        // a ref the provider cannot read. A ref that will not
                        // resolve fails the call, like any other bad request.
                        let prompt = crate::runtime::blob::resolve(
                            &task.request,
                            blobs.as_ref(),
                            &task.tenant_id,
                        )
                        .await;
                        let result = match prompt {
                            Err(err) => Err(err),
                            Ok(prompt) if task.stream => {
                                let (tx, rx) = mpsc::unbounded_channel();
                                let pump =
                                    spawn_delta_pump(&task, token_delta_transport.clone(), rx);
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
                    store.as_ref(),
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
        }));
    }
    handles
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
