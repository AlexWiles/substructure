use std::sync::Arc;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::providers::memory_queue::TaskQueue;
use crate::runtime::aggregate::{execute, Caller, ConflictRetry, ExecuteInput};
use crate::runtime::event_store::EventStore;
use crate::runtime::session::command::CommandPayload;
use crate::runtime::session::state::SessionState;

use super::{CallContext, ErrorCode, LlmProviderTrait, LlmTask, TokenDelta, TokenDeltaTransport};

pub fn spawn_llm_task_executor(
    store: Arc<dyn EventStore>,
    provider: Arc<dyn LlmProviderTrait>,
    queue: Arc<dyn TaskQueue<LlmTask>>,
    token_delta_transport: Arc<dyn TokenDeltaTransport>,
    worker_count: usize,
    cancel: CancellationToken,
) -> Vec<JoinHandle<()>> {
    let worker_count = worker_count.max(1);
    let mut handles = Vec::with_capacity(worker_count);
    for _ in 0..worker_count {
        let store = store.clone();
        let provider = provider.clone();
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

                let resolved = provider.resolve(&task.identity).await;

                let command = match resolved {
                    Ok(client) => {
                        let ctx = CallContext {
                            session_id: &task.session_id,
                            tenant_id: &task.tenant_id,
                            agent_id: &task.agent_id,
                            call_id: &task.call_id,
                            attempt: task.attempt,
                            identity: &task.identity,
                            ancestry: &task.ancestry,
                        };
                        let result = if task.stream {
                            let (tx, rx) = mpsc::unbounded_channel();
                            let pump = spawn_delta_pump(&task, token_delta_transport.clone(), rx);
                            let result = client.call_streaming(&task.request, &ctx, tx).await;
                            let _ = pump.await;
                            result
                        } else {
                            client.call(&task.request, &ctx).await
                        };
                        match result {
                            Ok(response) => CommandPayload::CompleteLlmCall {
                                call_id: task.call_id.clone(),
                                attempt: task.attempt,
                                response,
                            },
                            Err(err) => CommandPayload::FailLlmCall {
                                call_id: task.call_id.clone(),
                                attempt: task.attempt,
                                error: err.message,
                                retryable: err.retryable,
                                code: err.code,
                                detail: err.detail,
                            },
                        }
                    }
                    Err(err) => CommandPayload::FailLlmCall {
                        call_id: task.call_id.clone(),
                        attempt: task.attempt,
                        error: err,
                        retryable: false,
                        code: Some(ErrorCode::ProviderError),
                        detail: None,
                    },
                };

                let result = execute::<SessionState>(
                    store.as_ref(),
                    ExecuteInput {
                        aggregate_id: task.session_id.clone(),
                        tenant_id: task.tenant_id.clone(),
                        caller: Caller::System,
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
    mut rx: mpsc::UnboundedReceiver<super::StreamDelta>,
) -> JoinHandle<()> {
    let root_session_id = task
        .ancestry
        .first()
        .cloned()
        .unwrap_or_else(|| task.session_id.clone());
    let session_id = task.session_id.clone();
    let agent_id = task.agent_id.clone();
    let turn_id = task.turn_id.clone();
    let call_id = task.call_id.clone();
    let attempt = task.attempt;
    tokio::spawn(async move {
        let mut seq: u32 = 0;
        while let Some(delta) = rx.recv().await {
            transport
                .publish(TokenDelta {
                    root_session_id: root_session_id.clone(),
                    session_id: session_id.clone(),
                    agent_id: agent_id.clone(),
                    turn_id: turn_id.clone(),
                    call_id: call_id.clone(),
                    attempt,
                    seq,
                    text: delta.text,
                    finish_reason: delta.finish_reason,
                })
                .await;
            seq = seq.saturating_add(1);
        }
    })
}
