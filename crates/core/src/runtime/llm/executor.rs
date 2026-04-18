use std::sync::Arc;

use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::providers::memory_queue::TaskQueue;
use crate::runtime::aggregate::{execute, ConflictRetry, ExecuteInput};
use crate::runtime::event_store::EventStore;
use crate::runtime::session::command::CommandPayload;
use crate::runtime::session::state::SessionState;

use super::{LlmProviderTrait, LlmTask};

pub fn spawn_llm_task_executor(
    store: Arc<dyn EventStore>,
    provider: Arc<dyn LlmProviderTrait>,
    queue: Arc<dyn TaskQueue<LlmTask>>,
    worker_count: usize,
    cancel: CancellationToken,
) -> Vec<JoinHandle<()>> {
    let worker_count = worker_count.max(1);
    let mut handles = Vec::with_capacity(worker_count);
    for _ in 0..worker_count {
        let store = store.clone();
        let provider = provider.clone();
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

                let resolved = provider.resolve(&task.llm_client, &task.identity).await;

                let command = match resolved {
                    Ok(client) => {
                        let result = client.call(&task.request).await;
                        match result {
                            Ok(response) => CommandPayload::CompleteLlmCall {
                                call_id: task.call_id.clone(),
                                response,
                            },
                            Err(err) => CommandPayload::FailLlmCall {
                                call_id: task.call_id.clone(),
                                error: err.message,
                                retryable: err.retryable,
                            },
                        }
                    }
                    Err(err) => CommandPayload::FailLlmCall {
                        call_id: task.call_id.clone(),
                        error: err,
                        retryable: false,
                    },
                };

                let result = execute::<SessionState>(
                    store.as_ref(),
                    ExecuteInput {
                        aggregate_id: task.session_id.clone(),
                        tenant_id: task.tenant_id.clone(),
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
