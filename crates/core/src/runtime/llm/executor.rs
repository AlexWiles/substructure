use std::sync::Arc;

use tokio::task::JoinHandle;

use crate::runtime::aggregate::{execute, ConflictRetry, ExecuteInput};
use crate::runtime::event_store::EventStore;
use crate::runtime::session::command::CommandPayload;
use crate::runtime::session::state::SessionState;

use super::{LlmProviderTrait, LlmTaskQueue};

pub fn spawn_llm_task_executor(
    store: Arc<dyn EventStore>,
    provider: Arc<dyn LlmProviderTrait>,
    queue: Arc<dyn LlmTaskQueue>,
    worker_count: usize,
) -> Vec<JoinHandle<()>> {
    let worker_count = worker_count.max(1);
    let mut handles = Vec::with_capacity(worker_count);
    for _ in 0..worker_count {
        let store = store.clone();
        let provider = provider.clone();
        let queue = queue.clone();
        handles.push(tokio::spawn(async move {
            run_executor_loop(store, provider, queue).await;
        }));
    }
    handles
}

async fn run_executor_loop(
    store: Arc<dyn EventStore>,
    provider: Arc<dyn LlmProviderTrait>,
    queue: Arc<dyn LlmTaskQueue>,
) {
    while let Some(task) = queue.dequeue().await {
        let resolved = provider.resolve(&task.llm_client, &task.auth).await;

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
}
