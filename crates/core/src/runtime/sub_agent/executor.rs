use std::sync::Arc;

use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::providers::memory_queue::TaskQueue;
use crate::runtime::aggregate::{execute, ConflictRetry, ExecuteError, ExecuteInput};
use crate::runtime::event_store::EventStore;
use crate::runtime::session::command::{CommandPayload, SessionError};
use crate::runtime::session::state::SessionState;

use super::SubAgentTask;

pub fn spawn_sub_agent_task_executor(
    store: Arc<dyn EventStore>,
    queue: Arc<dyn TaskQueue<SubAgentTask>>,
    worker_count: usize,
    cancel: CancellationToken,
) -> Vec<JoinHandle<()>> {
    let worker_count = worker_count.max(1);
    let mut handles = Vec::with_capacity(worker_count);
    for _ in 0..worker_count {
        let store = store.clone();
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
                handle_task(store.as_ref(), task).await;
            }
        }));
    }
    handles
}

async fn handle_task(store: &dyn EventStore, task: SubAgentTask) {
    match task {
        SubAgentTask::SpawnSubAgent {
            parent_session_id,
            tenant_id,
            child_session_id,
            agent_id,
            identity,
            ancestry,
            retry,
            span,
            ..
        } => {
            let create_result = execute::<SessionState>(
                store,
                ExecuteInput {
                    aggregate_id: child_session_id.clone(),
                    tenant_id: tenant_id.clone(),
                    command: CommandPayload::CreateSession {
                        agent_id,
                        identity,
                        ancestry,
                        worker_retry: retry,
                    },
                    span: span.child("create_sub_agent"),
                },
                &ConflictRetry::default(),
            )
            .await;

            let parent_command = match create_result {
                Ok(_) => CommandPayload::StartSubAgent {
                    session_id: child_session_id.clone(),
                },
                Err(ExecuteError::Command(SessionError::SessionAlreadyCreated)) => {
                    CommandPayload::StartSubAgent {
                        session_id: child_session_id.clone(),
                    }
                }
                Err(err) => CommandPayload::FailSubAgent {
                    session_id: child_session_id.clone(),
                    error: format!("failed to create child session: {err}"),
                    retryable: false,
                },
            };

            let result = execute::<SessionState>(
                store,
                ExecuteInput {
                    aggregate_id: parent_session_id.clone(),
                    tenant_id: tenant_id.clone(),
                    command: parent_command,
                    span: span.child("sub_agent_parent_update"),
                },
                &ConflictRetry::default(),
            )
            .await;

            if let Err(err) = result {
                tracing::error!(
                    parent_session_id = %parent_session_id,
                    child_session_id = %child_session_id,
                    error = %err,
                    "failed to submit sub-agent parent command"
                );
            }
        }
        SubAgentTask::SendSessionMessage {
            tenant_id,
            target_session_id,
            message,
            span,
            ..
        } => {
            let result = execute::<SessionState>(
                store,
                ExecuteInput {
                    aggregate_id: target_session_id.clone(),
                    tenant_id,
                    command: CommandPayload::SendMessage {
                        message,
                        stream: false,
                        turn_id: Some(Uuid::now_v7().to_string()),
                    },
                    span: span.child("send_session_message"),
                },
                &ConflictRetry::default(),
            )
            .await;

            if let Err(err) = result {
                tracing::error!(
                    target_session_id = %target_session_id,
                    error = %err,
                    "failed to send cross-session message"
                );
            }
        }
        SubAgentTask::CompleteSubAgentTurn {
            parent_session_id,
            tenant_id,
            child_session_id,
            agent_id,
            turn_id,
            data,
            cost,
            token_usage,
            span,
            ..
        } => {
            let result = execute::<SessionState>(
                store,
                ExecuteInput {
                    aggregate_id: parent_session_id.clone(),
                    tenant_id,
                    command: CommandPayload::CompleteSubAgentTurn {
                        session_id: child_session_id,
                        agent_id,
                        turn_id,
                        data,
                        cost,
                        token_usage,
                    },
                    span: span.child("sub_agent_turn_complete"),
                },
                &ConflictRetry::default(),
            )
            .await;

            if let Err(err) = result {
                tracing::error!(
                    parent_session_id = %parent_session_id,
                    error = %err,
                    "failed to submit sub-agent turn completion"
                );
            }
        }
    }
}
