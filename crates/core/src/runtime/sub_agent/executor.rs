use std::sync::Arc;

use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::runtime::aggregate::{execute, ConflictRetry, ExecuteError, ExecuteInput};
use crate::runtime::event_store::EventStore;
use crate::runtime::session::command::{CommandPayload, SessionError};
use crate::runtime::session::state::SessionState;

use super::{SubAgentTask, SubAgentTaskQueue};

pub fn spawn_sub_agent_task_executor(
    store: Arc<dyn EventStore>,
    queue: Arc<dyn SubAgentTaskQueue>,
    worker_count: usize,
) -> Vec<JoinHandle<()>> {
    let worker_count = worker_count.max(1);
    let mut handles = Vec::with_capacity(worker_count);
    for _ in 0..worker_count {
        let store = store.clone();
        let queue = queue.clone();
        handles.push(tokio::spawn(async move {
            run_executor_loop(store, queue).await;
        }));
    }
    handles
}

async fn run_executor_loop(store: Arc<dyn EventStore>, queue: Arc<dyn SubAgentTaskQueue>) {
    while let Some(task) = queue.dequeue().await {
        match task {
            SubAgentTask::SpawnSubAgent {
                parent_session_id,
                tenant_id,
                child_session_id,
                agent_id,
                auth,
                ancestry,
                retry,
                span,
                ..
            } => {
                let create_result = execute::<SessionState>(
                    store.as_ref(),
                    ExecuteInput {
                        aggregate_id: child_session_id.clone(),
                        tenant_id: tenant_id.clone(),
                        command: CommandPayload::CreateSession {
                            agent_id,
                            auth,
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
                    store.as_ref(),
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
                    store.as_ref(),
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
            SubAgentTask::ResolveToolCall {
                tenant_id,
                target_session_id,
                tool_call_id,
                result,
                span,
                ..
            } => {
                let exec = execute::<SessionState>(
                    store.as_ref(),
                    ExecuteInput {
                        aggregate_id: target_session_id.clone(),
                        tenant_id,
                        command: CommandPayload::CompleteToolCall {
                            tool_call_id,
                            result,
                            worker_state: None,
                        },
                        span: span.child("resolve_tool_call"),
                    },
                    &ConflictRetry::default(),
                )
                .await;

                if let Err(err) = exec {
                    tracing::error!(
                        target_session_id = %target_session_id,
                        error = %err,
                        "failed to resolve cross-session tool call"
                    );
                }
            }
            SubAgentTask::CompleteSubAgentTurn {
                parent_session_id,
                tenant_id,
                child_session_id,
                agent_id,
                turn_id,
                artifacts,
                cost,
                token_usage,
                span,
                ..
            } => {
                let result = execute::<SessionState>(
                    store.as_ref(),
                    ExecuteInput {
                        aggregate_id: parent_session_id.clone(),
                        tenant_id,
                        command: CommandPayload::CompleteSubAgentTurn {
                            session_id: child_session_id,
                            agent_id,
                            turn_id,
                            artifacts,
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
}
