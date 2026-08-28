use std::sync::Arc;

use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::protocol::DraftMessage;
use crate::providers::memory_queue::TaskQueue;
use crate::runtime::event_store::EventStore;
use crate::runtime::executor::{spawn_bounded_executors, ExecutorPool};
use crate::runtime::session::command::{CommandPayload, Outcome, SessionError, SettleError};
use crate::runtime::session::state::EffectKind;
use crate::runtime::session::{execute, ConflictRetry, ExecuteError, ExecuteInput};
use crate::runtime::span::SpanContext;
use crate::runtime::Caller;

use super::SubagentTask;
use crate::protocol::ErrorInfo;

pub fn spawn_subagent_task_executor(
    store: Arc<dyn EventStore>,
    queue: Arc<dyn TaskQueue<SubagentTask>>,
    pool: ExecutorPool,
    cancel: CancellationToken,
) -> Vec<JoinHandle<()>> {
    let inner = store.clone();
    spawn_bounded_executors(store, queue, pool, cancel, move |task| {
        let store = inner.clone();
        async move { handle_task(store.as_ref(), task).await }
    })
}

async fn open(
    store: &dyn EventStore,
    tenant_id: &str,
    child_session_id: &str,
    message: Option<DraftMessage>,
    span: &SpanContext,
) -> Outcome {
    let Some(message) = message else {
        return Outcome::SubagentStarted;
    };
    match send_message(store, tenant_id, child_session_id, message, span).await {
        Ok(_) => Outcome::SubagentStarted,
        Err(err) => SettleError::new(
            ErrorInfo::internal(format!("failed to send the child's opening message: {err}")),
            false,
        )
        .into(),
    }
}

async fn send_message(
    store: &dyn EventStore,
    tenant_id: &str,
    session_id: &str,
    message: DraftMessage,
    span: &SpanContext,
) -> Result<(), ExecuteError> {
    execute(
        store,
        ExecuteInput {
            session_id: session_id.to_string(),
            caller: Caller::System {
                tenant_id: tenant_id.to_string(),
            },
            command: CommandPayload::SendMessage {
                message,
                stream: false,
                turn_id: Some(Uuid::now_v7().to_string()),
                parent_id: None,
            },
            span: span.child("send_session_message"),
        },
        &ConflictRetry::default(),
    )
    .await
    .map(|_| ())
}

async fn handle_task(store: &dyn EventStore, task: SubagentTask) {
    match task {
        SubagentTask::SpawnSubagent {
            parent_session_id,
            tenant_id,
            tool_call_id,
            child_session_id,
            agent_id,
            owner,
            ancestry,
            message,
            retry,
            span,
            ..
        } => {
            let created = execute(
                store,
                ExecuteInput {
                    session_id: child_session_id.clone(),
                    caller: Caller::System {
                        tenant_id: tenant_id.clone(),
                    },
                    command: CommandPayload::CreateSession {
                        agent_id,
                        owner,
                        ancestry,
                        worker_retry: retry,
                    },
                    span: span.child("create_subagent"),
                },
                &ConflictRetry::default(),
            )
            .await
            .map(|_| ())
            .or_else(|created| match created {
                ExecuteError::Command(SessionError::SessionAlreadyCreated) => Ok(()),
                err => Err(err),
            });
            let outcome = match created {
                Ok(()) => open(store, &tenant_id, &child_session_id, message, &span).await,
                Err(err) => SettleError::new(
                    ErrorInfo::internal(format!("failed to create child session: {err}")),
                    false,
                )
                .into(),
            };
            let parent_command =
                CommandPayload::settle(EffectKind::Subagent, tool_call_id.clone(), None, outcome);

            let result = execute(
                store,
                ExecuteInput {
                    session_id: parent_session_id.clone(),
                    caller: Caller::System {
                        tenant_id: tenant_id.clone(),
                    },
                    command: parent_command,
                    span: span.child("subagent_parent_update"),
                },
                &ConflictRetry::default(),
            )
            .await;

            if let Err(err) = result {
                tracing::error!(
                    parent_session_id = %parent_session_id,
                    child_session_id = %child_session_id,
                    error = %err,
                    "failed to submit subagent parent command"
                );
            }
        }
        SubagentTask::SendSessionMessage {
            tenant_id,
            target_session_id,
            message,
            span,
            ..
        } => {
            let result = send_message(store, &tenant_id, &target_session_id, message, &span).await;

            if let Err(err) = result {
                tracing::error!(
                    target_session_id = %target_session_id,
                    error = %err,
                    "failed to send cross-session message"
                );
            }
        }
        SubagentTask::CompleteSubagentTurn {
            parent_session_id,
            tenant_id,
            child_session_id,
            agent_id,
            turn_id,
            data,
            cost,
            token_usage,
            error,
            span,
            ..
        } => {
            let command = CommandPayload::CompleteSubagentTurn {
                session_id: child_session_id,
                agent_id,
                turn_id,
                data,
                cost,
                token_usage,
                error,
            };
            let result = execute(
                store,
                ExecuteInput {
                    session_id: parent_session_id.clone(),
                    caller: Caller::System { tenant_id },
                    command,
                    span: span.child("subagent_turn_complete"),
                },
                &ConflictRetry::default(),
            )
            .await;

            if let Err(err) = result {
                tracing::error!(
                    parent_session_id = %parent_session_id,
                    error = %err,
                    "failed to submit subagent turn completion"
                );
            }
        }
        SubagentTask::CancelSubagent {
            tenant_id,
            child_session_id,
            span,
            ..
        } => {
            let result = execute(
                store,
                ExecuteInput {
                    session_id: child_session_id.clone(),
                    caller: Caller::System { tenant_id },
                    command: CommandPayload::CancelSession,
                    span: span.child("cancel_subagent"),
                },
                &ConflictRetry::default(),
            )
            .await;

            if let Err(err) = result {
                if matches!(err, ExecuteError::Command(SessionError::SessionNotCreated)) {
                    tracing::debug!(
                        child_session_id = %child_session_id,
                        "cancel for a child session that was never created"
                    );
                } else {
                    tracing::error!(
                        child_session_id = %child_session_id,
                        error = %err,
                        "failed to cancel subagent session"
                    );
                }
            }
        }
    }
}
