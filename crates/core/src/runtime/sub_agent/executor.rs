use std::sync::Arc;

use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::protocol::DraftMessage;
use crate::providers::memory_queue::TaskQueue;
use crate::runtime::event_store::EventStore;
use crate::runtime::session::command::{CommandPayload, Outcome, SessionError, SettleError};
use crate::runtime::session::state::EffectKind;
use crate::runtime::session::{execute, ConflictRetry, ExecuteError, ExecuteInput};
use crate::runtime::span::SpanContext;
use crate::runtime::Caller;

use super::SubAgentTask;
use crate::protocol::ErrorInfo;

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

/// Deliver a freshly created child's opening message. Nothing to do when the
/// delegation carries none — a worker may open the child its own way.
async fn open_child(
    store: &dyn EventStore,
    tenant_id: &str,
    child_session_id: &str,
    message: Option<DraftMessage>,
    span: &SpanContext,
) -> Result<(), ExecuteError> {
    let Some(message) = message else {
        return Ok(());
    };
    send_message(store, tenant_id, child_session_id, message, span)
        .await
        .map(|_| ())
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

async fn handle_task(store: &dyn EventStore, task: SubAgentTask) {
    match task {
        SubAgentTask::SpawnSubAgent {
            parent_session_id,
            tenant_id,
            child_session_id,
            agent_id,
            owner,
            ancestry,
            message,
            retry,
            span,
            ..
        } => {
            let create_result = execute(
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
                    span: span.child("create_sub_agent"),
                },
                &ConflictRetry::default(),
            )
            .await;

            // The opening message rides with the spawn, so it lands on a
            // session that exists. A delegation whose message never arrives
            // would leave the child idle and the parent waiting forever, so a
            // failed send fails the delegation.
            let outcome = match create_result {
                Ok(_) | Err(ExecuteError::Command(SessionError::SessionAlreadyCreated)) => {
                    match open_child(store, &tenant_id, &child_session_id, message, &span).await {
                        Ok(()) => Outcome::SubAgentStarted,
                        Err(err) => SettleError::new(
                            ErrorInfo::internal(format!(
                                "failed to send the child's opening message: {err}"
                            )),
                            false,
                        )
                        .into(),
                    }
                }
                Err(err) => SettleError::new(
                    ErrorInfo::internal(format!("failed to create child session: {err}")),
                    false,
                )
                .into(),
            };
            let parent_command = CommandPayload::settle(
                EffectKind::SubAgent,
                child_session_id.clone(),
                None,
                outcome,
            );

            let result = execute(
                store,
                ExecuteInput {
                    session_id: parent_session_id.clone(),
                    caller: Caller::System {
                        tenant_id: tenant_id.clone(),
                    },
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
            let result = send_message(store, &tenant_id, &target_session_id, message, &span).await;

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
            error,
            span,
            ..
        } => {
            // A child whose run failed settles the delegation as an error; its
            // empty output is not an answer.
            let command = match error {
                Some(error) => CommandPayload::settle(
                    EffectKind::SubAgent,
                    child_session_id,
                    None,
                    SettleError::new(error, false),
                ),
                None => CommandPayload::CompleteSubAgentTurn {
                    session_id: child_session_id,
                    agent_id,
                    turn_id,
                    data,
                    cost,
                    token_usage,
                },
            };
            let result = execute(
                store,
                ExecuteInput {
                    session_id: parent_session_id.clone(),
                    caller: Caller::System { tenant_id },
                    command,
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
        SubAgentTask::CancelSubAgent {
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
                    span: span.child("cancel_sub_agent"),
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
                        "failed to cancel sub-agent session"
                    );
                }
            }
        }
    }
}
