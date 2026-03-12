//! Event handler that spawns child sessions for sub-agent requests
//! and confirms them with a SubAgentStarted event on the parent.

use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use crate::runtime::aggregate::{execute, DomainEvent, EventHandler, ExecuteInput};
use crate::runtime::event_store::EventStore;
use crate::runtime::session::command::CommandPayload;
use crate::runtime::session::events::{AncestryEntry, EventPayload};
use crate::runtime::session::state::SessionState;

pub struct SubAgentHandler {
    store: Arc<dyn EventStore>,
}

impl SubAgentHandler {
    pub fn new(store: Arc<dyn EventStore>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl EventHandler<SessionState> for SubAgentHandler {
    async fn on_event(&self, event: &DomainEvent<SessionState>) {
        match &event.payload {
            // SubAgent requested → spawn child session
            EventPayload::SubAgentRequested(req) => {
                let agent_id = req.agent_id.clone();
                let child_session_id = Uuid::now_v7();

                let auth = match event.derived.as_ref().and_then(|d| d.auth.as_ref()) {
                    Some(auth) => auth.clone(),
                    None => return,
                };

                // Build child ancestry: parent's ancestry + this link
                let mut ancestry = event
                    .derived
                    .as_ref()
                    .map(|d| d.ancestry.clone())
                    .unwrap_or_default();
                ancestry.push(AncestryEntry {
                    session_id: event.aggregate_id,
                    call_id: req.call_id.clone(),
                });

                // Create child session
                let create_result = execute::<SessionState>(
                    &*self.store,
                    ExecuteInput {
                        aggregate_id: child_session_id,
                        tenant_id: event.tenant_id.clone(),
                        command: CommandPayload::CreateSession {
                            agent_id,
                            auth,
                            ancestry,
                            worker_retry: req.retry.clone(),
                        },
                        span: event.span.child("create_sub_agent"),
                    },
                )
                .await;

                if create_result.is_err() {
                    let _ = execute::<SessionState>(
                        &*self.store,
                        ExecuteInput {
                            aggregate_id: event.aggregate_id,
                            tenant_id: event.tenant_id.clone(),
                            command: CommandPayload::FailSubAgent {
                                call_id: req.call_id.clone(),
                                error: "failed to create child session".to_string(),
                                retryable: false,
                            },
                            span: event.span.child("fail_sub_agent"),
                        },
                    )
                    .await;
                    return;
                }

                // Confirm start on the parent
                let _ = execute::<SessionState>(
                    &*self.store,
                    ExecuteInput {
                        aggregate_id: event.aggregate_id,
                        tenant_id: event.tenant_id.clone(),
                        command: CommandPayload::StartSubAgent {
                            call_id: req.call_id.clone(),
                            child_session_id,
                        },
                        span: event.span.child("start_sub_agent"),
                    },
                )
                .await;

                // Send the arguments as the user message to the child
                let _ = execute::<SessionState>(
                    &*self.store,
                    ExecuteInput {
                        aggregate_id: child_session_id,
                        tenant_id: event.tenant_id.clone(),
                        command: CommandPayload::SendUserMessage {
                            content: req.arguments.clone(),
                            stream: false,
                        },
                        span: event.span.child("send_to_sub_agent"),
                    },
                )
                .await;
            }

            _ => {}
        }
    }
}
