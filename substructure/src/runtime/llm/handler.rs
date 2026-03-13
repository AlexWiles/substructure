use std::sync::Arc;

use async_trait::async_trait;

use crate::runtime::aggregate::{execute, DomainEvent, EventHandler, ExecuteInput};
use crate::runtime::event_store::EventStore;
use crate::runtime::session::command::CommandPayload;
use crate::runtime::session::events::EventPayload;
use crate::runtime::session::state::SessionState;

use super::LlmProviderTrait;

pub struct LlmEventHandler {
    store: Arc<dyn EventStore>,
    provider: Arc<dyn LlmProviderTrait>,
}

impl LlmEventHandler {
    pub fn new(store: Arc<dyn EventStore>, provider: Arc<dyn LlmProviderTrait>) -> Self {
        Self { store, provider }
    }
}

#[async_trait]
impl EventHandler<SessionState> for LlmEventHandler {
    async fn on_event(&self, event: &DomainEvent<SessionState>) {
        let EventPayload::LlmCallRequested(req) = &event.payload else {
            return;
        };

        let auth = match event.derived.as_ref().and_then(|d| d.auth.as_ref()) {
            Some(auth) => auth,
            None => return,
        };

        let client = match self.provider.resolve(&req.llm_client, auth).await {
            Ok(c) => c,
            Err(err) => {
                let _ = execute::<SessionState>(
                    &*self.store,
                    ExecuteInput {
                        aggregate_id: event.aggregate_id,
                        tenant_id: event.tenant_id.clone(),
                        command: CommandPayload::FailLlmCall {
                            call_id: req.call_id.clone(),
                            error: err,
                            retryable: false,
                        },
                        span: event.span.child("llm_call"),
                    },
                )
                .await;
                return;
            }
        };

        let result = if req.stream {
            // TODO: wire up streaming channel
            client.call(&req.request).await
        } else {
            client.call(&req.request).await
        };

        let command = match result {
            Ok(response) => CommandPayload::CompleteLlmCall {
                call_id: req.call_id.clone(),
                response,
            },
            Err(err) => CommandPayload::FailLlmCall {
                call_id: req.call_id.clone(),
                error: err.message,
                retryable: err.retryable,
            },
        };

        let _ = execute::<SessionState>(
            &*self.store,
            ExecuteInput {
                aggregate_id: event.aggregate_id,
                tenant_id: event.tenant_id.clone(),
                command,
                span: event.span.child("llm_call"),
            },
        )
        .await;
    }
}
