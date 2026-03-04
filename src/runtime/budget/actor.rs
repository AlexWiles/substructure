use std::sync::Arc;

use chrono::Utc;
use ractor::ActorCell;

use crate::runtime::aggregate::actor::{spawn_aggregate_actor, AggregateActorArgs, AggregateActorHandle, AggregateMessage};
use crate::runtime::aggregate::{AggregateState, DomainEvent};
use crate::runtime::budget::{budget_aggregate_id, flatten_usage, BudgetCommand, BudgetLedger, UsageBreakdown};
use crate::runtime::config::BudgetPolicyConfig;
use crate::runtime::event::EventPayload;
use crate::runtime::llm::{LlmCallCompleted, LlmCallErrored, LlmResponse};
use crate::runtime::event_store::EventStore;
use crate::runtime::session::SessionState;

pub fn budget_actor_name(tenant_id: &str) -> String {
    format!("budget-{tenant_id}")
}

fn extract_usage_breakdown(response: &LlmResponse) -> UsageBreakdown {
    response
        .usage()
        .map(flatten_usage)
        .unwrap_or_default()
}

pub async fn spawn_budget_actor(
    tenant_id: String,
    policies: Vec<BudgetPolicyConfig>,
    store: Arc<dyn EventStore>,
    supervisor: ActorCell,
) -> Result<AggregateActorHandle<BudgetLedger>, ractor::SpawnErr> {
    let aggregate_id = budget_aggregate_id(&tenant_id);
    let tenant = tenant_id.clone();
    let policies_for_context = policies.clone();

    let handle = spawn_aggregate_actor(
        AggregateActorArgs {
            aggregate_id,
            store: store.clone(),
            tenant_id,
            init: Box::new(|_| BudgetLedger::default()),
            context_init: Box::new(move |_state| {
                let policies = policies_for_context.clone();
                Box::pin(async move { policies })
            }),
            idle_timeout: Some(std::time::Duration::from_secs(86400 * 365)),
        },
        supervisor,
    )
    .await?;

    // Cross-aggregate subscription: session events → budget commands → Cast
    let budget_ref = handle.actor.clone();
    store.events().subscribe(budget_ref, move |batch| {
        for raw in &batch {
            if raw.aggregate_type != SessionState::aggregate_type() || raw.tenant_id != tenant {
                continue;
            }
            let Ok(typed) = DomainEvent::<SessionState>::from_raw(raw) else {
                continue;
            };
            let session_id = typed.aggregate_id;
            let cmd = match &typed.payload {
                EventPayload::LlmCallCompleted(LlmCallCompleted { call_id, response }) => {
                    BudgetCommand::RecordUsage {
                        session_id,
                        call_id: call_id.clone(),
                        breakdown: extract_usage_breakdown(response),
                        recorded_at: Utc::now(),
                    }
                }
                EventPayload::LlmCallErrored(LlmCallErrored { call_id, .. }) => {
                    BudgetCommand::ReleaseReservation {
                        session_id,
                        call_id: call_id.clone(),
                    }
                }
                EventPayload::SessionCancelled | EventPayload::SessionDone(_) => {
                    BudgetCommand::ReleaseSessionReservations { session_id }
                }
                _ => continue,
            };
            return Some(AggregateMessage::Cast {
                cmd,
                span: typed.span.clone(),
                occurred_at: typed.occurred_at,
            });
        }
        None
    });

    Ok(handle)
}
