use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;
use ractor::{Actor, ActorCell, ActorProcessingErr, ActorRef, RpcReplyPort, SpawnErr};
use uuid::Uuid;

use crate::domain::aggregate::{Aggregate, DomainEvent};
use crate::domain::budget::{
    budget_aggregate_id, BudgetContext, BudgetDerived, BudgetEvent, BudgetLedger,
    ReservationEntry, ReservationResult, UsageRecorded,
};
use crate::domain::config::BudgetPolicyConfig;
use crate::domain::event::{EventPayload, LlmCallCompleted, LlmResponse};
use crate::domain::span::SpanContext;
use crate::runtime::event_store::{Event, EventBatch, EventStore, StoreError};

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

pub enum BudgetMessage {
    Reserve(ReserveRequest, RpcReplyPort<ReservationResult>),
    Events(EventBatch),
}

pub struct ReserveRequest {
    pub session_id: Uuid,
    pub call_id: String,
    pub context: BudgetContext,
    pub estimated_tokens: u64,
}

// ---------------------------------------------------------------------------
// Actor
// ---------------------------------------------------------------------------

pub struct BudgetActor;

pub struct BudgetActorArgs {
    pub tenant_id: String,
    pub policies: Vec<BudgetPolicyConfig>,
    pub store: Arc<dyn EventStore>,
}

pub struct BudgetActorState {
    tenant_id: String,
    aggregate_id: Uuid,
    ledger: BudgetLedger,
    reservations: HashMap<(Uuid, String), Vec<ReservationEntry>>,
    policies: Vec<BudgetPolicyConfig>,
    store: Arc<dyn EventStore>,
}

impl Actor for BudgetActor {
    type Msg = BudgetMessage;
    type State = BudgetActorState;
    type Arguments = BudgetActorArgs;

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        let aggregate_id = budget_aggregate_id(&args.tenant_id);

        let ledger = match args.store.load(aggregate_id, &args.tenant_id).await {
            Ok(loaded) => serde_json::from_value(loaded.snapshot)
                .map_err(|e| format!("budget snapshot deserialize: {e}"))?,
            Err(StoreError::StreamNotFound) => BudgetLedger::default(),
            Err(e) => return Err(format!("budget load: {e}").into()),
        };

        Ok(BudgetActorState {
            tenant_id: args.tenant_id,
            aggregate_id,
            ledger,
            reservations: HashMap::new(),
            policies: args.policies,
            store: args.store,
        })
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match message {
            BudgetMessage::Reserve(req, reply) => {
                let now = Utc::now();
                let (result, entries) = state.ledger.try_reserve(
                    &state.policies,
                    &req.context,
                    req.estimated_tokens,
                    &state.reservations,
                    now,
                );

                if matches!(result, ReservationResult::Granted) && !entries.is_empty() {
                    state
                        .reservations
                        .entry((req.session_id, req.call_id))
                        .or_default()
                        .extend(entries);
                }

                let _ = reply.send(result);
            }
            BudgetMessage::Events(batch) => {
                handle_events(state, &batch).await;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Process manager: settle reservations on session events
// ---------------------------------------------------------------------------

async fn handle_events(state: &mut BudgetActorState, batch: &[Arc<Event>]) {
    for raw in batch {
        // Only process session events for our tenant
        if raw.tenant_id != state.tenant_id || raw.aggregate_type != "session" {
            continue;
        }

        let session_id = raw.aggregate_id;

        match raw.event_type.as_str() {
            "llm.call.completed" => {
                let Ok(payload) =
                    serde_json::from_value::<EventPayload>(raw.payload.clone())
                else {
                    continue;
                };
                let EventPayload::LlmCallCompleted(LlmCallCompleted {
                    call_id,
                    response,
                }) = payload
                else {
                    continue;
                };

                let total_tokens = extract_total_tokens(&response);

                // Remove reservation
                let reservation_entries = state
                    .reservations
                    .remove(&(session_id, call_id.clone()))
                    .unwrap_or_default();

                if total_tokens > 0 {
                    settle_usage(state, session_id, &call_id, total_tokens, &reservation_entries)
                        .await;
                }
            }
            "llm.call.errored" => {
                // Extract call_id from payload
                if let Some(call_id) = raw.payload.get("call_id").and_then(|v| v.as_str()) {
                    state
                        .reservations
                        .remove(&(session_id, call_id.to_string()));
                }
            }
            "session.cancelled" | "session.done" => {
                // Remove all reservations for this session
                state
                    .reservations
                    .retain(|(sid, _), _| *sid != session_id);
            }
            _ => {}
        }
    }
}

/// Persist usage events and update the ledger.
async fn settle_usage(
    state: &mut BudgetActorState,
    session_id: Uuid,
    call_id: &str,
    total_tokens: u64,
    reservation_entries: &[ReservationEntry],
) {
    let now = Utc::now();

    // Build usage events — one per policy bucket that had a reservation
    let mut domain_events = Vec::new();
    let mut affected_keys: Vec<String> = Vec::new();

    if reservation_entries.is_empty() {
        // No reservation existed (e.g., fail-open case) — record against all matching policies
        // using a default context. We can't reconstruct the original context, but we can
        // record aggregate-level usage for all-tenant buckets.
        // Skip for now — only settle what was reserved.
        return;
    }

    // Deduplicate by composite_key and sum amounts (in practice one entry per policy)
    let mut by_key: HashMap<&str, u64> = HashMap::new();
    for entry in reservation_entries {
        *by_key.entry(&entry.composite_key).or_default() += entry.amount;
    }

    for (ck, _reserved) in &by_key {
        // Parse composite key back into policy_name and bucket_key
        let (policy_name, bucket_key) = match ck.split_once('|') {
            Some((p, b)) => (p, b),
            None => continue,
        };

        let event = BudgetEvent::UsageRecorded(UsageRecorded {
            policy_name: policy_name.to_string(),
            bucket_key: bucket_key.to_string(),
            session_id,
            call_id: call_id.to_string(),
            amount: total_tokens,
            recorded_at: now,
        });

        domain_events.push(event);
        affected_keys.push(ck.to_string());
    }

    if domain_events.is_empty() {
        return;
    }

    // Apply events to ledger and persist
    let expected_version = state.ledger.stream_version;
    for (i, event) in domain_events.iter().enumerate() {
        state
            .ledger
            .apply(event, expected_version + 1 + i as u64);
    }
    let new_version = state.ledger.stream_version;

    let raw_events: Vec<Event> = domain_events
        .into_iter()
        .enumerate()
        .map(|(i, payload)| {
            let domain_event = DomainEvent::<BudgetLedger> {
                id: Uuid::new_v4(),
                tenant_id: state.tenant_id.clone(),
                aggregate_id: state.aggregate_id,
                sequence: expected_version + 1 + i as u64,
                span: SpanContext::root(),
                occurred_at: now,
                payload,
                derived: Some(BudgetDerived {}),
            };
            domain_event.into_raw()
        })
        .collect();

    let snapshot =
        serde_json::to_value(&state.ledger).expect("budget ledger snapshot serialization");

    if let Err(e) = state
        .store
        .append(
            state.aggregate_id,
            &state.tenant_id,
            "budget",
            raw_events,
            snapshot,
            expected_version,
            new_version,
        )
        .await
    {
        tracing::error!(tenant = %state.tenant_id, error = %e, "failed to persist budget usage");
    }

    // Lazy eviction
    state.ledger.evict_expired(&state.policies, now);
}

fn extract_total_tokens(response: &LlmResponse) -> u64 {
    match response {
        LlmResponse::OpenAi(r) => r.usage.as_ref().map_or(0, |u| u.total_tokens as u64),
    }
}

// ---------------------------------------------------------------------------
// Spawning
// ---------------------------------------------------------------------------

pub fn budget_actor_name(tenant_id: &str) -> String {
    format!("budget-{tenant_id}")
}

pub async fn spawn_budget_actor(
    tenant_id: String,
    policies: Vec<BudgetPolicyConfig>,
    store: Arc<dyn EventStore>,
    supervisor: ActorCell,
) -> Result<ActorRef<BudgetMessage>, SpawnErr> {
    let actor_name = budget_actor_name(&tenant_id);
    let (actor_ref, _) = Actor::spawn_linked(
        Some(actor_name),
        BudgetActor,
        BudgetActorArgs {
            tenant_id,
            policies,
            store: store.clone(),
        },
        supervisor,
    )
    .await?;

    store
        .events()
        .subscribe(actor_ref.clone(), |batch| Some(BudgetMessage::Events(batch)));

    Ok(actor_ref)
}
