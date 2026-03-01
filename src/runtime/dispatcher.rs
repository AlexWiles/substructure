use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use ractor::{Actor, ActorCell, ActorProcessingErr, ActorRef, SpawnErr};
use uuid::Uuid;

use crate::domain::aggregate::{AggregateState, DomainEvent};

use super::event_store::{EventBatch, EventStore};

// ---------------------------------------------------------------------------
// AggregateDispatcher<A> — generic, per-aggregate-type event dispatcher
// ---------------------------------------------------------------------------

/// Routing closure: receives aggregate_id + typed events, delivers to actors.
pub type RouteTypedEvents<A> = Arc<dyn Fn(Uuid, Vec<Arc<DomainEvent<A>>>) + Send + Sync>;

pub enum AggregateDispatcherMsg {
    Events(EventBatch),
}

pub struct AggregateDispatcher<A: AggregateState> {
    _phantom: PhantomData<A>,
}

pub struct AggregateDispatcherState<A: AggregateState> {
    route: RouteTypedEvents<A>,
}

pub struct AggregateDispatcherArgs<A: AggregateState> {
    pub route: RouteTypedEvents<A>,
}

impl<A: AggregateState> Actor for AggregateDispatcher<A> {
    type Msg = AggregateDispatcherMsg;
    type State = AggregateDispatcherState<A>;
    type Arguments = AggregateDispatcherArgs<A>;

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        Ok(AggregateDispatcherState { route: args.route })
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match message {
            AggregateDispatcherMsg::Events(batch) => {
                let mut by_id: HashMap<Uuid, Vec<Arc<DomainEvent<A>>>> = HashMap::new();
                for raw in &batch {
                    if raw.aggregate_type != A::aggregate_type() {
                        continue;
                    }
                    if let Ok(typed) = DomainEvent::<A>::from_raw(raw) {
                        by_id
                            .entry(typed.aggregate_id)
                            .or_default()
                            .push(Arc::new(typed));
                    }
                }
                for (id, events) in by_id {
                    (state.route)(id, events);
                }
            }
        }
        Ok(())
    }
}

pub async fn spawn_aggregate_dispatcher<A: AggregateState>(
    store: &Arc<dyn EventStore>,
    route: RouteTypedEvents<A>,
    supervisor: ActorCell,
) -> Result<(), SpawnErr> {
    let name = format!("{}-dispatcher", A::aggregate_type());
    let (actor_ref, _) = Actor::spawn_linked(
        Some(name),
        AggregateDispatcher::<A> {
            _phantom: PhantomData,
        },
        AggregateDispatcherArgs { route },
        supervisor,
    )
    .await?;

    store.events().subscribe(actor_ref, |batch| {
        Some(AggregateDispatcherMsg::Events(batch))
    });

    Ok(())
}
