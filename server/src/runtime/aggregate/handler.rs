use async_trait::async_trait;

use super::state::{AggregateState, DomainEvent};

#[async_trait]
pub trait EventHandler<R: AggregateState>: Send + Sync {
    async fn on_event(&self, event: &DomainEvent<R>);
}
