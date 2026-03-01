#[cfg(feature = "sqlite")]
mod sqlite;
mod store;

#[cfg(feature = "sqlite")]
pub use sqlite::SqliteEventStore;
pub use store::{
    AggregateFilter, AggregateSort, AggregateSummary, Event, EventBatch, EventFilter, EventStore,
    StoreError, StreamLoad, Version,
};
