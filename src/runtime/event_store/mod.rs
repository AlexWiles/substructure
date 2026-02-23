mod store;
mod in_memory;

pub use store::{EventStore, StoreError, Version};
pub use in_memory::InMemoryEventStore;
