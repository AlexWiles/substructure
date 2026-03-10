use std::sync::Arc;

use tokio::task::JoinHandle;

use event_store::{spawn_handler_pool, EventBus, EventStore};
use llm::handler::LlmEventHandler;
use llm::LlmProviderTrait;
use session::state::SessionState;
use wake::spawn_wake_scheduler;
use worker::spawn_worker_enqueue;
use worker::WorkerQueue;

pub mod aggregate;
pub mod event_store;
pub mod identity;
pub mod llm;
pub mod retry;
pub mod serde_helpers;
pub mod session;
pub mod span;
pub mod wake;
pub mod worker;

pub struct RuntimeConfig {
    pub llm_pool_size: usize,
    pub wake_poll_interval: std::time::Duration,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            llm_pool_size: 4,
            wake_poll_interval: std::time::Duration::from_secs(30),
        }
    }
}

pub struct Runtime {
    pub bus: EventBus,
    pub store: Arc<dyn EventStore>,
    handles: Vec<JoinHandle<()>>,
}

impl Runtime {
    pub fn shutdown(self) {
        for handle in self.handles {
            handle.abort();
        }
    }
}

pub fn start(
    store: Arc<dyn EventStore>,
    llm_provider: Arc<dyn LlmProviderTrait>,
    worker_queue: Arc<dyn WorkerQueue>,
    config: RuntimeConfig,
) -> Runtime {
    let bus = EventBus::new(1024);

    let llm_handler = Arc::new(LlmEventHandler::new(store.clone(), llm_provider));
    let llm_handle = spawn_handler_pool::<SessionState>(&bus, llm_handler, config.llm_pool_size);

    let worker_handle = spawn_worker_enqueue(&bus, worker_queue);
    let wake_handle = spawn_wake_scheduler(&bus, store.clone(), config.wake_poll_interval);

    Runtime {
        bus,
        store,
        handles: vec![llm_handle, worker_handle, wake_handle],
    }
}
