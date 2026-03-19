use async_trait::async_trait;

use super::memory_queue::InMemoryQueue;
use crate::llm::{LlmTask, LlmTaskQueue};

pub type InMemoryLlmTaskQueue = InMemoryQueue<LlmTask>;

#[async_trait]
impl LlmTaskQueue for InMemoryQueue<LlmTask> {
    async fn enqueue(&self, task: LlmTask) -> Result<(), String> {
        self.enqueue(task).await
    }

    async fn dequeue(&self) -> Option<LlmTask> {
        self.dequeue().await
    }
}
