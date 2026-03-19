use async_trait::async_trait;

use super::memory_queue::InMemoryQueue;
use crate::sub_agent::{SubAgentTask, SubAgentTaskQueue};

pub type InMemorySubAgentTaskQueue = InMemoryQueue<SubAgentTask>;

#[async_trait]
impl SubAgentTaskQueue for InMemoryQueue<SubAgentTask> {
    async fn enqueue(&self, task: SubAgentTask) -> Result<(), String> {
        self.enqueue(task).await
    }

    async fn dequeue(&self) -> Option<SubAgentTask> {
        self.dequeue().await
    }
}
