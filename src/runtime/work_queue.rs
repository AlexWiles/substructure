//! Work queue trait and in-memory implementation.
//!
//! The `WorkQueue` trait abstracts the work distribution mechanism.
//! Producers enqueue work items, consumers pull matching items.
//! Implementations can be in-memory (default), Redis, SQS, etc.

use std::collections::{HashSet, VecDeque};

use async_trait::async_trait;
use ractor::{Actor, ActorProcessingErr, ActorRef, RpcReplyPort};

use crate::worker as proto;

// ---------------------------------------------------------------------------
// WorkQueue trait
// ---------------------------------------------------------------------------

#[async_trait]
pub trait WorkQueue: Send + Sync {
    /// Enqueue a work item for consumers to pick up.
    async fn enqueue(&self, item: proto::WorkItem, tenant_id: String);

    /// Block until a work item matching the given agent names (and optionally
    /// tenant) is available. Pass `None` for `tenant_id` to match all tenants.
    async fn get_work(
        &self,
        agent_names: &HashSet<String>,
        tenant_id: Option<&str>,
    ) -> proto::WorkItem;
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum WorkQueueError {
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

// ---------------------------------------------------------------------------
// QueueActor — ractor-based work queue with deferred replies
// ---------------------------------------------------------------------------

struct QueueEntry {
    item: proto::WorkItem,
    tenant_id: String,
}

struct WorkerWaiter {
    agent_names: HashSet<String>,
    tenant_id: Option<String>,
    reply: RpcReplyPort<proto::WorkItem>,
}

enum QueueMessage {
    Enqueue(QueueEntry),
    GetWork {
        agent_names: HashSet<String>,
        tenant_id: Option<String>,
        reply: RpcReplyPort<proto::WorkItem>,
    },
}

struct QueueActor;

struct QueueState {
    pending: VecDeque<QueueEntry>,
    waiting: VecDeque<WorkerWaiter>,
}

/// Extract the agent name from a work item.
fn item_agent_name(item: &proto::WorkItem) -> Option<&str> {
    match &item.item {
        Some(proto::work_item::Item::Decision(d)) => Some(&d.agent_name),
        Some(proto::work_item::Item::ToolCall(t)) => Some(&t.agent_name),
        None => None,
    }
}

/// Check whether a work item matches a waiter's filters.
fn matches(entry: &QueueEntry, agent_names: &HashSet<String>, tenant_id: Option<&str>) -> bool {
    let agent_match = item_agent_name(&entry.item)
        .map(|n| agent_names.contains(n))
        .unwrap_or(false);
    let tenant_match = tenant_id.is_none_or(|tid| entry.tenant_id == tid);
    agent_match && tenant_match
}

impl Actor for QueueActor {
    type Msg = QueueMessage;
    type State = QueueState;
    type Arguments = ();

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        _args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        Ok(QueueState {
            pending: VecDeque::new(),
            waiting: VecDeque::new(),
        })
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match message {
            QueueMessage::Enqueue(entry) => {
                // Try to match a waiting worker
                if let Some(pos) = state.waiting.iter().position(|w| {
                    matches(&entry, &w.agent_names, w.tenant_id.as_deref())
                }) {
                    let waiter = state.waiting.remove(pos).unwrap();
                    let _ = waiter.reply.send(entry.item);
                } else {
                    state.pending.push_back(entry);
                }
            }
            QueueMessage::GetWork {
                agent_names,
                tenant_id,
                reply,
            } => {
                // Try to find a matching pending item
                if let Some(pos) = state
                    .pending
                    .iter()
                    .position(|e| matches(e, &agent_names, tenant_id.as_deref()))
                {
                    let entry = state.pending.remove(pos).unwrap();
                    let _ = reply.send(entry.item);
                } else {
                    // Deferred reply — hold the port until a matching item is enqueued
                    state.waiting.push_back(WorkerWaiter {
                        agent_names,
                        tenant_id,
                        reply,
                    });
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// InMemoryWorkQueue — wraps QueueActor
// ---------------------------------------------------------------------------

/// In-memory FIFO work queue backed by a ractor actor.
pub struct InMemoryWorkQueue {
    actor: ActorRef<QueueMessage>,
}

impl InMemoryWorkQueue {
    pub async fn new(supervisor: ractor::ActorCell) -> Result<Self, String> {
        let (actor, _handle) = Actor::spawn_linked(None, QueueActor, (), supervisor)
            .await
            .map_err(|e| format!("failed to spawn queue actor: {e}"))?;
        Ok(Self { actor })
    }
}

#[async_trait]
impl WorkQueue for InMemoryWorkQueue {
    async fn enqueue(&self, item: proto::WorkItem, tenant_id: String) {
        let _ = self
            .actor
            .send_message(QueueMessage::Enqueue(QueueEntry { item, tenant_id }));
    }

    async fn get_work(
        &self,
        agent_names: &HashSet<String>,
        tenant_id: Option<&str>,
    ) -> proto::WorkItem {
        let names = agent_names.clone();
        let tid = tenant_id.map(String::from);
        match self
            .actor
            .call(
                |reply| QueueMessage::GetWork {
                    agent_names: names,
                    tenant_id: tid,
                    reply,
                },
                None, // no timeout — block until work is available
            )
            .await
        {
            Ok(ractor::rpc::CallResult::Success(item)) => item,
            _ => panic!("queue actor died"),
        }
    }
}
