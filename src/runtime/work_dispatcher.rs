//! Work dispatcher — enqueues work items into the shared decision queue.
//!
//! Pure queue dispatch: no tool execution, no sub-agent handling.
//! Workers pull items from the queue and execute them independently.

use std::sync::Arc;

use crate::runtime::work_queue;
use crate::runtime::session::{ToolCallDispatch, WorkerDispatch, WorkerExecutor};
use crate::worker as proto;

// ---------------------------------------------------------------------------
// WorkDispatcher
// ---------------------------------------------------------------------------

/// Enqueues work items (decisions and tool calls) into the shared WorkQueue.
///
/// No tool execution, no sub-agent handling — just proto conversion and enqueue.
/// Workers pull items from the queue and handle them independently.
pub struct WorkDispatcher {
    queue: Arc<dyn work_queue::WorkQueue>,
}

impl WorkDispatcher {
    pub fn new(queue: Arc<dyn work_queue::WorkQueue>) -> Self {
        Self { queue }
    }

    fn build_proto_dispatch(request: &WorkerDispatch) -> proto::WorkerDispatch {
        proto::WorkerDispatch {
            session_id: request.session_id.to_string(),
            decision_id: request.decision_id.clone(),
            trigger: Some((&request.trigger).into()),
            worker_state: request.worker_state.clone(),
            stream: request.stream,
            agent_name: request.agent_name.clone(),
            token_usage: request
                .token_usage
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
            tool_call_statuses: request
                .tool_call_statuses
                .iter()
                .map(|(id, s)| {
                    let cs: proto::CallStatus = s.into();
                    (id.clone(), cs as i32)
                })
                .collect(),
            llm_call_statuses: request
                .llm_call_statuses
                .iter()
                .map(|(id, s)| {
                    let cs: proto::CallStatus = s.into();
                    (id.clone(), cs as i32)
                })
                .collect(),
            span: Some((&request.span).into()),
            auth: Some((&request.auth).into()),
        }
    }

    fn build_proto_tool_call(request: &ToolCallDispatch) -> proto::ToolCallDispatch {
        proto::ToolCallDispatch {
            session_id: request.session_id.to_string(),
            tool_call_id: request.tool_call_id.clone(),
            name: request.name.clone(),
            arguments: request.arguments.clone(),
            context: if request.context.is_null() {
                None
            } else {
                serde_json::from_value(request.context.clone()).ok()
            },
            max_result_bytes: None,
            span: Some((&request.span).into()),
            agent_name: request.agent_name.clone(),
            auth: Some((&request.auth).into()),
        }
    }
}

impl WorkerExecutor for WorkDispatcher {
    fn dispatch_decision(&self, request: WorkerDispatch) {
        let tenant_id = request.auth.tenant_id.clone();
        let proto_dispatch = Self::build_proto_dispatch(&request);
        let queue = self.queue.clone();
        tokio::spawn(async move {
            queue
                .enqueue(
                    proto::WorkItem {
                        item: Some(proto::work_item::Item::Decision(proto_dispatch)),
                    },
                    tenant_id,
                )
                .await;
        });
    }

    fn dispatch_tool_call(&self, request: ToolCallDispatch) {
        let tenant_id = request.auth.tenant_id.clone();
        let proto_dispatch = Self::build_proto_tool_call(&request);
        let queue = self.queue.clone();
        tokio::spawn(async move {
            queue
                .enqueue(
                    proto::WorkItem {
                        item: Some(proto::work_item::Item::ToolCall(proto_dispatch)),
                    },
                    tenant_id,
                )
                .await;
        });
    }
}
