//! Local worker loop — calls the runtime directly, no transport layer.

use std::collections::HashSet;
use std::sync::Arc;

use super::Runtime;
use crate::worker::{self as proto, Worker};

/// Execute a single work item using the given worker.
async fn execute_work_item(worker: &dyn Worker, item: proto::WorkItem) -> Option<proto::WorkResult> {
    match item.item {
        Some(proto::work_item::Item::Decision(dispatch)) => {
            let ctx = proto::WorkerCtx::from(&dispatch);
            let trigger = match dispatch.trigger.as_ref() {
                Some(t) => t,
                None => {
                    tracing::warn!("dispatch with no trigger, skipping");
                    return None;
                }
            };

            let decision = worker.decide(trigger, &dispatch.worker_state, &ctx);

            Some(proto::WorkResult {
                result: Some(proto::work_result::Result::Decision(
                    proto::DecisionResult {
                        session_id: dispatch.session_id,
                        decision_id: dispatch.decision_id,
                        decision: Some(decision),
                        span: dispatch.span,
                    },
                )),
            })
        }
        Some(proto::work_item::Item::ToolCall(dispatch)) => {
            let tool_result = worker.execute_tool_call(&dispatch).await;

            Some(proto::WorkResult {
                result: Some(proto::work_result::Result::ToolCall(
                    proto::ToolCallSubmission {
                        session_id: dispatch.session_id.clone(),
                        result: Some(tool_result),
                        span: dispatch.span,
                    },
                )),
            })
        }
        None => {
            tracing::warn!("empty work item, skipping");
            None
        }
    }
}

/// Run a local worker loop that pulls work items via the runtime.
pub async fn run_local_worker(runtime: Arc<Runtime>, worker: Arc<dyn Worker>) {
    let agent_names: HashSet<String> = worker.agent_names().into_iter().collect();
    let mut pending_result: Option<proto::WorkResult> = None;
    loop {
        // Submit previous result (if any)
        if let Some(result) = pending_result.take() {
            let _ = runtime.submit_result(result).await;
        }
        // Block for next matching work item
        let item = runtime.get_work(&agent_names, None).await;
        pending_result = execute_work_item(&*worker, item).await;
    }
}
