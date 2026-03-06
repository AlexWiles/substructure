//! Decision queue — shared queue for worker decision dispatch.
//!
//! Both in-process and remote workers consume from the same queue.
//! In-process: a local worker loop calls `dequeue()` → `decide()` → `deliver()`.
//! Remote: gRPC `GetDecision` (long-poll) wraps `dequeue()`, `SubmitDecision` wraps `deliver()`.

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::Arc;

use ractor::ActorRef;
use tokio::sync::{Mutex, Notify};
use uuid::Uuid;

use crate::runtime::session::CommandPayload;
use crate::runtime::span::SpanContext;
use crate::runtime::types::RuntimeMessage;
use crate::worker::default::DefaultWorker;
use crate::worker::{self as proto, Worker};

// ---------------------------------------------------------------------------
// DecisionQueue
// ---------------------------------------------------------------------------

pub struct DecisionQueue {
    pending: Mutex<VecDeque<proto::WorkerDispatch>>,
    notify: Notify,
    runtime: ActorRef<RuntimeMessage>,
}

impl DecisionQueue {
    pub fn new(runtime: ActorRef<RuntimeMessage>) -> Self {
        Self {
            pending: Mutex::new(VecDeque::new()),
            notify: Notify::new(),
            runtime,
        }
    }

    /// Enqueue a decision dispatch for a worker to pick up.
    pub async fn enqueue(&self, dispatch: proto::WorkerDispatch) {
        self.pending.lock().await.push_back(dispatch);
        self.notify.notify_one();
    }

    /// Dequeue a decision dispatch, blocking until one is available.
    pub async fn dequeue(&self) -> proto::WorkerDispatch {
        loop {
            {
                let mut queue = self.pending.lock().await;
                if let Some(dispatch) = queue.pop_front() {
                    return dispatch;
                }
            }
            self.notify.notified().await;
        }
    }

    /// Deliver a worker decision back to the session.
    pub fn deliver(
        &self,
        session_id: Uuid,
        decision_id: String,
        decision: proto::WorkerDecision,
        span: SpanContext,
    ) {
        if let Err(e) = self.runtime.send_message(RuntimeMessage::DeliverToSession {
            session_id,
            payload: CommandPayload::SubmitWorkerDecision {
                decision_id,
                actions: decision.actions,
                state: decision.state,
            },
            span,
        }) {
            tracing::warn!(error = %e, %session_id, "failed to deliver decision to session");
        }
    }

    /// Start the gRPC server on the given address.
    pub async fn serve(
        self: Arc<Self>,
        addr: SocketAddr,
    ) -> Result<(), tonic::transport::Error> {
        tracing::info!(%addr, "starting worker gateway");
        tonic::transport::Server::builder()
            .add_service(
                proto::worker_gateway_server::WorkerGatewayServer::from_arc(self),
            )
            .serve(addr)
            .await
    }
}

// ---------------------------------------------------------------------------
// Local worker loop — in-process consumer
// ---------------------------------------------------------------------------

/// Run a local worker loop that pulls decisions from the queue and executes them.
pub async fn run_local_worker(queue: Arc<DecisionQueue>) {
    let worker = DefaultWorker;
    loop {
        let dispatch = queue.dequeue().await;

        let ctx = proto::WorkerCtx::from(&dispatch);
        let Some(trigger) = dispatch.trigger.as_ref() else {
            tracing::warn!("dequeued dispatch with no trigger, skipping");
            continue;
        };

        let decision = worker.decide(trigger, &dispatch.worker_state, &ctx);

        let Ok(session_id) = dispatch.session_id.parse::<Uuid>() else {
            tracing::warn!(
                session_id = %dispatch.session_id,
                "invalid session_id in dispatch, skipping"
            );
            continue;
        };
        let span = serde_json::from_str(&dispatch.span_json)
            .unwrap_or_else(|_| SpanContext::root());

        queue.deliver(session_id, dispatch.decision_id, decision, span);
    }
}

// ---------------------------------------------------------------------------
// gRPC trait implementation
// ---------------------------------------------------------------------------

#[tonic::async_trait]
impl proto::worker_gateway_server::WorkerGateway for DecisionQueue {
    async fn get_decision(
        &self,
        _request: tonic::Request<proto::GetDecisionRequest>,
    ) -> Result<tonic::Response<proto::WorkerDispatch>, tonic::Status> {
        let dispatch = self.dequeue().await;
        tracing::debug!(
            session_id = %dispatch.session_id,
            decision_id = %dispatch.decision_id,
            "dispatching decision to remote worker"
        );
        Ok(tonic::Response::new(dispatch))
    }

    async fn submit_decision(
        &self,
        request: tonic::Request<proto::SubmitDecisionRequest>,
    ) -> Result<tonic::Response<proto::SubmitDecisionResponse>, tonic::Status> {
        let req = request.into_inner();

        let session_id = req
            .session_id
            .parse::<Uuid>()
            .map_err(|e| tonic::Status::invalid_argument(format!("invalid session_id: {e}")))?;

        let decision = req
            .decision
            .ok_or_else(|| tonic::Status::invalid_argument("decision is required"))?;

        let span = if req.span_json.is_empty() {
            SpanContext::root().with_name("gateway.submit")
        } else {
            serde_json::from_str(&req.span_json).unwrap_or_else(|_| {
                tracing::warn!("invalid span_json in SubmitDecision, using root span");
                SpanContext::root().with_name("gateway.submit")
            })
        };

        self.deliver(session_id, req.decision_id, decision, span);

        Ok(tonic::Response::new(proto::SubmitDecisionResponse {}))
    }
}
