//! gRPC gateway — serves both workers (GetWork/SubmitResult) and clients (RunSession).

use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use uuid::Uuid;

use crate::runtime::config::ClientIdentity;
use crate::runtime::event::EventPayload;
use crate::runtime::session::{CommandPayload, IncomingMessage};
use crate::runtime::span::SpanContext;
use crate::runtime::Runtime;
use crate::worker as proto;

use super::auth::AuthResolver;
use super::SessionUpdate;

// ---------------------------------------------------------------------------
// GrpcGateway
// ---------------------------------------------------------------------------

pub struct GrpcGateway {
    auth: Arc<dyn AuthResolver>,
    runtime: Arc<Runtime>,
}

impl GrpcGateway {
    pub fn new(
        auth: Arc<dyn AuthResolver>,
        runtime: Arc<Runtime>,
    ) -> Self {
        Self { auth, runtime }
    }

    /// Start the gRPC server on the given address.
    pub async fn serve(self: Arc<Self>, addr: SocketAddr) -> Result<(), tonic::transport::Error> {
        tracing::info!(%addr, "starting gateway");
        tonic::transport::Server::builder()
            .add_service(proto::worker_gateway_server::WorkerGatewayServer::from_arc(
                self,
            ))
            .serve(addr)
            .await
    }

    /// Extract and validate a Bearer token from gRPC request metadata.
    async fn authenticate<T>(
        &self,
        request: &tonic::Request<T>,
    ) -> Result<ClientIdentity, tonic::Status> {
        let token = request
            .metadata()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));

        self.auth
            .resolve(token)
            .await
            .map_err(|e| tonic::Status::unauthenticated(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// tonic trait implementation
// ---------------------------------------------------------------------------

#[tonic::async_trait]
impl proto::worker_gateway_server::WorkerGateway for GrpcGateway {
    async fn get_work(
        &self,
        request: tonic::Request<proto::GetWorkRequest>,
    ) -> Result<tonic::Response<proto::WorkItem>, tonic::Status> {
        let identity = self.authenticate(&request).await?;
        let req = request.into_inner();

        // Submit previous result if present
        if let Some(result) = req.result {
            if let Err(e) = self.runtime.submit_result(result).await {
                tracing::warn!(error = %e, "failed to submit result from GetWork");
            }
        }

        let agent_names: HashSet<String> = req.agent_names.into_iter().collect();
        if agent_names.is_empty() {
            return Err(tonic::Status::invalid_argument("agent_names is required"));
        }

        let item = self
            .runtime
            .get_work(&agent_names, Some(&identity.tenant_id))
            .await;
        tracing::debug!(worker_id = %req.worker_id, tenant_id = %identity.tenant_id, "dispatching work item to worker");
        Ok(tonic::Response::new(item))
    }

    async fn submit_result(
        &self,
        request: tonic::Request<proto::WorkResult>,
    ) -> Result<tonic::Response<proto::SubmitResultResponse>, tonic::Status> {
        self.authenticate(&request).await?;
        self.runtime
            .submit_result(request.into_inner())
            .await
            .map_err(|e| tonic::Status::invalid_argument(e.to_string()))?;
        Ok(tonic::Response::new(proto::SubmitResultResponse {}))
    }

    type RunSessionStream = UnboundedReceiverStream<Result<proto::SessionEvent, tonic::Status>>;

    async fn run_session(
        &self,
        request: tonic::Request<proto::RunSessionRequest>,
    ) -> Result<tonic::Response<Self::RunSessionStream>, tonic::Status> {
        let identity = self.authenticate(&request).await?;
        let req = request.into_inner();

        if req.agent.is_empty() {
            return Err(tonic::Status::invalid_argument("agent is required"));
        }

        // Create the session via the runtime.
        let session = self
            .runtime
            .start_session(Uuid::new_v4(), &req.agent, identity.clone())
            .await
            .map_err(|e| tonic::Status::internal(e.to_string()))?;

        let session_id = session.session_id;
        let sid_str = session_id.to_string();

        // Clean up the default client actor from start_session (we use our own below).
        session.shutdown();

        // Unbounded channel bridges the actor callback into the gRPC stream.
        let (tx, rx) = mpsc::unbounded_channel::<Result<proto::SessionEvent, tonic::Status>>();

        // Connect a session client with a callback that feeds the channel.
        // The callback owns Option<Sender> — set to None on SessionDone to close the stream.
        let mut sender = Some(tx);
        let _client = self
            .runtime
            .connect(
                session_id,
                identity,
                Some(Box::new(move |update: &SessionUpdate| {
                    let Some(ref tx) = sender else { return };

                    if let SessionUpdate::Event(event) = update {
                        // Derive event_type from the serde tag.
                        let payload_value =
                            serde_json::to_value(&event.payload).unwrap_or_default();
                        let event_type = payload_value
                            .get("type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string();
                        let payload_json = payload_value.to_string();

                        let session_event = proto::SessionEvent {
                            session_id: sid_str.clone(),
                            event_type,
                            payload_json,
                            sequence: event.sequence,
                        };

                        let _ = tx.send(Ok(session_event));

                        if matches!(&event.payload, EventPayload::SessionDone(_)) {
                            sender = None; // drop sender → closes the stream
                        }
                    }
                })),
                true, // stop_on_done: actor self-terminates after SessionDone
            )
            .await
            .map_err(|e| tonic::Status::internal(e.to_string()))?;

        // Send the user message
        self.runtime.deliver(
            session_id,
            CommandPayload::SendMessage {
                message: IncomingMessage::User {
                    content: req.message,
                },
                stream: false,
            },
            SpanContext::root().with_name("grpc.run_session"),
        )
        .await;

        tracing::info!(%session_id, agent = %req.agent, "run_session started");

        Ok(tonic::Response::new(UnboundedReceiverStream::new(rx)))
    }
}
