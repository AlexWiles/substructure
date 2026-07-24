use std::sync::Arc;

use async_trait::async_trait;
use axum::Router;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::event_store::Seq;
use crate::processor::{EventProcessor, EventProcessorRunnerConfig};
use crate::protocol::TokenDelta;
use crate::session::subscriptions::SessionSubscriptionSpec;
use crate::session::{SessionAggregate, SessionEvent};
use crate::{Caller, ClientInputOutput, HandleClientInput, Runtime, RuntimeError};

/// The engine surface a channel gets: the client-input seam in, the event and
/// token-delta streams out, and session reads for catch-up rendering. Channels
/// hold this instead of the full runtime.
#[derive(Clone)]
pub struct ChannelContext {
    runtime: Arc<Runtime>,
    pub shutdown: CancellationToken,
}

impl ChannelContext {
    pub fn new(runtime: Arc<Runtime>, shutdown: CancellationToken) -> Self {
        Self { runtime, shutdown }
    }

    pub async fn handle_client_input(
        &self,
        input: HandleClientInput,
    ) -> Result<ClientInputOutput, RuntimeError> {
        self.runtime.handle_client_input(input).await
    }

    pub async fn stream(
        &self,
        spec: SessionSubscriptionSpec,
        after: Option<Seq>,
    ) -> Result<mpsc::Receiver<SessionEvent>, RuntimeError> {
        self.runtime.stream(spec, after).await
    }

    pub async fn subscribe_token_deltas(
        &self,
        caller: &Caller,
        root_session_id: &str,
    ) -> mpsc::Receiver<TokenDelta> {
        self.runtime
            .subscribe_token_deltas(caller, root_session_id)
            .await
    }

    pub async fn read_session_events(
        &self,
        caller: &Caller,
        session_id: &str,
        after: Option<Seq>,
        limit: Option<usize>,
    ) -> Result<Vec<SessionEvent>, RuntimeError> {
        self.runtime
            .read_session_events(caller, session_id, after, limit)
            .await
    }

    pub async fn get_session(
        &self,
        tenant_id: &str,
        session_id: &str,
    ) -> Result<SessionAggregate, RuntimeError> {
        self.runtime.get_session(tenant_id, session_id).await
    }

    /// A durable, checkpointed consumer of the event log — how a channel
    /// renders turns it must not lose across restarts.
    pub async fn spawn_processor(
        &self,
        processor: Arc<dyn EventProcessor>,
        config: EventProcessorRunnerConfig,
        start_at_tail: bool,
    ) -> Result<(), RuntimeError> {
        self.runtime
            .spawn_processor(processor, config, start_at_tail)
            .await
    }
}

/// A frontend for agent sessions. A channel maps its native inputs onto
/// [`ChannelContext::handle_client_input`] and renders the outbound event
/// stream into its own protocol. Request-driven channels contribute a router;
/// connection-driven channels (e.g. a socket-mode chat app) implement `run`.
#[async_trait]
pub trait Channel: Send + Sync {
    /// URL-safe name; the channel's routes are served under
    /// `/api/channels/{kind}`.
    fn kind(&self) -> &'static str;

    /// Routes relative to the channel's mount point.
    fn router(&self, ctx: ChannelContext) -> Option<Router> {
        let _ = ctx;
        None
    }

    async fn run(&self, ctx: ChannelContext) {
        let _ = ctx;
    }
}

/// Mount each channel's router under `/api/channels/{kind}` and spawn its run
/// loop.
pub fn start_channels(channels: Vec<Arc<dyn Channel>>, ctx: ChannelContext) -> Vec<Router> {
    let mut routers = Vec::new();
    for channel in channels {
        if let Some(router) = channel.router(ctx.clone()) {
            routers.push(Router::new().nest(&format!("/api/channels/{}", channel.kind()), router));
        }
        let ctx = ctx.clone();
        tokio::spawn(async move { channel.run(ctx).await });
    }
    routers
}
