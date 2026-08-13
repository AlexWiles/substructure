mod routes;
mod types;

use std::sync::Arc;

use axum::middleware;
use axum::routing::{get, post};
use axum::Router;

use tokio_util::sync::CancellationToken;

use crate::runtime::blob::BlobStore;
use crate::transport::auth::AuthResolver;
use crate::transport::http::{client_auth_middleware, client_cors};
use crate::Runtime;

#[derive(Clone)]
pub struct ClientHttpState {
    pub runtime: Arc<Runtime>,
    pub auth: Arc<dyn AuthResolver>,
    pub shutdown: CancellationToken,
    pub blobs: Option<Arc<dyn BlobStore>>,
}

pub fn router(state: ClientHttpState) -> Router {
    Router::new()
        .route("/api/client/sessions/input", post(routes::client_input))
        .route("/api/client/blobs", get(routes::get_blob))
        .route(
            "/api/client/sessions/{session_id}",
            get(routes::get_session),
        )
        .route(
            "/api/client/sessions/{session_id}/events/stream",
            get(routes::stream_session_events),
        )
        .route(
            "/api/client/sessions/{session_id}/interrupt",
            post(routes::interrupt_session),
        )
        .route_layer(middleware::from_fn_with_state(
            state.auth.clone(),
            client_auth_middleware,
        ))
        // Outermost so CORS preflight (OPTIONS) is answered before auth runs.
        .layer(client_cors())
        .with_state(state)
}
