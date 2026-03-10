mod routes;
mod types;

use std::sync::Arc;

use axum::routing::post;
use axum::Router;

use crate::runtime::Runtime;

pub fn router(runtime: Arc<Runtime>) -> Router {
    Router::new()
        .route("/workers/poll", post(routes::poll))
        .route("/workers/submit", post(routes::submit))
        .with_state(runtime)
}
