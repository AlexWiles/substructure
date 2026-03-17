mod routes;
mod types;

use std::sync::Arc;

use axum::routing::post;
use axum::Router;

use substructure_core::Runtime;

pub fn router(runtime: Arc<Runtime>) -> Router {
    Router::new()
        .route("/sessions/send", post(routes::send_message))
        .with_state(runtime)
}
