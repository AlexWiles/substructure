mod routes;
pub mod types;

use std::sync::Arc;

use axum::routing::post;
use axum::Router;

use crate::push::PushAdapter;

pub fn router(adapter: Arc<PushAdapter>) -> Router {
    Router::new()
        .route("/workers/submit", post(routes::submit))
        .route("/workers/register", post(routes::register))
        .with_state(adapter)
}
