mod routes;

use std::sync::Arc;

use axum::routing::get;
use axum::Router;

use substructure_core::Runtime;

pub fn router(runtime: Arc<Runtime>) -> Router {
    Router::new()
        .route("/admin/sessions", get(routes::list_sessions))
        .route("/admin/sessions/{session_id}", get(routes::get_session))
        .route(
            "/admin/sessions/{session_id}/events",
            get(routes::get_session_events),
        )
        .route(
            "/admin/sessions/{session_id}/events/stream",
            get(routes::stream_session_events),
        )
        .with_state(runtime)
}
