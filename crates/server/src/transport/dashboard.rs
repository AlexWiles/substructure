use axum::{
    extract::Path,
    http::{header, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::get,
    Router,
};
use rust_embed::Embed;

#[derive(Embed)]
#[folder = "../../dashboard/dist/client"]
struct DashboardAssets;

pub fn router() -> Router {
    Router::new()
        .route("/", get(index_handler))
        .route("/{*path}", get(static_handler))
}

async fn index_handler() -> impl IntoResponse {
    serve_file("index.html")
}

async fn static_handler(Path(path): Path<String>) -> impl IntoResponse {
    // Try the exact path first, then fall back to index.html for SPA routing
    if DashboardAssets::get(&path).is_some() {
        serve_file(&path)
    } else {
        serve_file("index.html")
    }
}

fn serve_file(path: &str) -> Response {
    match DashboardAssets::get(path) {
        Some(content) => {
            let mime = mime_guess::from_path(path)
                .first_or_octet_stream()
                .to_string();
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, mime)],
                content.data.to_vec(),
            )
                .into_response()
        }
        None => (StatusCode::NOT_FOUND, "Not found").into_response(),
    }
}
