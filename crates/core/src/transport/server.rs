use axum::Router;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

pub struct SubstructureServer {
    router: Router,
}

impl SubstructureServer {
    pub fn new(routers: Vec<Router>) -> Self {
        let mut router = Router::new();
        for r in routers {
            router = router.merge(r);
        }
        Self { router }
    }

    pub async fn serve(
        self,
        listener: TcpListener,
        shutdown: CancellationToken,
    ) -> anyhow::Result<()> {
        // Like `CorsLayer::permissive()`, but `allow_headers` mirrors the
        // request instead of sending `*`. Per the Fetch spec, a `*` in
        // `Access-Control-Allow-Headers` does NOT cover `Authorization`, so
        // browsers strip the bearer token on cross-origin requests (and the
        // streamed response read fails mid-flight). Mirroring lists the
        // requested headers explicitly, including `Authorization`.
        use tower_http::cors::{AllowHeaders, Any, CorsLayer};
        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(AllowHeaders::mirror_request())
            .expose_headers(Any);
        let app = self
            .router
            .layer(cors)
            .layer(tower_http::trace::TraceLayer::new_for_http());
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown.cancelled_owned())
            .await?;
        Ok(())
    }
}
