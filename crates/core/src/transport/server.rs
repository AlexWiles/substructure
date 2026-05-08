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
        let app = self
            .router
            .layer(tower_http::cors::CorsLayer::permissive())
            .layer(tower_http::trace::TraceLayer::new_for_http());
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown.cancelled_owned())
            .await?;
        Ok(())
    }
}
