use axum::Router;
use tokio::net::TcpListener;

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

    pub fn router(&self) -> Router {
        self.router.clone()
    }

    pub async fn serve(self, listener: TcpListener) -> anyhow::Result<()> {
        let app = self
            .router
            .layer(tower_http::cors::CorsLayer::permissive())
            .layer(tower_http::trace::TraceLayer::new_for_http());
        axum::serve(listener, app).await?;
        Ok(())
    }
}
