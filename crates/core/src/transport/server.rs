use axum::extract::DefaultBodyLimit;
use axum::Router;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

pub struct SubstructureServer {
    router: Router,
    max_body: u64,
}

impl SubstructureServer {
    pub fn new(routers: Vec<Router>, max_body: u64) -> Self {
        let mut router = Router::new();
        for r in routers {
            router = router.merge(r);
        }
        Self { router, max_body }
    }

    pub async fn serve(
        self,
        listener: TcpListener,
        shutdown: CancellationToken,
    ) -> anyhow::Result<()> {
        let app = self
            .router
            .layer(DefaultBodyLimit::max(
                usize::try_from(self.max_body).unwrap_or(usize::MAX),
            ))
            .layer(
                tower_http::trace::TraceLayer::new_for_http()
                    .make_span_with(
                        tower_http::trace::DefaultMakeSpan::new().level(tracing::Level::INFO),
                    )
                    .on_response(
                        tower_http::trace::DefaultOnResponse::new().level(tracing::Level::INFO),
                    ),
            );
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown.cancelled_owned())
            .await?;
        Ok(())
    }
}
