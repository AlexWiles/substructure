use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;

use crate::transport::worker_http::types::SubmitRequest;
use substructure_core::worker::push::{
    PushError, PushResponse, PushTransport, TransportConstructor,
};
use substructure_core::worker::WorkerDecisionRequest;

pub struct HttpPushTransport {
    http: Client,
    endpoint_url: String,
    timeout: Duration,
}

impl HttpPushTransport {
    pub fn new(endpoint_url: String, timeout: Option<Duration>) -> Self {
        Self {
            http: Client::new(),
            endpoint_url,
            timeout: timeout.unwrap_or(Duration::from_secs(30)),
        }
    }
}

#[async_trait]
impl PushTransport for HttpPushTransport {
    async fn push(&self, decision: &WorkerDecisionRequest) -> Result<PushResponse, PushError> {
        let resp = self
            .http
            .post(&self.endpoint_url)
            .json(decision)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| PushError {
                message: format!("HTTP request failed: {e}"),
                retryable: e.is_timeout() || e.is_connect(),
            })?;

        if !resp.status().is_success() {
            let retryable = resp.status().is_server_error();
            return Err(PushError {
                message: format!("endpoint returned {}", resp.status()),
                retryable,
            });
        }

        let submit: SubmitRequest = resp.json().await.map_err(|e| PushError {
            message: format!("failed to parse response: {e}"),
            retryable: false,
        })?;

        Ok(PushResponse {
            actions: submit.actions,
            state: submit.state,
        })
    }
}

#[derive(Deserialize)]
struct HttpTransportConfig {
    endpoint_url: String,
    #[serde(default)]
    timeout_secs: Option<u64>,
}

pub fn http_transport() -> (&'static str, TransportConstructor) {
    (
        "http",
        Box::new(|config| {
            let c: HttpTransportConfig =
                serde_json::from_value(config).map_err(|e| e.to_string())?;
            let timeout = c.timeout_secs.map(Duration::from_secs);
            Ok(Arc::new(HttpPushTransport::new(c.endpoint_url, timeout)))
        }),
    )
}
