use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use hmac::{Hmac, Mac};
use reqwest::Client;
use serde::Deserialize;
use sha2::Sha256;

use crate::transport::worker_http::types::SubmitRequest;
use crate::worker::push::{PushError, PushResponse, PushTransport, TransportConstructor};
use crate::worker::WorkerDecisionRequest;

type HmacSha256 = Hmac<Sha256>;

pub struct HttpPushTransport {
    http: Client,
    endpoint_url: String,
    timeout: Duration,
    signing_secret: Option<String>,
}

impl HttpPushTransport {
    pub fn new(
        endpoint_url: String,
        timeout: Option<Duration>,
        signing_secret: Option<String>,
    ) -> Self {
        Self {
            http: Client::new(),
            endpoint_url,
            timeout: timeout.unwrap_or(Duration::from_secs(30)),
            signing_secret,
        }
    }
}

#[async_trait]
impl PushTransport for HttpPushTransport {
    async fn push(&self, decision: &WorkerDecisionRequest) -> Result<PushResponse, PushError> {
        let body = serde_json::to_vec(decision).map_err(|e| PushError {
            message: format!("failed to serialize decision: {e}"),
            retryable: false,
        })?;

        let mut builder = self
            .http
            .post(&self.endpoint_url)
            .header("Content-Type", "application/json")
            .timeout(self.timeout);

        if let Some(ref secret) = self.signing_secret {
            let timestamp = chrono::Utc::now().timestamp();
            let body_str = String::from_utf8_lossy(&body);
            let signing_payload = format!("{timestamp}.{body_str}");

            let mut mac =
                HmacSha256::new_from_slice(secret.as_bytes()).expect("HMAC accepts any key length");
            mac.update(signing_payload.as_bytes());
            let signature = hex::encode(mac.finalize().into_bytes());

            builder = builder
                .header("X-Substructure-Timestamp", timestamp.to_string())
                .header("X-Substructure-Signature", format!("v1={signature}"));
        }

        let resp = builder.body(body).send().await.map_err(|e| PushError {
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
    #[serde(default)]
    signing_secret: Option<String>,
}

pub fn http_transport() -> (&'static str, TransportConstructor) {
    (
        "http",
        Box::new(|config| {
            let c: HttpTransportConfig =
                serde_json::from_value(config).map_err(|e| e.to_string())?;
            let timeout = c.timeout_secs.map(Duration::from_secs);
            Ok(Arc::new(HttpPushTransport::new(
                c.endpoint_url,
                timeout,
                c.signing_secret,
            )))
        }),
    )
}
