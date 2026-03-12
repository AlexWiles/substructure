use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;

use crate::runtime::worker::push::{PushError, PushResponse, PushTransport, TransportConstructor};
use crate::runtime::worker::PendingDecision;
use crate::transport::worker_http::types::{PollResponse, SubmitRequest};

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
    async fn push(&self, decision: &PendingDecision) -> Result<PushResponse, PushError> {
        let body = PollResponse {
            session_id: decision.session_id,
            tenant_id: decision.tenant_id.clone(),
            decision_id: decision.decision_id.clone(),
            agent_id: decision.agent_id.clone(),
            trigger: decision.trigger.clone(),
            worker_state: decision.worker_state.clone(),
            span: decision.span.clone(),
            attempts: decision.attempts,
            deadline: decision.deadline,
        };

        let resp = self
            .http
            .post(&self.endpoint_url)
            .json(&body)
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
    ("http", Box::new(|config| {
        let c: HttpTransportConfig =
            serde_json::from_value(config).map_err(|e| e.to_string())?;
        let timeout = c.timeout_secs.map(Duration::from_secs);
        Ok(Arc::new(HttpPushTransport::new(c.endpoint_url, timeout)))
    }))
}
