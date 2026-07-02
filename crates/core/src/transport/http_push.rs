use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use hmac::{Hmac, Mac};
use reqwest::header::{ACCEPT, CONTENT_TYPE};
use reqwest::Client;
use serde::Deserialize;
use sha2::Sha256;

use crate::runtime::llm::{StreamDelta, TokenDelta, TokenDeltaTransport};
use crate::runtime::session::decision::{DecisionTrigger, EffectWork};
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
    async fn push(
        &self,
        decision: &WorkerDecisionRequest,
        token_delta_transport: Arc<dyn TokenDeltaTransport>,
    ) -> Result<PushResponse, PushError> {
        let body = serde_json::to_vec(decision).map_err(|e| PushError {
            message: format!("failed to serialize decision: {e}"),
            retryable: false,
        })?;

        let mut builder = self
            .http
            .post(&self.endpoint_url)
            .header("Content-Type", "application/json")
            .header(ACCEPT, "text/event-stream, application/json")
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

        let content_type = resp
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_ascii_lowercase();

        let submit = if content_type.starts_with("text/event-stream") {
            read_sse_response(resp.bytes_stream(), decision, token_delta_transport).await?
        } else {
            resp.json().await.map_err(|e| PushError {
                message: format!("failed to parse response: {e}"),
                retryable: false,
            })?
        };

        if submit.session_id != decision.session_id || submit.decision_id != decision.decision_id {
            return Err(PushError {
                message: "worker response did not match requested session/decision".to_string(),
                retryable: false,
            });
        }

        Ok(PushResponse {
            transcript: submit.transcript,
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

#[derive(Deserialize)]
struct DecisionError {
    message: String,
    #[serde(default = "retryable_default")]
    retryable: bool,
}

fn retryable_default() -> bool {
    true
}

/// Reads a streaming worker response: interim `llm.token.delta` frames are
/// republished onto the token-delta transport; the terminal `decision.result`
/// frame carries the `SubmitRequest`. The actual SSE framing (UTF-8 across
/// chunk boundaries, multi-line data, comments) is handled by `eventsource_stream`.
async fn read_sse_response<S, B, E>(
    stream: S,
    decision: &WorkerDecisionRequest,
    token_delta_transport: Arc<dyn TokenDeltaTransport>,
) -> Result<SubmitRequest, PushError>
where
    S: futures_util::Stream<Item = Result<B, E>>,
    B: AsRef<[u8]>,
    E: std::error::Error,
{
    let mut events = Box::pin(stream.eventsource());
    let mut seq: u32 = 0;

    while let Some(event) = events.next().await {
        let event = event.map_err(|e| PushError {
            message: format!("failed to read streaming worker response: {e}"),
            retryable: true,
        })?;

        match event.event.as_str() {
            "llm.token.delta" => {
                let delta: StreamDelta =
                    serde_json::from_str(&event.data).map_err(|e| PushError {
                        message: format!("failed to parse llm.token.delta frame: {e}"),
                        retryable: false,
                    })?;
                publish_worker_delta(decision, delta, &token_delta_transport, &mut seq).await?;
            }
            "decision.result" => {
                return serde_json::from_str(&event.data).map_err(|e| PushError {
                    message: format!("failed to parse decision.result frame: {e}"),
                    retryable: false,
                });
            }
            "decision.error" => {
                let err: DecisionError =
                    serde_json::from_str(&event.data).map_err(|e| PushError {
                        message: format!("failed to parse decision.error frame: {e}"),
                        retryable: false,
                    })?;
                return Err(PushError {
                    message: err.message,
                    retryable: err.retryable,
                });
            }
            other => {
                return Err(PushError {
                    message: format!("unknown worker stream event: {other}"),
                    retryable: false,
                });
            }
        }
    }

    Err(PushError {
        message: "streaming worker response ended before decision.result".to_string(),
        retryable: true,
    })
}

async fn publish_worker_delta(
    decision: &WorkerDecisionRequest,
    delta: StreamDelta,
    token_delta_transport: &Arc<dyn TokenDeltaTransport>,
    seq: &mut u32,
) -> Result<(), PushError> {
    let DecisionTrigger::EffectExecute {
        id: call_id,
        attempt,
        work: EffectWork::LlmCall { stream, .. },
        ..
    } = &decision.trigger
    else {
        return Err(PushError {
            message: "llm.token.delta is only valid for llm effect.execute decisions".to_string(),
            retryable: false,
        });
    };

    if !stream {
        return Err(PushError {
            message: "llm.token.delta received for a non-streaming llm call".to_string(),
            retryable: false,
        });
    }

    token_delta_transport
        .publish(TokenDelta {
            tenant_id: decision.tenant_id.clone(),
            root_session_id: decision
                .ancestry
                .first()
                .cloned()
                .unwrap_or_else(|| decision.session_id.clone()),
            session_id: decision.session_id.clone(),
            agent_id: decision.agent_id.clone(),
            turn_id: decision.turn_id.clone(),
            call_id: call_id.clone(),
            attempt: *attempt,
            seq: *seq,
            text: delta.text,
            reasoning: delta.reasoning,
            tool_calls: delta.tool_calls,
            finish_reason: delta.finish_reason,
        })
        .await;
    *seq = seq.saturating_add(1);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use tokio::sync::mpsc;

    use crate::runtime::llm::{LlmRequest, TokenDelta};
    use crate::runtime::owner::SessionOwner;
    use crate::runtime::span::SpanContext;

    #[derive(Default)]
    struct RecordingTransport {
        deltas: Mutex<Vec<TokenDelta>>,
    }

    #[async_trait]
    impl TokenDeltaTransport for RecordingTransport {
        async fn publish(&self, delta: TokenDelta) {
            self.deltas.lock().unwrap().push(delta);
        }
        async fn subscribe(&self, _tenant_id: &str, _root: &str) -> mpsc::Receiver<TokenDelta> {
            unimplemented!("not needed for these tests")
        }
    }

    fn streaming_decision() -> WorkerDecisionRequest {
        WorkerDecisionRequest {
            session_id: "sess-1".to_string(),
            tenant_id: "tenant-a".to_string(),
            decision_id: "dec-1".to_string(),
            agent_id: "agent-1".to_string(),
            identity: SessionOwner {
                tenant_id: "tenant-a".to_string(),
                id: Some("user-1".to_string()),
                metadata: Default::default(),
            },
            trigger: DecisionTrigger::EffectExecute {
                id: "llm-1".to_string(),
                attempt: 0,
                deadline: None,
                work: EffectWork::LlmCall {
                    request: LlmRequest {
                        model: "test-model".to_string(),
                        messages: vec![],
                        tools: None,
                        temperature: None,
                        max_completion_tokens: None,
                        reasoning: None,
                    },
                    stream: true,
                },
            },
            worker_state: vec![].into(),
            effects: Default::default(),
            transcript: vec![],
            message_tree: Default::default(),
            ancestry: vec![],
            span: SpanContext::root(),
            attempts: 0,
            deadline: None,
            turn_id: None,
        }
    }

    /// Feed `body` as many tiny chunks so multibyte chars and SSE frames are
    /// split across chunk boundaries — the case the old hand-rolled parser got
    /// wrong.
    fn chunked(body: &str) -> impl futures_util::Stream<Item = Result<Vec<u8>, std::io::Error>> {
        let chunks: Vec<_> = body.as_bytes().chunks(3).map(|c| Ok(c.to_vec())).collect();
        futures_util::stream::iter(chunks)
    }

    #[tokio::test]
    async fn parses_deltas_then_terminal_result() {
        let transport = Arc::new(RecordingTransport::default());
        let decision = streaming_decision();
        // Note the multibyte token text; chunked() splits it mid-codepoint.
        let body = "event: llm.token.delta\ndata: {\"text\":\"héllo 🎉\"}\n\n\
                    event: llm.token.delta\ndata: {\"reasoning\":\"hmm\"}\n\n\
                    event: decision.result\ndata: {\"session_id\":\"sess-1\",\"decision_id\":\"dec-1\",\"actions\":[],\"state\":\"\"}\n\n";

        let submit = read_sse_response(chunked(body), &decision, transport.clone())
            .await
            .expect("should parse the stream");

        assert_eq!(submit.session_id, "sess-1");
        assert_eq!(submit.decision_id, "dec-1");

        let deltas = transport.deltas.lock().unwrap();
        assert_eq!(deltas.len(), 2);
        assert_eq!(deltas[0].text.as_deref(), Some("héllo 🎉"));
        assert_eq!(deltas[0].seq, 0);
        assert_eq!(deltas[0].call_id, "llm-1");
        assert_eq!(deltas[0].root_session_id, "sess-1");
        assert_eq!(deltas[1].reasoning.as_deref(), Some("hmm"));
        assert_eq!(deltas[1].seq, 1);
    }

    #[tokio::test]
    async fn unknown_event_is_a_hard_error() {
        let transport = Arc::new(RecordingTransport::default());
        let decision = streaming_decision();
        let body = "event: surprise\ndata: {}\n\n";
        let err = read_sse_response(chunked(body), &decision, transport)
            .await
            .expect_err("unknown event should error");
        assert!(!err.retryable, "unknown event is not retryable");
    }

    #[tokio::test]
    async fn decision_error_frame_surfaces_its_message_and_retryability() {
        let transport = Arc::new(RecordingTransport::default());
        let decision = streaming_decision();
        let body =
            "event: decision.error\ndata: {\"message\":\"handler threw\",\"retryable\":false}\n\n";
        let err = read_sse_response(chunked(body), &decision, transport)
            .await
            .expect_err("a decision.error frame should fail the read");
        assert_eq!(err.message, "handler threw");
        assert!(!err.retryable);
    }

    #[tokio::test]
    async fn decision_error_defaults_to_retryable() {
        let transport = Arc::new(RecordingTransport::default());
        let decision = streaming_decision();
        let body = "event: decision.error\ndata: {\"message\":\"boom\"}\n\n";
        let err = read_sse_response(chunked(body), &decision, transport)
            .await
            .expect_err("a decision.error frame should fail the read");
        assert!(err.retryable);
    }

    #[tokio::test]
    async fn missing_terminal_result_is_retryable() {
        let transport = Arc::new(RecordingTransport::default());
        let decision = streaming_decision();
        let body = "event: llm.token.delta\ndata: {\"text\":\"hi\"}\n\n";
        let err = read_sse_response(chunked(body), &decision, transport)
            .await
            .expect_err("stream without decision.result should error");
        assert!(err.retryable, "an interrupted stream is retryable");
    }
}
