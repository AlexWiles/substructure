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

use crate::protocol::{
    DecisionRequest, DecisionResponse, DecisionTrigger, StreamDelta, TokenDelta,
};
use crate::providers::format::DeltaParser;
use crate::runtime::llm::TokenDeltaTransport;
use crate::worker::push::{PushError, PushTransport, TransportConstructor};
use crate::worker::WorkerDecisionRequest;

type HmacSha256 = Hmac<Sha256>;

pub struct HttpPushTransport {
    http: Client,
    endpoint_url: String,
    /// Max silence, not total duration: request → first response, and between
    /// SSE events / body reads. A total timeout would kill long token streams;
    /// total duration is the decision deadline's job.
    idle_timeout: Duration,
    signing_secret: Option<String>,
}

impl HttpPushTransport {
    pub fn new(
        endpoint_url: String,
        idle_timeout: Option<Duration>,
        signing_secret: Option<String>,
    ) -> Self {
        Self {
            http: Client::builder()
                .connect_timeout(Duration::from_secs(10))
                .build()
                .unwrap_or_default(),
            endpoint_url,
            idle_timeout: idle_timeout.unwrap_or(Duration::from_secs(60)),
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
    ) -> Result<DecisionResponse, PushError> {
        let body = serde_json::to_vec(&DecisionRequest::from(decision)).map_err(|e| PushError {
            message: format!("failed to serialize decision: {e}"),
            retryable: false,
        })?;

        let mut builder = self
            .http
            .post(&self.endpoint_url)
            .header("Content-Type", "application/json")
            .header(ACCEPT, "text/event-stream, application/json")
            .header("traceparent", decision.span.traceparent());

        if let Some(ref secret) = self.signing_secret {
            let mut mac =
                HmacSha256::new_from_slice(secret.as_bytes()).expect("HMAC accepts any key length");
            mac.update(&body);
            let signature = hex::encode(mac.finalize().into_bytes());

            builder = builder.header("X-Substructure-Signature", format!("sha256={signature}"));
        }

        let started = std::time::Instant::now();
        tracing::info!(
            endpoint = %self.endpoint_url,
            decision_id = %decision.decision_id,
            agent_id = %decision.agent_id,
            trigger = decision.trigger.kind(),
            "dispatching decision to worker"
        );

        let resp = tokio::time::timeout(self.idle_timeout, builder.body(body).send())
            .await
            .map_err(|_| PushError {
                message: format!("no response within {:?}", self.idle_timeout),
                retryable: true,
            })?
            .map_err(|e| PushError {
                message: format!("HTTP request failed: {e}"),
                retryable: e.is_timeout() || e.is_connect(),
            })?;

        let status = resp.status();
        tracing::info!(
            decision_id = %decision.decision_id,
            status = status.as_u16(),
            elapsed_ms = started.elapsed().as_millis() as u64,
            "worker responded"
        );

        if !status.is_success() {
            let retryable = status.is_server_error();
            return Err(PushError {
                message: format!("endpoint returned {status}"),
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
            read_sse_response(
                resp.bytes_stream(),
                decision,
                token_delta_transport,
                Some(self.idle_timeout),
            )
            .await?
        } else {
            // `null` is the empty decision — the natural "nothing to add" reply
            // from JSON-language workers.
            tokio::time::timeout(self.idle_timeout, resp.json::<Option<DecisionResponse>>())
                .await
                .map_err(|_| PushError {
                    message: format!("no response body within {:?}", self.idle_timeout),
                    retryable: true,
                })?
                .map_err(|e| PushError {
                    message: format!("failed to parse response: {e}"),
                    retryable: false,
                })?
                .unwrap_or_default()
        };

        Ok(submit)
    }
}

#[derive(Deserialize)]
struct HttpTransportConfig {
    endpoint_url: String,
    /// Idle timeout: max silence before/between response reads.
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

/// Republishes streamed token deltas and returns the terminal decision.
/// `idle_timeout` bounds the silence between events, not the stream's length.
async fn read_sse_response<S, B, E>(
    stream: S,
    decision: &WorkerDecisionRequest,
    token_delta_transport: Arc<dyn TokenDeltaTransport>,
    idle_timeout: Option<Duration>,
) -> Result<DecisionResponse, PushError>
where
    S: futures_util::Stream<Item = Result<B, E>>,
    B: AsRef<[u8]>,
    E: std::error::Error,
{
    let mut events = Box::pin(stream.eventsource());
    let mut seq: u32 = 0;
    // A declared format ⇒ delta frames are raw provider events.
    let mut parser: Option<DeltaParser> = match &decision.trigger {
        DecisionTrigger::LlmExecute {
            format: Some(f), ..
        } => Some(f.delta_parser()),
        _ => None,
    };

    loop {
        let next = match idle_timeout {
            Some(t) => tokio::time::timeout(t, events.next())
                .await
                .map_err(|_| PushError {
                    message: format!("no stream event within {t:?}"),
                    retryable: true,
                })?,
            None => events.next().await,
        };
        let Some(event) = next else { break };
        let event = event.map_err(|e| PushError {
            message: format!("failed to read streaming worker response: {e}"),
            retryable: true,
        })?;

        match event.event.as_str() {
            "llm.token.delta" => match parser.as_mut() {
                Some(p) => {
                    for delta in p.parse_data(&event.data) {
                        publish_worker_delta(decision, delta, &token_delta_transport, &mut seq)
                            .await?;
                    }
                }
                None => {
                    let delta: StreamDelta =
                        serde_json::from_str(&event.data).map_err(|e| PushError {
                            message: format!("failed to parse llm.token.delta frame: {e}"),
                            retryable: false,
                        })?;
                    publish_worker_delta(decision, delta, &token_delta_transport, &mut seq).await?;
                }
            },
            "decision.result" => {
                return serde_json::from_str::<Option<DecisionResponse>>(&event.data)
                    .map(Option::unwrap_or_default)
                    .map_err(|e| PushError {
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
    let DecisionTrigger::LlmExecute {
        id: call_id,
        attempt,
        stream,
        ..
    } = &decision.trigger
    else {
        return Err(PushError {
            message: "llm.token.delta is only valid for llm.execute decisions".to_string(),
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
            tenant_id: decision.tenant_id().to_string(),
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

    use crate::protocol::{LlmRequest, SessionOwner};
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
            decision_id: "dec-1".to_string(),
            agent_id: "agent-1".to_string(),
            identity: SessionOwner {
                tenant_id: "tenant-a".to_string(),
                id: Some("user-1".to_string()),
                metadata: Default::default(),
            },
            trigger: DecisionTrigger::LlmExecute {
                id: "llm-1".to_string(),
                request: serde_json::to_value(LlmRequest {
                    model: "test-model".to_string(),
                    messages: vec![],
                    tools: None,
                    temperature: None,
                    max_completion_tokens: None,
                    reasoning: None,
                })
                .unwrap(),
                format: None,
                stream: true,
                attempt: 0,
                deadline: None,
            },
            proposed: DecisionResponse::default(),
            state: Default::default(),
            agent: None,
            calls: Default::default(),
            pending_calls: 0,
            transcript: vec![],
            message_tree: Default::default(),
            ancestry: vec![],
            span: SpanContext::root(),
            attempts: 0,
            deadline: None,
            turn_id: None,
        }
    }

    /// Split `body` into tiny chunks so multibyte chars and SSE frames straddle boundaries.
    fn chunked(body: &str) -> impl futures_util::Stream<Item = Result<Vec<u8>, std::io::Error>> {
        let chunks: Vec<_> = body.as_bytes().chunks(3).map(|c| Ok(c.to_vec())).collect();
        futures_util::stream::iter(chunks)
    }

    #[tokio::test]
    async fn parses_deltas_then_terminal_result() {
        let transport = Arc::new(RecordingTransport::default());
        let decision = streaming_decision();
        // Multibyte token text; chunked() splits it mid-codepoint.
        let body = "event: llm.token.delta\ndata: {\"text\":\"héllo 🎉\"}\n\n\
                    event: llm.token.delta\ndata: {\"reasoning\":\"hmm\"}\n\n\
                    event: decision.result\ndata: {\"actions\":[],\"state\":\"\"}\n\n";

        let submit = read_sse_response(chunked(body), &decision, transport.clone(), None)
            .await
            .expect("should parse the stream");

        // The terminal frame carries only the worker-authored decision — no ids.
        assert!(submit.actions.is_empty());
        assert!(submit.state.is_some());

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
    async fn a_null_decision_result_is_the_empty_decision() {
        let transport = Arc::new(RecordingTransport::default());
        let decision = streaming_decision();
        let body = "event: decision.result\ndata: null\n\n";

        let submit = read_sse_response(chunked(body), &decision, transport.clone(), None)
            .await
            .expect("null parses as the empty decision");

        assert!(submit.actions.is_empty());
        assert!(submit.messages.is_empty());
        assert!(submit.state.is_none());
        assert!(submit.agent.is_none());
    }

    #[tokio::test]
    async fn a_format_execute_streams_raw_provider_events() {
        let transport = Arc::new(RecordingTransport::default());
        let mut decision = streaming_decision();
        if let DecisionTrigger::LlmExecute { format, .. } = &mut decision.trigger {
            *format = Some(crate::protocol::LlmFormat::Anthropic);
        }
        let body = "event: llm.token.delta\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"m\"}}\n\n\
                    event: llm.token.delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n\
                    event: llm.token.delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n\
                    event: decision.result\ndata: {\"actions\":[]}\n\n";

        let submit = read_sse_response(chunked(body), &decision, transport.clone(), None)
            .await
            .expect("should parse the stream");
        assert!(submit.actions.is_empty());

        let deltas = transport.deltas.lock().unwrap();
        assert_eq!(deltas.len(), 2, "message_start streams nothing");
        assert_eq!(deltas[0].text.as_deref(), Some("hi"));
        assert_eq!(deltas[0].seq, 0);
        assert_eq!(deltas[1].finish_reason.as_deref(), Some("stop"));
        assert_eq!(deltas[1].seq, 1);
    }

    #[tokio::test]
    async fn unknown_event_is_a_hard_error() {
        let transport = Arc::new(RecordingTransport::default());
        let decision = streaming_decision();
        let body = "event: surprise\ndata: {}\n\n";
        let err = read_sse_response(chunked(body), &decision, transport, None)
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
        let err = read_sse_response(chunked(body), &decision, transport, None)
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
        let err = read_sse_response(chunked(body), &decision, transport, None)
            .await
            .expect_err("a decision.error frame should fail the read");
        assert!(err.retryable);
    }

    #[tokio::test]
    async fn missing_terminal_result_is_retryable() {
        let transport = Arc::new(RecordingTransport::default());
        let decision = streaming_decision();
        let body = "event: llm.token.delta\ndata: {\"text\":\"hi\"}\n\n";
        let err = read_sse_response(chunked(body), &decision, transport, None)
            .await
            .expect_err("stream without decision.result should error");
        assert!(err.retryable, "an interrupted stream is retryable");
    }
}
