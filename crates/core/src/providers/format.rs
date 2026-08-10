//! The worker-format seam: translate between the engine's neutral LLM types
//! and a provider's native wire format, for agents that declare `format`.
//! Reuses the server-side adapters' translation code.

use crate::protocol::{DeferToolsStrategy, LlmFormat, LlmRequest, LlmResponse, StreamDelta};

use super::{anthropic, openai};

impl LlmFormat {
    /// The provider-native request body for an `llm.execute` trigger.
    pub fn request_to_wire(
        &self,
        request: &LlmRequest,
        search: DeferToolsStrategy,
    ) -> serde_json::Value {
        match self {
            LlmFormat::Openai => openai::request_to_wire(request, search),
            LlmFormat::Anthropic => anthropic::request_to_wire(request, search),
        }
    }

    /// A provider-native `llm.result` response → the neutral `LlmResponse`.
    pub fn response_from_wire(&self, value: serde_json::Value) -> Result<LlmResponse, String> {
        match self {
            LlmFormat::Openai => openai::response_from_wire(value),
            LlmFormat::Anthropic => anthropic::response_from_wire(value),
        }
    }

    /// A stateful parser for provider-native `llm.token.delta` payloads
    /// (one per streamed call — provider events are contextual).
    pub fn delta_parser(&self) -> DeltaParser {
        DeltaParser(match self {
            LlmFormat::Openai => Inner::Openai(openai::StreamParser::new()),
            LlmFormat::Anthropic => Inner::Anthropic(anthropic::StreamParser::new()),
        })
    }
}

pub struct DeltaParser(Inner);

enum Inner {
    Openai(openai::StreamParser),
    Anthropic(anthropic::StreamParser),
}

impl DeltaParser {
    /// Fold one raw event payload into neutral deltas; payloads that carry
    /// nothing streamable yield an empty vec.
    pub fn parse_data(&mut self, data: &str) -> Vec<StreamDelta> {
        match &mut self.0 {
            Inner::Openai(p) => p.parse_data(data),
            Inner::Anthropic(p) => p.parse_data(data),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{Content, DraftMessage, Role};
    use serde_json::json;

    fn request() -> LlmRequest {
        LlmRequest {
            model: "m".to_string(),
            messages: vec![
                DraftMessage {
                    id: None,
                    role: Role::System,
                    content: Some(Content::Text("be nice".to_string())),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
                DraftMessage {
                    id: None,
                    role: Role::User,
                    content: Some(Content::Text("hi".to_string())),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
            ],
            tools: None,
            temperature: None,
            max_completion_tokens: Some(64),
            reasoning: None,
        }
    }

    #[test]
    fn anthropic_round_trip_omits_stream_and_maps_both_directions() {
        let body = LlmFormat::Anthropic.request_to_wire(&request(), DeferToolsStrategy::Full);
        assert_eq!(body["system"][0]["text"], "be nice");
        assert_eq!(body["max_tokens"], 64);
        assert!(body.get("stream").is_none(), "stream is the trigger's flag");

        let response = LlmFormat::Anthropic
            .response_from_wire(json!({
                "model": "claude-haiku-4-5",
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 2}
            }))
            .expect("parses");
        assert_eq!(response.content.as_deref(), Some("hello"));
        assert_eq!(response.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn openai_round_trip_omits_stream_and_maps_both_directions() {
        let body = LlmFormat::Openai.request_to_wire(&request(), DeferToolsStrategy::Full);
        assert_eq!(body["model"], "m");
        assert_eq!(body["messages"][0]["role"], "system");
        assert!(body.get("stream").is_none());

        let response = LlmFormat::Openai
            .response_from_wire(json!({
                "model": "gpt-4o",
                "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]
            }))
            .expect("parses");
        assert_eq!(response.content.as_deref(), Some("hello"));
        assert_eq!(response.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn a_bad_response_reports_instead_of_panicking() {
        assert!(LlmFormat::Anthropic
            .response_from_wire(json!({"content": "not blocks"}))
            .is_err());
        assert!(LlmFormat::Openai.response_from_wire(json!(42)).is_err());
    }

    #[test]
    fn anthropic_delta_parser_assembles_tool_call_chunks() {
        let mut parser = LlmFormat::Anthropic.delta_parser();
        assert!(parser
            .parse_data(r#"{"type":"message_start","message":{"model":"claude-haiku-4-5"}}"#)
            .is_empty());

        let text = parser.parse_data(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#,
        );
        assert_eq!(text[0].text.as_deref(), Some("hi"));

        parser.parse_data(
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tu_1","name":"get_weather","input":{}}}"#,
        );
        let args = parser.parse_data(
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"NYC\"}"}}"#,
        );
        assert_eq!(args[0].tool_calls[0].id, "tu_1");
        assert_eq!(args[0].tool_calls[0].name.as_deref(), Some("get_weather"));

        let finish = parser
            .parse_data(r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":7}}"#);
        assert_eq!(finish[0].finish_reason.as_deref(), Some("tool_calls"));

        assert!(parser.parse_data("not json").is_empty());
        assert!(parser.parse_data(r#"{"type":"ping"}"#).is_empty());
    }

    #[test]
    fn openai_delta_parser_passes_chunks_through() {
        let mut parser = LlmFormat::Openai.delta_parser();
        let text = parser.parse_data(
            r#"{"model":"gpt-4o","choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#,
        );
        assert_eq!(text[0].text.as_deref(), Some("hi"));

        let finish = parser.parse_data(r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#);
        assert_eq!(finish[0].finish_reason.as_deref(), Some("stop"));

        assert!(parser.parse_data("[DONE]").is_empty());
    }
}
