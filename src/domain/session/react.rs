use crate::domain::event::*;
use super::strategy::LlmResponseSummary;

pub fn extract_assistant_message(
    response: &LlmResponse,
) -> (Option<String>, Vec<ToolCall>, Option<u32>) {
    match response {
        LlmResponse::OpenAi(resp) => {
            let choice = &resp.choices[0];
            let content = choice.message.content.clone();
            let tool_calls = choice
                .message
                .tool_calls
                .as_ref()
                .map(|tcs| {
                    tcs.iter()
                        .map(|tc| ToolCall {
                            id: tc.id.clone(),
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.clone(),
                        })
                        .collect()
                })
                .unwrap_or_default();
            let token_count = resp.usage.as_ref().map(|u| u.total_tokens);
            (content, tool_calls, token_count)
        }
    }
}

/// Build an `LlmResponseSummary` from an `LlmCallCompleted` payload.
pub fn extract_response_summary(payload: &LlmCallCompleted) -> LlmResponseSummary {
    let (content, tool_calls, token_count) = extract_assistant_message(&payload.response);
    LlmResponseSummary {
        call_id: payload.call_id.clone(),
        content,
        tool_calls,
        token_count,
    }
}
