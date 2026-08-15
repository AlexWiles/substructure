//! Approval: a call a connection gates stops before it runs. One question per
//! call; the answer runs it or records a refusal.

use serde_json::Value;

use super::InterruptKind;
use crate::protocol::{
    ConnectorTool, ConnectorToolKind, Content, DecisionAction, DecisionResponse, DraftMessage,
    Role, ToolCall,
};
use crate::runtime::session::propose::{
    asked_about, recorded, reissue, resumed, route_tool_call, settled, siblings_answered, Proposing,
};
use crate::runtime::session::state::inner_arguments;

pub const PREFIX: &str = "mcp-approve:";

pub struct Approval;

impl InterruptKind for Approval {
    fn prefix(&self) -> &'static str {
        PREFIX
    }

    /// Answer the call this approval held, then ask about the next.
    fn resumed(&self, tail: &str, payload: &Value, p: &Proposing<'_>) -> Option<DecisionResponse> {
        let &Proposing {
            transcript,
            llm_calls,
            pending_calls,
            dispatched,
            config,
            connector_tools,
            decision_id,
        } = p;
        let Some((at, call)) = asked_about(tail, transcript, dispatched) else {
            return Some(match config {
                Some(c) => resumed(transcript, c),
                None => DecisionResponse {
                    messages: recorded(transcript),
                    ..Default::default()
                },
            });
        };

        // Nothing ran, so a decline has no effect to fail. The refusal is the result.
        let approved = approved(payload);
        let refusals = match approved {
            true => Vec::new(),
            false => vec![refusal(call)],
        };
        let mut actions = match approved {
            true => route_tool_call(call, config, decision_id),
            false => Vec::new(),
        };

        let waiting: Vec<&ToolCall> = transcript[at]
            .tool_calls
            .iter()
            .filter(|c| c.id != call.id && !settled(&c.id, at, transcript, dispatched))
            .collect();
        let mut gated = waiting.iter().filter(|c| needed(c, connector_tools));
        match gated.next() {
            Some(next) => actions.push(ask(next, connector_tools, gated.count())),
            None => actions.extend(
                waiting
                    .iter()
                    .flat_map(|c| route_tool_call(c, config, decision_id)),
            ),
        }

        // The last answer prompts the model. A call in flight does this when it settles.
        if actions.is_empty() && pending_calls == 0 && siblings_answered(&call.id, transcript) {
            actions.extend(reissue(&call.id, &refusals, transcript, llm_calls));
        }
        let mut messages = recorded(transcript);
        messages.extend(refusals);
        Some(DecisionResponse {
            messages,
            actions,
            ..Default::default()
        })
    }
}

pub fn needed(call: &ToolCall, connector_tools: &[ConnectorTool]) -> bool {
    target_of(call, connector_tools).is_some_and(|tool| tool.approve)
}

pub fn ask(held: &ToolCall, connector_tools: &[ConnectorTool], remaining: usize) -> DecisionAction {
    let name = ran_name(held, connector_tools);
    let arguments = ran_arguments(held, connector_tools);
    // Slack's mrkdwn renders a language tag as a line of the block.
    let shown = match &arguments {
        Value::Object(o) if o.is_empty() => String::new(),
        Value::Null => String::new(),
        value => format!(
            "\n\n```\n{}\n```",
            serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
        ),
    };
    let after = match remaining {
        0 => String::new(),
        1 => "\n\nOne more call waits behind it.".to_string(),
        n => format!("\n\n{n} more calls wait behind it."),
    };
    DecisionAction::Interrupt {
        interrupt_id: Some(format!("{PREFIX}{}", held.id)),
        reason: format!("`{name}` needs approval before it runs"),
        payload: serde_json::json!({
            "message": format!("Run `{name}`?{shown}{after}"),
            "tool_call_id": held.id,
            "metadata": {
                "type": "tool.approval",
                "tool": name,
                "arguments": arguments,
                "remaining": remaining,
                "options": [
                    { "label": "Run it", "value": { "approved": true }, "style": "primary" },
                    { "label": "Decline", "value": { "approved": false }, "style": "danger" },
                ],
            },
        }),
    }
}

fn refusal(call: &ToolCall) -> DraftMessage {
    DraftMessage {
        id: None,
        role: Role::Tool,
        content: Some(Content::Text(
            "A person declined this call. Do not try it again; \
             say what you were going to do and why it stopped."
                .to_string(),
        )),
        tool_calls: None,
        tool_call_id: Some(call.id.clone()),
        name: Some(call.function.name.clone()),
        reasoning: None,
    }
}

fn approved(payload: &Value) -> bool {
    if payload.get("status").and_then(|s| s.as_str()) == Some("cancelled") {
        return false;
    }
    payload
        .get("payload")
        .unwrap_or(payload)
        .get("approved")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn ran_name<'a>(call: &'a ToolCall, connector_tools: &'a [ConnectorTool]) -> &'a str {
    target_of(call, connector_tools)
        .map(|t| t.name.as_str())
        .unwrap_or(&call.function.name)
}

fn ran_arguments(call: &ToolCall, connector_tools: &[ConnectorTool]) -> Value {
    let written = serde_json::from_str::<Value>(&call.function.arguments)
        .unwrap_or_else(|_| Value::String(call.function.arguments.clone()));
    if target_of(call, connector_tools).is_none_or(|t| t.name == call.function.name) {
        return written;
    }
    let inner = inner_arguments(&written);
    serde_json::from_str(&inner).unwrap_or(Value::String(inner))
}

/// The tool the call runs.
fn target_of<'a>(
    call: &ToolCall,
    connector_tools: &'a [ConnectorTool],
) -> Option<&'a ConnectorTool> {
    let named = connector_tools
        .iter()
        .find(|t| t.name == call.function.name)?;
    if named.kind != ConnectorToolKind::Call {
        return Some(named);
    }
    let inner = serde_json::from_str::<Value>(&call.function.arguments)
        .ok()?
        .get("name")?
        .as_str()?
        .to_string();
    connector_tools
        .iter()
        .find(|t| t.name == inner && t.kind.is_remote())
}
