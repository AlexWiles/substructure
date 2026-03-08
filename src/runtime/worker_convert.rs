//! Conversions between internal runtime types and worker proto types.

use crate::runtime::config::ClientIdentity;
use crate::runtime::llm as internal_llm;
use crate::runtime::message as internal_msg;
use crate::runtime::session::types::{Artifact, Part};
use crate::runtime::session::{LlmCallStatus, ToolCallStatus, ToolResult};
use crate::runtime::span::{SpanContext, SpanId, TraceId};
use crate::worker as proto;

use proto::{CallStatus, Role};

// ---------------------------------------------------------------------------
// Message conversions
// ---------------------------------------------------------------------------

impl From<&internal_msg::Message> for proto::Message {
    fn from(m: &internal_msg::Message) -> Self {
        Self {
            role: match m.role {
                internal_msg::Role::System => Role::System as i32,
                internal_msg::Role::User => Role::User as i32,
                internal_msg::Role::Assistant => Role::Assistant as i32,
                internal_msg::Role::Tool => Role::Tool as i32,
            },
            content: m.content.clone(),
            tool_calls: m.tool_calls.iter().map(Into::into).collect(),
            tool_call_id: m.tool_call_id.clone(),
            call_id: m.call_id.clone(),
            usage: m
                .usage
                .as_ref()
                .and_then(|v| serde_json::from_value(v.clone()).ok()),
        }
    }
}

impl From<&proto::Message> for internal_msg::Message {
    fn from(m: &proto::Message) -> Self {
        Self {
            role: match Role::try_from(m.role).unwrap_or(Role::Unspecified) {
                Role::System => internal_msg::Role::System,
                Role::User => internal_msg::Role::User,
                Role::Assistant => internal_msg::Role::Assistant,
                Role::Tool => internal_msg::Role::Tool,
                _ => internal_msg::Role::User,
            },
            content: m.content.clone(),
            tool_calls: m.tool_calls.iter().map(Into::into).collect(),
            tool_call_id: m.tool_call_id.clone(),
            call_id: m.call_id.clone(),
            usage: m.usage.as_ref().and_then(|v| serde_json::to_value(v).ok()),
        }
    }
}

// ---------------------------------------------------------------------------
// ToolCall conversions
// ---------------------------------------------------------------------------

impl From<&internal_msg::ToolCall> for proto::ToolCall {
    fn from(tc: &internal_msg::ToolCall) -> Self {
        Self {
            id: tc.id.clone(),
            name: tc.name.clone(),
            arguments: tc.arguments.clone(),
        }
    }
}

impl From<&proto::ToolCall> for internal_msg::ToolCall {
    fn from(tc: &proto::ToolCall) -> Self {
        Self {
            id: tc.id.clone(),
            name: tc.name.clone(),
            arguments: tc.arguments.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// LlmTool conversions
// ---------------------------------------------------------------------------

impl From<&internal_llm::LlmTool> for proto::LlmTool {
    fn from(t: &internal_llm::LlmTool) -> Self {
        Self {
            tool_type: t.tool_type.clone(),
            function: Some(proto::LlmToolFunction {
                name: t.function.name.clone(),
                description: t.function.description.clone(),
                parameters: serde_json::from_value(t.function.parameters.clone()).ok(),
            }),
        }
    }
}

impl From<&proto::LlmTool> for internal_llm::LlmTool {
    fn from(t: &proto::LlmTool) -> Self {
        let function = t.function.as_ref().expect("LlmTool.function required");
        Self {
            tool_type: t.tool_type.clone(),
            function: internal_llm::LlmToolFunction {
                name: function.name.clone(),
                description: function.description.clone(),
                parameters: function
                    .parameters
                    .as_ref()
                    .and_then(|v| serde_json::to_value(v).ok())
                    .unwrap_or(serde_json::Value::Null),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// LlmRequest conversions
// ---------------------------------------------------------------------------

impl From<&internal_llm::LlmRequest> for proto::LlmRequest {
    fn from(r: &internal_llm::LlmRequest) -> Self {
        Self {
            model: r.model.clone(),
            messages: r.messages.iter().map(Into::into).collect(),
            tools: r
                .tools
                .as_ref()
                .map(|ts| ts.iter().map(Into::into).collect())
                .unwrap_or_default(),
            temperature: r.temperature,
            max_completion_tokens: r.max_completion_tokens,
        }
    }
}

impl From<&proto::LlmRequest> for internal_llm::LlmRequest {
    fn from(r: &proto::LlmRequest) -> Self {
        Self {
            model: r.model.clone(),
            messages: r.messages.iter().map(Into::into).collect(),
            tools: if r.tools.is_empty() {
                None
            } else {
                Some(r.tools.iter().map(Into::into).collect())
            },
            temperature: r.temperature,
            max_completion_tokens: r.max_completion_tokens,
        }
    }
}

// ---------------------------------------------------------------------------
// ToolResult conversions
// ---------------------------------------------------------------------------

impl From<&ToolResult> for proto::ToolResult {
    fn from(r: &ToolResult) -> Self {
        Self {
            tool_call_id: r.tool_call_id.clone(),
            name: r.name.clone(),
            content: r.content.clone(),
            is_error: r.is_error,
        }
    }
}

impl From<&proto::ToolResult> for ToolResult {
    fn from(r: &proto::ToolResult) -> Self {
        Self {
            tool_call_id: r.tool_call_id.clone(),
            name: r.name.clone(),
            content: r.content.clone(),
            is_error: r.is_error,
        }
    }
}

// ---------------------------------------------------------------------------
// Artifact / Part conversions
// ---------------------------------------------------------------------------

impl From<&Artifact> for proto::Artifact {
    fn from(a: &Artifact) -> Self {
        Self {
            name: a.name.clone(),
            description: a.description.clone(),
            parts: a.parts.iter().map(Into::into).collect(),
        }
    }
}

impl From<&proto::Artifact> for Artifact {
    fn from(a: &proto::Artifact) -> Self {
        Self {
            name: a.name.clone(),
            description: a.description.clone(),
            parts: a.parts.iter().map(Into::into).collect(),
        }
    }
}

impl From<&Part> for proto::Part {
    fn from(p: &Part) -> Self {
        match p {
            Part::Text { text } => Self {
                kind: Some(proto::part::Kind::Text(text.clone())),
            },
            Part::Data { data } => Self {
                kind: Some(proto::part::Kind::Data(
                    serde_json::from_value(data.clone()).unwrap_or_default(),
                )),
            },
        }
    }
}

impl From<&proto::Part> for Part {
    fn from(p: &proto::Part) -> Self {
        match &p.kind {
            Some(proto::part::Kind::Text(text)) => Part::Text { text: text.clone() },
            Some(proto::part::Kind::Data(data)) => Part::Data {
                data: serde_json::to_value(data).unwrap_or(serde_json::Value::Null),
            },
            None => Part::Text {
                text: String::new(),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Call status conversions
// ---------------------------------------------------------------------------

impl From<&ToolCallStatus> for CallStatus {
    fn from(s: &ToolCallStatus) -> Self {
        match s {
            ToolCallStatus::Pending => CallStatus::Pending,
            ToolCallStatus::Completed => CallStatus::Completed,
            ToolCallStatus::Failed => CallStatus::Failed,
            ToolCallStatus::RetryScheduled => CallStatus::RetryScheduled,
        }
    }
}

impl From<&LlmCallStatus> for CallStatus {
    fn from(s: &LlmCallStatus) -> Self {
        match s {
            LlmCallStatus::Pending => CallStatus::Pending,
            LlmCallStatus::Completed => CallStatus::Completed,
            LlmCallStatus::Failed => CallStatus::Failed,
            LlmCallStatus::RetryScheduled => CallStatus::RetryScheduled,
        }
    }
}

// ---------------------------------------------------------------------------
// DecisionTrigger conversions
// ---------------------------------------------------------------------------

impl From<&crate::runtime::session::decision::DecisionTrigger> for proto::DecisionTrigger {
    fn from(t: &crate::runtime::session::decision::DecisionTrigger) -> Self {
        use crate::runtime::session::decision::DecisionTrigger as DT;
        let trigger = match t {
            DT::UserMessage { stream, message } => {
                proto::decision_trigger::Trigger::UserMessage(proto::UserMessage {
                    stream: *stream,
                    message: Some(message.into()),
                })
            }
            DT::LlmCompleted {
                call_id,
                message,
                truncated,
            } => proto::decision_trigger::Trigger::LlmCompleted(proto::LlmCompleted {
                call_id: call_id.clone(),
                message: Some(message.into()),
                truncated: *truncated,
            }),
            DT::LlmFailed { call_id, error } => {
                proto::decision_trigger::Trigger::LlmFailed(proto::LlmFailed {
                    call_id: call_id.clone(),
                    error: error.clone(),
                })
            }
            DT::ToolResolved { result } => {
                proto::decision_trigger::Trigger::ToolResolved(proto::ToolResolved {
                    result: Some(result.into()),
                })
            }
            DT::InterruptResumed { interrupt_id } => {
                proto::decision_trigger::Trigger::InterruptResumed(proto::InterruptResumed {
                    interrupt_id: interrupt_id.clone(),
                })
            }
            DT::Stall => proto::decision_trigger::Trigger::Stall(proto::Stall {}),
        };
        Self {
            trigger: Some(trigger),
        }
    }
}

// ---------------------------------------------------------------------------
// ClientIdentity conversions
// ---------------------------------------------------------------------------

impl From<&ClientIdentity> for proto::ClientIdentity {
    fn from(c: &ClientIdentity) -> Self {
        Self {
            tenant_id: c.tenant_id.clone(),
            sub: c.sub.clone(),
            attrs: c.attrs.clone(),
        }
    }
}

impl From<&proto::ClientIdentity> for ClientIdentity {
    fn from(c: &proto::ClientIdentity) -> Self {
        Self {
            tenant_id: c.tenant_id.clone(),
            sub: c.sub.clone(),
            attrs: c.attrs.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// SpanContext conversions
// ---------------------------------------------------------------------------

impl From<&SpanContext> for proto::SpanContext {
    fn from(s: &SpanContext) -> Self {
        Self {
            trace_id: s.trace_id.as_bytes().to_vec(),
            span_id: s.span_id.as_bytes().to_vec(),
            parent_span_id: s.parent_span_id.map(|id| id.as_bytes().to_vec()),
            trace_flags: s.trace_flags as u32,
            trace_state: s.trace_state.clone(),
            name: s.name.clone(),
        }
    }
}

impl From<&proto::SpanContext> for SpanContext {
    fn from(s: &proto::SpanContext) -> Self {
        let trace_id = <[u8; 16]>::try_from(s.trace_id.as_slice())
            .map(TraceId::from_bytes)
            .unwrap_or_else(|_| TraceId::random());
        let span_id = <[u8; 8]>::try_from(s.span_id.as_slice())
            .map(SpanId::from_bytes)
            .unwrap_or_else(|_| SpanId::random());
        let parent_span_id = s
            .parent_span_id
            .as_ref()
            .and_then(|b| <[u8; 8]>::try_from(b.as_slice()).ok())
            .map(SpanId::from_bytes);
        Self {
            trace_id,
            span_id,
            parent_span_id,
            trace_flags: s.trace_flags as u8,
            trace_state: s.trace_state.clone(),
            name: s.name.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// WorkerCtx from WorkerDispatch (extract decision context from wire message)
// ---------------------------------------------------------------------------

impl From<&proto::WorkerDispatch> for proto::WorkerCtx {
    fn from(d: &proto::WorkerDispatch) -> Self {
        Self {
            session_id: d.session_id.clone(),
            stream: d.stream,
            agent_name: d.agent_name.clone(),
            token_usage: d.token_usage.clone(),
            tool_call_statuses: d.tool_call_statuses.clone(),
            llm_call_statuses: d.llm_call_statuses.clone(),
            auth: d.auth.clone(),
        }
    }
}
