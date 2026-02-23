use rand::RngExt;
use serde::{Deserialize, Serialize};

pub type TraceId = [u8; 16];
pub type SpanId = [u8; 8];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanContext {
    pub trace_id: TraceId,
    pub span_id: SpanId,
    pub parent_span_id: Option<SpanId>,
    pub trace_flags: u8,
    pub trace_state: Option<String>,
}

impl SpanContext {
    pub fn root() -> Self {
        SpanContext {
            trace_id: rand::rng().random(),
            span_id: rand::rng().random(),
            parent_span_id: None,
            trace_flags: 1,
            trace_state: None,
        }
    }

    pub fn child(&self) -> Self {
        SpanContext {
            trace_id: self.trace_id,
            span_id: rand::rng().random(),
            parent_span_id: Some(self.span_id),
            trace_flags: self.trace_flags,
            trace_state: self.trace_state.clone(),
        }
    }
}
