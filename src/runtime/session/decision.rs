//! Worker decision types — used for event serialization in the event store.
//!
//! The public worker interface (Worker trait, proto types) lives in
//! `crate::worker`. The executor trait lives in `session::dispatch`.
//! These types support event sourcing and command handling within
//! the session aggregate.

use serde_json::Value;

use serde::{Deserialize, Serialize};

use crate::runtime::llm::LlmRequest;
use crate::runtime::message::{Message, ToolCall};
use crate::runtime::session::state::ToolResult;
use crate::runtime::session::types::Artifact;

// ---------------------------------------------------------------------------
// Decision triggers — what caused the worker to be consulted
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionTrigger {
    UserMessage {
        stream: bool,
        message: Message,
    },
    LlmCompleted {
        call_id: String,
        message: Message,
        /// True when finish_reason was "length" (output truncated).
        truncated: bool,
    },
    LlmFailed {
        call_id: String,
        error: String,
    },
    ToolResolved {
        result: ToolResult,
    },
    InterruptResumed {
        interrupt_id: String,
    },
    Stall,
}

// ---------------------------------------------------------------------------
// Worker actions — what the worker wants the runtime to do
// ---------------------------------------------------------------------------

/// A tool call annotated with opaque worker context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallAction {
    pub tool_call: ToolCall,
    /// Opaque context from the worker, passed through to dispatch.
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub context: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkerAction {
    RequestLlm { request: LlmRequest, stream: bool },
    RequestToolCalls { tool_calls: Vec<ToolCallAction> },
    Done { artifacts: Vec<Artifact> },
}

// ---------------------------------------------------------------------------
// Worker decision events (persisted in event store)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerDecisionRequested {
    pub decision_id: String,
    pub trigger: DecisionTrigger,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerDecisionCompleted {
    pub decision_id: String,
    /// Opaque worker state — session stores but never interprets.
    #[serde(with = "base64_bytes")]
    pub state: Vec<u8>,
}

/// Serde helper: encode Vec<u8> as base64 in JSON for event store compatibility.
mod base64_bytes {
    use base64::Engine;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(bytes: &Vec<u8>, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&base64::engine::general_purpose::STANDARD.encode(bytes))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let s = String::deserialize(d)?;
        base64::engine::general_purpose::STANDARD
            .decode(&s)
            .map_err(serde::de::Error::custom)
    }
}
