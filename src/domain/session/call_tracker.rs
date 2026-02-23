use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::domain::event::*;
use super::strategy::ToolResult;

// ---------------------------------------------------------------------------
// Types — tracking state for LLM and tool call lifecycles
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LlmCallStatus {
    Pending,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCall {
    pub call_id: String,
    pub request: LlmRequest,
    pub status: LlmCallStatus,
    pub response: Option<LlmResponse>,
    pub response_processed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ToolCallStatus {
    Pending,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackedToolCall {
    pub tool_call_id: String,
    pub name: String,
    pub status: ToolCallStatus,
}

// ---------------------------------------------------------------------------
// CallTracker — runtime-owned state machine
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallTracker {
    llm_calls: HashMap<String, LlmCall>,
    tool_calls: HashMap<String, TrackedToolCall>,
    pub pending_tool_results: usize,
    pub dirty: bool,
    pub stream: bool,
    tool_result_batch: Vec<ToolResult>,
    active_interrupt: Option<String>,
}

impl CallTracker {
    pub fn new() -> Self {
        CallTracker {
            llm_calls: HashMap::new(),
            tool_calls: HashMap::new(),
            pending_tool_results: 0,
            dirty: false,
            stream: false,
            tool_result_batch: Vec::new(),
            active_interrupt: None,
        }
    }

    pub fn apply(&mut self, event: &Event) {
        match &event.payload {
            EventPayload::MessageUser(payload) => {
                self.stream = payload.stream;
                if self.active_llm_call().is_some() {
                    self.dirty = true;
                }
            }
            EventPayload::MessageAssistant(payload) => {
                self.pending_tool_results = payload.message.tool_calls.len();
                if let Some(call) = self.llm_calls.get_mut(&payload.call_id) {
                    call.response_processed = true;
                }
            }
            EventPayload::MessageTool(_) => {
                self.pending_tool_results = self.pending_tool_results.saturating_sub(1);
            }
            EventPayload::LlmCallRequested(payload) => {
                self.llm_calls.insert(
                    payload.call_id.clone(),
                    LlmCall {
                        call_id: payload.call_id.clone(),
                        request: payload.request.clone(),
                        status: LlmCallStatus::Pending,
                        response: None,
                        response_processed: false,
                    },
                );
            }
            EventPayload::LlmCallCompleted(payload) => {
                if let Some(call) = self.llm_calls.get_mut(&payload.call_id) {
                    call.status = LlmCallStatus::Completed;
                    call.response = Some(payload.response.clone());
                }
            }
            EventPayload::LlmCallErrored(payload) => {
                if let Some(call) = self.llm_calls.get_mut(&payload.call_id) {
                    call.status = LlmCallStatus::Failed;
                }
            }
            EventPayload::ToolCallRequested(payload) => {
                self.tool_calls.insert(
                    payload.tool_call_id.clone(),
                    TrackedToolCall {
                        tool_call_id: payload.tool_call_id.clone(),
                        name: payload.name.clone(),
                        status: ToolCallStatus::Pending,
                    },
                );
            }
            EventPayload::ToolCallCompleted(payload) => {
                if let Some(tc) = self.tool_calls.get_mut(&payload.tool_call_id) {
                    tc.status = ToolCallStatus::Completed;
                }
                self.tool_result_batch.push(ToolResult {
                    tool_call_id: payload.tool_call_id.clone(),
                    name: payload.name.clone(),
                    content: payload.result.clone(),
                    is_error: false,
                });
            }
            EventPayload::ToolCallErrored(payload) => {
                if let Some(tc) = self.tool_calls.get_mut(&payload.tool_call_id) {
                    tc.status = ToolCallStatus::Failed;
                }
                self.tool_result_batch.push(ToolResult {
                    tool_call_id: payload.tool_call_id.clone(),
                    name: payload.name.clone(),
                    content: payload.error.clone(),
                    is_error: true,
                });
            }
            EventPayload::SessionInterrupted(payload) => {
                self.active_interrupt = Some(payload.interrupt_id.clone());
            }
            EventPayload::InterruptResumed(_) => {
                self.active_interrupt = None;
            }
            _ => {}
        }
    }

    pub fn active_interrupt(&self) -> Option<&str> {
        self.active_interrupt.as_deref()
    }

    pub fn active_llm_call(&self) -> Option<&LlmCall> {
        self.llm_calls
            .values()
            .find(|c| c.status == LlmCallStatus::Pending)
    }

    // --- Query methods for handle() guards ---

    pub fn has_llm_call(&self, call_id: &str) -> bool {
        self.llm_calls.contains_key(call_id)
    }

    pub fn llm_call_status(&self, call_id: &str) -> Option<&LlmCallStatus> {
        self.llm_calls.get(call_id).map(|c| &c.status)
    }

    pub fn is_response_processed(&self, call_id: &str) -> bool {
        self.llm_calls.get(call_id).is_some_and(|c| c.response_processed)
    }

    pub fn has_tool_call(&self, tool_call_id: &str) -> bool {
        self.tool_calls.contains_key(tool_call_id)
    }

    pub fn tool_call_status(&self, tool_call_id: &str) -> Option<&ToolCallStatus> {
        self.tool_calls.get(tool_call_id).map(|tc| &tc.status)
    }

    /// Drain accumulated tool results for the strategy callback.
    pub fn take_tool_result_batch(&mut self) -> Vec<ToolResult> {
        std::mem::take(&mut self.tool_result_batch)
    }

    // --- Recovery queries ---

    pub fn llm_calls(&self) -> impl Iterator<Item = &LlmCall> {
        self.llm_calls.values()
    }

    pub fn tool_calls(&self) -> impl Iterator<Item = &TrackedToolCall> {
        self.tool_calls.values()
    }

    pub fn has_pending_llm(&self) -> bool {
        self.llm_calls.values().any(|c| c.status == LlmCallStatus::Pending)
    }
}
