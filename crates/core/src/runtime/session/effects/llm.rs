//! An LLM call.
//!
//! ```text
//! Queued ─dispatch→ Pending ─complete────────────→ Completed ⇒ queue llm.finished(ok)
//!    ↑                 │ ─error[retries left]────→ RetryScheduled ─due→ Queued
//!    │                 │ ─error[exhausted]───────→ Failed    ⇒ queue llm.finished(err)
//!    └──── requeue ────┘ ─void──────────────────→ Failed
//! ```
//!
//! Dispatch merges the connector tools in force, so a call waits on every fetch
//! the config at its anchor owes ([`Dep::ConnectorSettled`](super::super::schedule::Dep)).
//! A `Worker` call is handed to the worker as `llm.execute`; a `Server` one the
//! engine runs itself.

use super::{decision_queued, fail, mismatched, void_events, KindSpec, Outcome, SettleError};
use crate::protocol::{
    Content, ContentPart, DraftMessage, EffectStatus, ErrorCode, ImageUrl, LlmFormat, LlmRequest,
    LlmResponse, RetryPolicy, Role,
};
use crate::runtime::session::command::SessionError;
use crate::runtime::session::decision::{LlmHandler, Trigger};
use crate::runtime::session::events::*;
use crate::runtime::session::schedule::Dep;
use crate::runtime::session::state::{EffectKind, QueueEntry, SessionState};
use crate::runtime::Caller;

pub struct LlmSpec;

impl KindSpec for LlmSpec {
    fn kind(&self) -> EffectKind {
        EffectKind::LlmCall
    }

    fn authorize(
        &self,
        state: &SessionState,
        id: &str,
        caller: &Caller,
    ) -> Result<(), SessionError> {
        state.check_llm_call_caller(state.llm_call(id), caller)
    }

    fn settle(&self, state: &SessionState, id: &str, outcome: Outcome) -> Vec<EventPayload> {
        match outcome {
            Outcome::Llm(response) => complete(state, id, *response),
            Outcome::Error(e) => fail(self, state, id, &e),
            other => mismatched(self.kind(), &other),
        }
    }

    fn errored(&self, state: &SessionState, id: &str, e: &SettleError) -> Option<EventPayload> {
        Some(EventPayload::LlmCallErrored(LlmCallErrored {
            id: id.to_string(),
            attempt: attempt(state, id),
            error: e.error.clone(),
            retryable: e.retryable,
            code: e.code.clone(),
            detail: e.detail.clone(),
        }))
    }

    /// The worker decides whether a turn that lost its model call is salvageable.
    fn terminal(&self, _state: &SessionState, id: &str, e: &SettleError) -> Vec<EventPayload> {
        vec![decision_queued(Trigger::llm_err(
            id.to_string(),
            e.error.clone(),
            e.code.clone(),
            e.detail.clone(),
        ))]
    }

    fn timeout_error(&self) -> SettleError {
        SettleError::new(super::DEADLINE, true).with_detail(Some(ErrorCode::DeadlineExceeded), None)
    }

    fn dispatch(&self, state: &SessionState, id: &str) -> Vec<EventPayload> {
        // A queue entry with no call is inconsistent state; void it so the walk
        // makes progress instead of spinning.
        let Some(effect) = state.effect(EffectKind::LlmCall, id) else {
            return void_events(EffectKind::LlmCall, id.to_string());
        };
        vec![EventPayload::LlmCallDispatched(LlmCallDispatched {
            id: id.to_string(),
            attempt: effect.tracking.retry.attempts,
        })]
    }

    fn execute_trigger(&self, state: &SessionState, id: &str) -> Option<Trigger> {
        let effect = state.effect(EffectKind::LlmCall, id)?;
        let call = effect.llm().filter(|c| c.handler == LlmHandler::Worker)?;
        Some(Trigger::LlmExecute {
            id: id.to_string(),
            request: call.spec.to_request(call.prompt.clone()),
            format: call.format,
            stream: call.stream,
            attempt: effect.tracking.retry.attempts,
            deadline: effect.tracking.deadline,
        })
    }

    /// A call cannot be authored against tools the engine has not fetched, and
    /// dispatch merges the tools in force, so it waits for every fetch the
    /// config *at its own anchor* owes — not the head's, which may have moved.
    fn deps(&self, state: &SessionState, entry: &QueueEntry) -> Vec<Dep> {
        let leaf = state
            .effect(EffectKind::LlmCall, &entry.id)
            .and_then(|e| e.anchor.clone())
            .or_else(|| state.head_id.clone());
        super::connector::owed(state, leaf.as_deref())
    }

    fn retry(&self, state: &SessionState, id: &str) -> Vec<EventPayload> {
        let Some(effect) = state.effect(EffectKind::LlmCall, id) else {
            return Vec::new();
        };
        let (t, Some(call)) = (&effect.tracking, effect.llm()) else {
            return Vec::new();
        };
        vec![EventPayload::LlmCallRequested(LlmCallRequested {
            id: id.to_string(),
            attempt: t.retry.attempts,
            llm: call.llm.clone(),
            request: call.spec.to_request(call.prompt.clone()),
            stream: call.stream,
            retry: t.retry_policy.clone(),
            handler: call.handler,
            format: call.format,
        })]
    }
}

fn attempt(state: &SessionState, id: &str) -> u32 {
    state
        .tracking(EffectKind::LlmCall, id)
        .map(|t| t.retry.attempts)
        .unwrap_or_default()
}

/// A model result: record it and queue `llm.finished` with the assistant node.
fn complete(state: &SessionState, id: &str, response: LlmResponse) -> Vec<EventPayload> {
    let message = assistant_message(id, &response);
    let truncated = response.finish_reason.as_deref() == Some("length");
    let usage = response.usage.clone();
    let cost = response.cost;
    vec![
        EventPayload::LlmCallCompleted(LlmCallCompleted {
            id: id.to_string(),
            attempt: attempt(state, id),
            response,
        }),
        decision_queued(Trigger::llm_ok(
            id.to_string(),
            message,
            truncated,
            usage,
            cost,
        )),
    ]
}

/// Issue a call. Idempotent by id: (re-)issue only for a new, `Failed`, or
/// `RetryScheduled` call, so a repeat request writes nothing.
#[allow(clippy::too_many_arguments)]
pub(in crate::runtime::session) fn request(
    state: &SessionState,
    call_id: String,
    llm: String,
    request: LlmRequest,
    stream: bool,
    retry: RetryPolicy,
    handler: LlmHandler,
    format: Option<LlmFormat>,
    caller: &Caller,
) -> Result<Vec<EventPayload>, SessionError> {
    SessionState::ensure_internal(caller)?;
    let issue = matches!(
        state
            .tracking(EffectKind::LlmCall, &call_id)
            .map(|t| t.status()),
        None | Some(EffectStatus::Failed) | Some(EffectStatus::RetryScheduled)
    );
    if !issue {
        tracing::debug!(
            %call_id,
            "llm call id already issued; request no-ops (idempotent)"
        );
        return Ok(Vec::new());
    }
    // Mint ids now so the stored prompt (and its retries) is deterministic
    // across replay; keep the wire form. Connector tools are merged at
    // dispatch, when they are current; the execute decision for a
    // worker-handled call queues there too.
    let request = LlmRequest {
        messages: request
            .messages
            .into_iter()
            .map(|m| DraftMessage::from(m.record()))
            .collect(),
        ..request
    };
    Ok(vec![EventPayload::LlmCallRequested(LlmCallRequested {
        id: call_id,
        attempt: 0,
        llm,
        request,
        stream,
        retry,
        handler,
        format,
    })])
}

/// The assistant node for a completed call. Recorded under the call id: it's
/// globally unique (so reconcile still records a new node) and it's the id the
/// client was already streamed — AG-UI keys the assistant message on the call id
/// — so a client's full-view echo matches this node instead of forking.
fn assistant_message(call_id: &str, response: &LlmResponse) -> DraftMessage {
    let tool_calls = (!response.tool_calls.is_empty()).then(|| response.tool_calls.clone());
    let content = if response.images.is_empty() {
        response.content.clone().map(Content::Text)
    } else {
        let mut parts: Vec<ContentPart> = Vec::new();
        if let Some(text) = &response.content {
            parts.push(ContentPart::Text { text: text.clone() });
        }
        for img in &response.images {
            parts.push(ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: img.url.clone(),
                },
            });
        }
        Some(Content::Parts(parts))
    };
    DraftMessage {
        id: Some(call_id.to_string()),
        role: Role::Assistant,
        content,
        tool_calls,
        tool_call_id: None,
        name: None,
    }
}
