//! Client input and the message tree.
//!
//! Cross-kind: what a client submits can settle a tool call, move the head, and
//! queue a decision, so it belongs to no one kind. The tree changes only on
//! decision submit, and [`reconcile_transcript`](SessionState::reconcile_transcript)
//! is its sole writer.

use super::{Completion, SessionError};
use crate::protocol::{
    ClientAppend, ClientContext, ClientMessage, ClientMessages, ClientPayload, Content,
    DraftMessage, EffectStatus, NewMessage, Role,
};
use crate::runtime::session::decision::{ToolHandler, Trigger};
use crate::runtime::session::effects::{decision_queued, tool};
use crate::runtime::session::events::*;
use crate::runtime::session::reconcile::{landing_leaf, plan_reconcile};
use crate::runtime::session::state::{EffectKind, SessionState};
use crate::runtime::Caller;

impl SessionState {
    // Sole producer of NewMessage; the tree changes only on decision submit.
    // Executes the plan from `plan_reconcile` — the one interpreter of "what
    // recording this list writes" — so submit-time classification and delivery
    // annotation can't drift from what actually lands in the tree.
    //
    // Returns the events and the post-batch head. A non-empty view writing
    // nothing selects its leaf as the branch: the head rebases via `HeadMoved`.
    pub(super) fn reconcile_transcript(
        &self,
        transcript: Vec<DraftMessage>,
    ) -> (Vec<EventPayload>, Option<String>) {
        // Normalize again at the write seam: a view frozen into a queued trigger
        // can race the decision that records a tool node (the echo predates it),
        // so folding at client-submit time alone would fork the tree here.
        let transcript = self.normalize_client_view(transcript);
        let known: std::collections::HashSet<&str> =
            self.nodes.iter().map(|n| n.message.id.as_str()).collect();
        let plan = plan_reconcile(&known, &transcript);
        let mut events = Vec::with_capacity(plan.len());
        let mut plan_iter = plan.iter().peekable();
        let mut parent: Option<String> = None;
        for (index, msg) in transcript.into_iter().enumerate() {
            match plan_iter.peek() {
                Some(write) if write.index == index => {
                    let rerecord = write.rerecord;
                    plan_iter.next();
                    // Re-record known ids in the news region as fresh nodes so
                    // the branch stays a chain instead of grafting onto the old.
                    let msg = if rerecord {
                        msg.rerecord()
                    } else {
                        msg.record()
                    };
                    let id = msg.id.clone();
                    events.push(EventPayload::NewMessage(NewMessage {
                        message: msg,
                        parent_id: parent.take(),
                    }));
                    parent = Some(id);
                }
                // Known prefix before the news: advance the parent cursor only.
                _ => parent = msg.id,
            }
        }
        if !events.is_empty() {
            return (events, parent);
        }
        match parent {
            Some(leaf) if self.head_id.as_deref() != Some(leaf.as_str()) => {
                let moved = EventPayload::HeadMoved(HeadMoved {
                    head_id: leaf.clone(),
                });
                (vec![moved], Some(leaf))
            }
            _ => (events, self.head_id.clone()),
        }
    }

    /// `(tool_call_id, name, content)` for a transcript tool message answering a
    /// pending client tool call.
    fn pending_client_result(&self, m: &DraftMessage) -> Option<(String, String, String)> {
        if m.role != Role::Tool {
            return None;
        }
        let id = m.tool_call_id.as_deref()?;
        let effect = self.effect(EffectKind::ToolCall, id)?;
        let tc = effect.tool()?;
        if tc.handler != ToolHandler::Client || effect.tracking.status() != EffectStatus::Pending {
            return None;
        }
        let content = m
            .content
            .as_ref()
            .map(Content::text_owned)
            .unwrap_or_default();
        Some((id.to_string(), tc.name.clone(), content))
    }

    /// The id of a recorded tool-result node answering `tool_call_id`, if any —
    /// preferring one on the active path, then the most recently recorded.
    fn recorded_result_node(
        &self,
        tool_call_id: &str,
        on_path: &std::collections::HashSet<&str>,
    ) -> Option<String> {
        let mut best: Option<&str> = None;
        let mut best_on_path = false;
        for m in self.nodes.iter().map(|n| &n.message) {
            if m.role != Role::Tool || m.tool_call_id.as_deref() != Some(tool_call_id) {
                continue;
            }
            let on = on_path.contains(m.id.as_str());
            if on {
                best = Some(&m.id);
                best_on_path = true;
            } else if !best_on_path {
                best = Some(&m.id);
            }
        }
        best.map(str::to_string)
    }

    /// Fold client echoes of already-recorded tool results back onto their
    /// nodes: an unknown-id tool message whose call already has a recorded node
    /// IS that node's echo, so it adopts the node's id and the tree sees a
    /// resend rather than a fork. Identification, not deletion — nothing is
    /// dropped, so the reconcile plan stays coherent.
    pub fn normalize_client_view(&self, messages: Vec<DraftMessage>) -> Vec<DraftMessage> {
        let known: std::collections::HashSet<&str> =
            self.nodes.iter().map(|n| n.message.id.as_str()).collect();
        let on_path: std::collections::HashSet<&str> = self
            .head_id
            .as_deref()
            .map(|h| self.path_ids(h))
            .unwrap_or_default();
        messages
            .into_iter()
            .map(|m| {
                let is_known = m.id.as_deref().is_some_and(|id| known.contains(id));
                if m.role == Role::Tool && !is_known {
                    if let Some(node_id) = m
                        .tool_call_id
                        .as_deref()
                        .and_then(|tcid| self.recorded_result_node(tcid, &on_path))
                    {
                        return DraftMessage {
                            id: Some(node_id),
                            ..m
                        };
                    }
                }
                m
            })
            .collect()
    }

    /// Client input, by shape. Each arm validates against the live world and
    /// returns what accepting it writes.
    ///
    /// `deferred_turn` is the turn a queued submit holds for later: it rides
    /// the message trigger and opens the turn when that decision dispatches.
    /// Only the message shapes can carry one — the caller decides which.
    pub(super) fn client_payload_events(
        &self,
        payload: ClientPayload,
        caller: &Caller,
        deferred_turn: Option<String>,
    ) -> Result<Vec<EventPayload>, SessionError> {
        self.ensure_owns_session(caller)?;
        match payload {
            ClientPayload::Message(ClientMessage { message, stream: _ }) => {
                let is_user = message.role == Role::User;
                if self.head_parked() && is_user {
                    return Err(SessionError::SessionInterrupted);
                }
                let mut events = Vec::new();
                if self.session_start_failed && is_user {
                    // A failed session.start left no config. Re-queue the start
                    // ahead of the message: arrival order and the prerequisite
                    // gate make the retry run first. Recovery is the next
                    // message, not a reset.
                    events.push(decision_queued(Trigger::SessionStart));
                }
                if is_user {
                    events.push(decision_queued(Trigger::ClientMessage {
                        messages: vec![message],
                        client: ClientContext::default(),
                        turn_id: deferred_turn,
                    }));
                }
                Ok(events)
            }
            ClientPayload::Append(ClientAppend {
                messages,
                stream: _,
                client,
            }) => {
                let any_user = messages.iter().any(|m| m.role == Role::User);
                if self.head_parked() && any_user {
                    return Err(SessionError::SessionInterrupted);
                }
                let mut events = Vec::new();
                if self.session_start_failed && any_user {
                    events.push(decision_queued(Trigger::SessionStart));
                }
                events.push(decision_queued(Trigger::ClientMessage {
                    messages,
                    client,
                    turn_id: deferred_turn,
                }));
                Ok(events)
            }
            ClientPayload::Messages(ClientMessages {
                messages,
                stream: _,
                client,
            }) => self.client_view_events(messages, client),
            // Actions pass an interrupted session: a click may be what
            // answers the prompt.
            ClientPayload::Action(action) => Ok(vec![decision_queued(Trigger::ClientAction {
                name: action.name,
                args: action.args,
            })]),
        }
    }

    /// A full client view: fold echoes onto their recorded nodes, settle the
    /// answers to still-pending client calls, and either mirror the settle
    /// endpoint or hand the worker the whole view as one frozen transcript.
    fn client_view_events(
        &self,
        messages: Vec<DraftMessage>,
        client: ClientContext,
    ) -> Result<Vec<EventPayload>, SessionError> {
        // Fold client echoes of already-recorded results onto their nodes so
        // the tree sees a resend, not a fork.
        let messages = self.normalize_client_view(messages);

        // Answers to still-pending client calls, first-wins per call id (a
        // repeat within one view has no recorded node to fold onto, so the
        // later copy is dropped).
        let mut seen = std::collections::HashSet::new();
        let completions: Vec<Completion> = messages
            .iter()
            .enumerate()
            .filter_map(|(index, m)| {
                self.pending_client_result(m)
                    .map(|(tc, name, result)| Completion {
                        index,
                        tool_call_id: tc,
                        name,
                        result,
                    })
            })
            .filter(|c| seen.insert(c.tool_call_id.clone()))
            .collect();

        // What recording this view would write, by the one reconcile
        // interpreter; classification reads this plan rather than re-walking
        // the tree.
        let (landing, single_answer) = {
            let known: std::collections::HashSet<&str> =
                self.nodes.iter().map(|n| n.message.id.as_str()).collect();
            let plan = plan_reconcile(&known, &messages);
            // Sharing no prefix with a non-empty tree is almost always a lost
            // or mis-built client view, not an intentional fork.
            if plan.first().map(|w| w.index) == Some(0) && !self.nodes.is_empty() {
                tracing::warn!(
                    "client view shares no prefix with the session; recording will fork at the root"
                );
            }
            let landing = landing_leaf(&messages, &plan);
            // Fast path iff the view's only change is one answer to one pending
            // call — provable from the plan.
            let single_answer = plan.len() == 1
                && completions.len() == 1
                && plan.first().map(|w| w.index) == completions.first().map(|c| c.index);
            (landing, single_answer)
        };

        // Gate where the view lands: a view escaping the parked head
        // dispatches; answers to pending work still queue.
        if completions.is_empty() && self.active_interrupt_for(landing.as_deref()).is_some() {
            return Err(SessionError::SessionInterrupted);
        }

        if single_answer {
            // Mirror the settle endpoint: settle + tool.finished, the worker
            // appends the node, the view is discarded.
            return Ok(match completions.into_iter().next() {
                Some(c) => tool::complete(c.tool_call_id, c.name, c.result),
                None => Vec::new(),
            });
        }

        // Bedrock: settle every answer silently (no tool.finished), then hand
        // the worker the whole normalized view as one frozen transcript. The
        // worker's echo records the client's messages. Covers plan-empty views
        // (no-op resend / regenerate): still delivered, the worker decides.
        let mut events: Vec<EventPayload> = completions
            .into_iter()
            .map(|c| {
                EventPayload::ToolCallCompleted(ToolCallCompleted {
                    id: c.tool_call_id,
                    name: c.name,
                    result: c.result,
                })
            })
            .collect();
        events.push(decision_queued(Trigger::ClientTranscript {
            messages,
            new_from: 0,
            client,
        }));
        Ok(events)
    }
}
