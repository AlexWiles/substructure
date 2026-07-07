use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::runtime::aggregate::{AggregateState, DomainEvent};
use crate::runtime::event_store::{Event, EventStore};
use crate::runtime::processor::{
    EventProcessor, EventProcessorRunner, EventProcessorRunnerConfig, ProcessorCheckpointStore,
    ProcessorError,
};
use crate::runtime::session::decision::DecisionTrigger;
use crate::runtime::session::events::{EventPayload, MessageTree};
use crate::runtime::session::message::Message;
use crate::runtime::session::reconcile::news_start;
use crate::runtime::session::state::SessionState;

use super::{WorkerDecisionRequest, WorkerQueue};

struct WorkerDecisionProjection {
    queue: Arc<dyn WorkerQueue>,
}

impl WorkerDecisionProjection {
    fn new(queue: Arc<dyn WorkerQueue>) -> Self {
        Self { queue }
    }
}

#[async_trait::async_trait]
impl EventProcessor for WorkerDecisionProjection {
    fn name(&self) -> &'static str {
        "worker_decision_enqueue"
    }

    fn shard_key(&self, event: &Event) -> Option<String> {
        if event.aggregate_type != SessionState::AGGREGATE_TYPE {
            return None;
        }
        Some(event.aggregate_id.clone())
    }

    async fn apply(&self, event: &Event) -> Result<(), ProcessorError> {
        if let Some(decision) = try_extract(event) {
            tracing::debug!(
                session_id = %decision.session_id,
                decision_id = %decision.decision_id,
                agent_id = %decision.agent_id,
                trigger_type = ?decision.trigger,
                "enqueuing worker decision"
            );
            self.queue.enqueue(decision).await;
        }
        Ok(())
    }
}

pub fn spawn_worker_processor(
    store: Arc<dyn EventStore>,
    checkpoint_store: Arc<dyn ProcessorCheckpointStore>,
    queue: Arc<dyn WorkerQueue>,
    cancel: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    let projection = Arc::new(WorkerDecisionProjection::new(queue));
    EventProcessorRunner::new(
        store,
        checkpoint_store,
        projection,
        EventProcessorRunnerConfig::default(),
        cancel,
    )
    .spawn()
}

/// Client triggers unify at delivery: a bare message becomes a full proposed
/// transcript, and `new_from` marks where the unrecorded suffix starts. The tree
/// is frozen while this decision is pending, so the result is stable across
/// redeliveries and matches what reconciling the echo will write.
fn materialize_client_trigger(
    trigger: DecisionTrigger,
    active_path: &[Message],
    tree: &MessageTree,
) -> DecisionTrigger {
    match trigger {
        DecisionTrigger::ClientMessage { message } => {
            let mut messages = active_path.to_vec();
            let new_from = messages.len();
            messages.push(message);
            DecisionTrigger::ClientTranscript { messages, new_from }
        }
        DecisionTrigger::ClientTranscript { messages, .. } => {
            let known: std::collections::HashSet<&str> =
                tree.nodes.iter().map(|n| n.id()).collect();
            let new_from = news_start(&known, &messages);
            DecisionTrigger::ClientTranscript { messages, new_from }
        }
        other => other,
    }
}

fn try_extract(raw: &Event) -> Option<WorkerDecisionRequest> {
    if raw.aggregate_type != SessionState::AGGREGATE_TYPE {
        return None;
    }
    let event = DomainEvent::<SessionState>::from_raw(raw).ok()?;
    let req = match &event.payload {
        EventPayload::WorkerDecisionRequested(req) => req,
        _ => return None,
    };
    let derived = event.derived.as_ref()?;
    let agent_id = derived.agent_id.as_ref()?;
    let owner = derived.owner.as_ref()?;
    let wd = derived.worker_decisions.get(&req.decision_id)?;

    let pending_calls = derived.pending_work(&req.decision_id);

    let transcript = derived
        .message_tree
        .head_id
        .as_deref()
        .map(|h| derived.message_tree.path_to(h))
        .unwrap_or_default();

    Some(WorkerDecisionRequest {
        session_id: event.aggregate_id.clone(),
        decision_id: req.decision_id.clone(),
        agent_id: agent_id.clone(),
        identity: owner.clone(),
        trigger: materialize_client_trigger(
            req.trigger.clone(),
            &transcript,
            &derived.message_tree,
        ),
        state: derived.worker_state.clone(),
        calls: derived.calls.clone(),
        pending_calls,
        transcript,
        message_tree: derived.message_tree.clone(),
        ancestry: derived.ancestry.clone(),
        span: event.span.clone(),
        attempts: wd.tracking.retry.attempts,
        deadline: wd.tracking.deadline,
        turn_id: derived.turn_id.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::session::events::{NewMessage, Node};
    use crate::runtime::session::message::{Content, Role};

    fn msg(id: &str, role: Role, text: &str) -> Message {
        Message {
            id: id.to_string(),
            role,
            content: Some(Content::Text(text.to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    fn linear_tree(messages: &[Message]) -> MessageTree {
        let nodes: Vec<Node> = messages
            .iter()
            .enumerate()
            .map(|(i, m)| {
                Node::Message(NewMessage {
                    message: m.clone(),
                    parent_id: (i > 0).then(|| messages[i - 1].id.clone()),
                })
            })
            .collect();
        MessageTree {
            head_id: messages.last().map(|m| m.id.clone()),
            nodes,
        }
    }

    fn transcript_of(trigger: DecisionTrigger) -> (Vec<Message>, usize) {
        match trigger {
            DecisionTrigger::ClientTranscript { messages, new_from } => (messages, new_from),
            t => panic!("expected a client.messages trigger; got {t:?}"),
        }
    }

    #[test]
    fn materializes_a_client_message_onto_the_active_path() {
        let path = vec![
            msg("u1", Role::User, "hi"),
            msg("a1", Role::Assistant, "yo"),
        ];
        let tree = linear_tree(&path);

        let (messages, new_from) = transcript_of(materialize_client_trigger(
            DecisionTrigger::ClientMessage {
                message: msg("u2", Role::User, "more"),
            },
            &path,
            &tree,
        ));

        assert_eq!(
            messages.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            vec!["u1", "a1", "u2"]
        );
        assert_eq!(new_from, 2);
    }

    #[test]
    fn annotates_an_appending_full_view() {
        let path = vec![
            msg("u1", Role::User, "hi"),
            msg("a1", Role::Assistant, "yo"),
        ];
        let tree = linear_tree(&path);
        let view = vec![
            msg("u1", Role::User, "hi"),
            msg("a1", Role::Assistant, "yo"),
            msg("u2", Role::User, "more"),
        ];

        let (_, new_from) = transcript_of(materialize_client_trigger(
            DecisionTrigger::ClientTranscript {
                messages: view,
                new_from: 0,
            },
            &path,
            &tree,
        ));
        assert_eq!(new_from, 2);
    }

    #[test]
    fn annotates_an_edit_at_its_divergence_point() {
        let path = vec![
            msg("u1", Role::User, "hi"),
            msg("a1", Role::Assistant, "yo"),
            msg("u2", Role::User, "more"),
        ];
        let tree = linear_tree(&path);
        // An edit of a1: fresh id, resent up to that point.
        let view = vec![msg("u1", Role::User, "hi"), msg("e1", Role::User, "edited")];

        let (_, new_from) = transcript_of(materialize_client_trigger(
            DecisionTrigger::ClientTranscript {
                messages: view,
                new_from: 0,
            },
            &path,
            &tree,
        ));
        assert_eq!(new_from, 1);
    }

    #[test]
    fn annotates_a_no_op_resend_as_all_recorded() {
        let path = vec![
            msg("u1", Role::User, "hi"),
            msg("a1", Role::Assistant, "yo"),
        ];
        let tree = linear_tree(&path);

        let (_, new_from) = transcript_of(materialize_client_trigger(
            DecisionTrigger::ClientTranscript {
                messages: path.clone(),
                new_from: 0,
            },
            &path,
            &tree,
        ));
        assert_eq!(new_from, 2, "empty news: nothing to write");
    }

    #[test]
    fn annotates_an_idless_view_as_all_new() {
        let path = vec![msg("u1", Role::User, "hi")];
        let tree = linear_tree(&path);
        let view = vec![msg("", Role::User, "hi"), msg("", Role::User, "more")];

        let (_, new_from) = transcript_of(materialize_client_trigger(
            DecisionTrigger::ClientTranscript {
                messages: view,
                new_from: 0,
            },
            &path,
            &tree,
        ));
        assert_eq!(new_from, 0);
    }

    #[test]
    fn passes_non_client_triggers_through() {
        let tree = MessageTree::default();
        let out = materialize_client_trigger(
            DecisionTrigger::ToolFinished {
                id: "tc-1".to_string(),
                ok: true,
                name: "t".to_string(),
                result: Some("r".to_string()),
                error: None,
            },
            &[],
            &tree,
        );
        assert!(matches!(out, DecisionTrigger::ToolFinished { .. }));
    }
}
