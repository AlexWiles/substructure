use std::collections::HashSet;

use super::message::Message;

/// A message the reconcile pass would record as a new node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannedWrite {
    /// Index into the input `messages` slice.
    pub index: usize,
    /// A known id re-recorded past the news boundary (fresh node on the branch),
    /// as opposed to a genuinely new message.
    pub rerecord: bool,
}

/// The single interpreter of "what would recording this list write against a
/// tree that already holds `known` ids".
///
/// A known-id prefix advances the parent cursor and writes nothing; the first
/// id-less or unknown message begins the news; the news doesn't stop — every
/// message from there on is written, and a known id among them is re-recorded
/// as a fresh node so the new branch stays a chain instead of grafting back
/// onto the old one. The writes are therefore the contiguous suffix that starts
/// at the news boundary.
pub fn plan_reconcile(known: &HashSet<&str>, messages: &[Message]) -> Vec<PlannedWrite> {
    let mut writes = Vec::new();
    let mut news_started = false;
    for (index, msg) in messages.iter().enumerate() {
        let is_known = !msg.id.is_empty() && known.contains(msg.id.as_str());
        if is_known && !news_started {
            continue;
        }
        news_started = true;
        writes.push(PlannedWrite {
            index,
            rerecord: is_known,
        });
    }
    writes
}

/// Index where the news begins: the first message reconcile would write, or the
/// list length when nothing is new.
pub fn news_start(known: &HashSet<&str>, messages: &[Message]) -> usize {
    plan_reconcile(known, messages)
        .first()
        .map(|w| w.index)
        .unwrap_or(messages.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::session::message::{Content, Role};

    fn msg(id: &str) -> Message {
        Message {
            id: id.to_string(),
            role: Role::User,
            content: Some(Content::Text(String::new())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    fn known_of<'a>(ids: &'a [&'a str]) -> HashSet<&'a str> {
        ids.iter().copied().collect()
    }

    #[test]
    fn append_writes_only_the_tail() {
        let known = known_of(&["u1", "a1"]);
        let view = [msg("u1"), msg("a1"), msg("u2")];
        let plan = plan_reconcile(&known, &view);
        assert_eq!(
            plan,
            vec![PlannedWrite {
                index: 2,
                rerecord: false
            }]
        );
        assert_eq!(news_start(&known, &view), 2);
    }

    #[test]
    fn no_op_resend_writes_nothing() {
        let known = known_of(&["u1", "a1"]);
        let view = [msg("u1"), msg("a1")];
        assert!(plan_reconcile(&known, &view).is_empty());
        assert_eq!(news_start(&known, &view), 2);
    }

    #[test]
    fn news_does_not_stop_and_re_records_known_ids() {
        let known = known_of(&["u1", "a1", "u2"]);
        // Edit of a1: fresh id at index 1, then the known u2 trails the edit.
        let view = [msg("u1"), msg("e1"), msg("u2")];
        let plan = plan_reconcile(&known, &view);
        assert_eq!(
            plan,
            vec![
                PlannedWrite {
                    index: 1,
                    rerecord: false
                },
                PlannedWrite {
                    index: 2,
                    rerecord: true
                },
            ]
        );
    }

    #[test]
    fn idless_head_is_all_new() {
        let known = known_of(&["u1"]);
        let view = [msg(""), msg("")];
        assert_eq!(news_start(&known, &view), 0);
    }
}
