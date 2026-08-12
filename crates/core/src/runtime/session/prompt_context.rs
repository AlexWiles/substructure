//! What the engine adds to a prompt that the worker did not write.
//!
//! Merged at dispatch, beside the connector tools, into the stored prompt — so
//! a retry and a replay carry what the first attempt carried.
//!
//! Every source is pure over [`SessionState`]. One that needs the network
//! fetches it as its own effect and reads the settled result here, the way a
//! connector sync settles before its tools are merged.
//!
//! The source decides what to say and whether the session has heard it. This
//! module decides where it lands and whether this request already holds it.

use crate::protocol::{Announce, Content, Message, Role};
use crate::runtime::session::state::SessionState;

/// Where context lands. Each rung falls to the next when it cannot be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// The system prefix, until a call commits it. The Anthropic wire hoists a
    /// system message into the prefix whatever its position, so a later one
    /// would rewrite what is cached.
    System,
    /// A block on the last user message. Falls to [`Placement::Own`] otherwise:
    /// text on an assistant turn reads as the model's own words.
    Inline,
    /// A message of its own.
    Own,
}

/// One piece of engine-derived context, and where it goes.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptContext {
    /// Unique to what it says. Keeps a retry from merging the same text twice.
    pub id: String,
    pub placement: Placement,
    pub content: String,
}

/// Everything the engine owes this call, in a fixed order — two replays that
/// disagreed would send two different prompts.
pub fn owed(state: &SessionState, leaf: Option<&str>, call_id: &str) -> Vec<PromptContext> {
    announce_servers(state, leaf, call_id)
}

/// The connections this path has not announced yet.
///
/// Once means once in the prompt. The record is the context id on an earlier
/// call of this path, so a fork that never held the server announces it.
fn announce_servers(state: &SessionState, leaf: Option<&str>, call_id: &str) -> Vec<PromptContext> {
    let Some(config) = state.resolve_agent_for(leaf) else {
        return Vec::new();
    };
    if config.announce_mcp == Announce::Never {
        return Vec::new();
    }
    let said = state.context_ids_on_path(leaf, call_id);
    config
        .mcp
        .iter()
        .map(|server| (server, format!("mcp:{}", server.id)))
        .filter(|(_, id)| !said.contains(id))
        .filter_map(|(server, id)| {
            Some(PromptContext {
                id,
                placement: Placement::System,
                // The `mcp_server` key is the label, so there is no prose here
                // to go stale or to need configuring.
                content: state.connection_summary(&server.id, leaf)?,
            })
        })
        .collect()
}

/// Merge what this call does not already hold. `applied` is the record, not the
/// text, so it survives a change of wording.
pub fn merge(
    prompt: &mut Vec<Message>,
    applied: &mut Vec<String>,
    call_id: &str,
    system_open: bool,
    owed: Vec<PromptContext>,
) {
    for c in owed {
        if applied.contains(&c.id) {
            continue;
        }
        let placement = resolve(c.placement, prompt, system_open);
        place(prompt, call_id, &c, placement);
        applied.push(c.id);
    }
}

/// The first rung this one can use.
fn resolve(wanted: Placement, prompt: &[Message], system_open: bool) -> Placement {
    match wanted {
        Placement::System if system_open => Placement::System,
        Placement::System | Placement::Inline => {
            let host = prompt.last().is_some_and(|m| m.role == Role::User);
            if host {
                Placement::Inline
            } else {
                Placement::Own
            }
        }
        Placement::Own => Placement::Own,
    }
}

fn place(prompt: &mut Vec<Message>, call_id: &str, c: &PromptContext, placement: Placement) {
    match placement {
        Placement::System => {
            let at = prompt
                .iter()
                .position(|m| m.role != Role::System)
                .unwrap_or(prompt.len());
            prompt.insert(at, message(call_id, &c.id, Role::System, &c.content));
        }
        Placement::Inline => {
            let Some(last) = prompt.last_mut() else {
                return;
            };
            let head = match &last.content {
                Some(Content::Text(t)) => t.clone(),
                _ => String::new(),
            };
            last.content = Some(Content::Text(format!("{head}\n\n{}", c.content)));
        }
        Placement::Own => prompt.push(message(call_id, &c.id, Role::User, &c.content)),
    }
}

/// An id a replay mints the same way.
fn message(call_id: &str, context: &str, role: Role, content: &str) -> Message {
    Message {
        id: format!("{call_id}-ctx-{context}"),
        role,
        content: Some(Content::Text(content.to_string())),
        tool_calls: Vec::new(),
        tool_call_id: None,
        name: None,
        reasoning: None,
    }
}
