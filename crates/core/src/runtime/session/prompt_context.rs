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

/// A source of engine-derived context, pure over [`SessionState`].
pub type Contributor = fn(&SessionState, Option<&str>, &str) -> Vec<PromptContext>;

/// Every source, in a fixed order — two replays that disagreed would send two
/// different prompts. A new source is one entry here.
const CONTRIBUTORS: &[Contributor] = &[plugin_catalog, announce_servers];

/// Everything the engine owes this call.
pub fn owed(state: &SessionState, leaf: Option<&str>, call_id: &str) -> Vec<PromptContext> {
    CONTRIBUTORS
        .iter()
        .flat_map(|contribute| contribute(state, leaf, call_id))
        .collect()
}

/// The connections this path has not announced yet — a plugin's servers
/// included, since they behave as `mcp` entries.
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
    state
        .servers_for(&config)
        .into_iter()
        .map(|server| (format!("mcp:{}", server.id), server))
        .filter(|(id, _)| !said.contains(id))
        .filter_map(|(id, server)| {
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

/// One catalog entry per plugin the path has not seen: what it is, its
/// skills, and that the `skill` tool is the way in. Each plugin is its own
/// context, so one declared mid-session introduces itself through the ladder
/// without touching what an earlier call cached.
fn plugin_catalog(state: &SessionState, leaf: Option<&str>, call_id: &str) -> Vec<PromptContext> {
    let Some(config) = state.resolve_agent_for(leaf) else {
        return Vec::new();
    };
    let said = state.context_ids_on_path(leaf, call_id);
    config
        .plugins
        .iter()
        .map(|plugin| (plugin, format!("plugin:{}", plugin.id)))
        .filter(|(_, id)| !said.contains(id))
        .map(|(plugin, id)| {
            let skills: Vec<SkillListing> = plugin
                .skills
                .iter()
                .map(|s| SkillListing {
                    name: format!("{}:{}", plugin.id, s.name),
                    description: s.description.clone(),
                })
                .collect();
            let content = serde_json::to_string(&PluginListing {
                plugin: &plugin.id,
                about: (!plugin.description.is_empty()).then_some(&plugin.description),
                skills,
                usage: "load a skill with the `skill` tool",
            })
            .unwrap_or_default();
            PromptContext {
                id,
                placement: Placement::System,
                content,
            }
        })
        .collect()
}

/// Struct, not a `json!` map: key order is the wire contract's stability.
#[derive(serde::Serialize)]
struct PluginListing<'a> {
    plugin: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    about: Option<&'a String>,
    skills: Vec<SkillListing>,
    usage: &'static str,
}

#[derive(serde::Serialize)]
struct SkillListing {
    name: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    description: String,
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
            last.content = Some(match last.content.take() {
                // Non-text parts stay; the context lands as one more part.
                Some(Content::Parts(mut parts)) => {
                    parts.push(crate::protocol::StoredContent::Text {
                        text: c.content.clone(),
                    });
                    Content::Parts(parts)
                }
                Some(Content::Text(head)) => Content::Text(format!("{head}\n\n{}", c.content)),
                None => Content::Text(format!("\n\n{}", c.content)),
            });
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectors::RemoteTool;
    use crate::protocol::{AgentConfig, AgentPlugin, RetryPolicy, SkillMeta, StoredContent};
    use crate::runtime::session::events::{
        AgentConfigUpdated, ConnectorSyncCompleted, ConnectorSyncRequested, EventPayload,
    };
    use crate::session::state::ApplyContext;

    fn apply(state: &mut SessionState, seq: u64, payload: EventPayload) {
        state.apply(
            &payload,
            &ApplyContext {
                occurred_at: chrono::Utc::now(),
                sequence: seq,
            },
        );
    }

    fn plugin_state(settled: bool) -> SessionState {
        let mut s = SessionState::new("sess-1".to_string());
        apply(
            &mut s,
            1,
            EventPayload::AgentConfigUpdated(AgentConfigUpdated {
                config: AgentConfig {
                    llm: None,
                    model: "m1".to_string(),
                    system: None,
                    effort: None,
                    retry: None,
                    tools: vec![],
                    sub_agents: vec![],
                    mcp: vec![],
                    plugins: vec![AgentPlugin {
                        id: "pdf".to_string(),
                        description: "PDF work.".to_string(),
                        skills: vec![
                            SkillMeta {
                                name: "form-filling".to_string(),
                                description: "long ".repeat(500),
                            },
                            SkillMeta {
                                name: "text-extraction".to_string(),
                                description: "also long ".repeat(500),
                            },
                        ],
                        servers: vec!["pdf-renderer".to_string()],
                        tools: None,
                        auth_failure: Default::default(),
                        approve: Default::default(),
                    }],
                    defer_tools: None,
                    announce_mcp: Default::default(),
                },
                anchor: None,
            }),
        );
        if settled {
            apply(
                &mut s,
                3,
                EventPayload::ConnectorSyncRequested(ConnectorSyncRequested {
                    id: "pdf-renderer".to_string(),
                    attempt: 0,
                    retry: RetryPolicy::no_retry(),
                }),
            );
            apply(
                &mut s,
                4,
                EventPayload::ConnectorSyncCompleted(Box::new(ConnectorSyncCompleted {
                    id: "pdf-renderer".to_string(),
                    prefix: Some("pdf-renderer".to_string()),
                    tools: vec![RemoteTool {
                        name: "fill_form".to_string(),
                        description: String::new(),
                        input: None,
                        output: None,
                        annotations: Default::default(),
                    }],
                    instructions: None,
                })),
            );
        }
        s
    }

    fn owed_ids(state: &SessionState) -> Vec<String> {
        owed(state, None, "call-1")
            .into_iter()
            .map(|c| c.id)
            .collect()
    }

    #[test]
    fn a_plugins_settled_server_is_announced_like_any_connection() {
        let state = plugin_state(true);
        assert_eq!(owed_ids(&state), ["plugin:pdf", "mcp:pdf-renderer"]);
        let owed = owed(&state, None, "call-1");
        assert!(
            owed[1]
                .content
                .starts_with("{\"mcp_server\":\"pdf-renderer\""),
            "{}",
            owed[1].content
        );
    }

    #[test]
    fn a_plugins_unsettled_server_is_not_announced() {
        let state = plugin_state(false);
        assert_eq!(
            owed_ids(&state),
            ["plugin:pdf"],
            "the catalog speaks; the server has not answered yet"
        );
    }

    #[test]
    fn the_catalog_carries_every_skill_description() {
        let state = plugin_state(false);
        let catalog = &owed(&state, None, "call-1")[0].content;
        assert!(catalog.matches("long long").count() >= 2, "{catalog}");
        assert!(catalog.contains("pdf:text-extraction"), "{catalog}");
    }

    #[test]
    fn inline_context_keeps_image_parts() {
        let mut prompt = vec![Message {
            id: "m1".into(),
            role: Role::User,
            content: Some(Content::Parts(vec![
                StoredContent::Text {
                    text: "look".into(),
                },
                StoredContent::Blob {
                    uri: "blob://t/ab?mime=image%2Fpng&size=1".into(),
                },
            ])),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
            reasoning: None,
        }];
        let c = PromptContext {
            id: "ctx1".into(),
            placement: Placement::Inline,
            content: "server up".into(),
        };
        place(&mut prompt, "c1", &c, Placement::Inline);
        match prompt[0].content.as_ref().unwrap() {
            Content::Parts(parts) => {
                assert_eq!(parts.len(), 3);
                assert!(matches!(&parts[1], StoredContent::Blob { .. }));
                assert!(matches!(&parts[2], StoredContent::Text { text } if text == "server up"));
            }
            _ => panic!("expected parts"),
        }
    }
}
