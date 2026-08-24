//! The engine's own tools. Each answer is a function of [`SessionState`], so a
//! replay gives the same answer. Definitions live in [`filter`].
//! An answer is a [`StoredResult`], the same as a connection's or a worker's.

use super::state::SessionStateAtNode;
use crate::connectors::filter;
use crate::plugins::{PluginBundle, Skill};
use crate::protocol::{ConnectorToolKind, StoredContent, StoredResult};

/// `None` hands the call to its target.
pub fn answer(
    at: SessionStateAtNode,
    kind: ConnectorToolKind,
    arguments: &str,
) -> Option<StoredResult> {
    match kind {
        ConnectorToolKind::Remote => None,
        ConnectorToolKind::Find => Some(find(at, arguments)),
        ConnectorToolKind::Call => Some(call(at, arguments)),
        // The bundle is not in state; the executor answers this one.
        ConnectorToolKind::Skill => None,
    }
}

/// BM25 over every tool the agent can reach.
fn find(at: SessionStateAtNode, arguments: &str) -> StoredResult {
    StoredResult::text(filter::find_answer(
        &at.searchable_tools(),
        &argument(arguments, "query"),
        at.resolve_agent_for()
            .map(|c| c.defer_settings())
            .unwrap_or_default()
            .max_matches,
    ))
}

/// A `call_tool` that gets here could not be routed.
fn call(at: SessionStateAtNode, arguments: &str) -> StoredResult {
    StoredResult::error(
        at.call_tool_fault(arguments)
            .unwrap_or_else(|| "the call could not be routed".to_string()),
    )
}

fn argument(arguments: &str, key: &str) -> String {
    serde_json::from_str::<serde_json::Value>(arguments)
        .ok()
        .and_then(|v| v.get(key)?.as_str().map(str::to_string))
        .unwrap_or_default()
}

/// The two halves of a `<plugin>:<skill>` name. Without a colon, the plugin
/// half is empty.
pub fn split_skill(named: &str) -> (&str, &str) {
    named.split_once(':').unwrap_or(("", named))
}

/// A skill body, or one of its files. Each fault lists what the model can ask
/// for.
pub fn skill_answer(
    at: SessionStateAtNode,
    bundle: Option<&PluginBundle>,
    arguments: &str,
) -> StoredResult {
    let named = argument(arguments, "name");
    let file = Some(argument(arguments, "file")).filter(|f| !f.is_empty());
    let Some(config) = at.resolve_agent_for() else {
        return StoredResult::error("this agent has no config on this branch");
    };
    let catalog = || {
        config
            .plugins
            .iter()
            .map(|p| p.id.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };

    let (plugin_id, skill_name) = split_skill(&named);
    if plugin_id.is_empty() {
        return StoredResult::error(format!(
            "`{named}` is not a skill name. Skills are named `<plugin>:<skill>`; this agent's \
             plugins: {}.",
            catalog()
        ));
    }
    let Some(plugin) = config.plugins.iter().find(|p| p.id == plugin_id) else {
        return StoredResult::error(format!(
            "no plugin `{plugin_id}` for this agent. Declared: {}.",
            catalog()
        ));
    };
    let Some(bundle) = bundle else {
        return StoredResult::error(format!(
            "plugin `{plugin_id}` has no data on this engine; re-apply the project"
        ));
    };
    let Some(skill) = bundle.skill(skill_name) else {
        let listed = plugin
            .skills
            .iter()
            .map(|s| format!("{}:{} — {}", plugin.id, s.name, s.description))
            .collect::<Vec<_>>()
            .join("\n");
        return StoredResult::error(format!(
            "no skill `{skill_name}` in plugin `{plugin_id}`. Its skills:\n{listed}"
        ));
    };

    match file {
        Some(path) => file_answer(skill, &named, &path),
        None => {
            let mut answer = format!("Skill {named} — {}\n\n{}", skill.description, skill.body);
            let listed = skill_files(skill);
            if !listed.is_empty() {
                answer.push_str(&format!(
                    "\n\nFiles (read one with `skill` and `file`):\n{listed}"
                ));
            }
            StoredResult::text(answer)
        }
    }
}

/// Text inlines; a binary answers with its blob.
fn file_answer(skill: &Skill, named: &str, path: &str) -> StoredResult {
    if let Some(content) = skill.files.get(path) {
        return StoredResult::text(content.clone());
    }
    match skill.binaries.get(path) {
        Some(binary) => StoredResult {
            content: vec![StoredContent::Blob {
                uri: binary.uri.clone(),
            }],
            ..Default::default()
        },
        None => StoredResult::error(format!(
            "no file `{path}` in skill `{named}`. Its files:\n{}",
            skill_files(skill)
        )),
    }
}

/// Text files by path, binaries with what they are.
fn skill_files(skill: &Skill) -> String {
    let text = skill.files.keys().map(|path| format!("- {path}"));
    let binaries = skill
        .binaries
        .iter()
        .map(|(path, b)| format!("- {path} ({}, {} bytes)", b.mime, b.size));
    text.chain(binaries).collect::<Vec<_>>().join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugins::{PluginSet, Skill};
    use crate::protocol::{AgentConfig, AgentPlugin, SkillMeta};
    use crate::runtime::session::events::{AgentConfigUpdated, EventPayload};
    use crate::session::state::{ApplyContext, SessionState};

    fn state_with_plugin() -> SessionState {
        let mut s = SessionState::new("sess-1".to_string());
        s.apply(
            &EventPayload::AgentConfigUpdated(AgentConfigUpdated {
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
                        skills: vec![SkillMeta {
                            name: "form-filling".to_string(),
                            description: "Fill out PDF forms.".to_string(),
                        }],
                        servers: vec![],
                        tools: None,
                        auth_failure: Default::default(),
                        approve: Default::default(),
                    }],
                    defer_tools: None,
                    announce_mcp: Default::default(),
                },
                anchor: None,
            }),
            &ApplyContext {
                occurred_at: chrono::Utc::now(),
                sequence: 1,
            },
        );
        s
    }

    fn bundles() -> PluginSet {
        [(
            "pdf".to_string(),
            PluginBundle {
                name: "pdf-tools".to_string(),
                skills: vec![Skill {
                    name: "form-filling".to_string(),
                    description: "Fill out PDF forms.".to_string(),
                    body: "Read references/FORMS.md first.".to_string(),
                    files: [("references/FORMS.md".to_string(), "field rules".to_string())].into(),
                    binaries: [(
                        "references/sample.pdf".to_string(),
                        crate::plugins::Binary {
                            uri: "blob://t1/1?mime=application/pdf&size=3".to_string(),
                            mime: "application/pdf".to_string(),
                            size: 3,
                        },
                    )]
                    .into(),
                }],
                ..Default::default()
            },
        )]
        .into()
    }

    #[test]
    fn a_skill_answers_with_its_body_and_file_listing() {
        let answer = skill_answer(
            state_with_plugin().at(None),
            bundles().get("pdf"),
            r#"{"name":"pdf:form-filling"}"#,
        );
        assert!(!answer.is_error, "expected a result; got {answer:?}");
        let text = answer.as_text();
        assert!(text.contains("Read references/FORMS.md first."), "{text}");
        assert!(text.contains("- references/FORMS.md"), "{text}");
        assert!(
            text.contains("- references/sample.pdf (application/pdf, 3 bytes)"),
            "a binary is listed with what it is: {text}"
        );
    }

    #[test]
    fn a_file_read_answers_with_the_content() {
        let answer = skill_answer(
            state_with_plugin().at(None),
            bundles().get("pdf"),
            r#"{"name":"pdf:form-filling","file":"references/FORMS.md"}"#,
        );
        assert_eq!(answer, StoredResult::text("field rules"));
    }

    #[test]
    fn a_binary_answers_with_its_blob() {
        let answer = skill_answer(
            state_with_plugin().at(None),
            bundles().get("pdf"),
            r#"{"name":"pdf:form-filling","file":"references/sample.pdf"}"#,
        );
        assert_eq!(
            answer.content,
            vec![StoredContent::Blob {
                uri: "blob://t1/1?mime=application/pdf&size=3".to_string()
            }],
            "the bytes reach the model the way a connection's do"
        );
        assert!(!answer.is_error);
    }

    #[test]
    fn a_wrong_name_answers_with_the_directory() {
        let cases = [
            (r#"{"name":"form-filling"}"#, "not a skill name"),
            (r#"{"name":"typo:form-filling"}"#, "no plugin `typo`"),
            (
                r#"{"name":"pdf:typo"}"#,
                "pdf:form-filling — Fill out PDF forms.",
            ),
            (
                r#"{"name":"pdf:form-filling","file":"typo.md"}"#,
                "- references/FORMS.md",
            ),
        ];
        for (arguments, expected) in cases {
            let answer = skill_answer(
                state_with_plugin().at(None),
                bundles().get("pdf"),
                arguments,
            );
            assert!(
                answer.is_error,
                "{arguments}: expected a fault; got {answer:?}"
            );
            let text = answer.as_text();
            assert!(text.contains(expected), "{arguments}: {text}");
        }
    }
}

#[cfg(test)]
mod split_tests {
    use super::split_skill;

    #[test]
    fn one_split_rule_for_routing_and_answering() {
        assert_eq!(split_skill("pdf:form-filling"), ("pdf", "form-filling"));
        assert_eq!(split_skill("form-filling"), ("", "form-filling"));
        assert_eq!(
            split_skill("a:b:c"),
            ("a", "b:c"),
            "only the first colon splits"
        );
        assert_eq!(split_skill(""), ("", ""));
    }
}
