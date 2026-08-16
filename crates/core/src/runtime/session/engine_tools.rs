//! The engine's own tools. Each answer is a function of [`SessionState`], so a
//! replay gives the same answer. Definitions live in [`filter`].

use super::state::{LocalAnswer, SessionState};
use crate::connectors::filter;
use crate::plugins::PluginBundle;
use crate::protocol::ConnectorToolKind;

/// `None` hands the call to its target.
pub fn answer(
    state: &SessionState,
    kind: ConnectorToolKind,
    leaf: Option<&str>,
    arguments: &str,
) -> Option<LocalAnswer> {
    match kind {
        ConnectorToolKind::Remote => None,
        ConnectorToolKind::Find => Some(find(state, leaf, arguments)),
        ConnectorToolKind::Call => Some(call(state, leaf, arguments)),
        // The bundle is not in state; the executor answers this one.
        ConnectorToolKind::Skill => None,
    }
}

/// BM25 over every tool the agent can reach.
fn find(state: &SessionState, leaf: Option<&str>, arguments: &str) -> LocalAnswer {
    LocalAnswer::Result(filter::find_answer(
        &state.searchable_tools(leaf),
        &argument(arguments, "query"),
        state
            .resolve_agent_for(leaf)
            .map(|c| c.defer_settings())
            .unwrap_or_default()
            .max_matches,
    ))
}

/// A `call_tool` that gets here could not be routed.
fn call(state: &SessionState, leaf: Option<&str>, arguments: &str) -> LocalAnswer {
    LocalAnswer::Error(
        state
            .call_tool_fault(arguments, leaf)
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
    state: &SessionState,
    bundle: Option<&PluginBundle>,
    leaf: Option<&str>,
    arguments: &str,
) -> LocalAnswer {
    let named = argument(arguments, "name");
    let file = Some(argument(arguments, "file")).filter(|f| !f.is_empty());
    let Some(config) = state.resolve_agent_for(leaf) else {
        return LocalAnswer::Error("this agent has no config on this branch".to_string());
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
        return LocalAnswer::Error(format!(
            "`{named}` is not a skill name. Skills are named `<plugin>:<skill>`; this agent's \
             plugins: {}.",
            catalog()
        ));
    }
    let Some(plugin) = config.plugins.iter().find(|p| p.id == plugin_id) else {
        return LocalAnswer::Error(format!(
            "no plugin `{plugin_id}` for this agent. Declared: {}.",
            catalog()
        ));
    };
    let Some(bundle) = bundle else {
        return LocalAnswer::Error(format!(
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
        return LocalAnswer::Error(format!(
            "no skill `{skill_name}` in plugin `{plugin_id}`. Its skills:\n{listed}"
        ));
    };

    match file {
        Some(path) => match skill.files.get(&path) {
            Some(content) => LocalAnswer::Result(content.clone()),
            None => LocalAnswer::Error(format!(
                "no file `{path}` in skill `{named}`. Its files:\n{}",
                skill_files(skill)
            )),
        },
        None => {
            let mut answer = format!("Skill {named} — {}\n\n{}", skill.description, skill.body);
            if !skill.files.is_empty() {
                answer.push_str(&format!(
                    "\n\nFiles (read one with `skill` and `file`):\n{}",
                    skill_files(skill)
                ));
            }
            LocalAnswer::Result(answer)
        }
    }
}

fn skill_files(skill: &crate::plugins::Skill) -> String {
    skill
        .files
        .keys()
        .map(|k| format!("- {k}"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugins::{PluginSet, Skill};
    use crate::protocol::{AgentConfig, AgentPlugin, SkillMeta};
    use crate::runtime::session::events::{AgentConfigUpdated, EventPayload};
    use crate::session::state::ApplyContext;

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
                }],
                ..Default::default()
            },
        )]
        .into()
    }

    #[test]
    fn a_skill_answers_with_its_body_and_file_listing() {
        let answer = skill_answer(
            &state_with_plugin(),
            bundles().get("pdf"),
            None,
            r#"{"name":"pdf:form-filling"}"#,
        );
        let LocalAnswer::Result(text) = answer else {
            panic!("expected a result; got {answer:?}");
        };
        assert!(text.contains("Read references/FORMS.md first."), "{text}");
        assert!(text.contains("- references/FORMS.md"), "{text}");
    }

    #[test]
    fn a_file_read_answers_with_the_content() {
        let answer = skill_answer(
            &state_with_plugin(),
            bundles().get("pdf"),
            None,
            r#"{"name":"pdf:form-filling","file":"references/FORMS.md"}"#,
        );
        assert!(matches!(answer, LocalAnswer::Result(t) if t == "field rules"));
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
            let answer = skill_answer(&state_with_plugin(), bundles().get("pdf"), None, arguments);
            let LocalAnswer::Error(text) = answer else {
                panic!("{arguments}: expected a fault; got {answer:?}");
            };
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
