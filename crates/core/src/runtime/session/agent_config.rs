//! Behavior over the agent config — the static half of an agent's identity,
//! declared as a document the engine reads. The types live in [`crate::protocol`].
//!
//! Unlike opaque worker `state`, this is a typed document the engine interprets:
//! it lets the engine propose the `client.messages` LLM request (model + tools +
//! system) that the worker would otherwise have to hand-author. It is versioned
//! and anchored exactly like worker state (see [`AgentVersion`](super::state::AgentVersion)).

use crate::protocol::{AgentConfig, AgentTool, Content, DraftMessage, LlmTool, Role, SubAgent};

impl AgentTool {
    /// The model-facing contract, with routing stripped.
    pub fn to_llm_tool(&self) -> LlmTool {
        LlmTool {
            name: self.name.clone(),
            description: self.description.clone(),
            input: self.input.clone(),
            output: self.output.clone(),
        }
    }
}

impl SubAgent {
    /// The model-facing delegation tool: the sub-agent's id as the tool name and a
    /// fixed single-`message` input schema.
    pub fn to_llm_tool(&self) -> LlmTool {
        let description = if self.description.is_empty() {
            format!("Delegate to {}", self.id)
        } else {
            self.description.clone()
        };
        LlmTool {
            name: self.id.clone(),
            description,
            input: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "message": { "type": "string", "description": "The message to send to the agent" }
                },
                "required": ["message"]
            })),
            output: None,
        }
    }
}

impl AgentConfig {
    /// The model-facing tool list: declared function tools plus one delegation tool
    /// per sub-agent. `None` when neither is declared. The two share the model's
    /// single tool namespace, so a name should not appear in both.
    pub fn tools_as_llm(&self) -> Option<Vec<LlmTool>> {
        if self.tools.is_empty() && self.sub_agents.is_empty() {
            None
        } else {
            let mut tools: Vec<LlmTool> = self.tools.iter().map(AgentTool::to_llm_tool).collect();
            tools.extend(self.sub_agents.iter().map(SubAgent::to_llm_tool));
            Some(tools)
        }
    }

    pub fn tool(&self, tool_name: &str) -> Option<&AgentTool> {
        self.tools.iter().find(|t| t.name == tool_name)
    }

    pub fn sub_agent(&self, name: &str) -> Option<&SubAgent> {
        self.sub_agents.iter().find(|s| s.id == name)
    }

    /// The prompt for a proposed call over `view`: the configured system message
    /// (if any) prepended to the view.
    pub fn prompt_for(&self, view: &[DraftMessage]) -> Vec<DraftMessage> {
        match &self.system {
            Some(system) => {
                let mut messages = Vec::with_capacity(view.len() + 1);
                messages.push(DraftMessage {
                    id: None,
                    role: Role::System,
                    content: Some(Content::Text(system.clone())),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                });
                messages.extend(view.iter().cloned());
                messages
            }
            None => view.to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::ToolHandler;

    fn sub(id: &str, description: &str) -> SubAgent {
        SubAgent {
            id: id.to_string(),
            description: description.to_string(),
        }
    }

    fn function_tool(name: &str, handler: Option<ToolHandler>) -> AgentTool {
        AgentTool {
            name: name.to_string(),
            description: String::new(),
            input: None,
            output: None,
            handler,
        }
    }

    fn config(tools: Vec<AgentTool>, sub_agents: Vec<SubAgent>) -> AgentConfig {
        AgentConfig {
            model: "m".to_string(),
            system: None,
            stream: false,
            retry: None,
            tools,
            sub_agents,
        }
    }

    #[test]
    fn no_tools_or_sub_agents_offers_nothing() {
        assert!(config(vec![], vec![]).tools_as_llm().is_none());
    }

    #[test]
    fn tools_as_llm_merges_function_tools_then_sub_agents() {
        let cfg = config(
            vec![function_tool("get_time", None)],
            vec![sub("researcher", "")],
        );
        let names: Vec<_> = cfg
            .tools_as_llm()
            .expect("declared ⇒ some")
            .iter()
            .map(|t| t.name.clone())
            .collect();
        assert_eq!(
            names,
            ["get_time", "researcher"],
            "function tools first, then sub-agents"
        );
    }

    #[test]
    fn sub_agent_tool_has_the_conventional_message_schema() {
        let tool = sub("researcher", "").to_llm_tool();
        assert_eq!(tool.name, "researcher", "the id is the tool name");
        assert_eq!(tool.description, "Delegate to researcher");
        let input = tool.input.expect("delegation input schema");
        assert_eq!(input["type"], "object");
        assert_eq!(input["required"], serde_json::json!(["message"]));
        assert_eq!(input["properties"]["message"]["type"], "string");
    }

    #[test]
    fn sub_agent_description_overrides_the_default() {
        assert_eq!(
            sub("researcher", "Find sources").to_llm_tool().description,
            "Find sources"
        );
    }

    #[test]
    fn lookups_keep_the_two_namespaces_distinct() {
        let cfg = config(
            vec![function_tool("confirm", Some(ToolHandler::Client))],
            vec![sub("researcher", "")],
        );
        assert_eq!(
            cfg.tool("confirm").and_then(|t| t.handler.clone()),
            Some(ToolHandler::Client)
        );
        assert!(
            cfg.tool("researcher").is_none(),
            "a sub-agent is not a function tool"
        );
        assert!(cfg.sub_agent("researcher").is_some());
        assert!(cfg.sub_agent("confirm").is_none());
    }
}
