use crate::protocol::{
    AgentConfig, AgentTool, ConnectorTool, Content, DraftMessage, LlmTool, Role, SubAgent,
};

impl AgentTool {
    pub fn to_llm_tool(&self, default: bool) -> LlmTool {
        LlmTool {
            name: self.name.clone(),
            description: self.description.clone(),
            input: self.input.clone(),
            output: self.output.clone(),
            defer: self.defer.unwrap_or(default),
        }
    }
}

impl ConnectorTool {
    pub fn to_llm_tool(&self) -> LlmTool {
        LlmTool {
            name: self.name.clone(),
            description: self.description.clone(),
            input: self.input.clone(),
            output: self.output.clone(),
            defer: self.defer,
        }
    }
}

impl SubAgent {
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
            defer: false,
        }
    }
}

impl AgentConfig {
    pub fn tools_as_llm(&self) -> Option<Vec<LlmTool>> {
        if self.tools.is_empty() && self.sub_agents.is_empty() {
            None
        } else {
            let mut tools: Vec<LlmTool> = self
                .tools
                .iter()
                .map(|t| t.to_llm_tool(self.defers_tools()))
                .collect();
            tools.extend(self.sub_agents.iter().map(SubAgent::to_llm_tool));
            Some(tools)
        }
    }

    pub fn with_client_tools(&self, client_tools: &[AgentTool]) -> Option<AgentConfig> {
        let mut tools = self.tools.clone();
        for t in client_tools {
            let taken = tools.iter().any(|e| e.name == t.name)
                || self.sub_agents.iter().any(|s| s.id == t.name);
            if !taken {
                tools.push(t.clone());
            }
        }
        if tools.len() == self.tools.len() {
            None
        } else {
            Some(AgentConfig {
                tools,
                ..self.clone()
            })
        }
    }

    pub fn tool(&self, tool_name: &str) -> Option<&AgentTool> {
        self.tools.iter().find(|t| t.name == tool_name)
    }

    pub fn sub_agent(&self, name: &str) -> Option<&SubAgent> {
        self.sub_agents.iter().find(|s| s.id == name)
    }

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
                    reasoning: None,
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
    use crate::protocol::Handler;

    fn sub(id: &str, description: &str) -> SubAgent {
        SubAgent {
            id: id.to_string(),
            description: description.to_string(),
        }
    }

    fn function_tool(name: &str, handler: Option<Handler>) -> AgentTool {
        AgentTool {
            name: name.to_string(),
            description: String::new(),
            input: None,
            output: None,
            handler,
            defer: None,
        }
    }

    fn config(tools: Vec<AgentTool>, sub_agents: Vec<SubAgent>) -> AgentConfig {
        AgentConfig {
            llm: Some("claude".to_string()),
            model: "m".to_string(),
            system: None,
            retry: None,
            tools,
            sub_agents,
            mcp: Vec::new(),
            defer_tools: None,
            mcp_announce: Default::default(),
            plugins: Vec::new(),
            effort: None,
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
    fn with_client_tools_appends_new_tools_and_is_idempotent() {
        let cfg = config(vec![function_tool("get_time", None)], vec![]);
        let client = [function_tool("get_tz", Some(Handler::Client))];
        let merged = cfg
            .with_client_tools(&client)
            .expect("a new tool ⇒ a rewrite");
        assert_eq!(
            merged
                .tools
                .iter()
                .map(|t| t.name.clone())
                .collect::<Vec<_>>(),
            ["get_time", "get_tz"],
            "client tools appended after config tools"
        );
        assert_eq!(
            merged.tool("get_tz").and_then(|t| t.handler),
            Some(Handler::Client)
        );
        assert!(
            merged.with_client_tools(&client).is_none(),
            "already present ⇒ no rewrite"
        );
        assert!(
            cfg.with_client_tools(&[]).is_none(),
            "nothing declared ⇒ no rewrite"
        );
    }

    #[test]
    fn with_client_tools_never_shadows_a_taken_name() {
        let cfg = config(
            vec![function_tool("confirm", Some(Handler::Client))],
            vec![sub("researcher", "")],
        );
        let client = [
            function_tool("confirm", Some(Handler::Client)),
            function_tool("researcher", Some(Handler::Client)),
        ];
        assert!(
            cfg.with_client_tools(&client).is_none(),
            "names taken by a tool or sub-agent are skipped"
        );
    }

    #[test]
    fn lookups_keep_the_two_namespaces_distinct() {
        let cfg = config(
            vec![function_tool("confirm", Some(Handler::Client))],
            vec![sub("researcher", "")],
        );
        assert_eq!(
            cfg.tool("confirm").and_then(|t| t.handler),
            Some(Handler::Client)
        );
        assert!(
            cfg.tool("researcher").is_none(),
            "a sub-agent is not a function tool"
        );
        assert!(cfg.sub_agent("researcher").is_some());
        assert!(cfg.sub_agent("confirm").is_none());
    }
}
