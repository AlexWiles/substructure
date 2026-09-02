use std::num::NonZeroUsize;

use crate::connectors::RemoteTool;
use crate::protocol::{
    Approve, BoundServer, ConnectionPath, ConnectorProtocol, ConnectorTool, ConnectorToolKind,
    DeferToolsStrategy, LlmTool, McpTools, SpawnMode, Subagent, SubagentMode,
    SubagentToolsStrategy, SUBAGENT_WAIT,
};

const SEPARATOR: &str = "__";

pub const MAX_NAME: usize = 64;

#[derive(Debug, Clone, PartialEq)]
pub struct Resolution {
    pub tools: Vec<ConnectorTool>,
    pub offered: usize,
    pub unannotated: usize,
    pub unmatched_include: Vec<String>,
    pub oversized: Vec<String>,
}

impl Resolution {
    pub fn of(tools: Vec<ConnectorTool>) -> Self {
        Self {
            tools,
            offered: 0,
            unannotated: 0,
            unmatched_include: Vec::new(),
            oversized: Vec::new(),
        }
    }
}

pub fn resolve(
    connector: &BoundServer,
    offered: &[RemoteTool],
    prefix: Option<&str>,
    defer: bool,
) -> Resolution {
    let filter = connector.tools.clone().unwrap_or_default();

    let unannotated = offered
        .iter()
        .filter(|tool| matches!(capability_verdict(&filter, tool), Verdict::Unannotated))
        .count();
    let kept: Vec<&RemoteTool> = offered
        .iter()
        .filter(|tool| passes(&filter, tool))
        .collect();

    let unmatched_include = filter
        .include
        .iter()
        .filter(|g| !offered.iter().any(|t| glob_match(g, &t.name)))
        .cloned()
        .collect();

    let mut tools = Vec::with_capacity(kept.len());
    let mut oversized = Vec::new();
    for tool in kept {
        match expand(
            &connector.path,
            tool,
            prefix,
            defer,
            approves(connector.approve, tool),
        ) {
            Some(expanded) => tools.push(expanded),
            None => oversized.push(tool.name.clone()),
        }
    }

    Resolution {
        tools,
        offered: offered.len(),
        unannotated,
        unmatched_include,
        oversized,
    }
}

pub fn approves(policy: Approve, tool: &RemoteTool) -> bool {
    match policy {
        Approve::Never => false,
        Approve::Always => true,
        Approve::Destructive => tool.annotations.destructive == Some(true),
    }
}

pub fn defers(connector: &BoundServer, default: bool) -> bool {
    connector
        .tools
        .as_ref()
        .and_then(|t| t.defer)
        .unwrap_or(default)
}

pub fn callable<'a>(connector: &BoundServer, offered: &'a [RemoteTool]) -> Vec<&'a RemoteTool> {
    let filter = connector.tools.clone().unwrap_or_default();
    offered
        .iter()
        .filter(|tool| passes(&filter, tool))
        .collect()
}

pub use crate::protocol::{CALL_TOOL, SKILL, SUBAGENT, TOOL_SEARCH};

const SESSION_DESCRIPTION: &str =
    "Session id of an earlier call to this agent, to continue that conversation";

const MODE_DESCRIPTION: &str = "How the call returns [default: blocking]. `blocking` waits and \
     answers with the result. `detached` answers at once with the session id; the result \
     arrives later as a message.";

const DETACHED_NOTE: &str = " Runs detached: the call answers at once with the session id, and \
     the result arrives as a later message.";

pub fn subagent_input() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "message": { "type": "string", "description": "The message to send to the agent" },
            "session": { "type": "string", "description": SESSION_DESCRIPTION }
        },
        "required": ["message"]
    })
}

fn subagent_input_with_modes(modes: &[SpawnMode]) -> serde_json::Value {
    let mut input = subagent_input();
    if modes.len() > 1 {
        input["properties"]["mode"] = serde_json::json!({
            "type": "string",
            "enum": modes,
            "description": MODE_DESCRIPTION
        });
    }
    input
}

pub fn subagent_tools(
    subagents: &[Subagent],
    default_defer: bool,
    strategy: SubagentToolsStrategy,
    wait: bool,
) -> Resolution {
    let can_detach = subagents
        .iter()
        .any(|s| s.resolved_mode().offered().contains(&SpawnMode::Detached));
    let wait = wait && can_detach;
    let mut r = match strategy {
        SubagentToolsStrategy::PerAgent => per_agent_subagent_tools(subagents, default_defer),
        SubagentToolsStrategy::Single => Resolution::of(match subagents.is_empty() {
            true => Vec::new(),
            false => vec![subagent_switch(subagents, default_defer)],
        }),
    };
    if wait {
        r.tools.push(wait_tool(default_defer));
    }
    r
}

fn wait_tool(defer: bool) -> ConnectorTool {
    ConnectorTool {
        defer,
        via: ConnectorProtocol::Agent,
        ..engine_tool(
            SUBAGENT_WAIT,
            "Wait for the result of an earlier detached subagent call. Pass the `session` that \
             call answered with. Answers at once when the result is already in, and holds \
             until the child finishes when it is not."
                .to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "session": { "type": "string", "description": "The detached call's session id" }
                },
                "required": ["session"]
            }),
            ConnectorToolKind::Subagent,
        )
    }
}

fn subagent_switch(subagents: &[Subagent], defer: bool) -> ConnectorTool {
    let listed = subagents
        .iter()
        .map(|s| {
            let mode = match s.resolved_mode() {
                SubagentMode::Any => "",
                SubagentMode::Blocking => " [always blocking]",
                SubagentMode::Detached => " [always detached]",
            };
            match s.description.is_empty() {
                true => format!("- {}{mode}", s.id),
                false => format!("- {} — {}{mode}", s.id, s.description),
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let ids: Vec<&str> = subagents.iter().map(|s| s.id.as_str()).collect();
    let modes: Vec<SpawnMode> = [SpawnMode::Blocking, SpawnMode::Detached]
        .into_iter()
        .filter(|m| {
            subagents
                .iter()
                .any(|s| s.resolved_mode().offered().contains(m))
        })
        .collect();
    let mut input = subagent_input_with_modes(&modes);
    input["properties"]["agent"] = serde_json::json!({
        "type": "string",
        "enum": ids,
        "description": "The agent to delegate to."
    });
    input["required"] = serde_json::json!(["agent", "message"]);
    ConnectorTool {
        defer,
        via: ConnectorProtocol::Agent,
        ..engine_tool(
            SUBAGENT,
            format!(
                "Delegate a turn to one of this agent's subagents. The call runs in the named \
                 agent's own session and answers with its result and that session's id. Pass \
                 the id back as `session` to continue where it left off. Agents:\n{listed}"
            ),
            input,
            ConnectorToolKind::Subagent,
        )
    }
}

fn per_agent_subagent_tools(subagents: &[Subagent], default_defer: bool) -> Resolution {
    let mut tools = Vec::with_capacity(subagents.len());
    let mut oversized = Vec::new();
    for sub in subagents {
        let name = sub.offered_name();
        let defer = sub.defers(default_defer);
        if !defer && name.len() > MAX_NAME {
            oversized.push(name);
            continue;
        }
        let mut description = match sub.description.is_empty() {
            true => format!("Delegate to {}", sub.id),
            false => sub.description.clone(),
        };
        let modes = sub.resolved_mode().offered();
        if modes == [SpawnMode::Detached] {
            description.push_str(DETACHED_NOTE);
        }
        tools.push(ConnectorTool {
            name,
            description,
            input: Some(subagent_input_with_modes(modes)),
            output: None,
            connector: Some(ConnectionPath::Agent(sub.id.clone())),
            via: ConnectorProtocol::Agent,
            remote_name: sub.id.clone(),
            kind: ConnectorToolKind::Subagent,
            defer,
            approve: false,
        });
    }
    Resolution {
        oversized,
        ..Resolution::of(tools)
    }
}

pub fn skill_tool() -> ConnectorTool {
    engine_tool(
        SKILL,
        "Load a skill's instructions. The plugin catalog lists each plugin and its skills; \
         name one as `<plugin>:<skill>` and follow what comes back. Pass `file` to read one \
         of the files the instructions reference."
            .to_string(),
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The skill, as `<plugin>:<skill>` from the catalog."
                },
                "file": {
                    "type": "string",
                    "description": "A file the skill's instructions reference, by its listed path."
                }
            },
            "required": ["name"]
        }),
        ConnectorToolKind::Skill,
    )
}

pub fn search_tools(search: DeferToolsStrategy) -> Vec<ConnectorTool> {
    match search {
        DeferToolsStrategy::Search => vec![find_tool(), call_tool()],
    }
}

pub(crate) fn engine_tool(
    name: &str,
    description: String,
    input: serde_json::Value,
    kind: ConnectorToolKind,
) -> ConnectorTool {
    ConnectorTool {
        name: name.to_string(),
        description,
        input: Some(input),
        output: None,
        connector: None,
        via: ConnectorProtocol::Mcp,
        remote_name: String::new(),
        kind,
        defer: false,
        approve: false,
    }
}

fn find_tool() -> ConnectorTool {
    engine_tool(
        TOOL_SEARCH,
        "Search the tools this agent can reach. They are not listed up front. Answers with the \
         name, the description, and the input schema of each match, best match first. Any word \
         can match, so a plain description of the task searches well. A connector's name is a \
         word in every one of its tools, so adding it keeps the search to that connection. An \
         empty query matches every tool: start there when you do not know what is available."
            .to_string(),
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Words to match against tool names and descriptions."
                }
            },
            "required": ["query"]
        }),
        ConnectorToolKind::Find,
    )
}

fn call_tool() -> ConnectorTool {
    engine_tool(
        CALL_TOOL,
        format!(
            "Run one tool this agent can reach. Take `name` from `{TOOL_SEARCH}` exactly as it \
             was given, and pass that tool's own arguments."
        ),
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The tool's name, exactly as a search gave it."
                },
                "arguments": {
                    "type": "object",
                    "description": "The arguments that tool's own input schema declares."
                }
            },
            "required": ["name"]
        }),
        ConnectorToolKind::Call,
    )
}

const K1: f64 = 1.2;
const B: f64 = 0.75;

pub fn find<'a>(tools: &'a [LlmTool], query: &str) -> Vec<&'a LlmTool> {
    let terms = words(query);
    if terms.is_empty() || tools.is_empty() {
        return tools.iter().collect();
    }

    let docs: Vec<Vec<String>> = tools
        .iter()
        .map(|tool| words(&format!("{} {}", tool.name, tool.description)))
        .collect();
    let total = docs.len() as f64;
    let average_length = docs.iter().map(Vec::len).sum::<usize>() as f64 / total;
    let idf: Vec<f64> = terms
        .iter()
        .map(|term| {
            let carrying = docs.iter().filter(|doc| frequency(doc, term) > 0.0).count() as f64;
            (1.0 + (total - carrying + 0.5) / (carrying + 0.5)).ln()
        })
        .collect();

    let mut scored: Vec<(f64, usize, &LlmTool)> = Vec::new();
    for (order, (tool, doc)) in tools.iter().zip(&docs).enumerate() {
        let length = doc.len() as f64 / average_length;
        let score: f64 = terms
            .iter()
            .zip(&idf)
            .map(|(term, idf)| {
                let f = frequency(doc, term);
                idf * (f * (K1 + 1.0)) / (f + K1 * (1.0 - B + B * length))
            })
            .sum();
        if score > 0.0 {
            scored.push((score, order, tool));
        }
    }
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    scored.into_iter().map(|(_, _, tool)| tool).collect()
}

fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn frequency(doc: &[String], term: &str) -> f64 {
    doc.iter().filter(|word| word.starts_with(term)).count() as f64
}

pub fn find_answer(
    tools: &[LlmTool],
    query: &str,
    max_matches: NonZeroUsize,
    unavailable: &[ConnectionPath],
) -> String {
    let matched = find(tools, query);
    let shown: Vec<serde_json::Value> = matched
        .iter()
        .take(max_matches.get())
        .map(|tool| {
            serde_json::json!({
                "name": tool.name,
                "description": tool.description,
                "input": tool.input,
            })
        })
        .collect();
    let mut answer = serde_json::json!({
        "tools": shown,
        "matched": matched.len(),
        "searched": tools.len(),
        "call_with": CALL_TOOL,
    });
    if matched.is_empty() {
        answer["note"] = serde_json::json!(crate::copy::NOTHING_MATCHED);
    } else if matched.len() > shown.len() {
        answer["note"] =
            serde_json::json!(crate::copy::matches_truncated(shown.len(), matched.len()));
    }
    if !unavailable.is_empty() {
        answer["unavailable"] = serde_json::json!(unavailable);
    }
    answer.to_string()
}

fn passes(filter: &McpTools, tool: &RemoteTool) -> bool {
    matches!(capability_verdict(filter, tool), Verdict::Pass)
        && (filter.include.is_empty() || filter.include.iter().any(|g| glob_match(g, &tool.name)))
        && !filter.exclude.iter().any(|g| glob_match(g, &tool.name))
}

#[derive(Debug, Clone, PartialEq)]
pub struct Merged {
    pub tools: Vec<ConnectorTool>,
    pub collisions: Vec<String>,
}

pub fn merge<'a>(
    resolutions: impl IntoIterator<Item = Resolution>,
    taken: impl IntoIterator<Item = &'a str>,
) -> Merged {
    let taken: Vec<&str> = taken.into_iter().collect();
    let all: Vec<ConnectorTool> = resolutions.into_iter().flat_map(|r| r.tools).collect();

    let mut collisions = Vec::new();
    let mut tools = Vec::with_capacity(all.len());
    for tool in &all {
        let claimed_by_config = taken.contains(&tool.name.as_str());
        let claimed_by_sibling = all.iter().filter(|t| t.name == tool.name).count() > 1;
        if claimed_by_config || claimed_by_sibling {
            if !collisions.contains(&tool.name) {
                collisions.push(tool.name.clone());
            }
        } else {
            tools.push(tool.clone());
        }
    }
    Merged { tools, collisions }
}

pub fn qualified_name(prefix: Option<&str>, remote_name: &str) -> String {
    match prefix {
        Some(prefix) => format!("{}{SEPARATOR}{remote_name}", name_prefix(prefix)),
        None => remote_name.to_string(),
    }
}

fn expand(
    path: &ConnectionPath,
    tool: &RemoteTool,
    prefix: Option<&str>,
    defer: bool,
    approve: bool,
) -> Option<ConnectorTool> {
    let name = qualified_name(prefix, &tool.name);
    if !defer && name.len() > MAX_NAME {
        return None;
    }
    Some(ConnectorTool {
        name,
        description: tool.description.clone(),
        input: tool.input.clone(),
        output: tool.output.clone(),
        connector: Some(path.clone()),
        via: ConnectorProtocol::Mcp,
        remote_name: tool.name.clone(),
        kind: ConnectorToolKind::Remote,
        defer,
        approve,
    })
}

fn name_prefix(connector_id: &str) -> String {
    connector_id
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

enum Verdict {
    Pass,
    Fail,
    Unannotated,
}

fn capability_verdict(filter: &McpTools, tool: &RemoteTool) -> Verdict {
    let a = &tool.annotations;
    let checks = [
        (filter.read_only, a.read_only),
        (filter.non_destructive, a.destructive.map(|d| !d)),
        (filter.idempotent, a.idempotent),
    ];

    let mut unannotated = false;
    for (want, got) in checks {
        let Some(want) = want else { continue };
        match got {
            Some(got) if got == want => {}
            Some(_) => return Verdict::Fail,
            None => unannotated = true,
        }
    }
    if unannotated {
        Verdict::Unannotated
    } else {
        Verdict::Pass
    }
}

fn glob_match(pattern: &str, value: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let v: Vec<char> = value.chars().collect();
    let (mut pi, mut vi) = (0usize, 0usize);
    let (mut star, mut resume) = (None, 0usize);

    while vi < v.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == v[vi]) {
            pi += 1;
            vi += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            resume = vi;
            pi += 1;
        } else if let Some(s) = star {
            pi = s + 1;
            resume += 1;
            vi = resume;
        } else {
            return false;
        }
    }
    p[pi..].iter().all(|c| *c == '*')
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectors::registry::ConnectionPath;
    use crate::connectors::ToolAnnotations;

    fn tool(name: &str, annotations: ToolAnnotations) -> RemoteTool {
        RemoteTool {
            name: name.to_string(),
            title: None,
            description: format!("does {name}"),
            input: Some(serde_json::json!({ "type": "object" })),
            output: None,
            annotations,
        }
    }

    fn read_only(name: &str) -> RemoteTool {
        tool(
            name,
            ToolAnnotations {
                read_only: Some(true),
                destructive: Some(false),
                ..Default::default()
            },
        )
    }

    fn writer(name: &str) -> RemoteTool {
        tool(
            name,
            ToolAnnotations {
                read_only: Some(false),
                destructive: Some(true),
                ..Default::default()
            },
        )
    }

    fn bare(name: &str) -> RemoteTool {
        tool(name, ToolAnnotations::default())
    }

    fn connector(id: &str, tools: Option<McpTools>) -> BoundServer {
        BoundServer {
            path: ConnectionPath::Mcp(id.to_string()),
            tools,
            auth_failure: Default::default(),
            tool_sync_failure: Default::default(),
            approve: Default::default(),
        }
    }

    fn asking(id: &str, approve: Approve) -> BoundServer {
        BoundServer {
            approve,
            ..connector(id, None)
        }
    }

    fn names(r: &Resolution) -> Vec<&str> {
        r.tools.iter().map(|t| t.name.as_str()).collect()
    }

    #[test]
    fn no_filter_takes_everything_the_connection_offers() {
        let offered = [read_only("search"), writer("delete")];
        let r = resolve(&connector("sentry", None), &offered, Some("sentry"), false);
        assert_eq!(names(&r), vec!["sentry__search", "sentry__delete"]);
        assert_eq!(r.offered, 2);
    }

    #[test]
    fn tools_are_prefixed_so_two_connections_cannot_collide() {
        let offered = [read_only("search")];
        let a = resolve(&connector("sentry", None), &offered, Some("sentry"), false);
        let b = resolve(&connector("github", None), &offered, Some("github"), false);
        assert_eq!(names(&a), vec!["sentry__search"]);
        assert_eq!(names(&b), vec!["github__search"]);
    }

    #[test]
    fn the_remote_name_is_kept_for_the_executor_to_call() {
        let r = resolve(
            &connector("sentry", None),
            &[read_only("search_issues")],
            Some("sentry"),
            false,
        );
        assert_eq!(r.tools[0].remote_name, "search_issues");
        assert_eq!(
            r.tools[0].connector,
            Some(ConnectionPath::Mcp("sentry".into()))
        );
        assert_eq!(r.tools[0].name, "sentry__search_issues");
    }

    #[test]
    fn read_only_keeps_readers_and_drops_writers() {
        let offered = [read_only("search"), writer("delete")];
        let filter = McpTools {
            read_only: Some(true),
            ..Default::default()
        };
        let r = resolve(
            &connector("sentry", Some(filter)),
            &offered,
            Some("sentry"),
            false,
        );
        assert_eq!(names(&r), vec!["sentry__search"]);
        assert_eq!(r.unannotated, 0, "both tools were annotated");
    }

    #[test]
    fn an_unannotated_server_yields_nothing_under_a_predicate_and_says_so() {
        let offered = [bare("search"), bare("delete")];
        let filter = McpTools {
            read_only: Some(true),
            ..Default::default()
        };
        let r = resolve(
            &connector("custom", Some(filter)),
            &offered,
            Some("custom"),
            false,
        );
        assert!(
            r.tools.is_empty(),
            "silence is not a promise of being read-only"
        );
        assert_eq!(r.unannotated, 2, "the drop is counted, not hidden");
    }

    #[test]
    fn non_destructive_reads_the_inverted_hint() {
        let offered = [read_only("search"), writer("delete")];
        let filter = McpTools {
            non_destructive: Some(true),
            ..Default::default()
        };
        let r = resolve(
            &connector("sentry", Some(filter)),
            &offered,
            Some("sentry"),
            false,
        );
        assert_eq!(names(&r), vec!["sentry__search"]);
    }

    #[test]
    fn include_globs_match_the_remote_name_not_the_prefixed_one() {
        let offered = [
            read_only("search_issues"),
            read_only("get_issue"),
            read_only("update_issue"),
        ];
        let filter = McpTools {
            include: vec!["search_*".to_string(), "get_*".to_string()],
            ..Default::default()
        };
        let r = resolve(
            &connector("sentry", Some(filter)),
            &offered,
            Some("sentry"),
            false,
        );
        assert_eq!(
            names(&r),
            vec!["sentry__search_issues", "sentry__get_issue"]
        );
        assert!(r.unmatched_include.is_empty());
    }

    #[test]
    fn exclude_wins_over_include() {
        let offered = [read_only("search_issues"), read_only("search_secrets")];
        let filter = McpTools {
            include: vec!["search_*".to_string()],
            exclude: vec!["*_secrets".to_string()],
            ..Default::default()
        };
        let r = resolve(
            &connector("sentry", Some(filter)),
            &offered,
            Some("sentry"),
            false,
        );
        assert_eq!(names(&r), vec!["sentry__search_issues"]);
    }

    #[test]
    fn an_include_that_matches_nothing_is_reported() {
        let offered = [read_only("search_issues")];
        let filter = McpTools {
            include: vec!["search_*".to_string(), "listProjects".to_string()],
            ..Default::default()
        };
        let r = resolve(
            &connector("sentry", Some(filter)),
            &offered,
            Some("sentry"),
            false,
        );
        assert_eq!(
            r.unmatched_include,
            vec!["listProjects".to_string()],
            "a rename upstream must not look like a working filter"
        );
    }

    #[test]
    fn a_filter_never_widens_what_the_connection_offered() {
        let offered = [read_only("search")];
        let filter = McpTools {
            include: vec!["*".to_string(), "anything_else".to_string()],
            ..Default::default()
        };
        let r = resolve(
            &connector("sentry", Some(filter)),
            &offered,
            Some("sentry"),
            false,
        );
        assert_eq!(names(&r), vec!["sentry__search"]);
    }

    #[test]
    fn the_separator_is_doubled_so_the_name_parses_back() {
        let r = resolve(
            &connector("sentry", None),
            &[read_only("search_issues")],
            Some("sentry"),
            false,
        );
        let name = &r.tools[0].name;
        assert_eq!(name, "sentry__search_issues");
        assert_eq!(
            name.split_once("__"),
            Some(("sentry", "search_issues")),
            "a single underscore would be ambiguous on both sides"
        );
    }

    #[test]
    fn a_name_too_long_for_a_provider_is_dropped_and_reported() {
        let long = "a".repeat(60);
        let offered = [read_only(&long), read_only("search")];
        let r = resolve(&connector("sentry", None), &offered, Some("sentry"), false);
        assert_eq!(
            names(&r),
            vec!["sentry__search"],
            "the over-long name is left out rather than truncated into a collision"
        );
        assert_eq!(r.oversized, vec![long]);
    }

    #[test]
    fn a_connection_can_offer_its_tools_unprefixed() {
        let offered = [read_only("search_issues")];
        let r = resolve(&connector("sentry", None), &offered, None, false);
        assert_eq!(names(&r), vec!["search_issues"]);
        assert_eq!(
            r.tools[0].connector,
            Some(ConnectionPath::Mcp("sentry".into())),
            "provenance survives even when the name does not carry it"
        );
        assert_eq!(r.tools[0].remote_name, "search_issues");
    }

    #[test]
    fn unprefixed_connections_that_collide_both_lose_the_name() {
        let a = resolve(
            &connector("sentry", None),
            &[read_only("search")],
            None,
            false,
        );
        let b = resolve(
            &connector("github", None),
            &[read_only("search")],
            None,
            false,
        );
        let merged = merge([a, b], []);
        assert!(
            merged.tools.is_empty(),
            "picking one of two would route the model to an arbitrary connection"
        );
        assert_eq!(merged.collisions, vec!["search".to_string()]);
    }

    #[test]
    fn a_declared_tool_keeps_its_name_against_a_connector() {
        let r = resolve(
            &connector("sentry", None),
            &[read_only("search")],
            None,
            false,
        );
        let merged = merge([r], ["search"]);
        assert!(merged.tools.is_empty(), "the declared tool wins its name");
        assert_eq!(merged.collisions, vec!["search".to_string()]);
    }

    #[test]
    fn prefixing_is_what_lets_two_connections_share_a_tool_name() {
        let a = resolve(
            &connector("sentry", None),
            &[read_only("search")],
            Some("sentry"),
            false,
        );
        let b = resolve(
            &connector("github", None),
            &[read_only("search")],
            Some("github"),
            false,
        );
        let merged = merge([a, b], ["search"]);
        assert_eq!(
            merged
                .tools
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>(),
            vec!["sentry__search", "github__search"],
            "both survive, and neither shadows the declared `search`"
        );
        assert!(merged.collisions.is_empty());
    }

    #[test]
    fn merging_keeps_everything_when_nothing_overlaps() {
        let a = resolve(
            &connector("sentry", None),
            &[read_only("issues")],
            None,
            false,
        );
        let b = resolve(
            &connector("github", None),
            &[read_only("repos")],
            None,
            false,
        );
        let merged = merge([a, b], ["get_time"]);
        assert_eq!(
            merged
                .tools
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>(),
            vec!["issues", "repos"]
        );
    }

    #[test]
    fn globs_handle_stars_questions_and_backtracking() {
        assert!(glob_match("*", "anything"));
        assert!(glob_match("search_*", "search_issues"));
        assert!(glob_match("*_issues", "search_issues"));
        assert!(glob_match("*_*", "search_issues"));
        assert!(glob_match("get_issue", "get_issue"));
        assert!(glob_match("get_issu?", "get_issue"));
        assert!(glob_match("a*b*c", "axxbyyc"));
        assert!(!glob_match("search_*", "get_issue"));
        assert!(!glob_match("get_issue", "get_issues"));
        assert!(!glob_match("a*b*c", "axxbyy"));
        assert!(glob_match("a*", "a"));
        assert!(!glob_match("a?", "a"));
    }

    #[test]
    fn nothing_asks_until_a_connection_says_to() {
        let offered = [read_only("search"), writer("delete")];
        let r = resolve(&connector("sentry", None), &offered, Some("sentry"), false);
        assert!(
            r.tools.iter().all(|t| !t.approve),
            "nothing asks by default"
        );
    }

    #[test]
    fn destructive_asks_about_the_writer_and_not_the_reader() {
        let offered = [read_only("search"), writer("delete")];
        let r = resolve(
            &asking("sentry", Approve::Destructive),
            &offered,
            Some("sentry"),
            false,
        );
        assert_eq!(names(&r), vec!["sentry__search", "sentry__delete"]);
        assert!(!r.tools[0].approve, "a read-only tool destroys nothing");
        assert!(r.tools[1].approve);
    }

    #[test]
    fn a_tool_the_server_says_nothing_about_is_not_one_it_called_destructive() {
        let r = resolve(
            &asking("custom", Approve::Destructive),
            &[bare("run")],
            Some("custom"),
            false,
        );
        assert!(!r.tools[0].approve, "silence is not a claim");
        let all = resolve(
            &asking("custom", Approve::Always),
            &[bare("run")],
            Some("custom"),
            false,
        );
        assert!(all.tools[0].approve);
    }

    #[test]
    fn a_read_only_tool_that_says_nothing_else_asks_nothing() {
        let tool = tool(
            "search",
            crate::connectors::ToolAnnotations {
                read_only: Some(true),
                ..Default::default()
            },
        );
        let r = resolve(
            &asking("sentry", Approve::Destructive),
            &[tool],
            Some("sentry"),
            false,
        );
        assert!(!r.tools[0].approve);
    }

    #[test]
    fn always_asks_whatever_the_server_says_about_itself() {
        let offered = [read_only("search"), writer("delete")];
        let r = resolve(
            &asking("sentry", Approve::Always),
            &offered,
            Some("sentry"),
            false,
        );
        assert!(r.tools.iter().all(|t| t.approve));
    }

    #[test]
    fn a_deferred_tool_still_asks() {
        let r = resolve(
            &asking("sentry", Approve::Destructive),
            &[writer("delete")],
            Some("sentry"),
            true,
        );
        assert!(
            r.tools[0].approve,
            "how a tool reaches the model says nothing about what it does"
        );
        assert!(
            search_tools(DeferToolsStrategy::Search)
                .iter()
                .all(|t| !t.approve),
            "the engine's own tools reach nothing on their own"
        );
    }

    fn llm(name: &str, description: &str) -> LlmTool {
        LlmTool {
            name: name.to_string(),
            description: description.to_string(),
            input: Some(serde_json::json!({ "type": "object" })),
            output: None,
            defer: false,
        }
    }

    fn found(tools: &[&LlmTool]) -> Vec<String> {
        tools.iter().map(|t| t.name.clone()).collect()
    }

    #[test]
    fn no_defer_setting_offers_every_tool_as_before() {
        let offered = [read_only("search"), writer("delete")];
        let r = resolve(&connector("sentry", None), &offered, Some("sentry"), false);
        assert_eq!(names(&r), vec!["sentry__search", "sentry__delete"]);
        assert!(
            r.tools.iter().all(|t| !t.defer),
            "the default is the behaviour that predates the setting"
        );
    }

    #[test]
    fn a_searched_connection_keeps_its_tools_and_marks_them_deferred() {
        let offered = [read_only("search"), writer("delete")];
        let r = resolve(&connector("sentry", None), &offered, Some("sentry"), true);
        assert_eq!(
            names(&r),
            vec!["sentry__search", "sentry__delete"],
            "the engine still holds them; only the request leaves them out"
        );
        assert!(r.tools.iter().all(|t| t.defer));
    }

    #[test]
    fn a_deferred_name_has_no_length_limit() {
        let long = "a".repeat(60);
        let offered = [read_only(&long)];
        let listed = resolve(&connector("sentry", None), &offered, Some("sentry"), false);
        assert_eq!(
            listed.oversized,
            vec![long.clone()],
            "a name in the request must fit what a provider accepts"
        );
        let deferred = resolve(&connector("sentry", None), &offered, Some("sentry"), true);
        assert!(
            deferred.oversized.is_empty() && deferred.tools.len() == 1,
            "a name that never reaches the request has nothing to fit"
        );
    }

    #[test]
    fn the_agent_default_applies_where_a_connection_says_nothing() {
        let bare = connector("sentry", None);
        assert!(!defers(&bare, false));
        assert!(defers(&bare, true));

        let listed = connector(
            "sentry",
            Some(McpTools {
                defer: Some(false),
                ..Default::default()
            }),
        );
        assert!(!defers(&listed, true), "the connection overrides the agent");
    }

    #[test]
    fn one_set_of_tools_serves_the_whole_agent() {
        let tools = search_tools(DeferToolsStrategy::Search);
        assert_eq!(
            tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
            vec![TOOL_SEARCH, CALL_TOOL]
        );
        assert!(
            !tools[0].description.contains("sentry"),
            "no connection reaches a definition, so adding one cannot rewrite it"
        );
        assert!(tools.iter().all(|t| t.connector.is_none()));
    }

    #[test]
    fn a_search_answer_and_a_call_use_one_word_for_one_thing() {
        let call = &search_tools(DeferToolsStrategy::Search)[1];
        let properties = call.input.as_ref().unwrap()["properties"].clone();
        assert_eq!(
            call.input.as_ref().unwrap()["required"],
            serde_json::json!(["name"])
        );
        assert!(properties.get("name").is_some());
        assert!(properties.get("arguments").is_some());
        assert!(properties.get("connector").is_none());
    }

    #[test]
    fn find_matches_any_word_and_ranks_by_how_many() {
        let tools = [
            llm("list_projects", "does list_projects"),
            llm("search_issues", "does search_issues"),
        ];
        let matched = |q: &str| found(&find(&tools, q));
        assert_eq!(matched("issues"), vec!["search_issues"]);
        assert_eq!(matched("ISSUES"), vec!["search_issues"], "case is ignored");
        assert_eq!(
            matched("does search_issues"),
            vec!["search_issues", "list_projects"],
            "the description is searched too, and the better match leads"
        );
        assert_eq!(matched("").len(), 2, "an empty query lists everything");
    }

    #[test]
    fn a_sentence_finds_the_tool_and_puts_it_first() {
        let tools = [llm("list_projects", ""), llm("search_issues", "")];
        assert_eq!(
            found(&find(&tools, "find all the open issues")),
            vec!["search_issues"],
            "`issues` matched; the other words matched nothing and cost nothing"
        );
    }

    #[test]
    fn a_word_in_every_tool_does_not_beat_a_word_in_one() {
        let tools = [
            llm("issue_list", ""),
            llm("issue_get", ""),
            llm("issue_resolve", ""),
            llm("issue_search", ""),
        ];
        assert_eq!(
            found(&find(&tools, "issue resolve"))[0],
            "issue_resolve",
            "`issue` is in each of the four, so `resolve` decides"
        );
    }

    #[test]
    fn a_term_matches_a_longer_word_that_starts_with_it() {
        let tools = [llm("search_issues", "")];
        assert_eq!(
            find(&tools, "issue").len(),
            1,
            "a model writes the singular"
        );
    }

    #[test]
    fn ties_keep_the_order_of_the_list() {
        let tools = [llm("b_issues", ""), llm("a_issues", "")];
        assert_eq!(
            found(&find(&tools, "issues")),
            vec!["b_issues", "a_issues"],
            "a replay must answer the same, so a tie is never broken by chance"
        );
    }

    fn cap() -> NonZeroUsize {
        crate::protocol::DeferTools::default().max_matches
    }

    #[test]
    fn an_empty_query_answers_with_every_tool() {
        let tools = [
            llm("sentry__search_issues", "Search the issues."),
            llm("github__create_pr", "Open a pull request."),
        ];
        let answer: serde_json::Value =
            serde_json::from_str(&find_answer(&tools, "", cap(), &[])).expect("json");
        assert_eq!(answer["matched"], 2, "no query is not no match");
        assert_eq!(
            answer["tools"][0]["input"]["type"], "object",
            "the only door carries the schema, or a name is a dead end"
        );
    }

    #[test]
    fn a_search_that_matches_nothing_gives_the_next_move() {
        let tools = [llm("search_issues", "")];
        let answer: serde_json::Value =
            serde_json::from_str(&find_answer(&tools, "kubernetes helm", cap(), &[]))
                .expect("json");
        assert_eq!(answer["matched"], 0);
        assert!(
            answer["note"].as_str().unwrap().contains("empty query"),
            "an empty answer must give the model a move it can make"
        );
    }

    #[test]
    fn a_find_answers_up_to_the_cap_and_says_how_many_matched() {
        let tools: Vec<LlmTool> = (0..15)
            .map(|i| llm(&format!("issue_tool_{i}"), ""))
            .collect();
        let answer: serde_json::Value =
            serde_json::from_str(&find_answer(&tools, "issue", cap(), &[])).expect("json");
        assert_eq!(answer["tools"].as_array().unwrap().len(), cap().get());
        assert_eq!(answer["matched"], 15, "the count is of every match");
        assert!(answer["note"].is_string());
    }

    #[test]
    fn a_match_reads_as_a_tool_definition() {
        let tools = [llm("sentry__search_issues", "Search the issues.")];
        let answer: serde_json::Value =
            serde_json::from_str(&find_answer(&tools, "issues", cap(), &[])).expect("json");
        assert_eq!(answer["tools"][0]["name"], "sentry__search_issues");
        assert_eq!(answer["tools"][0]["input"]["type"], "object");
        assert_eq!(answer["call_with"], CALL_TOOL);
    }

    #[test]
    fn search_cannot_reach_a_tool_the_filter_removed() {
        let offered = [read_only("search_issues"), read_only("search_secrets")];
        let c = connector(
            "sentry",
            Some(McpTools {
                exclude: vec!["*_secrets".to_string()],
                defer: Some(true),
                ..Default::default()
            }),
        );
        let r = resolve(&c, &offered, Some("sentry"), true);
        assert_eq!(
            names(&r),
            vec!["sentry__search_issues"],
            "deferral re-presents what the filter kept; it never reaches past it"
        );
    }

    fn subagent(id: &str, defer: Option<bool>, prefix: Option<bool>) -> Subagent {
        Subagent {
            id: id.to_string(),
            description: String::new(),
            defer,
            prefix,
            mode: None,
        }
    }

    #[test]
    fn a_subagent_becomes_a_subagent_tool_with_the_message_schema() {
        let r = subagent_tools(
            &[subagent("researcher", None, None)],
            false,
            SubagentToolsStrategy::PerAgent,
            false,
        );
        let tool = &r.tools[0];
        assert_eq!(tool.name, "researcher");
        assert_eq!(tool.description, "Delegate to researcher");
        assert_eq!(tool.kind, ConnectorToolKind::Subagent);
        assert_eq!(tool.via, ConnectorProtocol::Agent);
        assert_eq!(
            tool.connector,
            Some(ConnectionPath::Agent("researcher".into()))
        );
        assert_eq!(tool.remote_name, "researcher");
        assert!(!tool.defer && !tool.approve);
        let input = tool.input.as_ref().expect("subagent input schema");
        assert_eq!(input["required"], serde_json::json!(["message"]));
        assert_eq!(input["properties"]["message"]["type"], "string");
    }

    #[test]
    fn a_subagent_description_overrides_the_default() {
        let mut sub = subagent("researcher", None, None);
        sub.description = "Find sources".to_string();
        assert_eq!(
            subagent_tools(&[sub], false, SubagentToolsStrategy::PerAgent, false).tools[0]
                .description,
            "Find sources"
        );
    }

    #[test]
    fn prefix_offers_the_tool_under_the_agent_prefix() {
        let r = subagent_tools(
            &[subagent("researcher", None, Some(true))],
            false,
            SubagentToolsStrategy::PerAgent,
            false,
        );
        assert_eq!(r.tools[0].name, "agent__researcher");
        assert_eq!(
            r.tools[0].remote_name, "researcher",
            "the spawn still keys on the agent id"
        );
    }

    #[test]
    fn a_subagent_defers_by_the_agent_default_and_overrides_it() {
        let subs = [
            subagent("a", None, None),
            subagent("b", Some(false), None),
            subagent("c", Some(true), None),
        ];
        let defaulted: Vec<bool> =
            subagent_tools(&subs, true, SubagentToolsStrategy::PerAgent, false)
                .tools
                .iter()
                .map(|t| t.defer)
                .collect();
        assert_eq!(defaulted, [true, false, true]);
        let bare: Vec<bool> = subagent_tools(&subs, false, SubagentToolsStrategy::PerAgent, false)
            .tools
            .iter()
            .map(|t| t.defer)
            .collect();
        assert_eq!(bare, [false, false, true]);
    }

    #[test]
    fn an_over_long_subagent_name_is_dropped_unless_it_defers() {
        let long = "a".repeat(70);
        let listed = subagent_tools(
            &[subagent(&long, None, None)],
            false,
            SubagentToolsStrategy::PerAgent,
            false,
        );
        assert!(listed.tools.is_empty());
        assert_eq!(listed.oversized, vec![long.clone()]);
        let deferred = subagent_tools(
            &[subagent(&long, Some(true), None)],
            false,
            SubagentToolsStrategy::PerAgent,
            false,
        );
        assert_eq!(deferred.tools.len(), 1);
        assert!(deferred.oversized.is_empty());
    }

    #[test]
    fn the_single_strategy_offers_one_tool_naming_every_agent() {
        let mut helper = subagent("helper", None, None);
        helper.description = "Does the work.".to_string();
        let r = subagent_tools(
            &[helper, subagent("scribe", None, None)],
            false,
            SubagentToolsStrategy::Single,
            false,
        );

        assert_eq!(r.tools.len(), 1, "one tool for all of them");
        let tool = &r.tools[0];
        assert_eq!(tool.name, SUBAGENT);
        assert_eq!(tool.kind, ConnectorToolKind::Subagent);
        assert!(
            tool.connector.is_none() && tool.remote_name.is_empty(),
            "the tool names no agent of its own; the call does"
        );
        assert_eq!(tool.via, ConnectorProtocol::Agent);
        assert!(
            tool.description.contains("- helper — Does the work.")
                && tool.description.contains("- scribe"),
            "the roster rides in the description: {}",
            tool.description
        );
        let input = tool.input.as_ref().expect("input schema");
        assert_eq!(
            input["properties"]["agent"]["enum"],
            serde_json::json!(["helper", "scribe"])
        );
        assert_eq!(input["required"], serde_json::json!(["agent", "message"]));
        assert_eq!(input["properties"]["session"]["type"], "string");
    }

    #[test]
    fn the_single_tool_defers_with_the_agent() {
        let subs = [subagent("helper", None, None)];
        assert!(subagent_tools(&subs, true, SubagentToolsStrategy::Single, false).tools[0].defer);
        assert!(!subagent_tools(&subs, false, SubagentToolsStrategy::Single, false).tools[0].defer);
    }

    #[test]
    fn a_pin_narrows_the_modes_a_tool_offers() {
        use crate::protocol::SubagentMode;
        let plain = subagent("plain", None, None);
        let mut detached = subagent("detached", None, None);
        detached.mode = Some(SubagentMode::Detached);
        let mut blocking = subagent("blocking", None, None);
        blocking.mode = Some(SubagentMode::Blocking);

        let r = subagent_tools(
            &[plain, detached, blocking],
            false,
            SubagentToolsStrategy::PerAgent,
            false,
        );
        let modes = |i: usize| r.tools[i].input.as_ref().unwrap()["properties"]["mode"].clone();
        assert_eq!(
            modes(0)["enum"],
            serde_json::json!(["blocking", "detached"]),
            "unpinned, the model picks"
        );
        assert!(modes(1).is_null(), "one possible mode is no choice at all");
        assert!(
            r.tools[1].description.contains("Runs detached"),
            "the pin rides in the description: {}",
            r.tools[1].description
        );
        assert!(
            modes(2).is_null(),
            "pinned blocking is the schema the tool always had"
        );
    }

    #[test]
    fn the_single_tool_offers_the_union_of_its_agents_modes() {
        use crate::protocol::SubagentMode;
        let mut detached = subagent("detached", None, None);
        detached.mode = Some(SubagentMode::Detached);
        let mut blocking = subagent("blocking", None, None);
        blocking.mode = Some(SubagentMode::Blocking);

        let r = subagent_tools(
            &[detached, blocking.clone()],
            false,
            SubagentToolsStrategy::Single,
            false,
        );
        assert!(
            r.tools[0].input.as_ref().unwrap()["properties"]["mode"].is_null(),
            "no agent grants a choice, so no field"
        );
        assert!(
            r.tools[0]
                .description
                .contains("- detached [always detached]"),
            "the roster says whose mode is whose: {}",
            r.tools[0].description
        );
        assert!(
            r.tools[0]
                .description
                .contains("- blocking [always blocking]"),
            "{}",
            r.tools[0].description
        );
        let plain = subagent_tools(
            &[subagent("plain", None, None)],
            false,
            SubagentToolsStrategy::Single,
            false,
        );
        assert!(
            !plain.tools[0].description.contains("- plain ["),
            "the default carries no annotation: {}",
            plain.tools[0].description
        );
        assert_eq!(
            plain.tools[0].input.as_ref().unwrap()["properties"]["mode"]["enum"],
            serde_json::json!(["blocking", "detached"])
        );

        let bare = subagent_tools(&[blocking], false, SubagentToolsStrategy::Single, false);
        assert!(
            bare.tools[0].input.as_ref().unwrap()["properties"]["mode"].is_null(),
            "an all-blocking roster keeps the schema it always had"
        );
    }

    #[test]
    fn the_wait_tool_rides_along_and_claims_the_wait_mode() {
        use crate::protocol::SubagentMode;
        let r = subagent_tools(
            &[subagent("plain", None, None)],
            false,
            SubagentToolsStrategy::PerAgent,
            true,
        );
        assert_eq!(names(&r), vec!["plain", SUBAGENT_WAIT]);
        assert_eq!(
            r.tools[0].input.as_ref().unwrap()["properties"]["mode"]["enum"],
            serde_json::json!(["blocking", "detached"]),
            "waiting lives on its own tool"
        );
        let wait = &r.tools[1];
        assert_eq!(wait.kind, ConnectorToolKind::Subagent);
        assert!(wait.connector.is_none() && wait.remote_name.is_empty());
        assert_eq!(
            wait.input.as_ref().unwrap()["required"],
            serde_json::json!(["session"])
        );

        let mut pinned = subagent("pinned", None, None);
        pinned.mode = Some(SubagentMode::Detached);
        let r = subagent_tools(&[pinned], false, SubagentToolsStrategy::PerAgent, true);
        assert!(
            r.tools[0].input.as_ref().unwrap()["properties"]["mode"].is_null(),
            "one possible mode is no choice at all"
        );
        assert!(
            r.tools[0].description.contains("Runs detached"),
            "the description says what the call will do: {}",
            r.tools[0].description
        );

        let mut blocking = subagent("blocking", None, None);
        blocking.mode = Some(SubagentMode::Blocking);
        let none = subagent_tools(&[blocking], false, SubagentToolsStrategy::PerAgent, true);
        assert_eq!(
            names(&none),
            vec!["blocking"],
            "nothing detaches, nothing waits"
        );
    }

    #[test]
    fn the_single_strategy_offers_nothing_without_subagents() {
        assert!(
            subagent_tools(&[], false, SubagentToolsStrategy::Single, false)
                .tools
                .is_empty()
        );
    }

    #[test]
    fn a_declared_tool_still_wins_its_name_against_a_search_tool() {
        let r = Resolution::of(search_tools(DeferToolsStrategy::Search));
        let merged = merge([r], [TOOL_SEARCH]);
        assert_eq!(
            merged
                .tools
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>(),
            vec![CALL_TOOL],
            "the config's own name wins, the same as against any connector tool"
        );
        assert_eq!(merged.collisions, vec![TOOL_SEARCH.to_string()]);
    }
}
