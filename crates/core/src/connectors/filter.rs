//! Turning what a connection offers into what the model sees.
//!
//! Two steps. [`resolve`] filters — capability predicates, then `include`, then
//! `exclude` — and every step only removes, so a filter can never widen what
//! the connection granted. [`discover`] then puts what survived in front of the
//! model.
//!
//! Only the first step is a boundary, which is why [`kept`] asks it.
//!
//! Nothing here is silent: a predicate that drops unannotated tools and an
//! `include` that matches nothing are both reported, because both usually mean
//! the far side changed under us.

use crate::connectors::RemoteTool;
use crate::protocol::{
    ConnectorProtocol, ConnectorTool, ConnectorToolKind, McpServer, McpTools, ToolDiscovery,
};

/// Separates the connector id from the tool's own name. Doubled because both
/// halves routinely contain single underscores, and a single separator cannot
/// be parsed back into its parts — the same reason other MCP clients settled on
/// `mcp__server__tool`.
const SEPARATOR: &str = "__";

/// Providers cap function names; OpenAI's limit is 64. An over-long name gets
/// the whole request rejected, so it is caught here rather than at call time.
const MAX_NAME: usize = 64;

/// What a connector resolved to, plus what was dropped getting there.
#[derive(Debug, Clone, PartialEq)]
pub struct Resolution {
    pub tools: Vec<ConnectorTool>,
    /// How many tools the connection offered before filtering.
    pub offered: usize,
    /// Tools a capability predicate dropped because they carry no annotation at
    /// all. A whole server landing here means it annotates nothing, not that it
    /// has nothing to offer.
    pub unannotated: usize,
    /// `include` globs that matched nothing — usually a rename upstream.
    pub unmatched_include: Vec<String>,
    /// Remote names whose prefixed form would exceed what a provider accepts.
    pub oversized: Vec<String>,
}

/// Expand one connector's offered tools into model-facing tools.
///
/// `prefix` is the connection's, not the agent's: `Some(id)` gives the model
/// `<id>__<remote name>`, so two connections offering a `search` cannot
/// collide; `None` offers the connection's own names, and [`merge`] is then
/// what keeps a collision from shadowing anything.
///
/// Filters always match the *remote* name — the one the connection's own
/// documentation uses — whether or not the model sees a prefix.
pub fn resolve(connector: &McpServer, offered: &[RemoteTool], prefix: Option<&str>) -> Resolution {
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
        match expand(&connector.id, tool, prefix) {
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

/// How what the filter kept is put in front of the model.
///
/// A `Search` connection puts no tool of its own in the list. One pair of
/// search tools stands for every searched connection of the agent, and
/// [`search_tools`] gives it.
///
/// `oversized` clears under `Search`: `call_tool` carries a name as an
/// argument, where no name limit applies.
pub fn discover(connector: &McpServer, resolution: Resolution) -> Resolution {
    match discovery(connector) {
        ToolDiscovery::All => resolution,
        ToolDiscovery::Search => Resolution {
            tools: Vec::new(),
            oversized: Vec::new(),
            ..resolution
        },
    }
}

pub fn discovery(connector: &McpServer) -> ToolDiscovery {
    connector
        .tools
        .as_ref()
        .and_then(|t| t.discovery)
        .unwrap_or_default()
}

/// The tool a connection knows as `remote_name`, if the filter kept it.
pub fn kept<'a>(
    connector: &McpServer,
    offered: &'a [RemoteTool],
    remote_name: &str,
) -> Option<&'a RemoteTool> {
    let filter = connector.tools.clone().unwrap_or_default();
    offered
        .iter()
        .find(|tool| tool.name == remote_name && passes(&filter, tool))
}

/// Every tool the agent may reach, in offer order.
pub fn callable<'a>(connector: &McpServer, offered: &'a [RemoteTool]) -> Vec<&'a RemoteTool> {
    let filter = connector.tools.clone().unwrap_or_default();
    offered
        .iter()
        .filter(|tool| passes(&filter, tool))
        .collect()
}

pub const FIND_TOOLS: &str = "find_tools";
pub const CALL_TOOL: &str = "call_tool";

/// The pair the engine answers for every searched connection of one agent.
///
/// One pair, not one for each connection: a model that does not know which
/// connection holds a tool would otherwise search each one in turn.
///
/// Both definitions are constant. Nothing about which connections exist reaches
/// them, so a connection added during a session does not rewrite the tool list,
/// and the provider's cache holds. The catalog rides in the answer instead.
pub fn search_tools() -> Vec<ConnectorTool> {
    let find = ConnectorTool {
        name: FIND_TOOLS.to_string(),
        description: "Search the tools of the connections this agent can reach. Their tools are \
             not listed up front. Answers with the connections, and with the name, the \
             description, and the input schema of each tool that matches. Leave the query empty \
             to list every connection and every tool."
            .to_string(),
        input: Some(serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Words to match against tool names and descriptions. Empty lists every tool."
                }
            },
            "required": ["query"]
        })),
        output: None,
        connector: String::new(),
        via: ConnectorProtocol::Mcp,
        remote_name: String::new(),
        kind: ConnectorToolKind::Find,
    };
    let call = ConnectorTool {
        name: CALL_TOOL.to_string(),
        description: format!(
            "Run one tool of a connection this agent can reach. Take `connector` and `tool` from \
             `{FIND_TOOLS}`, and pass that tool's own arguments."
        ),
        input: Some(serde_json::json!({
            "type": "object",
            "properties": {
                "connector": {
                    "type": "string",
                    "description": "The connection that holds the tool."
                },
                "tool": {
                    "type": "string",
                    "description": "The tool's name on that connection."
                },
                "arguments": {
                    "type": "object",
                    "description": "The arguments that tool's own input schema declares."
                }
            },
            "required": ["connector", "tool"]
        })),
        output: None,
        connector: String::new(),
        via: ConnectorProtocol::Mcp,
        remote_name: String::new(),
        kind: ConnectorToolKind::Call,
    };
    vec![find, call]
}

/// More than this is longer than the tool list a search replaced.
const MAX_MATCHES: usize = 10;

/// The usual BM25 defaults.
const K1: f64 = 1.2;
const B: f64 = 0.75;

/// One tool a search matched, and the connection that holds it.
#[derive(Debug, Clone, PartialEq)]
pub struct Match<'a> {
    pub connector: &'a str,
    pub tool: &'a RemoteTool,
}

/// The tools a query matches across every searched connection, best first, by
/// BM25 over the name and the description. An empty query matches everything.
///
/// Any term, not all: a model writes "find all open issues", not keywords. BM25
/// is what keeps that useful — a term in every tool scores near zero.
pub fn find<'a>(searched: &'a [(McpServer, Vec<RemoteTool>)], query: &str) -> Vec<Match<'a>> {
    let candidates: Vec<Match<'a>> = searched
        .iter()
        .flat_map(|(connector, offered)| {
            callable(connector, offered)
                .into_iter()
                .map(|tool| Match {
                    connector: connector.id.as_str(),
                    tool,
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let terms = words(query);
    if terms.is_empty() || candidates.is_empty() {
        return candidates;
    }

    let docs: Vec<Vec<String>> = candidates
        .iter()
        .map(|m| words(&format!("{} {}", m.tool.name, m.tool.description)))
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

    let mut scored: Vec<(f64, usize, Match<'a>)> = Vec::new();
    for (order, (found, doc)) in candidates.into_iter().zip(&docs).enumerate() {
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
            scored.push((score, order, found));
        }
    }
    // The offer order breaks a tie, so a replay answers the same.
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    scored.into_iter().map(|(_, _, found)| found).collect()
}

/// Lower case words. `search_issues` is two.
fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn frequency(doc: &[String], term: &str) -> f64 {
    doc.iter().filter(|word| word.starts_with(term)).count() as f64
}

/// The result the model reads from a `find_tools` call. Each match is a tool
/// definition, in the shape of the tool list, so one find is the full distance
/// to a call. `matched` counts every match, including those past
/// [`MAX_MATCHES`].
pub fn find_answer(searched: &[(McpServer, Vec<RemoteTool>)], query: &str) -> String {
    let matched = find(searched, query);
    let tools: Vec<serde_json::Value> = matched
        .iter()
        .take(MAX_MATCHES)
        .map(|m| {
            serde_json::json!({
                "connector": m.connector,
                "name": m.tool.name,
                "description": m.tool.description,
                "input": m.tool.input,
                "output": m.tool.output,
            })
        })
        .collect();
    let connections: Vec<serde_json::Value> = searched
        .iter()
        .map(|(server, offered)| {
            serde_json::json!({
                "connector": server.id,
                "tools": callable(server, offered).len(),
            })
        })
        .collect();
    let mut answer = serde_json::json!({
        "connections": connections,
        "tools": tools,
        "matched": matched.len(),
        "searched": find(searched, "").len(),
        "call_with": CALL_TOOL,
    });
    if matched.len() > tools.len() {
        answer["hint"] = serde_json::json!(format!(
            "{} of {} matches shown. Narrow the query to see the rest.",
            tools.len(),
            matched.len()
        ));
    }
    answer.to_string()
}

fn passes(filter: &McpTools, tool: &RemoteTool) -> bool {
    matches!(capability_verdict(filter, tool), Verdict::Pass)
        && (filter.include.is_empty() || filter.include.iter().any(|g| glob_match(g, &tool.name)))
        && !filter.exclude.iter().any(|g| glob_match(g, &tool.name))
}

/// Every connector's tools in one namespace, with the ambiguous ones removed.
///
/// `taken` is what the agent config already occupies — its declared tool names
/// and its sub-agent ids. A connector tool that lands on one of those loses, in
/// keeping with a declared tool always winning its own name. Two connector
/// tools that land on each other both lose: keeping either would be arbitrary,
/// and routing the model to the wrong connection is worse than being one tool
/// short. Every dropped name is reported.
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

/// `None` when the name would be longer than a provider accepts. Truncating
/// would risk two tools collapsing onto one name, which is worse than the
/// connector being one tool short and saying so.
fn expand(connector_id: &str, tool: &RemoteTool, prefix: Option<&str>) -> Option<ConnectorTool> {
    let name = match prefix {
        Some(prefix) => format!("{}{SEPARATOR}{}", name_prefix(prefix), tool.name),
        None => tool.name.clone(),
    };
    if name.len() > MAX_NAME {
        return None;
    }
    Some(ConnectorTool {
        name,
        description: tool.description.clone(),
        input: tool.input.clone(),
        output: tool.output.clone(),
        connector: connector_id.to_string(),
        via: ConnectorProtocol::Mcp,
        remote_name: tool.name.clone(),
        kind: ConnectorToolKind::Remote,
    })
}

/// Connection ids are operator-chosen, so anything a provider would reject in a
/// tool name is flattened rather than passed through.
fn name_prefix(connector_id: &str) -> String {
    connector_id
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

enum Verdict {
    Pass,
    Fail,
    /// Failed only because the tool says nothing about the capability asked for.
    Unannotated,
}

/// A predicate is a requirement on the connection's own annotation, and an
/// absent annotation never satisfies one. MCP defaults `readOnlyHint` to false
/// and `destructiveHint` to true, so treating silence as a pass would hand the
/// model exactly the tools the filter was written to keep away from it.
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

/// `*` matches any run of characters, `?` exactly one. Enough for tool names,
/// and small enough not to need a dependency.
fn glob_match(pattern: &str, value: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let v: Vec<char> = value.chars().collect();
    // `star` remembers where to resume the pattern after the last `*`, so a
    // failed match backtracks by consuming one more character instead of
    // rescanning from the start.
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
    use crate::connectors::ToolAnnotations;

    fn tool(name: &str, annotations: ToolAnnotations) -> RemoteTool {
        RemoteTool {
            name: name.to_string(),
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

    fn connector(id: &str, tools: Option<McpTools>) -> McpServer {
        McpServer {
            id: id.to_string(),
            tools,
            auth_failure: Default::default(),
        }
    }

    fn names(r: &Resolution) -> Vec<&str> {
        r.tools.iter().map(|t| t.name.as_str()).collect()
    }

    fn searching(filter: Option<McpTools>) -> McpTools {
        McpTools {
            discovery: Some(ToolDiscovery::Search),
            ..filter.unwrap_or_default()
        }
    }

    /// The two steps, as every caller runs them.
    fn offer(connector: &McpServer, offered: &[RemoteTool], prefix: Option<&str>) -> Resolution {
        discover(connector, resolve(connector, offered, prefix))
    }

    #[test]
    fn no_filter_takes_everything_the_connection_offers() {
        let offered = [read_only("search"), writer("delete")];
        let r = resolve(&connector("sentry", None), &offered, Some("sentry"));
        assert_eq!(names(&r), vec!["sentry__search", "sentry__delete"]);
        assert_eq!(r.offered, 2);
    }

    #[test]
    fn tools_are_prefixed_so_two_connections_cannot_collide() {
        let offered = [read_only("search")];
        let a = resolve(&connector("sentry", None), &offered, Some("sentry"));
        let b = resolve(&connector("github", None), &offered, Some("github"));
        assert_eq!(names(&a), vec!["sentry__search"]);
        assert_eq!(names(&b), vec!["github__search"]);
    }

    #[test]
    fn the_remote_name_is_kept_for_the_executor_to_call() {
        let r = resolve(
            &connector("sentry", None),
            &[read_only("search_issues")],
            Some("sentry"),
        );
        assert_eq!(r.tools[0].remote_name, "search_issues");
        assert_eq!(r.tools[0].connector, "sentry");
        assert_eq!(r.tools[0].name, "sentry__search_issues");
    }

    #[test]
    fn read_only_keeps_readers_and_drops_writers() {
        let offered = [read_only("search"), writer("delete")];
        let filter = McpTools {
            read_only: Some(true),
            ..Default::default()
        };
        let r = resolve(&connector("sentry", Some(filter)), &offered, Some("sentry"));
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
        let r = resolve(&connector("custom", Some(filter)), &offered, Some("custom"));
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
        let r = resolve(&connector("sentry", Some(filter)), &offered, Some("sentry"));
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
        let r = resolve(&connector("sentry", Some(filter)), &offered, Some("sentry"));
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
        let r = resolve(&connector("sentry", Some(filter)), &offered, Some("sentry"));
        assert_eq!(names(&r), vec!["sentry__search_issues"]);
    }

    #[test]
    fn an_include_that_matches_nothing_is_reported() {
        let offered = [read_only("search_issues")];
        let filter = McpTools {
            include: vec!["search_*".to_string(), "listProjects".to_string()],
            ..Default::default()
        };
        let r = resolve(&connector("sentry", Some(filter)), &offered, Some("sentry"));
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
        let r = resolve(&connector("sentry", Some(filter)), &offered, Some("sentry"));
        assert_eq!(names(&r), vec!["sentry__search"]);
    }

    #[test]
    fn the_separator_is_doubled_so_the_name_parses_back() {
        let r = resolve(
            &connector("sentry", None),
            &[read_only("search_issues")],
            Some("sentry"),
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
        let r = resolve(&connector("sentry", None), &offered, Some("sentry"));
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
        let r = resolve(&connector("sentry", None), &offered, None);
        assert_eq!(names(&r), vec!["search_issues"]);
        assert_eq!(
            r.tools[0].connector, "sentry",
            "provenance survives even when the name does not carry it"
        );
        assert_eq!(r.tools[0].remote_name, "search_issues");
    }

    #[test]
    fn unprefixed_connections_that_collide_both_lose_the_name() {
        let a = resolve(&connector("sentry", None), &[read_only("search")], None);
        let b = resolve(&connector("github", None), &[read_only("search")], None);
        let merged = merge([a, b], []);
        assert!(
            merged.tools.is_empty(),
            "picking one of two would route the model to an arbitrary connection"
        );
        assert_eq!(merged.collisions, vec!["search".to_string()]);
    }

    #[test]
    fn a_declared_tool_keeps_its_name_against_a_connector() {
        let r = resolve(&connector("sentry", None), &[read_only("search")], None);
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
        );
        let b = resolve(
            &connector("github", None),
            &[read_only("search")],
            Some("github"),
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
        let a = resolve(&connector("sentry", None), &[read_only("issues")], None);
        let b = resolve(&connector("github", None), &[read_only("repos")], None);
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

    // ── discovery ────────────────────────────────────────────────────────

    fn searched_pair(
        id: &str,
        filter: Option<McpTools>,
        offered: &[RemoteTool],
    ) -> Vec<(McpServer, Vec<RemoteTool>)> {
        vec![(connector(id, Some(searching(filter))), offered.to_vec())]
    }

    fn found(matches: &[Match<'_>]) -> Vec<String> {
        matches
            .iter()
            .map(|m| format!("{}:{}", m.connector, m.tool.name))
            .collect()
    }

    #[test]
    fn no_discovery_setting_offers_every_tool_as_before() {
        let offered = [read_only("search"), writer("delete")];
        let r = offer(&connector("sentry", None), &offered, Some("sentry"));
        assert_eq!(
            names(&r),
            vec!["sentry__search", "sentry__delete"],
            "the default is the behaviour that predates the setting"
        );
    }

    #[test]
    fn a_searched_connection_puts_no_tool_of_its_own_in_the_list() {
        let offered = [read_only("search"), writer("delete"), bare("list")];
        let r = offer(
            &connector("sentry", Some(searching(None))),
            &offered,
            Some("sentry"),
        );
        assert!(r.tools.is_empty(), "one shared pair stands for them all");
        assert_eq!(
            r.offered, 3,
            "what the connection offered is still reported"
        );
    }

    #[test]
    fn one_pair_serves_every_searched_connection() {
        let tools = search_tools();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names,
            vec![FIND_TOOLS, CALL_TOOL],
            "two definitions, whatever the agent connects to"
        );
        assert!(
            !tools[0].description.contains("sentry"),
            "no connection reaches a definition, so adding one cannot rewrite it: {}",
            tools[0].description
        );
        assert_eq!(tools[0].kind, ConnectorToolKind::Find);
        assert_eq!(tools[1].kind, ConnectorToolKind::Call);
        assert!(
            tools.iter().all(|t| t.connector.is_empty()),
            "neither belongs to one connection"
        );
    }

    #[test]
    fn call_tool_asks_for_the_connection() {
        let call = &search_tools()[1];
        let required = call.input.as_ref().unwrap()["required"].clone();
        assert_eq!(
            required,
            serde_json::json!(["connector", "tool"]),
            "the name alone cannot say which connection holds the tool"
        );
    }

    #[test]
    fn a_find_ranks_across_connections() {
        let searched = vec![
            (
                connector("sentry", Some(searching(None))),
                vec![read_only("list_projects")],
            ),
            (
                connector("linear", Some(searching(None))),
                vec![read_only("search_issues")],
            ),
        ];
        assert_eq!(
            found(&find(&searched, "issues")),
            vec!["linear:search_issues"],
            "one call reaches every searched connection"
        );
        assert_eq!(
            found(&find(&searched, "")).len(),
            2,
            "an empty query lists them all"
        );
    }

    #[test]
    fn search_cannot_reach_a_tool_the_filter_removed() {
        let offered = [read_only("search_issues"), read_only("search_secrets")];
        let searched = searched_pair(
            "sentry",
            Some(McpTools {
                exclude: vec!["*_secrets".to_string()],
                ..Default::default()
            }),
            &offered,
        );
        assert!(
            kept(&searched[0].0, &offered, "search_secrets").is_none(),
            "discovery re-presents what the filter kept; it never reaches past it"
        );
        assert!(kept(&searched[0].0, &offered, "search_issues").is_some());
        assert_eq!(
            found(&find(&searched, "search")),
            vec!["sentry:search_issues"],
            "an excluded tool is not findable either"
        );
    }

    #[test]
    fn search_rescues_a_name_too_long_to_offer_directly() {
        let long = "a".repeat(60);
        let offered = [read_only(&long)];
        let direct = offer(&connector("sentry", None), &offered, Some("sentry"));
        assert_eq!(direct.oversized, vec![long.clone()], "too long to name");

        let c = connector("sentry", Some(searching(None)));
        let r = offer(&c, &offered, Some("sentry"));
        assert!(
            r.oversized.is_empty(),
            "call_tool carries the name as an argument, where no limit applies"
        );
        assert!(
            kept(&c, &offered, &long).is_some(),
            "and the filter still keeps it"
        );
    }

    #[test]
    fn a_word_in_every_tool_does_not_beat_a_word_in_one() {
        let offered = [
            read_only("issue_list"),
            read_only("issue_get"),
            read_only("issue_resolve"),
            read_only("issue_search"),
        ];
        let searched = searched_pair("sentry", None, &offered);
        assert_eq!(
            found(&find(&searched, "issue resolve"))[0],
            "sentry:issue_resolve",
            "`issue` is in each of the four, so `resolve` decides"
        );
    }

    #[test]
    fn a_term_matches_a_longer_word_that_starts_with_it() {
        let offered = [read_only("search_issues")];
        let searched = searched_pair("sentry", None, &offered);
        assert_eq!(
            find(&searched, "issue").len(),
            1,
            "a model writes the singular"
        );
    }

    #[test]
    fn find_matches_any_word_and_ranks_by_how_many() {
        let offered = [read_only("list_projects"), read_only("search_issues")];
        let searched = searched_pair("sentry", None, &offered);
        let matched = |q: &str| found(&find(&searched, q));

        assert_eq!(matched("issues"), vec!["sentry:search_issues"]);
        assert_eq!(
            matched("ISSUES"),
            vec!["sentry:search_issues"],
            "case is ignored"
        );
        assert_eq!(
            matched("does search_issues"),
            vec!["sentry:search_issues", "sentry:list_projects"],
            "the description is searched too, and the better match leads"
        );
        assert_eq!(matched("").len(), 2, "an empty query lists the connection");
    }

    #[test]
    fn a_sentence_finds_the_tool_and_puts_it_first() {
        let offered = [read_only("list_projects"), read_only("search_issues")];
        let searched = searched_pair("sentry", None, &offered);
        assert_eq!(
            found(&find(&searched, "find all the open issues")),
            vec!["sentry:search_issues"],
            "`issues` matched; `find`, `all`, `the`, and `open` matched nothing and cost nothing"
        );
    }

    #[test]
    fn ties_keep_the_order_the_connection_offered() {
        let offered = [read_only("b_issues"), read_only("a_issues")];
        let searched = searched_pair("sentry", None, &offered);
        assert_eq!(
            found(&find(&searched, "issues")),
            vec!["sentry:b_issues", "sentry:a_issues"],
            "a replay must answer the same, so a tie is never broken by chance"
        );
    }

    #[test]
    fn a_find_answers_with_at_most_ten_and_says_how_many_matched() {
        let offered: Vec<RemoteTool> = (0..15)
            .map(|i| read_only(&format!("issue_tool_{i}")))
            .collect();
        let searched = searched_pair("sentry", None, &offered);
        let answer: serde_json::Value =
            serde_json::from_str(&find_answer(&searched, "issue")).expect("json");
        assert_eq!(
            answer["tools"].as_array().unwrap().len(),
            MAX_MATCHES,
            "an answer longer than the tool list defeats the point of a search"
        );
        assert_eq!(answer["matched"], 15, "the count is of every match");
        assert!(
            answer["hint"].is_string(),
            "and the model is told to narrow rather than left to guess"
        );
    }

    #[test]
    fn a_find_answer_carries_what_the_model_needs_to_call() {
        let offered = [read_only("search_issues")];
        let searched = searched_pair("sentry", None, &offered);
        let answer: serde_json::Value =
            serde_json::from_str(&find_answer(&searched, "issues")).expect("json");
        assert_eq!(answer["matched"], 1);
        assert_eq!(answer["searched"], 1);
        assert_eq!(
            answer["tools"][0]["connector"], "sentry",
            "the match names its connection, which `call_tool` then needs"
        );
        assert_eq!(
            answer["tools"][0]["name"], "search_issues",
            "a match reads as a tool definition, the shape the model already knows"
        );
        assert_eq!(
            answer["tools"][0]["input"]["type"], "object",
            "the input schema rides along, so a find is the only round trip"
        );
        assert_eq!(answer["call_with"], CALL_TOOL);
    }

    #[test]
    fn a_declared_tool_still_wins_its_name_against_a_search_tool() {
        let r = Resolution {
            tools: search_tools(),
            offered: 0,
            unannotated: 0,
            unmatched_include: Vec::new(),
            oversized: Vec::new(),
        };
        let merged = merge([r], [FIND_TOOLS]);
        assert_eq!(
            merged
                .tools
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>(),
            vec![CALL_TOOL],
            "the config's own name wins, the same as against any connector tool"
        );
        assert_eq!(merged.collisions, vec![FIND_TOOLS.to_string()]);
    }
}
