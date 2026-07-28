//! Turning what a connection offers into what the model sees.
//!
//! The order is fixed — capability predicates, then `include`, then `exclude` —
//! and every step only removes, so a filter can never widen what the connection
//! granted. Nothing here is silent: a predicate that drops unannotated tools and
//! an `include` that matches nothing are both reported, because both usually
//! mean the far side changed under us.

use crate::connectors::RemoteTool;
use crate::protocol::{ConnectorProtocol, ConnectorTool, McpServer, McpTools};

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
    let mut unannotated = 0;

    let kept: Vec<&RemoteTool> = offered
        .iter()
        .filter(|tool| match capability_verdict(&filter, tool) {
            Verdict::Pass => true,
            Verdict::Fail => false,
            Verdict::Unannotated => {
                unannotated += 1;
                false
            }
        })
        .filter(|tool| {
            filter.include.is_empty() || filter.include.iter().any(|g| glob_match(g, &tool.name))
        })
        .filter(|tool| !filter.exclude.iter().any(|g| glob_match(g, &tool.name)))
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
        }
    }

    fn names(r: &Resolution) -> Vec<&str> {
        r.tools.iter().map(|t| t.name.as_str()).collect()
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
}
