use crate::connectors::filter::TOOL_SEARCH;

pub const NOTHING_MATCHED: &str =
    "Nothing matched. Search again with an empty query for every tool.";

pub const CALL_NOT_ROUTED: &str = "the call could not be routed";

pub const PLUGIN_USAGE: &str = "load a skill with the `skill` tool";

pub fn matches_truncated(shown: usize, matched: usize) -> String {
    format!(
        "{shown} of {matched} matches shown. Narrow the query to see the rest: a connector's \
         name is a word in every one of its tools, so adding it keeps the search to that \
         connection."
    )
}

pub fn no_such_tool(named: &str) -> String {
    format!(
        "no tool `{named}` for this agent. Call `{TOOL_SEARCH}` with an empty query for every tool."
    )
}

pub fn no_such_tool_near(named: &str, near: &[&str]) -> String {
    format!(
        "no tool `{named}` for this agent. The closest are: {}. Call `{TOOL_SEARCH}` for the \
         schema of one.",
        near.join(", ")
    )
}

pub fn bad_arguments(named: &str, error: &str, schema: Option<&serde_json::Value>) -> String {
    match schema {
        Some(schema) => format!("`{named}`: {error}. Its input schema is: {schema}"),
        None => format!("`{named}`: {error}"),
    }
}

pub fn unavailable_reason(auth: Option<crate::connectors::AuthNeed>) -> &'static str {
    match auth {
        Some(_) => "needs_authorization",
        None => "unreachable",
    }
}

pub fn unavailable_note(connection: &str) -> String {
    format!("`{connection}` is unavailable, so its tools are not listed.")
}

pub fn needs_authorization(connection: &str, need: crate::connectors::AuthNeed) -> String {
    match need {
        crate::connectors::AuthNeed::NeverAuthorized => {
            format!("`{connection}` is not authorized yet, so I cannot use it.")
        }
        crate::connectors::AuthNeed::Reauthorize => {
            format!("`{connection}` needs to be authorized again. Its access expired.")
        }
        crate::connectors::AuthNeed::TokenRejected => format!(
            "`{connection}` rejected its token. An operator must set a new one \
             with `subs auth {connection}`."
        ),
    }
}
