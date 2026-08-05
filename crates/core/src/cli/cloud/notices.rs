//! What a project still needs, and the one way it is printed.
//!
//! A deployment computes its own: it holds the keys, the consents and the
//! workspaces, so it decides what it says and how loudly, and a CLI older than
//! the deployment still prints what it is given. An engine you run here has no
//! deployment to ask, so `subs doctor` writes the same notices off this machine
//! — see [`super::super::doctor`].
//!
//! The grouping, the order, and the `-c` a reader would have to repeat are
//! decided here, so every command that reports a step reports it the same way.

use anyhow::Result;

use crate::api::v1::{Notice, NoticeLevel, NoticesResponse};

use super::context::Context;
use super::project_config::{self, Found};
use super::CloudGlobals;

/// The feature a deployment advertises when it reports what a project still
/// needs. One that does not is older than this CLI, and there is no earlier
/// shape to fall back to.
pub const FEATURE: &str = "notices";

/// What the deployment says this project still needs, or `None` from one that
/// does not report it — which is a deployment too old to ask, not a project
/// with nothing left to do.
pub async fn fetch(ctx: &Context, project: &str) -> Result<Option<Vec<Notice>>> {
    if !ctx.meta().await?.has(FEATURE) {
        return Ok(None);
    }
    let res: NoticesResponse = ctx
        .client
        .get(&format!("/api/v1/projects/{project}/notices"))
        .await?;
    Ok(Some(res.notices))
}

/// What is still unfinished, printed after a command has finished one step.
///
/// Best effort and actions only: the step succeeded, so a status read that
/// fails must not read as a failure of the thing that just worked, and a
/// warning about something else is not this command's news.
pub async fn remaining(ctx: &Context, project: &str, globals: &CloudGlobals) {
    if globals.json {
        return;
    }
    let Ok(Some(notices)) = fetch(ctx, project).await else {
        return;
    };
    let left: Vec<&Notice> = notices
        .iter()
        .filter(|n| n.level == NoticeLevel::Action)
        .collect();
    if left.is_empty() {
        return;
    }
    println!();
    println!("{}", NoticeLevel::Action.heading());
    let flag = flag(globals);
    for notice in left {
        entry(notice, &flag);
    }
}

/// Every notice under the heading its level belongs to, loudest first, keeping
/// the order they were written in. Empty levels are absent rather than printed
/// as a heading with nothing under it.
pub fn print(notices: &[Notice], flag: &str) {
    for (level, group) in grouped(notices) {
        println!();
        println!("{}", level.heading());
        for notice in group {
            entry(notice, flag);
        }
    }
}

/// One notice: what it is, and the one thing to do about it.
fn entry(notice: &Notice, flag: &str) {
    println!();
    println!("  {}", notice.message);
    if let Some(hint) = hint(notice, flag) {
        println!("    {hint}");
    }
}

/// The one thing to do about a notice: the command when there is one, and the
/// URL only when there is not. A reader who can stay on the command line is not
/// sent to a browser as well.
///
/// The `-c` goes on this CLI's own commands and nothing else: a `export VAR=…`
/// takes no such flag.
fn hint(notice: &Notice, flag: &str) -> Option<String> {
    match &notice.command {
        Some(command) if command.starts_with("subs ") => Some(format!("{command}{flag}")),
        Some(command) => Some(command.clone()),
        None => notice.url.clone(),
    }
}

fn grouped(notices: &[Notice]) -> Vec<(NoticeLevel, Vec<&Notice>)> {
    NoticeLevel::ORDER
        .into_iter()
        .filter_map(|level| {
            let group: Vec<&Notice> = notices.iter().filter(|n| n.level == level).collect();
            (!group.is_empty()).then_some((level, group))
        })
        .collect()
}

/// The `-c` the reader would have to repeat, omitted when discovery finds the
/// same file anyway.
pub fn flag(globals: &CloudGlobals) -> String {
    let Some(path) = globals.config.as_deref() else {
        return String::new();
    };
    let discovered = project_config::find()
        .ok()
        .flatten()
        .map(|found: Found| found.path);
    match discovered {
        Some(d) if d == path => String::new(),
        _ => format!(" -c {}", path.display()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::cloud::CloudGlobals;
    use axum::routing::get;
    use axum::{Json, Router};
    use serde_json::json;

    /// A deployment answering `/meta` with `features` and, if it has any, the
    /// notices for every project.
    async fn deployment(features: Vec<&'static str>) -> String {
        let app =
            Router::new()
                .route(
                    "/api/v1/meta",
                    get(move || async move {
                        Json(json!({ "singleTenant": false, "features": features }))
                    }),
                )
                .route(
                    "/api/v1/projects/{project}/notices",
                    get(|| async {
                        Json(json!({ "notices": [
                            { "level": "action", "message": "Connect a Slack workspace",
                              "command": "subs slack connect" },
                            { "level": "warn", "message": "#general was taken" },
                        ]}))
                    }),
                );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        format!("http://{addr}")
    }

    fn ctx(url: &str) -> Context {
        let globals = CloudGlobals {
            url: Some(url.to_string()),
            ..Default::default()
        };
        Context::with_config(&globals, None).unwrap()
    }

    /// The deployment holds the state, so it is the deployment that says what is
    /// left — and one that cannot is not a project with nothing to do.
    #[tokio::test]
    async fn what_is_left_is_read_from_the_deployment_that_reports_it() {
        let url = deployment(vec!["projects", "notices"]).await;
        let notices = fetch(&ctx(&url), "p1").await.unwrap().expect("reported");
        assert_eq!(notices.len(), 2);
        assert_eq!(notices[0].command.as_deref(), Some("subs slack connect"));

        let older = deployment(vec!["projects"]).await;
        assert!(
            fetch(&ctx(&older), "p1").await.unwrap().is_none(),
            "a deployment that does not advertise it is not asked"
        );
    }

    fn notice(level: &str, message: &str) -> Notice {
        serde_json::from_value(json!({ "level": level, "message": message })).unwrap()
    }

    /// The deployment decides what it says and how loudly; the order of the
    /// headings, and which ones appear at all, are decided here.
    #[test]
    fn notices_are_grouped_by_level_loudest_first() {
        let notices = [
            notice("info", "minted a secret"),
            notice("action", "connect a workspace"),
            notice("warn", "#general was taken"),
            notice("action", "set a key"),
        ];
        let shape: Vec<(&str, Vec<&str>)> = grouped(&notices)
            .iter()
            .map(|(level, group)| {
                (
                    level.heading(),
                    group.iter().map(|n| n.message.as_str()).collect(),
                )
            })
            .collect();
        assert_eq!(
            shape,
            [
                ("Action required:", vec!["connect a workspace", "set a key"]),
                ("Warnings:", vec!["#general was taken"]),
                ("Notes:", vec!["minted a secret"]),
            ]
        );
        // A level with nothing in it is not a heading over an empty list.
        assert_eq!(grouped(&[notice("warn", "only this")]).len(), 1);
    }

    /// A browser is the fallback, not the companion: a notice that names a
    /// command is done on the command line.
    #[test]
    fn a_notice_sends_the_reader_to_a_browser_only_when_there_is_no_command() {
        let both: Notice = serde_json::from_value(json!({
            "message": "authorize it",
            "command": "subs mcp login sentry",
            "url": "https://app.test/mcp",
        }))
        .unwrap();
        assert_eq!(
            hint(&both, " -c other.toml").as_deref(),
            Some("subs mcp login sentry -c other.toml")
        );

        let url_only: Notice = serde_json::from_value(
            json!({ "message": "connect Slack", "url": "https://app.test" }),
        )
        .unwrap();
        assert_eq!(hint(&url_only, "").as_deref(), Some("https://app.test"));

        assert_eq!(hint(&notice("info", "nothing to do"), ""), None);
    }

    /// A deployment newer than this CLI must still be heard: an unknown level
    /// reads as the loudest rather than failing the whole response.
    #[test]
    fn an_unknown_level_is_shown_rather_than_dropped() {
        let parsed: Notice =
            serde_json::from_value(json!({ "level": "critical", "message": "something new" }))
                .expect("an unknown level must not fail the response");
        assert_eq!(parsed.level, NoticeLevel::Action);
        // And a deployment that sends no level at all still lands somewhere.
        let bare: Notice = serde_json::from_value(json!({ "message": "no level" })).unwrap();
        assert_eq!(bare.level, NoticeLevel::Action);
    }

    /// The file is this CLI's own argument, so it is repeated on this CLI's own
    /// commands and on nothing else.
    #[test]
    fn the_file_flag_goes_on_a_subs_command_and_not_on_a_shell_line() {
        let export =
            Notice::action("Set $ANTHROPIC_API_KEY").with_command("export ANTHROPIC_API_KEY=...");
        assert_eq!(
            hint(&export, " -c dev.toml").as_deref(),
            Some("export ANTHROPIC_API_KEY=...")
        );
    }
}
