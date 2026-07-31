//! `subs apply` and `subs config log`: the manifest as the only way
//! configuration enters a deployment.
//!
//! Apply is additive and idempotent — it creates and updates what the file
//! names and removes nothing, so running it on every merge is safe. The app
//! itself is part of that: an unpinned file gets one created and the pin
//! written back, which is why this is also how an app is born.

use anyhow::{bail, Context as _, Result};
use clap::Subcommand;
use serde::{Deserialize, Serialize};

use crate::api::v1::{
    App, ApplyResponse, ConfigConnectionRef, ConfigEvent, ConfigUpdate, ConfigWorkerUpdate, Page,
};

use super::context::Context;
use super::pickers;
use super::print;
use super::project_config::{self, EnvConfig, Found};
use super::CloudGlobals;

#[derive(Debug, clap::Args)]
pub struct ApplyCommand {
    /// Name the app, when the file does not. Written back with the pin.
    #[arg(long)]
    pub name: Option<String>,
    /// Org to create the app in. Only consulted when nothing is pinned.
    #[arg(long)]
    pub org: Option<String>,
    #[command(flatten)]
    pub globals: CloudGlobals,
}

#[derive(Subcommand)]
pub enum ConfigCommand {
    /// Show what has changed this app's configuration, newest first.
    Log {
        /// Resume from a `next_cursor` a previous page reported.
        #[arg(long)]
        cursor: Option<String>,
        /// Entries per page.
        #[arg(long, default_value_t = 20)]
        limit: usize,
        #[command(flatten)]
        globals: CloudGlobals,
    },
}

#[derive(Debug, Serialize)]
struct NamePayload<'a> {
    name: &'a str,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CreateAppResponse {
    app: App,
    signing_secret: String,
}

/// What apply reports, and what `--json` emits verbatim.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Applied {
    app_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    created: Option<Created>,
    changes: Vec<ConfigEvent>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Created {
    org_id: String,
    name: String,
    /// Shown once, here and never again.
    signing_secret: String,
}

pub async fn run(cmd: ApplyCommand) -> Result<()> {
    let found = project_config::resolve(cmd.globals.config.as_deref())?
        .context("no substructure.toml found. Write one with `subs init`, or pass -c.")?;
    let path = found.path.clone();
    let env = found.config;
    local_credentials(&env)?;
    let ctx = Context::with_project(&cmd.globals, Some(env.clone()))?;

    let (app_id, created) = match env.app().map(str::to_string) {
        Some(app_id) => (app_id, None),
        None => {
            let (app, secret) = create(&ctx, &cmd, &env).await?;
            let org_id = app.organization_id.clone();
            let name = app.name.clone();
            // Written back before the document is pushed: an apply that fails
            // halfway must not leave an app nobody can find again.
            pin(&path, &env, &app)?;
            (
                app.id,
                Some(Created {
                    org_id,
                    name,
                    signing_secret: secret,
                }),
            )
        }
    };

    let update = ConfigUpdate {
        name: env
            .name
            .clone()
            .or_else(|| created.as_ref().map(|c| c.name.clone())),
        worker: app_worker(&env)?.map(|url| ConfigWorkerUpdate { url }),
        mcp: Some(
            env.mcp
                .iter()
                .map(|(id, spec)| ConfigConnectionRef {
                    id: id.clone(),
                    url: spec.url.clone(),
                })
                .collect(),
        ),
    };
    let applied: ApplyResponse = ctx
        .client
        .put_json(&format!("/api/v1/apps/{app_id}/config"), &update)
        .await?;

    let result = Applied {
        app_id: applied.app_id,
        created,
        changes: applied.changes,
    };
    if cmd.globals.json {
        return print::json(&result);
    }
    report(&result, &path);
    Ok(())
}

/// The one worker URL a deployment can hold today.
///
/// Hosting is per-agent in the file, but a deployment still keeps a single
/// worker for the whole app, so apply can carry the file's hosting only while
/// the agents agree on one URL. Two different ones is not something to pick a
/// winner from — it is a file this deployment cannot serve yet, and it says so.
fn app_worker(env: &EnvConfig) -> Result<Option<String>> {
    let mut urls: Vec<&str> = env
        .agent
        .values()
        .filter_map(|a| a.worker.as_deref())
        .collect();
    urls.sort_unstable();
    urls.dedup();
    match urls.as_slice() {
        [] => Ok(None),
        [url] => Ok(Some((*url).to_string())),
        many => bail!(
            "this deployment holds one worker for the whole app, and the file gives {} agents \
             {} different workers ({}). Point them at one URL, or run them on an engine that \
             hosts per-agent workers.",
            env.agent.values().filter(|a| a.worker.is_some()).count(),
            many.len(),
            many.join(", ")
        ),
    }
}

/// A credential the deployment cannot reach. `token_env` names a variable on
/// this machine, which is the engine's half of a connection; consent for the
/// deployment's copy is `subs mcp login`.
fn local_credentials(env: &EnvConfig) -> Result<()> {
    let named: Vec<&str> = env
        .mcp
        .iter()
        .filter(|(_, spec)| spec.auth.is_some())
        .map(|(id, _)| id.as_str())
        .collect();
    if let Some(id) = named.first() {
        bail!(
            "[mcp.{id}] names `token_env`, which a deployment cannot read: it holds its own \
             credential. Drop `auth` and authorize the connection there — `subs mcp login {id}`."
        );
    }
    Ok(())
}

/// Create the app this file describes. The org comes from the pin, then the
/// server's own default, then a picker.
async fn create(ctx: &Context, cmd: &ApplyCommand, env: &EnvConfig) -> Result<(App, String)> {
    let interactive = pickers::interactive(&cmd.globals);
    let org = if let Some(org) = cmd.org.clone().or_else(|| env.org().map(str::to_string)) {
        org
    } else if let Some(org) = ctx.server_default_org().await {
        org
    } else if interactive {
        pickers::pick_org(ctx).await?
    } else {
        bail!("no org to create the app in. Pass --org <id>, or set `org` in the file.")
    };

    let name = match cmd.name.clone().or_else(|| env.name.clone()) {
        Some(name) => name,
        None if interactive => {
            let default = default_name();
            let prompt = match &default {
                Some(d) => format!("App name [{d}]"),
                None => "App name".to_string(),
            };
            let entered = pickers::prompt_text(&prompt)?;
            match entered.trim() {
                "" => default.context("no app name given")?,
                given => given.to_string(),
            }
        }
        None => bail!("no app name. Set `name` in the file, or pass --name."),
    };

    let res: CreateAppResponse = ctx
        .client
        .post_json(
            &format!("/api/v1/orgs/{org}/apps"),
            &NamePayload { name: &name },
        )
        .await?;
    Ok((res.app, res.signing_secret))
}

/// The directory's own name, which is what a project is usually called.
fn default_name() -> Option<String> {
    std::env::current_dir()
        .ok()?
        .file_name()?
        .to_str()
        .map(str::to_string)
}

/// Write the pin back, and the org and name with it, so a second apply is a
/// no-op rather than a second app.
fn pin(path: &std::path::Path, env: &EnvConfig, app: &App) -> Result<()> {
    let mut pinned = env.clone();
    pinned.name = Some(app.name.clone());
    let deployment = pinned.deployment_mut();
    deployment.app = Some(app.id.clone());
    deployment.org = Some(app.organization_id.clone());
    project_config::write(path, &pinned)
}

fn report(result: &Applied, path: &std::path::Path) {
    let file = path.display();
    if let Some(created) = &result.created {
        println!(
            "Created app {} ({}) in {}",
            created.name, result.app_id, created.org_id
        );
        println!("  signing secret (shown once): {}", created.signing_secret);
        println!("Pinned {} in {file}", result.app_id);
    }

    if result.changes.is_empty() {
        println!("No changes.");
        return;
    }

    // `config.applied` is the record of the document, not one of its changes.
    let changes: Vec<&ConfigEvent> = result
        .changes
        .iter()
        .filter(|c| c.kind != "config.applied")
        .collect();
    println!("Applied {} changes:", changes.len());
    for change in &changes {
        println!("  {:<24}{}", change.kind, summarize(change));
    }

    // A declared connection reaches nothing until a human consents, so the
    // next command is part of the output rather than something to go look up.
    let pending: Vec<&str> = result
        .changes
        .iter()
        .filter(|c| c.kind == "mcp.connection_declared")
        .filter_map(|c| c.data.get("id").and_then(|v| v.as_str()))
        .collect();
    if !pending.is_empty() {
        println!("Pending authorization:");
        let flag = config_flag(path);
        for id in pending {
            println!("  {id} — run `subs mcp login {id}{flag}` or authorize in the dashboard");
        }
    }
}

/// The `-c` the user would have to repeat, omitted when discovery finds the
/// same file anyway.
fn config_flag(path: &std::path::Path) -> String {
    let discovered = project_config::find()
        .ok()
        .flatten()
        .map(|found: Found| found.path);
    match discovered {
        Some(d) if d == path => String::new(),
        _ => format!(" -c {}", path.display()),
    }
}

fn summarize(change: &ConfigEvent) -> String {
    let field = |key: &str| {
        change
            .data
            .get(key)
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string()
    };
    match change.kind.as_str() {
        "app.created" => field("name"),
        "app.renamed" => format!("{} -> {}", field("from"), field("to")),
        "worker.updated" => format!("url={}", field("url")),
        "mcp.connection_declared" => format!("{} (pending authorization)", field("id")),
        "mcp.grant_added" | "mcp.grant_removed" => field("id"),
        _ => String::new(),
    }
}

pub async fn config(command: ConfigCommand) -> Result<()> {
    match command {
        ConfigCommand::Log {
            cursor,
            limit,
            globals,
        } => log(cursor, limit, globals).await,
    }
}

async fn log(cursor: Option<String>, limit: usize, globals: CloudGlobals) -> Result<()> {
    let ctx = Context::load(&globals)?;
    let app = ctx
        .pinned_app(None)
        .await
        .context("no app. Pin one with `subs apply`, or pass -c <file>.")?;

    let mut path = format!("/api/v1/apps/{app}/config/events?limit={limit}");
    if let Some(cursor) = &cursor {
        path.push_str(&format!("&cursor={cursor}"));
    }
    let page: Page<ConfigEvent> = ctx.client.get(&path).await?;

    if globals.json {
        return print::json(&page);
    }

    let columns = [
        print::Column::left("WHEN"),
        print::Column::left("WHO"),
        print::Column::left("HOW"),
        print::Column::left("WHAT"),
        print::Column::left(""),
    ];
    let rows: Vec<Vec<String>> = page
        .items
        .iter()
        .map(|e| {
            vec![
                when(&e.created_at),
                e.actor_email.clone().unwrap_or_else(|| "system".into()),
                e.source.clone(),
                e.kind.clone(),
                detail(e),
            ]
        })
        .collect();
    print::table(&columns, &rows);
    if let Some(next) = page.next_cursor {
        println!();
        println!("More: --cursor {next}");
    }
    Ok(())
}

/// `2026-07-30 14:02`, or the timestamp as sent when it is not one we parse.
fn when(created_at: &str) -> String {
    match chrono::DateTime::parse_from_rfc3339(created_at) {
        Ok(t) => t
            .with_timezone(&chrono::Local)
            .format("%Y-%m-%d %H:%M")
            .to_string(),
        Err(_) => created_at.to_string(),
    }
}

fn detail(event: &ConfigEvent) -> String {
    match event.kind.as_str() {
        "config.applied" => match event.data.get("hash").and_then(|v| v.as_str()) {
            Some(hash) => format!("hash {}", &hash[..hash.len().min(12)]),
            None => String::new(),
        },
        _ => summarize(event),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn event(kind: &str, data: serde_json::Value) -> ConfigEvent {
        ConfigEvent {
            seq: 1,
            kind: kind.into(),
            data,
            actor_email: Some("alex@test".into()),
            source: "apply".into(),
            created_at: "2026-07-30T14:02:00.000000Z".into(),
        }
    }

    #[test]
    fn a_change_reads_as_what_it_did() {
        assert_eq!(
            summarize(&event("worker.updated", json!({"url": "https://w.test"}))),
            "url=https://w.test"
        );
        assert_eq!(
            summarize(&event("mcp.connection_declared", json!({"id": "sentry"}))),
            "sentry (pending authorization)"
        );
        assert_eq!(
            summarize(&event("app.renamed", json!({"from": "a", "to": "b"}))),
            "a -> b"
        );
        // A kind this build does not know is still a row, just without detail.
        assert_eq!(summarize(&event("something.new", json!({}))), "");
    }

    #[test]
    fn an_applied_document_is_identified_by_its_hash() {
        let long = "a".repeat(64);
        assert_eq!(
            detail(&event("config.applied", json!({ "hash": long }))),
            "hash aaaaaaaaaaaa"
        );
    }

    #[test]
    fn a_timestamp_that_is_not_one_is_shown_as_sent() {
        assert_eq!(when("not-a-time"), "not-a-time");
        assert!(when("2026-07-30T14:02:00Z").starts_with("2026-07-"));
    }
}
