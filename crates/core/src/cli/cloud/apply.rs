//! `subs apply` and `subs config log`: the manifest as the only way
//! configuration enters a deployment.
//!
//! Apply replaces rather than merges — the file is the whole declaration, so an
//! agent, block, or channel absent from it is one that was removed. It is still
//! idempotent, so running it on every merge is safe. The project itself is part
//! of that: an unpinned file gets one created and the pin written back, which
//! is why this is also how a project is born.

use anyhow::{bail, Context as _, Result};
use clap::Subcommand;
use serde::{Deserialize, Serialize};

use crate::api::v1::{ApplyResponse, ConfigEvent, Notice, Page, Project};

use super::context::Context;
use super::credentials;
use super::notices;
use super::pickers;
use super::print;
use super::project_config::{self, ProjectConfig};
use super::CloudGlobals;

#[derive(Debug, clap::Args)]
pub struct ApplyCommand {
    /// Name the project, when the file does not. Written back with the pin.
    #[arg(long)]
    pub name: Option<String>,
    /// Org to create the project in. Only consulted when nothing is pinned.
    #[arg(long)]
    pub org: Option<String>,
    #[command(flatten)]
    pub globals: CloudGlobals,
}

#[derive(Subcommand)]
pub enum ConfigCommand {
    /// Show what has changed this project's configuration, newest first.
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
struct CreateProjectResponse {
    project: Project,
}

/// What apply reports, and what `--json` emits verbatim.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Applied {
    project_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    project_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    created: Option<Created>,
    changes: Vec<ConfigEvent>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    notices: Vec<Notice>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Created {
    org_id: String,
    name: String,
}

pub async fn run(cmd: ApplyCommand) -> Result<()> {
    let found = project_config::resolve(cmd.globals.config.as_deref())?
        .context("no substructure.toml found. Write one, or pass -c.")?;
    let path = found.path.clone();
    let config = found.config;
    let ctx = Context::with_config(&cmd.globals, Some(config.clone()))?;
    require_agents(&ctx).await?;

    let (project_id, created) = match config.project().map(str::to_string) {
        Some(project_id) => (project_id, None),
        None => {
            let project = create(&ctx, &cmd, &config).await?;
            let org_id = project.organization_id.clone();
            let name = project.name.clone();
            // Written back before the document is pushed: an apply that fails
            // halfway must not leave a project nobody can find again.
            pin(&path, &config, &project)?;
            (project.id, Some(Created { org_id, name }))
        }
    };

    // The file's own manifest, minus the two fields that name variables on this
    // machine. A deployment holds its own key and mints its own secret, so
    // sending either would be sending a name it cannot resolve.
    let mut manifest = config.manifest().for_wire();
    manifest.name = manifest
        .name
        .or_else(|| created.as_ref().map(|c| c.name.clone()));

    let applied: ApplyResponse = ctx
        .client
        .put_json(&format!("/api/v1/projects/{project_id}/config"), &manifest)
        .await?;

    let result = Applied {
        project_url: project_url(&applied, ctx.client.base_url()),
        project_id: applied.project_id,
        created,
        changes: applied.changes,
        notices: applied.notices,
    };
    if cmd.globals.json {
        return print::json(&result);
    }
    report(&result, &path, &cmd.globals);
    Ok(())
}

/// The page to see this project on, if there is one to send a reader to.
///
/// The deployment's own answer wins: only it knows the origin a browser
/// reaches it at, which is not always the one the API answers on. Absent that,
/// the hosted cloud is the one deployment whose page this CLI knows — anything
/// else gets no line rather than a guessed one.
fn project_url(applied: &ApplyResponse, api_url: &str) -> Option<String> {
    if let Some(url) = &applied.project_url {
        return Some(url.clone());
    }
    let hosted = api_url.trim_end_matches('/') == credentials::DEFAULT_API_URL;
    hosted.then(|| print::admin_url(api_url, &applied.project_id))
}

/// A deployment that does not declare agents cannot hold this document, and
/// says so before anything is created rather than 400-ing halfway.
///
/// The gate is the feature, not its absence: a deployment that advertises
/// nothing is one this CLI predates, and there is no older shape to fall back
/// to. Only the deployment's own answer decides that — see [`Context::meta`].
async fn require_agents(ctx: &Context) -> Result<()> {
    if ctx.meta().await?.has("agents") {
        return Ok(());
    }
    bail!(
        "this deployment does not hold per-agent configuration, so it cannot take this file. \
         Upgrade it, or use the CLI it shipped with."
    )
}

/// Create the project this file describes. The org comes from the pin, then the
/// server's own default, then a picker.
async fn create(ctx: &Context, cmd: &ApplyCommand, config: &ProjectConfig) -> Result<Project> {
    let interactive = pickers::interactive(&cmd.globals);
    let org = if let Some(org) = cmd.org.clone().or_else(|| config.org().map(str::to_string)) {
        org
    } else if let Some(org) = ctx.server_default_org().await? {
        org
    } else if interactive {
        pickers::pick_org(ctx).await?
    } else {
        bail!("no org to create the project in. Pass --org <id>, or set `org` in the file.")
    };

    let name = match cmd.name.clone().or_else(|| config.name.clone()) {
        Some(name) => name,
        None if interactive => {
            let default = default_name();
            let prompt = match &default {
                Some(d) => format!("Project name [{d}]"),
                None => "Project name".to_string(),
            };
            let entered = pickers::prompt_text(&prompt)?;
            match entered.trim() {
                "" => default.context("no project name given")?,
                given => given.to_string(),
            }
        }
        None => bail!("no project name. Set `name` in the file, or pass --name."),
    };

    let res: CreateProjectResponse = ctx
        .client
        .post_json(
            &format!("/api/v1/orgs/{org}/projects"),
            &NamePayload { name: &name },
        )
        .await?;
    Ok(res.project)
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
/// no-op rather than a second project.
fn pin(path: &std::path::Path, env: &ProjectConfig, project: &Project) -> Result<()> {
    let mut pinned = env.clone();
    pinned.name = Some(project.name.clone());
    let remote = pinned.remote_mut();
    remote.project = Some(project.id.clone());
    remote.org = Some(project.organization_id.clone());
    project_config::write(path, &pinned)
}

fn report(result: &Applied, path: &std::path::Path, globals: &CloudGlobals) {
    let file = path.display();
    // Whatever ran before this — a login, most of it a browser's — is not part
    // of the report.
    println!();
    if let Some(created) = &result.created {
        println!(
            "Created project {} ({}) in {}",
            created.name, result.project_id, created.org_id
        );
        println!("Pinned {} in {file}", result.project_id);
    }

    if result.changes.is_empty() {
        println!("No changes.");
    } else {
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

        // A worker-hosted agent gets a secret minted on its first apply. It is
        // retrievable, so this points at the command rather than printing it.
        let minted: Vec<&str> = result
            .changes
            .iter()
            .filter(|c| c.kind == "agent.updated")
            .filter(|c| c.data.get("secretMinted").and_then(|v| v.as_bool()) == Some(true))
            .filter_map(|c| c.data.get("id").and_then(|v| v.as_str()))
            .collect();
        if !minted.is_empty() {
            println!("Signing secrets minted:");
            for id in minted {
                println!("  {id} — `subs agents show {id}`");
            }
        }
    }

    if let Some(url) = &result.project_url {
        println!();
        println!("View this project: {url}");
        // Notices bring their own blank line; nothing else does.
        if result.notices.is_empty() {
            println!();
        }
    }

    // What the deployment has to say, as it sees it. Printed whether or not
    // this apply changed anything: it is current state, not a record of the run.
    notices::print(&result.notices, &notices::flag(globals));
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
        "project.created" => field("name"),
        "project.renamed" => format!("{} -> {}", field("from"), field("to")),
        "agent.updated" => match field("workerUrl").as_str() {
            "" => format!("{} (engine)", field("id")),
            url => format!("{} -> {url}", field("id")),
        },
        "agent.removed" | "agent.secret_rotated" => field("id"),
        "llm.updated" => format!("{} ({})", field("name"), field("type")),
        "llm.removed" | "llm.key_set" | "llm.key_removed" => field("name"),
        "slack.routing_updated" => field("summary"),
        "mcp.connection_declared" => format!("{} (pending authorization)", field("id")),
        "mcp.grant_added" | "mcp.grant_removed" | "mcp.connection_removed" => field("id"),
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
    let project = ctx
        .pinned_project(None)
        .await?
        .context("no project. Pin one with `subs apply`, or pass -c <file>.")?;

    let mut path = format!("/api/v1/projects/{project}/config/events?limit={limit}");
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
            summarize(&event(
                "agent.updated",
                json!({"id": "triage", "workerUrl": "https://w.test"})
            )),
            "triage -> https://w.test"
        );
        // No worker URL is not missing data: the engine decides for it.
        assert_eq!(
            summarize(&event("agent.updated", json!({"id": "support"}))),
            "support (engine)"
        );
        assert_eq!(
            summarize(&event(
                "llm.updated",
                json!({"name": "claude", "type": "anthropic"})
            )),
            "claude (anthropic)"
        );
        assert_eq!(
            summarize(&event("mcp.connection_declared", json!({"id": "sentry"}))),
            "sentry (pending authorization)"
        );
        assert_eq!(
            summarize(&event("project.renamed", json!({"from": "a", "to": "b"}))),
            "a -> b"
        );
        // A kind this build does not know is still a row, just without detail.
        assert_eq!(summarize(&event("something.new", json!({}))), "");
    }

    /// The deployment knows the origin a browser reaches it at; the CLI only
    /// knows the hosted cloud's.
    #[test]
    fn the_page_to_view_a_project_comes_from_the_deployment_that_has_one() {
        let response = |url: Option<&str>| ApplyResponse {
            project_id: "p1".into(),
            project_url: url.map(str::to_string),
            changes: vec![],
            notices: vec![],
        };

        assert_eq!(
            project_url(
                &response(Some("https://subs.test/p/p1")),
                "http://localhost:5174"
            ),
            Some("https://subs.test/p/p1".to_string())
        );
        assert_eq!(
            project_url(&response(None), credentials::DEFAULT_API_URL),
            Some("https://app.substructure.ai/projects/p1".to_string())
        );
        // A deployment that says nothing gets no line rather than a guess.
        assert_eq!(project_url(&response(None), "http://localhost:5174"), None);
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
