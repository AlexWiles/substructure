//! `subs projects`: what exists, and removing one.
//!
//! Creating and renaming are absent on purpose — `subs apply` owns both, since
//! the file is the source of truth for a project's existence and its name.

use anyhow::Result;
use clap::Subcommand;
use serde::Serialize;

use crate::api::v1::Project;

use super::context::Context;
use super::print;
use super::{OrgScope, ProjectScope};

#[derive(Subcommand)]
pub enum ProjectsCommand {
    /// List projects in the current org.
    #[command(name = "list", visible_alias = "ls")]
    List {
        #[command(flatten)]
        scope: OrgScope,
    },
    /// Show a project's details.
    Show {
        project_id: Option<String>,
        #[command(flatten)]
        scope: ProjectScope,
    },
    /// Delete a project (owner only).
    Delete {
        project_id: Option<String>,
        #[command(flatten)]
        scope: ProjectScope,
    },
}

#[derive(Debug, Serialize)]
struct DeleteResult<'a> {
    deleted: bool,
    id: &'a str,
}

pub async fn run(command: ProjectsCommand) -> Result<()> {
    match command {
        ProjectsCommand::List { scope } => list(scope).await,
        ProjectsCommand::Show { project_id, scope } => show(project_id, scope).await,
        ProjectsCommand::Delete { project_id, scope } => delete(project_id, scope).await,
    }
}

async fn list(scope: OrgScope) -> Result<()> {
    let (ctx, org) = Context::from_org(&scope).await?;
    let projects: Vec<Project> = ctx
        .client
        .get(&format!("/api/v1/orgs/{org}/projects"))
        .await?;

    if scope.globals.json {
        return print::json(&projects);
    }

    let pinned = ctx.config.as_ref().and_then(|p| p.project());
    let columns = [
        print::Column::left(""),
        print::Column::left("ID"),
        print::Column::left("NAME"),
        print::Column::right("SESSIONS"),
        print::Column::right("BALANCE"),
    ];
    let rows: Vec<Vec<String>> = projects
        .iter()
        .map(|a| {
            let marker = if pinned == Some(a.id.as_str()) {
                "*"
            } else {
                ""
            };
            let sessions = a
                .session_count
                .map(|n| n.to_string())
                .unwrap_or_else(|| "-".into());
            let balance = print::fmt_usd(a.balance_usd.as_deref().unwrap_or("0"));
            vec![
                marker.into(),
                a.id.clone(),
                a.name.clone(),
                sessions,
                balance,
            ]
        })
        .collect();
    print::table(&columns, &rows);
    Ok(())
}

async fn show(project_id: Option<String>, scope: ProjectScope) -> Result<()> {
    let scope = ProjectScope {
        project: project_id.or(scope.project.clone()),
        ..scope
    };
    let (ctx, project_id) = Context::from_project(&scope).await?;
    let a: Project = ctx
        .client
        .get(&format!("/api/v1/projects/{project_id}"))
        .await?;

    if scope.globals.json {
        return print::json(&a);
    }

    println!("id:              {}", a.id);
    println!("name:            {}", a.name);
    println!("organization_id: {}", a.organization_id);
    if let Some(ca) = &a.created_at {
        println!("created_at:      {ca}");
    }
    let balance_raw = a.balance_usd.as_deref().unwrap_or("0");
    println!("balance:         {}", print::fmt_usd(balance_raw));
    if let Some(s) = a.session_count {
        println!("session_count:   {s}");
    }
    Ok(())
}

async fn delete(project_id: Option<String>, scope: ProjectScope) -> Result<()> {
    let scope = ProjectScope {
        project: project_id.or(scope.project.clone()),
        ..scope
    };
    let (ctx, project_id) = Context::from_project(&scope).await?;
    ctx.client
        .delete_discard(&format!("/api/v1/projects/{project_id}"))
        .await?;

    if scope.globals.json {
        return print::json(&DeleteResult {
            deleted: true,
            id: &project_id,
        });
    }

    println!("Project {project_id} deleted");
    Ok(())
}
