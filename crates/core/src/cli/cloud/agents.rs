//! `subs agents`: what the deployment holds for each agent the file declares.
//!
//! Read-only but for rotation — an agent exists because `substructure.toml`
//! declares it, so there is nothing to create here. What this shows is the half
//! the file cannot: the signing secret the deployment minted for a
//! worker-hosted agent.

use anyhow::Result;
use clap::Subcommand;
use serde::Serialize;

use crate::api::v1::Agent;

use super::context::Context;
use super::print;
use super::ProjectScope;

#[derive(Subcommand)]
pub enum AgentsCommand {
    /// List the agents this project declares.
    #[command(name = "list", visible_alias = "ls")]
    List {
        #[command(flatten)]
        scope: ProjectScope,
    },
    /// Show one agent, including its signing secret.
    Show {
        agent_id: String,
        #[command(flatten)]
        scope: ProjectScope,
    },
    /// Mint a new signing secret for one agent (owner only). The old one stops
    /// working as soon as this returns.
    RotateSecret {
        agent_id: String,
        #[command(flatten)]
        scope: ProjectScope,
    },
}

#[derive(Debug, Serialize)]
struct Rotated<'a> {
    rotated: bool,
    id: &'a str,
    signing_secret: String,
}

pub async fn run(command: AgentsCommand) -> Result<()> {
    match command {
        AgentsCommand::List { scope } => list(scope).await,
        AgentsCommand::Show { agent_id, scope } => show(agent_id, scope).await,
        AgentsCommand::RotateSecret { agent_id, scope } => rotate(agent_id, scope).await,
    }
}

async fn list(scope: ProjectScope) -> Result<()> {
    let (ctx, project) = Context::from_project(&scope).await?;
    let agents: Vec<Agent> = ctx
        .client
        .get(&format!("/api/v1/projects/{project}/agents"))
        .await?;

    if scope.globals.json {
        return print::json(&agents);
    }

    let columns = [
        print::Column::left("ID"),
        print::Column::left("LLM"),
        print::Column::left("MODEL"),
        print::Column::left("HOSTING"),
        print::Column::left("SECRET"),
    ];
    let rows: Vec<Vec<String>> = agents
        .iter()
        .map(|a| {
            let llm = a
                .config
                .as_ref()
                .and_then(|c| c.llm.clone())
                .unwrap_or_else(|| "-".into());
            let model = a
                .config
                .as_ref()
                .map(|c| c.model.clone())
                .unwrap_or_else(|| "-".into());
            let hosting = a.worker_url.clone().unwrap_or_else(|| "engine".into());
            let secret = match a.secret_set {
                true => "set",
                false => "-",
            };
            vec![a.id.clone(), llm, model, hosting, secret.into()]
        })
        .collect();
    print::table(&columns, &rows);
    Ok(())
}

async fn show(agent_id: String, scope: ProjectScope) -> Result<()> {
    let (ctx, project) = Context::from_project(&scope).await?;
    let agent: Agent = ctx
        .client
        .get(&format!("/api/v1/projects/{project}/agents/{agent_id}"))
        .await?;

    if scope.globals.json {
        return print::json(&agent);
    }

    println!("id:       {}", agent.id);
    match &agent.worker_url {
        Some(url) => println!("hosting:  {url}"),
        None => println!("hosting:  engine"),
    }
    if let Some(config) = &agent.config {
        if let Some(llm) = &config.llm {
            println!("llm:      {llm}");
        }
        println!("model:    {}", config.model);
    }
    match &agent.signing_secret {
        Some(secret) => println!("secret:   {secret}"),
        // Only a worker-hosted agent has one to show; the engine signs nothing.
        None => println!("secret:   none (the engine decides for this agent)"),
    }
    Ok(())
}

async fn rotate(agent_id: String, scope: ProjectScope) -> Result<()> {
    let (ctx, project) = Context::from_project(&scope).await?;
    let agent: Agent = ctx
        .client
        .post_json(
            &format!("/api/v1/projects/{project}/agents/{agent_id}/rotate-secret"),
            &serde_json::json!({}),
        )
        .await?;

    let secret = agent.signing_secret.unwrap_or_default();
    if scope.globals.json {
        return print::json(&Rotated {
            rotated: true,
            id: &agent_id,
            signing_secret: secret,
        });
    }

    println!("Rotated the signing secret for {agent_id}.");
    println!("  {secret}");
    println!("Update your worker before its next decision.");
    Ok(())
}
