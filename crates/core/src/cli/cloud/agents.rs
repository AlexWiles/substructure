//! `subs agents`: shows each agent the file declares.
//!
//! Read-only but for rotation — an agent exists because `subs.toml`
//! declares it, so there is nothing to create here. Printing a signing secret
//! is its own command: no other output carries one, so a secret reaches a
//! terminal or a pipe only where that was the point.
//!
//! An engine here holds no secret. The file names the variable that signs a
//! decision, and this machine holds it.

use anyhow::Result;
use clap::Subcommand;
use serde::Serialize;

use crate::api::v1::{Agent, AgentSecret};
use crate::cli::env_value;
use crate::cli::target::target;
use crate::manifest::AgentSection;

use super::context::Context;
use super::print;
use super::project_config::ProjectConfig;
use super::ProjectScope;

#[derive(Subcommand)]
pub enum AgentsCommand {
    /// List the agents this project declares.
    #[command(name = "list", visible_alias = "ls")]
    List {
        #[command(flatten)]
        scope: ProjectScope,
    },
    /// Show one agent. Never its signing secret; see `subs agents secret`.
    Show {
        agent_id: String,
        #[command(flatten)]
        scope: ProjectScope,
    },
    /// Print the signing secret for one agent, for a worker to verify with.
    Secret {
        agent_id: String,
        #[command(flatten)]
        scope: ProjectScope,
    },
    /// Mint a new signing secret for one agent. The old one stops working as
    /// soon as this returns.
    RotateSecret {
        agent_id: String,
        #[command(flatten)]
        scope: ProjectScope,
    },
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Rotated<'a> {
    rotated: bool,
    id: &'a str,
    signing_secret: String,
}

pub async fn run(command: AgentsCommand) -> Result<()> {
    match command {
        AgentsCommand::List { scope } => match target(&scope.globals)?.here() {
            Some(config) => list_here(&config, &scope),
            None => list(scope).await,
        },
        AgentsCommand::Show { agent_id, scope } => match target(&scope.globals)?.here() {
            Some(config) => show_here(&agent_id, &config, &scope),
            None => show(agent_id, scope).await,
        },
        AgentsCommand::Secret { agent_id, scope } => match target(&scope.globals)?.here() {
            Some(config) => secret_here(&agent_id, &config),
            None => secret(agent_id, scope).await,
        },
        AgentsCommand::RotateSecret { agent_id, scope } => match target(&scope.globals)?.here() {
            Some(config) => rotate_here(&agent_id, &config),
            None => rotate(agent_id, scope).await,
        },
    }
}

/// The columns the deployment sends, read from the file.
fn declared<'a>(id: &'a str, section: &'a AgentSection) -> [String; 5] {
    let hosting = section
        .worker
        .clone()
        .unwrap_or_else(|| "engine".to_string());
    // An agent that names no variable does not sign. That is a dash, not a
    // missing key.
    let secret = match &section.signing_secret_env {
        None => "-".to_string(),
        Some(var) => match env_value(var).is_some() {
            true => format!("${var}"),
            false => format!("${var} (not set)"),
        },
    };
    [
        id.to_string(),
        section.llm.clone().unwrap_or_else(|| "-".into()),
        section.model.clone().unwrap_or_else(|| "-".into()),
        hosting,
        secret,
    ]
}

fn section<'a>(agent_id: &str, config: &'a ProjectConfig) -> Result<&'a AgentSection> {
    config.agent.get(agent_id).ok_or_else(|| {
        anyhow::anyhow!(
            "no [agent.{agent_id}] in subs.toml. Declared: {}",
            crate::worker::directory::declared(&config.agent_ids())
        )
    })
}

fn list_here(config: &ProjectConfig, scope: &ProjectScope) -> Result<()> {
    if scope.globals.json {
        let agents: Vec<serde_json::Value> = config
            .agent
            .iter()
            .map(|(id, section)| {
                let [id, llm, model, hosting, secret] = declared(id, section);
                serde_json::json!({
                    "id": id,
                    "llm": llm,
                    "model": model,
                    "hosting": hosting,
                    "secret": secret,
                })
            })
            .collect();
        return print::json(&agents);
    }

    let columns = [
        print::Column::left("ID"),
        print::Column::left("LLM"),
        print::Column::left("MODEL"),
        print::Column::left("HOSTING"),
        print::Column::left("SECRET"),
    ];
    let rows: Vec<Vec<String>> = config
        .agent
        .iter()
        .map(|(id, section)| declared(id, section).to_vec())
        .collect();
    print::table(&columns, &rows);
    Ok(())
}

fn show_here(agent_id: &str, config: &ProjectConfig, scope: &ProjectScope) -> Result<()> {
    let section = section(agent_id, config)?;
    let [id, llm, model, hosting, secret] = declared(agent_id, section);

    if scope.globals.json {
        return print::json(&serde_json::json!({
            "id": id,
            "llm": llm,
            "model": model,
            "hosting": hosting,
            "secret": secret,
        }));
    }

    println!("id:       {id}");
    println!("hosting:  {hosting}");
    if section.llm.is_some() {
        println!("llm:      {llm}");
    }
    if section.model.is_some() {
        println!("model:    {model}");
    }
    println!("secret:   {secret}");
    Ok(())
}

/// There is no secret to fetch here. Says which variable holds it.
fn secret_here(agent_id: &str, config: &ProjectConfig) -> Result<()> {
    let section = section(agent_id, config)?;
    let Some(var) = &section.signing_secret_env else {
        anyhow::bail!(
            "[agent.{agent_id}] names no `signing_secret_env`, so an engine here sends its \
             decisions unsigned. Name a variable to sign them."
        );
    };
    match env_value(var) {
        Some(_) => anyhow::bail!(
            "an engine here signs [agent.{agent_id}] with ${var}, which this machine's \
             environment holds — this command does not print it.\n  echo ${var}"
        ),
        None => anyhow::bail!(
            "[agent.{agent_id}] signs with ${var}, which holds nothing.\n  export {var}=..."
        ),
    }
}

fn rotate_here(agent_id: &str, config: &ProjectConfig) -> Result<()> {
    let section = section(agent_id, config)?;
    let Some(var) = &section.signing_secret_env else {
        anyhow::bail!(
            "[agent.{agent_id}] names no `signing_secret_env`, so an engine here sends its \
             decisions unsigned. There is nothing to rotate."
        );
    };
    anyhow::bail!(
        "there is no secret here to rotate: an engine here signs [agent.{agent_id}] with \
         ${var}, so rotating it means setting a new value and restarting.\n  export {var}=..."
    )
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
            // A secret exists for exactly the worker-hosted agents.
            let secret = match a.worker_url.is_some() {
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
    Ok(())
}

/// The secret alone, on stdout: what a worker's environment needs, pipeable.
async fn secret(agent_id: String, scope: ProjectScope) -> Result<()> {
    let (ctx, project) = Context::from_project(&scope).await?;
    let secret: AgentSecret = ctx
        .client
        .get(&format!(
            "/api/v1/projects/{project}/agents/{agent_id}/secret"
        ))
        .await?;

    if scope.globals.json {
        return print::json(&secret);
    }

    println!("{}", secret.signing_secret);
    Ok(())
}

async fn rotate(agent_id: String, scope: ProjectScope) -> Result<()> {
    let (ctx, project) = Context::from_project(&scope).await?;
    let rotated: AgentSecret = ctx
        .client
        .post_json(
            &format!("/api/v1/projects/{project}/agents/{agent_id}/rotate-secret"),
            &serde_json::json!({}),
        )
        .await?;

    let secret = rotated.signing_secret;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn config(body: &str) -> ProjectConfig {
        toml::from_str(body).unwrap()
    }

    const WORKER_AGENT: &str = "[llm.byo]\ntype = \"worker\"\n\
         [agent.support]\nllm = \"byo\"\nmodel = \"m\"\nworker = \"https://w.test\"\n\
         signing_secret_env = \"NOT_SET_SECRET\"\n";

    #[test]
    fn an_agent_here_reads_its_row_off_the_file() {
        let config = config(WORKER_AGENT);
        let [id, llm, model, hosting, secret] =
            declared("support", config.agent.get("support").unwrap());
        assert_eq!(id, "support");
        assert_eq!(llm, "byo");
        assert_eq!(model, "m");
        assert_eq!(hosting, "https://w.test");
        assert_eq!(secret, "$NOT_SET_SECRET (not set)");
    }

    #[test]
    fn an_engine_hosted_agent_reports_no_secret() {
        let config =
            config("[llm.l]\ntype = \"anthropic\"\n[agent.a]\nllm = \"l\"\nmodel = \"m\"\n");
        let [_, _, _, hosting, secret] = declared("a", config.agent.get("a").unwrap());
        assert_eq!(hosting, "engine");
        assert_eq!(secret, "-");
    }

    #[test]
    fn the_secret_here_is_named_not_printed() {
        let config = config(WORKER_AGENT);
        let err = secret_here("support", &config).unwrap_err().to_string();
        assert!(err.contains("$NOT_SET_SECRET"), "{err}");
        assert!(err.contains("export NOT_SET_SECRET"), "{err}");

        let err = rotate_here("support", &config).unwrap_err().to_string();
        assert!(err.contains("no secret here to rotate"), "{err}");
        assert!(err.contains("$NOT_SET_SECRET"), "{err}");
    }

    #[test]
    fn an_undeclared_agent_lists_the_declared_ones() {
        let config = config(WORKER_AGENT);
        let err = section("suport", &config).unwrap_err().to_string();
        assert!(err.contains("no [agent.suport]"), "{err}");
        assert!(err.contains("support"), "{err}");
    }
}
