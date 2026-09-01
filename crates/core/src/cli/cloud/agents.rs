use anyhow::Result;
use clap::Subcommand;

use crate::api::v1::Agent;
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
    /// Show one agent. Its worker's secret lives at `subs secret worker.<id>`.
    Show {
        agent_id: String,
        #[command(flatten)]
        scope: ProjectScope,
    },
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
    }
}

fn declared<'a>(id: &'a str, section: &'a AgentSection) -> [String; 4] {
    let hosting = match &section.worker {
        None => "engine".to_string(),
        Some(worker) => format!("worker.{worker}"),
    };
    [
        id.to_string(),
        section.llm.clone().unwrap_or_else(|| "-".into()),
        section.model.clone().unwrap_or_else(|| "-".into()),
        hosting,
    ]
}

fn section<'a>(agent_id: &str, config: &'a ProjectConfig) -> Result<&'a AgentSection> {
    config.agent.get(agent_id).ok_or_else(|| {
        anyhow::anyhow!(
            "no [agent.{agent_id}] in subs.toml. Declared: {}",
            crate::copy::declared(config.agent_ids())
        )
    })
}

fn list_here(config: &ProjectConfig, scope: &ProjectScope) -> Result<()> {
    if scope.globals.json {
        let agents: Vec<serde_json::Value> = config
            .agent
            .iter()
            .map(|(id, section)| {
                let [id, llm, model, hosting] = declared(id, section);
                serde_json::json!({
                    "id": id,
                    "llm": llm,
                    "model": model,
                    "hosting": hosting,
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
    let [id, llm, model, hosting] = declared(agent_id, section);

    if scope.globals.json {
        return print::json(&serde_json::json!({
            "id": id,
            "llm": llm,
            "model": model,
            "hosting": hosting,
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
    Ok(())
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
            let hosting = a
                .worker
                .as_ref()
                .map(|w| format!("worker.{w}"))
                .unwrap_or_else(|| "engine".into());
            vec![a.id.clone(), llm, model, hosting]
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
    match &agent.worker {
        Some(w) => println!("hosting:  worker.{w}"),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn config(body: &str) -> ProjectConfig {
        toml::from_str(body).unwrap()
    }

    const WORKER_AGENT: &str = "[llm.byo]\ntype = \"worker\"\n\
         [worker.main]\nurl = \"https://w.test\"\n\
         [agent.support]\nllm = \"byo\"\nmodel = \"m\"\nworker = \"main\"\n";

    #[test]
    fn an_agent_here_reads_its_row_off_the_file() {
        let config = config(WORKER_AGENT);
        let [id, llm, model, hosting] = declared("support", config.agent.get("support").unwrap());
        assert_eq!(id, "support");
        assert_eq!(llm, "byo");
        assert_eq!(model, "m");
        assert_eq!(hosting, "worker.main");
    }

    #[test]
    fn an_engine_hosted_agent_reports_engine_hosting() {
        let config =
            config("[llm.l]\ntype = \"anthropic\"\n[agent.a]\nllm = \"l\"\nmodel = \"m\"\n");
        let [_, _, _, hosting] = declared("a", config.agent.get("a").unwrap());
        assert_eq!(hosting, "engine");
    }

    #[test]
    fn an_undeclared_agent_lists_the_declared_ones() {
        let config = config(WORKER_AGENT);
        let err = section("suport", &config).unwrap_err().to_string();
        assert!(err.contains("no [agent.suport]"), "{err}");
        assert!(err.contains("support"), "{err}");
    }
}
