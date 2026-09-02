use anyhow::{bail, Result};
use serde::Deserialize;

use super::address::Address;
use super::cloud::context::Context;
use super::cloud::{print, project_config, ProjectScope};
use super::target::{target, Target};
use super::DEFAULT_TENANT;
use crate::providers::sqlite::{SqliteDb, SqliteSecretStore};
use crate::runtime::secret::{get_or_mint, mint_secret, SecretPath, SecretStore};

#[derive(Debug, clap::Args)]
pub struct SecretCommand {
    /// The path whose secret to print: `worker.main`. Omit to list every
    /// secret-bearing path without printing a value.
    pub path: Option<String>,
    /// Mint a new secret in place of the current one. The engine signs with
    /// it from its next push.
    #[arg(long)]
    pub rotate: bool,
    #[command(flatten)]
    pub scope: ProjectScope,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WorkerSecretView {
    signing_secret: String,
}

#[derive(Debug, Deserialize)]
struct WorkerRow {
    id: String,
}

/// Which store holds the secret is decided the same way every other command
/// decides what it acts on: a `[remote]` means the deployment owns it. Reading
/// the local store for a deployed project would hand back a secret nothing
/// signs with — see the `Target::Here` arm for the case where it is real.
pub async fn run(cmd: SecretCommand) -> Result<()> {
    match target(&cmd.scope.globals)? {
        Target::Deployment => run_deployment(cmd).await,
        Target::Here(cfg) => run_here(cmd, *cfg).await,
    }
}

async fn run_deployment(cmd: SecretCommand) -> Result<()> {
    let (ctx, project) = Context::from_project(&cmd.scope).await?;

    let Some(path) = cmd.path else {
        if cmd.rotate {
            bail!("--rotate needs a path: `subs secret worker.<id> --rotate`");
        }
        let workers: Vec<WorkerRow> = ctx
            .client
            .get(&format!("/api/v1/projects/{project}/workers"))
            .await?;
        return list_deployment(workers, cmd.scope.globals.json);
    };

    let id = match Address::parse(&path) {
        Some(Address::Worker(id)) => id,
        Some(_) => {
            bail!("`{path}` holds no minted secret; for a credential, use `subs auth {path}`")
        }
        None => bail!("`{path}` holds no minted secret. Today only a `[worker.*]` block does."),
    };

    let view: WorkerSecretView = match cmd.rotate {
        false => {
            ctx.client
                .get(&format!("/api/v1/projects/{project}/workers/{id}/secret"))
                .await?
        }
        true => {
            let view = ctx
                .client
                .post_empty(&format!(
                    "/api/v1/projects/{project}/workers/{id}/rotate-secret"
                ))
                .await?;
            eprintln!("Rotated. The engine signs with it from its next push.");
            view
        }
    };

    if cmd.scope.globals.json {
        return print::json(&serde_json::json!({
            "path": SecretPath::Worker(id).to_string(),
            "secret": view.signing_secret,
        }));
    }
    println!("{}", view.signing_secret);
    Ok(())
}

fn list_deployment(workers: Vec<WorkerRow>, json: bool) -> Result<()> {
    let rows: Vec<(String, bool)> = workers
        .into_iter()
        .map(|w| (SecretPath::Worker(w.id).to_string(), true))
        .collect();
    if json {
        let rows: Vec<serde_json::Value> = rows
            .iter()
            .map(|(path, set)| serde_json::json!({ "path": path, "set": set }))
            .collect();
        return print::json(&rows);
    }
    if rows.is_empty() {
        println!("This deployment declares no [worker.*], so nothing holds a minted secret.");
        return Ok(());
    }
    let columns = [print::Column::left("PATH"), print::Column::left("SECRET")];
    let rows: Vec<Vec<String>> = rows
        .into_iter()
        .map(|(path, _)| vec![path, "minted".to_string()])
        .collect();
    print::table(&columns, &rows);
    Ok(())
}

async fn run_here(cmd: SecretCommand, cfg: project_config::ProjectConfig) -> Result<()> {
    let Some(path) = cmd.path else {
        if cmd.rotate {
            bail!("--rotate needs a path: `subs secret worker.<id> --rotate`");
        }
        return list(&cfg, cmd.scope.globals.json).await;
    };

    let id = match Address::parse(&path) {
        Some(Address::Worker(id)) => id,
        Some(_) => {
            bail!("`{path}` holds no minted secret; for a credential, use `subs auth {path}`")
        }
        None => bail!(
            "`{path}` holds no minted secret. Today only a `[worker.*]` block does: {}",
            declared(&cfg)
        ),
    };
    if !cfg.worker.contains_key(&id) {
        bail!(
            "no [worker.{id}] in subs.toml. Declared: {}",
            declared(&cfg)
        );
    }

    let store = open_store(&cfg)?;
    let r = SecretPath::Worker(id).secret_ref();
    let secret = match cmd.rotate {
        false => get_or_mint(&store, DEFAULT_TENANT, &r).await?,
        true => {
            let minted = mint_secret();
            store.put(DEFAULT_TENANT, &r, minted.as_bytes()).await?;
            eprintln!("Rotated. The engine signs with it from its next push.");
            minted
        }
    };
    println!("{secret}");
    Ok(())
}

async fn list(cfg: &project_config::ProjectConfig, json: bool) -> Result<()> {
    let store = open_store(cfg)?;
    let mut rows = Vec::new();
    for id in cfg.worker.keys() {
        let path = SecretPath::Worker(id.clone());
        let set = store
            .get(DEFAULT_TENANT, &path.secret_ref())
            .await?
            .is_some();
        rows.push((path.to_string(), set));
    }
    if json {
        let rows: Vec<serde_json::Value> = rows
            .iter()
            .map(|(path, set)| serde_json::json!({ "path": path, "set": set }))
            .collect();
        return print::json(&rows);
    }
    if rows.is_empty() {
        println!("subs.toml declares no [worker.*], so nothing holds a minted secret.");
        return Ok(());
    }
    let columns = [print::Column::left("PATH"), print::Column::left("SECRET")];
    let rows: Vec<Vec<String>> = rows
        .into_iter()
        .map(|(path, set)| {
            vec![
                path,
                match set {
                    true => "minted".to_string(),
                    false => "on first use".to_string(),
                },
            ]
        })
        .collect();
    print::table(&columns, &rows);
    Ok(())
}

fn declared(cfg: &project_config::ProjectConfig) -> String {
    crate::copy::declared(
        cfg.worker
            .keys()
            .map(|id| SecretPath::Worker(id.clone()).to_string()),
    )
}

fn open_store(cfg: &project_config::ProjectConfig) -> Result<SqliteSecretStore> {
    let path = cfg.db_path();
    project_config::ensure_parent(&path)?;
    let db = SqliteDb::open(&path, std::time::Duration::from_secs(5))?;
    Ok(SqliteSecretStore::new(db, super::connections::cipher()?))
}
