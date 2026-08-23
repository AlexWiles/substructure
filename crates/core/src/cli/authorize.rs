use anyhow::{bail, Result};

use super::cloud::project_config::ProjectConfig;
use super::cloud::{llm, print, ProjectScope};
use super::connections;
use crate::connectors::registry::{AuthKind, ConnectionPath, ConnectionSpec};

enum Target {
    Connection(ConnectionSpec),
    Block(String),
}

fn target(path: Option<String>, cfg: &ProjectConfig) -> Result<Target> {
    if let Some(block) = path.as_deref().and_then(|p| p.strip_prefix("llm.")) {
        return Ok(Target::Block(block.to_string()));
    }
    let connections = cfg.resolved_connections()?;
    let parsed = match &path {
        Some(written) => match ConnectionPath::parse(written) {
            Some(path) => Some(path),
            None => bail!(
                "`{written}` names nothing. Declared: {}",
                connections::list(&connections)
            ),
        },
        None => None,
    };
    Ok(Target::Connection(connections::pick(&connections, parsed)?))
}

#[derive(Debug, clap::Args)]
pub struct AuthCommand {
    /// Where it is declared: `mcp.sentry`, `plugin.reggu.mcp.code`,
    /// `llm.openrouter`. Optional when the file declares exactly one
    /// connection.
    pub path: Option<String>,
    /// Read the credential from this environment variable instead of stdin.
    #[arg(long, value_name = "VAR")]
    pub env: Option<String>,
    /// Print the authorization URL instead of opening a browser.
    #[arg(long)]
    pub no_browser: bool,
    #[command(flatten)]
    pub scope: ProjectScope,
}

#[derive(Debug, clap::Args)]
pub struct RevokeCommand {
    /// Where it is declared. Optional when the file declares exactly one
    /// connection.
    pub path: Option<String>,
    #[command(flatten)]
    pub scope: ProjectScope,
}

#[derive(Debug, clap::Args)]
pub struct ListCommand {
    #[command(flatten)]
    pub scope: ProjectScope,
}

pub async fn auth(cmd: AuthCommand) -> Result<()> {
    let AuthCommand {
        path,
        env,
        no_browser,
        scope,
    } = cmd;
    let cfg = connections::environment(&scope.globals)?;

    let spec = match target(path, &cfg)? {
        Target::Block(block) => {
            if no_browser {
                bail!("`llm.{block}` takes a key rather than a consent, so --no-browser says nothing.");
            }
            return llm::set_key_at(block, env, scope).await;
        }
        Target::Connection(spec) => spec,
    };

    let path = &spec.path;
    match spec.decl.auth {
        Some(AuthKind::None) => {
            println!("`{path}` declares `auth = \"none\"`, so it needs no credential.");
            Ok(())
        }
        Some(AuthKind::Token) => {
            if no_browser {
                bail!(
                    "`{path}` declares `auth = \"token\"`, so it is typed rather than consented \
                     to. --no-browser says nothing here."
                );
            }
            match cfg.remote.is_none() {
                true => connections::set_token_local(&spec, env, &scope.globals, &cfg).await,
                false => connections::set_token_remote(&spec, env, scope).await,
            }
        }
        Some(AuthKind::Oauth) | None => {
            if env.is_some() {
                bail!("{}", no_variable(&spec));
            }
            let no_browser = no_browser || scope.globals.no_interaction;
            match cfg.remote.is_none() {
                true => connections::login_local(&spec, no_browser, &cfg).await,
                false => connections::login_remote(&spec, no_browser, scope).await,
            }
        }
    }
}

fn no_variable(spec: &ConnectionSpec) -> String {
    let path = &spec.path;
    match spec.decl.auth {
        Some(AuthKind::Oauth) => format!(
            "`{path}` declares `auth = \"oauth\"`, so its credential comes from consent rather \
             than a variable. Drop --env, or write `auth = \"token\"` on [{path}]."
        ),
        _ => format!(
            "`{path}` declares no `auth`, so a token set here would never be sent. Write \
             `auth = \"token\"` on [{path}] first."
        ),
    }
}

pub async fn revoke(cmd: RevokeCommand) -> Result<()> {
    let RevokeCommand { path, scope } = cmd;
    let cfg = connections::environment(&scope.globals)?;
    let path = match target(path, &cfg)? {
        Target::Block(block) => return llm::delete_key_at(block, scope).await,
        Target::Connection(spec) => spec.path,
    };
    match cfg.remote.is_none() {
        true => connections::delete_token_local(&path, &cfg).await,
        false => connections::delete_token_remote(&path, scope).await,
    }
}

pub async fn list(cmd: ListCommand) -> Result<()> {
    let ListCommand { scope } = cmd;
    let cfg = connections::environment(&scope.globals)?;
    let here = cfg.remote.is_none();

    let mut rows = match here {
        true => connections::list_local(&cfg).await?,
        false => connections::list_remote(&scope, &cfg).await?,
    };
    rows.extend(llm::rows(&cfg, &scope, here).await?);
    rows.sort_by(|a, b| a.path.cmp(&b.path));

    if rows.is_empty() {
        bail!("substructure.toml declares no connections and no `[llm.*]` blocks");
    }
    if scope.globals.json {
        return print::json(&rows);
    }

    let columns = [
        print::Column::left("PATH"),
        print::Column::left("WHAT"),
        print::Column::left("CREDENTIAL"),
    ];
    let table: Vec<Vec<String>> = rows
        .iter()
        .map(|r| vec![r.path.clone(), r.what.clone(), r.credential.clone()])
        .collect();
    print::table(&columns, &table);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::cloud::{CloudGlobals, ProjectScope};

    fn globals() -> CloudGlobals {
        CloudGlobals {
            no_interaction: true,
            ..Default::default()
        }
    }

    fn scope_at(dir: &std::path::Path, body: &str) -> ProjectScope {
        let path = dir.join(crate::cli::cloud::project_config::FILENAME);
        std::fs::write(&path, body).unwrap();
        ProjectScope {
            org: None,
            project: None,
            globals: CloudGlobals {
                config: Some(path.clone()),
                ..globals()
            },
        }
    }

    fn tmpdir() -> std::path::PathBuf {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("subs-auth-test-{nanos}-{seq}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn cmd(scope: ProjectScope, path: &str, env: Option<&str>, no_browser: bool) -> AuthCommand {
        AuthCommand {
            path: Some(path.to_string()),
            env: env.map(str::to_string),
            no_browser,
            scope,
        }
    }

    #[tokio::test]
    async fn a_flag_the_method_cannot_use_is_refused() {
        let dir = tmpdir();
        let oauth = scope_at(
            &dir,
            "[mcp.sentry]\nurl = \"https://mcp.sentry.dev/mcp\"\nauth = \"oauth\"\n",
        );
        let err = auth(cmd(oauth, "mcp.sentry", Some("PATH"), false))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("comes from consent"), "{err}");

        let dir = tmpdir();
        let token = scope_at(
            &dir,
            "[mcp.sentry]\nurl = \"https://mcp.sentry.dev/mcp\"\nauth = \"token\"\n",
        );
        let err = auth(cmd(token, "mcp.sentry", None, true))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("--no-browser says nothing"), "{err}");
    }

    #[tokio::test]
    async fn a_token_for_a_connection_that_declares_none_is_refused() {
        let dir = tmpdir();
        let bare = scope_at(&dir, "[mcp.sentry]\nurl = \"https://mcp.sentry.dev/mcp\"\n");
        let err = auth(cmd(bare, "mcp.sentry", Some("PATH"), false))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("`auth = \"token\"` on [mcp.sentry]"), "{err}");
    }

    #[tokio::test]
    async fn a_connection_that_needs_no_credential_says_so() {
        let dir = tmpdir();
        let none = scope_at(
            &dir,
            "[mcp.public]\nurl = \"https://public.test/mcp\"\nauth = \"none\"\n",
        );
        auth(cmd(none, "mcp.public", None, false)).await.unwrap();
    }

    #[tokio::test]
    async fn a_plugins_server_is_authorized_by_its_path() {
        let dir = tmpdir();
        std::fs::create_dir_all(dir.join("plugin")).unwrap();
        std::fs::write(
            dir.join("plugin/plugin.json"),
            r#"{ "name": "reggu.admin", "version": "1.0.0", "description": "Admin." }"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("plugin/mcp.json"),
            r#"{ "mcpServers": { "admin": {
                "type": "streamable-http", "url": "https://reggu.test/mcp" } } }"#,
        )
        .unwrap();
        let scope = scope_at(
            &dir,
            "[plugin.reggu]\npath = \"./plugin\"\n[remote]\nurl = \"https://api.test\"\n",
        );

        let err = auth(cmd(scope, "plugin.reggu.mcp.admin", Some("PATH"), false))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("`auth = \"token\"` on [plugin.reggu.mcp.admin]"),
            "{err}"
        );
    }
}
