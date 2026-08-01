use anyhow::Result;
use serde::Serialize;

use super::context::Context;
use super::print;
use super::ProjectScope;

#[derive(Debug, Serialize)]
struct OpenResult<'a> {
    project_id: &'a str,
    url: &'a str,
    opened: bool,
}

pub async fn run(project_id: Option<String>, no_browser: bool, scope: ProjectScope) -> Result<()> {
    let scope = ProjectScope {
        project: project_id.or(scope.project.clone()),
        ..scope
    };
    let (ctx, project_id) = Context::from_project(&scope).await?;
    let url = print::admin_url(ctx.client.base_url(), &project_id);

    // --no-interaction implies --no-browser; opening a browser is never
    // appropriate in non-interactive contexts (CI, scripts).
    let no_browser = no_browser || scope.globals.no_interaction;
    if !scope.globals.json {
        println!("{url}");
    }
    let opened = !no_browser && webbrowser::open(&url).is_ok();

    if scope.globals.json {
        return print::json(&OpenResult {
            project_id: &project_id,
            url: &url,
            opened,
        });
    }

    if opened {
        println!("Opened in your browser.");
    }
    Ok(())
}
