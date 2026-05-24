use anyhow::Result;
use serde::Serialize;

use super::context::Context;
use super::print;
use super::AppScope;

#[derive(Debug, Serialize)]
struct OpenResult<'a> {
    app_id: &'a str,
    url: &'a str,
    opened: bool,
}

pub async fn run(app_id: Option<String>, no_browser: bool, scope: AppScope) -> Result<()> {
    let scope = AppScope {
        app: app_id.or(scope.app.clone()),
        ..scope
    };
    let (ctx, app_id) = Context::from_app(&scope).await?;
    let url = print::admin_url(ctx.client.base_url(), &app_id);

    // --no-interaction implies --no-browser; opening a browser is never
    // appropriate in non-interactive contexts (CI, scripts).
    let no_browser = no_browser || scope.globals.no_interaction;
    let opened = !no_browser && webbrowser::open(&url).is_ok();

    if scope.globals.json {
        return print::json(&OpenResult {
            app_id: &app_id,
            url: &url,
            opened,
        });
    }

    if opened {
        println!("Opened {url}");
    } else {
        println!("{url}");
    }
    Ok(())
}
