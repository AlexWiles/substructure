//! Plugin content, sent as a unit and named by its hash.
//!
//! A plugin is the only thing in a project that is bytes rather than a
//! declaration, so it travels on its own: the config names a plugin by hash,
//! and this is what puts the thing the hash names where the deployment can
//! read it. Apply does both, in that order.

use anyhow::{Context as _, Result};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;

use crate::api::v1::{PluginBinary, PluginHeads, PluginPush, PluginPushed};
use crate::manifest::{Manifest, ResolvedPlugins};
use crate::plugins::Pending;

use super::context::Context;

/// Send each plugin the deployment does not already hold, and answer with what
/// was sent. An unchanged directory sends nothing.
pub async fn push_missing(
    ctx: &Context,
    project_id: &str,
    manifest: &Manifest,
    resolved: &ResolvedPlugins,
) -> Result<Vec<PluginPushed>> {
    if manifest.plugin.is_empty() {
        return Ok(Vec::new());
    }
    // A deployment that does not answer is one that holds none; the push
    // below reports what it thinks of that with its own status.
    let held = ctx
        .client
        .get::<PluginHeads>(&format!("/api/v1/projects/{project_id}/plugins"))
        .await
        .map(|h| h.plugins)
        .unwrap_or_default();

    let mut pushed = Vec::new();
    for (id, spec) in &manifest.plugin {
        let (Some(bundle), Some(hash)) = (&spec.bundle, &spec.hash) else {
            continue;
        };
        if held.iter().any(|h| &h.id == id && &h.hash == hash) {
            continue;
        }
        let body = PluginPush {
            hash: hash.clone(),
            bundle: serde_json::to_value(bundle)?,
            binaries: binaries(resolved.pending.get(id)),
        };
        let sent: PluginPushed = ctx
            .client
            .put_json(
                &format!("/api/v1/projects/{project_id}/plugins/{id}"),
                &body,
            )
            .await
            .with_context(|| format!("[plugin.{id}]: the deployment did not take the plugin"))?;
        pushed.push(PluginPushed {
            id: id.clone(),
            hash: hash.clone(),
            binaries: body.binaries.len().max(sent.binaries),
        });
    }
    Ok(pushed)
}

fn binaries(pending: Option<&Vec<Pending>>) -> Vec<PluginBinary> {
    pending
        .into_iter()
        .flatten()
        .map(|file| PluginBinary {
            skill: file.skill.clone(),
            path: file.path.clone(),
            mime: file.mime.clone(),
            bytes: BASE64.encode(&file.bytes),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_binary_travels_beside_the_bundle_it_belongs_to() {
        let pending = vec![Pending {
            skill: "respond".to_string(),
            path: "references/map.png".to_string(),
            mime: "image/png".to_string(),
            bytes: vec![0x89, b'P', b'N', b'G'],
        }];
        let sent = binaries(Some(&pending));
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].skill, "respond");
        assert_eq!(sent[0].path, "references/map.png");
        assert_eq!(sent[0].mime, "image/png");
        assert_eq!(
            BASE64.decode(&sent[0].bytes).unwrap(),
            [0x89, b'P', b'N', b'G'],
            "the bytes arrive as they left"
        );
    }

    #[test]
    fn a_plugin_with_no_binaries_sends_none() {
        assert!(binaries(None).is_empty());
    }
}
