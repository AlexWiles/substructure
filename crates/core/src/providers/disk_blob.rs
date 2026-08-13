//! Disk-backed [`BlobStore`]: `{root}/{tenant}/{id}` plus a `.meta.json`
//! sidecar. Local engines use this; a hosted deployment implements the trait
//! against an object store.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use percent_encoding::{utf8_percent_encode, NON_ALPHANUMERIC};

use crate::runtime::blob::{BlobError, BlobRef, BlobStore, NewBlob};

pub struct DiskBlobStore {
    root: PathBuf,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Meta {
    mime: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    size: u64,
}

impl DiskBlobStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn dir(&self, tenant_id: &str) -> PathBuf {
        // Tenant ids come from config; escape them out of caution.
        self.root
            .join(utf8_percent_encode(tenant_id, NON_ALPHANUMERIC).to_string())
    }

    fn paths(&self, tenant_id: &str, id: &str) -> (PathBuf, PathBuf) {
        let dir = self.dir(tenant_id);
        (dir.join(id), dir.join(format!("{id}.meta.json")))
    }
}

fn io_err(e: impl std::fmt::Display) -> BlobError {
    BlobError::Io(e.to_string())
}

async fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), BlobError> {
    let tmp = path.with_extension("tmp");
    tokio::fs::write(&tmp, bytes).await.map_err(io_err)?;
    tokio::fs::rename(&tmp, path).await.map_err(io_err)
}

#[async_trait]
impl BlobStore for DiskBlobStore {
    async fn put(&self, blob: NewBlob) -> Result<BlobRef, BlobError> {
        let r = BlobRef {
            tenant_id: blob.tenant_id,
            id: uuid::Uuid::now_v7().to_string(),
            mime: blob.mime,
            name: blob.name,
            size: blob.bytes.len() as u64,
        };
        let (data, meta) = self.paths(&r.tenant_id, &r.id);
        tokio::fs::create_dir_all(data.parent().unwrap())
            .await
            .map_err(io_err)?;
        write_atomic(&data, &blob.bytes).await?;
        let meta_json = serde_json::to_vec(&Meta {
            mime: r.mime.clone(),
            name: r.name.clone(),
            size: r.size,
        })
        .map_err(io_err)?;
        write_atomic(&meta, &meta_json).await?;
        Ok(r)
    }

    async fn get(&self, r: &BlobRef) -> Result<Vec<u8>, BlobError> {
        let (data, _) = self.paths(&r.tenant_id, &r.id);
        match tokio::fs::read(&data).await {
            Ok(bytes) => Ok(bytes),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(BlobError::NotFound),
            Err(e) => Err(io_err(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn put_get_round_trips_and_tenants_stay_separate() {
        let dir = tempfile::tempdir().unwrap();
        let store = DiskBlobStore::new(dir.path());
        let bytes = b"same bytes".to_vec();
        let a = store
            .put(NewBlob {
                tenant_id: "t1".into(),
                mime: "image/png".into(),
                name: Some("a.png".into()),
                bytes: bytes.clone(),
            })
            .await
            .unwrap();
        let b = store
            .put(NewBlob {
                tenant_id: "t2".into(),
                mime: "image/png".into(),
                name: None,
                bytes: bytes.clone(),
            })
            .await
            .unwrap();
        assert_ne!(a.id, b.id);
        assert_eq!(store.get(&a).await.unwrap(), bytes);
        assert_eq!(store.get(&b).await.unwrap(), bytes);

        // A ref pointing at a tenant that never stored it resolves to nothing.
        let foreign = BlobRef {
            tenant_id: "t3".into(),
            ..a.clone()
        };
        assert!(matches!(
            store.get(&foreign).await,
            Err(BlobError::NotFound)
        ));
    }

    #[tokio::test]
    async fn every_put_is_its_own_entry() {
        let dir = tempfile::tempdir().unwrap();
        let store = DiskBlobStore::new(dir.path());
        let blob = || NewBlob {
            tenant_id: "t1".into(),
            mime: "image/png".into(),
            name: None,
            bytes: b"x".to_vec(),
        };
        let a = store.put(blob()).await.unwrap();
        let b = store.put(blob()).await.unwrap();
        assert_ne!(a.id, b.id);
        let files: Vec<_> = std::fs::read_dir(dir.path().join("t1"))
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| !e.file_name().to_string_lossy().ends_with(".meta.json"))
            .collect();
        assert_eq!(files.len(), 2);
    }
}
