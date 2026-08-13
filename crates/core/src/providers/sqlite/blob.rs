//! SQLite-backed [`BlobStore`]: the bytes live in the same file as
//! everything else, so a local engine is one database to back up. A hosted
//! deployment implements the trait against an object store instead.

use async_trait::async_trait;
use chrono::Utc;
use rusqlite::OptionalExtension;

use super::SqliteDb;
use crate::runtime::blob::{BlobError, BlobRef, BlobStore, NewBlob};

pub struct SqliteBlobStore {
    db: SqliteDb,
}

impl SqliteBlobStore {
    pub fn new(db: SqliteDb) -> Self {
        Self { db }
    }
}

fn io_err(e: impl std::fmt::Display) -> BlobError {
    BlobError::Io(e.to_string())
}

#[async_trait]
impl BlobStore for SqliteBlobStore {
    async fn put(&self, blob: NewBlob) -> Result<BlobRef, BlobError> {
        let r = BlobRef {
            tenant_id: blob.tenant_id,
            id: uuid::Uuid::now_v7().to_string(),
            mime: blob.mime,
            name: blob.name,
            size: blob.bytes.len() as u64,
        };
        let row = r.clone();
        let writer = self.db.writer.clone();
        tokio::task::spawn_blocking(move || {
            let conn = writer.lock().map_err(io_err)?;
            conn.execute(
                "INSERT INTO blobs (tenant_id, id, mime, name, size, bytes, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                rusqlite::params![
                    row.tenant_id,
                    row.id,
                    row.mime,
                    row.name,
                    row.size,
                    blob.bytes,
                    Utc::now().to_rfc3339(),
                ],
            )
            .map_err(io_err)?;
            Ok(())
        })
        .await
        .map_err(io_err)??;
        Ok(r)
    }

    async fn get(&self, r: &BlobRef) -> Result<Vec<u8>, BlobError> {
        let (tenant_id, id) = (r.tenant_id.clone(), r.id.clone());
        let reader = self.db.reader.clone();
        tokio::task::spawn_blocking(move || {
            let conn = reader.open().map_err(io_err)?;
            conn.query_row(
                "SELECT bytes FROM blobs WHERE tenant_id = ?1 AND id = ?2",
                rusqlite::params![tenant_id, id],
                |row| row.get(0),
            )
            .optional()
            .map_err(io_err)?
            .ok_or(BlobError::NotFound)
        })
        .await
        .map_err(io_err)?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn store(name: &str) -> SqliteBlobStore {
        let path =
            std::env::temp_dir().join(format!("core-blobs-{name}-{}.db", uuid::Uuid::now_v7()));
        let db = SqliteDb::open(path.to_str().unwrap(), Duration::from_secs(5)).unwrap();
        SqliteBlobStore::new(db)
    }

    #[tokio::test]
    async fn put_get_round_trips_and_tenants_stay_separate() {
        let store = store("roundtrip");
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

        // A ref rewritten to a tenant that never stored it resolves to nothing.
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
    async fn every_put_is_its_own_row() {
        let store = store("rows");
        let blob = || NewBlob {
            tenant_id: "t1".into(),
            mime: "image/png".into(),
            name: None,
            bytes: b"x".to_vec(),
        };
        let a = store.put(blob()).await.unwrap();
        let b = store.put(blob()).await.unwrap();
        assert_ne!(a.id, b.id);
        assert_eq!(store.get(&a).await.unwrap(), b"x");
        assert_eq!(store.get(&b).await.unwrap(), b"x");
    }
}
