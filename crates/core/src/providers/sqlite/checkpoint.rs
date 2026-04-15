use async_trait::async_trait;
use chrono::Utc;
use rusqlite::{Connection, OptionalExtension};

use crate::event_store::StoreError;
use crate::processor::{CheckpointError, ProcessorCheckpoint, ProcessorCheckpointStore};

use super::{parse_dt, SqliteDb};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS projection_checkpoints (
    projection_name TEXT NOT NULL,
    shard_id        INTEGER NOT NULL,
    position        INTEGER NOT NULL,
    version         INTEGER NOT NULL,
    owner_id        TEXT,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (projection_name, shard_id)
);
";

pub struct SqliteCheckpointStore {
    db: SqliteDb,
}

impl SqliteCheckpointStore {
    pub fn new(db: SqliteDb) -> Result<Self, StoreError> {
        db.run_schema(SCHEMA)?;
        Ok(Self { db })
    }
}

#[async_trait]
impl ProcessorCheckpointStore for SqliteCheckpointStore {
    async fn load_checkpoint(
        &self,
        projection: &str,
        shard_id: u32,
    ) -> Result<ProcessorCheckpoint, CheckpointError> {
        let projection = projection.to_string();
        let reader = self.db.reader.clone();
        tokio::task::spawn_blocking(move || {
            let conn = reader
                .open()
                .map_err(|e| CheckpointError::Message(e.to_string()))?;
            do_load_checkpoint(&conn, &projection, shard_id)
        })
        .await
        .map_err(|e| CheckpointError::Message(e.to_string()))?
    }

    async fn compare_and_set_checkpoint(
        &self,
        projection: &str,
        shard_id: u32,
        expected_version: u64,
        new_position: u64,
        owner_id: Option<&str>,
    ) -> Result<bool, CheckpointError> {
        let projection = projection.to_string();
        let owner_id = owner_id.map(str::to_string);
        let writer = self.db.writer.clone();
        tokio::task::spawn_blocking(move || {
            let mut conn = writer
                .lock()
                .map_err(|e| CheckpointError::Message(e.to_string()))?;
            do_compare_and_set(
                &mut conn,
                &projection,
                shard_id,
                expected_version,
                new_position,
                owner_id.as_deref(),
            )
        })
        .await
        .map_err(|e| CheckpointError::Message(e.to_string()))?
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn do_load_checkpoint(
    conn: &Connection,
    projection: &str,
    shard_id: u32,
) -> Result<ProcessorCheckpoint, CheckpointError> {
    let row = conn
        .query_row(
            "SELECT position, version, updated_at FROM projection_checkpoints WHERE projection_name = ?1 AND shard_id = ?2",
            rusqlite::params![projection, shard_id],
            |row| {
                Ok((
                    row.get::<_, u64>(0)?,
                    row.get::<_, u64>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        )
        .optional()
        .map_err(|e| CheckpointError::Message(e.to_string()))?;

    match row {
        Some((position, version, updated_at)) => Ok(ProcessorCheckpoint {
            position,
            version,
            updated_at: parse_dt(&updated_at).unwrap_or_else(Utc::now),
        }),
        None => Ok(ProcessorCheckpoint {
            position: 0,
            version: 0,
            updated_at: Utc::now(),
        }),
    }
}

fn do_compare_and_set(
    conn: &mut Connection,
    projection: &str,
    shard_id: u32,
    expected_version: u64,
    new_position: u64,
    owner_id: Option<&str>,
) -> Result<bool, CheckpointError> {
    let tx = conn
        .transaction()
        .map_err(|e| CheckpointError::Message(e.to_string()))?;

    if expected_version == 0 {
        tx.execute(
            "INSERT OR IGNORE INTO projection_checkpoints (projection_name, shard_id, position, version, owner_id, updated_at)
             VALUES (?1, ?2, 0, 0, NULL, ?3)",
            rusqlite::params![projection, shard_id, Utc::now().to_rfc3339()],
        )
        .map_err(|e| CheckpointError::Message(e.to_string()))?;
    }

    let updated = tx
        .execute(
            "UPDATE projection_checkpoints
             SET position = ?1,
                 version = version + 1,
                 owner_id = ?2,
                 updated_at = ?3
             WHERE projection_name = ?4
               AND shard_id = ?5
               AND version = ?6",
            rusqlite::params![
                new_position,
                owner_id,
                Utc::now().to_rfc3339(),
                projection,
                shard_id,
                expected_version,
            ],
        )
        .map_err(|e| CheckpointError::Message(e.to_string()))?;

    tx.commit()
        .map_err(|e| CheckpointError::Message(e.to_string()))?;

    Ok(updated == 1)
}
